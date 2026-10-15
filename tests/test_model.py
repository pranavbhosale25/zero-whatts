from dataclasses import replace
from io import StringIO
import numpy as np
import pytest
from model import (Battery, Tariff, Economics, Scenario, DEFAULT_SCENARIOS, build_load,
                   read_load_csv, optimize_month, analyze, run_study, stable_payback)
from reporting import html_report


def dispatch(load, prices, battery, demand=0, om=0):
    return optimize_month(np.stack([load,load]),np.stack([prices,prices]),np.array([22,8]),
                          battery.capacity_kwh,battery.power_kw,battery,battery.round_trip_efficiency,demand,om)


def test_flat_tariff_no_demand_never_cycles():
    b=Battery()
    result=dispatch(np.full(24,100.),np.full(24,.15),b)
    np.testing.assert_allclose(result['charge'],0,atol=1e-7)
    np.testing.assert_allclose(result['discharge'],0,atol=1e-7)


def test_peak_shaving_matches_energy_balance_closed_form():
    b=Battery(100,100,1,0,1,0,0,2)
    load=np.full(24,100.);load[18]=200
    result=dispatch(load,np.full(24,.1),b,demand=20)
    np.testing.assert_allclose(result['grid'],load.mean(),atol=1e-6)
    assert result['peak']==pytest.approx(2500/24)


def test_round_trip_arbitrage_matches_closed_form():
    b=Battery(100,200,.81,0,1,0,0,1)
    prices=np.full(24,.1);prices[18]=.4
    result=dispatch(np.full(24,200.),prices,b)
    assert result['discharge'][0].sum()==pytest.approx(90)
    assert result['charge'][0].sum()==pytest.approx(100/.9)
    savings=np.sum((result['discharge'][0]-result['charge'][0])*prices)
    assert savings==pytest.approx(90*.4-100/.9*.1)


def test_constraints_and_no_free_initial_energy():
    b=Battery(500,80,.85,.1,.9,.3,.02,.7)
    load=np.array([40.]*6+[100.]*12+[180.]*6)
    prices=np.array([.05]*8+[.10]*8+[.30]*8)
    result=dispatch(load,prices,b,demand=20)
    eta=np.sqrt(b.round_trip_efficiency)
    assert result['grid'].min()>=-1e-7
    assert (result['charge']+result['discharge']).max()<=b.power_kw+1e-7
    assert result['energy'].min()>=b.reserve_soc*b.capacity_kwh-1e-7
    assert result['energy'].max()<=b.max_soc*b.capacity_kwh+1e-7
    np.testing.assert_allclose(np.diff(result['energy']),eta*result['charge']-result['discharge']/eta,atol=1e-7)
    np.testing.assert_allclose(result['energy'][:,0],result['energy'][:,-1],atol=1e-7)
    np.testing.assert_allclose(result['discharge'].sum(axis=1),result['charge'].sum(axis=1)*.85,atol=1e-7)
    assert np.max(result['discharge'].sum(axis=1)/eta)<=500*.6*.7+1e-7
    assert np.max(np.minimum(result['charge'],result['discharge']))<1e-7


@pytest.mark.parametrize('year,days',[(2027,365),(2028,366)])
def test_annual_bills_have_correct_calendar_and_monthly_units(year,days):
    b=Battery()
    e=Economics(years=1,start_year=year)
    r=analyze(np.full(24,100.),[1]*12,1,b,Tariff(.1,.1,demand_rate=15),e,Scenario('Base'))
    y=r['yearly'].iloc[0]
    assert y.baseline_energy==pytest.approx(100*24*days*.1)
    assert y.baseline_demand==pytest.approx(100*15*12)
    assert y.optimized_energy==pytest.approx(y.baseline_energy)
    assert y.optimized_demand==pytest.approx(y.baseline_demand)
    assert y.net_cashflow==pytest.approx(-e.fixed_om_per_kw_year*b.power_kw)


def test_npv_replacement_and_capacity_reset():
    b=Battery(annual_degradation=.05)
    e=Economics(years=4,replacement_year=3,residual_fraction=.1)
    r=analyze(np.full(24,100.),[1]*12,.5,b,Tariff(),e,Scenario('Base'))
    y=r['yearly']
    np.testing.assert_allclose(y.capacity_remaining_pct,[100,95,100,95])
    assert y.iloc[2].replacement==pytest.approx(r['gross_capex']*.5*(1+e.om_escalation)**2)
    expected=-r['net_capex']+sum(y.net_cashflow.to_numpy()/(1+e.discount_rate)**np.arange(1,5))
    assert r['npv']==pytest.approx(expected)
    assert y.iloc[-1].residual==pytest.approx(r['gross_capex']*.1)


def test_unreached_and_stable_payback():
    assert stable_payback([-100,10,10]) is None
    assert stable_payback([-100,120,-80,80])==pytest.approx(2.75)
    assert stable_payback([-100,40,40,40])==pytest.approx(2.5)


def test_small_battery_remains_small_in_all_scenarios():
    b=Battery(100,25)
    results=run_study(np.array([200.]*18+[500.]*6),[1]*12,.5,b,Tariff(),Economics(years=1))
    for r in results:
        assert r['dispatch'].discharge_kw.max()<=25+1e-7
    assert [r['scenario']['name'] for r in results]==['Conservative','Base','Aggressive']


@pytest.mark.parametrize('mutation',[
    {'battery':Battery(reserve_soc=.95)}, {'load':np.zeros(24)},
    {'monthly_factors':[1]*11}, {'tariff':Tariff(offpeak_rate=.3,peak_rate=.1)},
    {'economics':Economics(upfront_incentive=1e9)},
    {'scenarios':[replace(s,efficiency_delta=.3) for s in DEFAULT_SCENARIOS]},
])
def test_bad_inputs_are_rejected(mutation):
    args=dict(load=np.ones(24)*100,monthly_factors=[1]*12,weekend_factor=.5,battery=Battery(),
              tariff=Tariff(),economics=Economics(years=1),scenarios=DEFAULT_SCENARIOS)
    args.update(mutation)
    with pytest.raises(ValueError):
        run_study(**args)


def test_builder_supports_overnight_and_csv_rejects_bad_data():
    load=build_load([{'Rated kW':100,'Load factor':.5,'Start hour':22,'End hour':6}])
    assert load.sum()==400 and load[23]==50 and load[7]==0
    text='hour,load_kw\n'+'\n'.join(f'{h},{h+1}' for h in range(24))
    np.testing.assert_allclose(read_load_csv(StringIO(text)),np.arange(1,25))
    with pytest.raises(ValueError):
        read_load_csv(StringIO(text.replace('23,24','22,24')))
    with pytest.raises(ValueError):
        read_load_csv(StringIO(text.replace('23,24','23,-1')))


def test_export_is_self_contained_and_escapes_user_text():
    results=run_study(np.full(24,100.),[1]*12,.5,Battery(),Tariff(),Economics(years=1))
    report=html_report(results,{'study_name':'<script>alert(1)</script>'})
    assert '<script>' not in report
    assert '&lt;script&gt;' in report
    assert 'Sources and reference values' in report and 'Conservative' in report
