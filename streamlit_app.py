"""Zero Whatts: industrial battery configuration and technoeconomic report."""
from dataclasses import asdict
import calendar
import hashlib
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from model import Battery, Tariff, Economics, Scenario, DEFAULT_SCENARIOS, build_load, read_load_csv, run_study
from ui_help import HELP
from reporting import SOURCES, REFERENCE_VALUES, METHODOLOGY, summary_table, html_report

st.set_page_config(page_title="Zero Whatts | Industrial Storage", page_icon="⚡", layout="wide")
st.markdown("""<style>
.block-container{max-width:1320px;padding-top:2rem;padding-bottom:3rem}
[data-testid="stMetricValue"]{font-size:1.75rem}
</style>""", unsafe_allow_html=True)

COLORS = {"Conservative": "#94A3B8", "Base": "#FF4B4B", "Aggressive": "#2563EB"}


def money(value):
    return f"-${abs(value):,.0f}" if value < 0 else f"${value:,.0f}"


def chart_style(fig, title=None, height=360):
    fig.update_layout(title=title, height=height, template="plotly_white", margin=dict(l=10,r=10,t=45,b=15),
                      font=dict(family="Arial",color="#31333F"), legend=dict(orientation="h",y=1.12),
                      paper_bgcolor="rgba(0,0,0,0)", hovermode="x unified")
    return fig


@st.cache_data(show_spinner=False)
def calculate(snapshot):
    return run_study(snapshot['load_kw'], snapshot['monthly_factors'], snapshot['weekend_factor'],
                     Battery(**snapshot['battery']), Tariff(**snapshot['tariff']), Economics(**snapshot['economics']),
                     [Scenario(**s) for s in snapshot['scenarios']])


st.title("⚡️ Zero Whatts")
st.subheader("Battery Projects Screening for Industrial Loads")
st.markdown("""Steps to use this tool:

1. **Build your study**
2. **Configure scenarios**
3. **View results**
""")

st.subheader("1. Build your study")
load_tab, battery_tab, finance_tab, tariff_tab = st.tabs([
    "Industrial load", "Battery", "Economics", "Utility tariff"])

input_error = None
with load_tab:
    st.markdown("#### Operating day")
    st.caption("Add equipment and its operating hours. Hover over column headings for help.")
    source = st.radio("Load input", ["Equipment builder", "24-hour CSV"], horizontal=True,help=HELP['Load input'])
    components = pd.DataFrame([
        {"Equipment":"Facility baseload", "Rated kW":200., "Load factor":1., "Start hour":0, "End hour":24},
        {"Equipment":"Process machinery", "Rated kW":1000., "Load factor":0.8, "Start hour":7, "End hour":19},
        {"Equipment":"HVAC / heat pumps", "Rated kW":500., "Load factor":0.7, "Start hour":6, "End hour":22},
        {"Equipment":"Fleet charging", "Rated kW":300., "Load factor":0.8, "Start hour":17, "End hour":23}])
    load = np.zeros(24)
    equipment = []
    if source == "Equipment builder":
        components = st.data_editor(components, num_rows="dynamic", hide_index=True, width="stretch", key="equipment",
            column_config={"Equipment":st.column_config.TextColumn(help="Name of the machine, process or group of loads. Add one row per operating schedule."), "Rated kW":st.column_config.NumberColumn(min_value=0.,required=True,help="Electrical nameplate power before the load factor is applied. Use the equipment rating or measured operating power; values depend on the site."),
                           "Load factor":st.column_config.NumberColumn(min_value=0.,max_value=1.,step=0.05,required=True,help="Average fraction of rated power used during operating hours. 0.8 means 80%. Starting example: 0.8 for process machinery, illustrative; use measured data."),
                           "Start hour":st.column_config.NumberColumn(min_value=0,max_value=23,step=1,required=True,help="First operating hour in 24-hour local time, inclusive. Example: 7 means 07:00. Equal start/end hours mean continuous operation."),
                           "End hour":st.column_config.NumberColumn(min_value=0,max_value=24,step=1,required=True,help="End of operating window, exclusive. Example: 19 means stop at 19:00; 24 means midnight. An end earlier than the start crosses midnight.")})
        equipment = components.to_dict('records')
        try:
            load = build_load(equipment)
        except (ValueError, TypeError) as exc:
            input_error = str(exc)
    else:
        st.caption("Columns: hour,load_kw. Exactly 24 hourly average power readings (0–23). Monthly and weekend multipliers below also apply to uploaded data.")
        example = pd.DataFrame({"hour":np.arange(24),"load_kw":build_load(components.to_dict('records'))})
        st.download_button("Download CSV template", example.to_csv(index=False), "load_template.csv", "text/csv")
        uploaded = st.file_uploader("Upload representative weekday", type="csv",help=HELP['Upload representative weekday'])
        if uploaded:
            try:
                load = read_load_csv(uploaded)
            except (ValueError, TypeError) as exc:
                input_error = str(exc)
        else:
            input_error = "Upload a 24-hour load CSV before generating the report."
    weekend_factor = st.slider("Weekend load relative to weekday (%)",0,150,50,5,help=HELP['Weekend load relative to weekday (%)'])/100
    with st.expander("Seasonality · monthly load multipliers"):
        season = st.data_editor(pd.DataFrame({"Month":list(calendar.month_abbr)[1:],"Load multiplier":[1.]*12}),
            hide_index=True,disabled=["Month"],width="stretch",key="seasonality",
            column_config={"Month":st.column_config.TextColumn(help="Calendar month to which the load multiplier applies."), "Load multiplier":st.column_config.NumberColumn(min_value=0.1,max_value=3.,step=0.05,required=True,help="Scales weekday and weekend demand for this month. Suggested starting value: 1.0 (no seasonality); 1.2 means 20% higher demand.")})
    factors = season['Load multiplier'].tolist()
    a,b,c = st.columns(3)
    a.metric("Typical weekday peak",f"{np.max(load):,.0f} kW")
    b.metric("Typical weekday energy",f"{np.sum(load):,.0f} kWh")
    c.metric("Weekday load factor",f"{np.mean(load)/np.max(load)*100 if np.max(load)>0 else 0:.0f}%")
    fig = go.Figure()
    fig.add_scatter(x=np.arange(24),y=load,name="Weekday",fill="tozeroy",line_color="#FF4B4B")
    fig.add_scatter(x=np.arange(24),y=load*weekend_factor,name="Weekend",line=dict(color="#94a3b8",dash="dot"))
    fig.update_xaxes(title="Hour of day");fig.update_yaxes(title="Facility demand (kW)")
    st.plotly_chart(chart_style(fig,height=280),width="stretch",key="load_preview")

with battery_tab:
    st.markdown("#### Battery specifications")
    st.caption("Use the same nameplate battery in all three scenarios.")
    a,b,c = st.columns(3)
    capacity = a.number_input("Nameplate energy (kWh)",100.,1000000.,2000.,100.,help=HELP['Nameplate energy (kWh)'])
    power = b.number_input("AC inverter power (kW)",25.,100000.,500.,25.,help=HELP['AC inverter power (kW)'])
    rte = c.number_input("AC round-trip efficiency (%)",50.,100.,85.,1.,help=HELP['AC round-trip efficiency (%)'])/100
    reserve = st.number_input("Reserve floor (%)",0.,95.,20.,5.,help=HELP['Reserve floor (%)'])/100
    with st.expander("Advanced battery settings"):
        a,b = st.columns(2)
        min_soc = a.number_input("Minimum SOC (%)",0.,95.,10.,5.,help=HELP['Minimum SOC (%)'])/100
        max_soc = b.number_input("Maximum SOC (%)",5.,100.,90.,5.,help=HELP['Maximum SOC (%)'])/100
        a,b = st.columns(2)
        degradation = a.number_input("Annual capacity degradation (%)",0.,10.,2.,0.25,help=HELP['Annual capacity degradation (%)'])/100
        cycles = b.number_input("Maximum equivalent cycles / day",0.1,4.,1.,0.1,help=HELP['Maximum equivalent cycles / day'])
    battery = Battery(capacity,power,rte,min_soc,max_soc,reserve,degradation,cycles)
    a,b,c = st.columns(3)
    a.metric("Nameplate duration",f"{capacity/power:.1f} h")
    b.metric("Base dispatchable energy",f"{max(0,max_soc-reserve)*capacity:,.0f} kWh")
    c.metric("Deliverable at full power",f"{max(0,max_soc-reserve)*capacity*np.sqrt(rte)/power:.1f} h")


with finance_tab:
    st.markdown("#### Project economics")
    a,b = st.columns(2)
    installed_cost = a.number_input("Total installed cost ($)",1000.,100000000.,825000.,10000.,help=HELP['Total installed cost ($)'])
    fixed_om = b.number_input("Fixed O&M ($/kW-year)",0.,1000.,10.,1.,help=HELP['Fixed O&M ($/kW-year)'])
    a,b = st.columns(2)
    years = a.number_input("Analysis period (years)",1,30,15,help=HELP['Analysis period (years)'])
    discount = b.number_input("Nominal discount rate (%)",0.,50.,8.,0.5,help=HELP['Nominal discount rate (%)'])/100
    with st.expander("Advanced financial settings"):
        a,b,c = st.columns(3)
        incentive = a.number_input("Confirmed upfront incentive ($)",0.,100000000.,0.,10000.,help=HELP['Confirmed upfront incentive ($)'])
        variable_om = b.number_input("Variable O&M ($/AC kWh discharged)",0.,1.,0.005,0.001,format="%.3f",help=HELP['Variable O&M ($/AC kWh discharged)'])
        om_escalation = c.number_input("Annual O&M escalation (%)",0.,20.,2.,0.5,help=HELP['Annual O&M escalation (%)'])/100
        a,b,c = st.columns(3)
        replacement_year = a.number_input("Replacement year (0 = disabled)",0,int(years),0,help=HELP['Replacement year (0 = disabled)'])
        replacement_fraction = b.number_input("Replacement / gross installed cost (%)",0.,100.,50.,5.,help=HELP['Replacement / gross installed cost (%)'])/100
        residual = c.number_input("Terminal residual / initial installed cost (%)",0.,100.,0.,5.,help=HELP['Terminal residual / initial installed cost (%)'])/100
        start_year = st.number_input("First operating calendar year",2000,2100,2027,help=HELP['First operating calendar year'])
    economics = Economics(energy_cost_per_kwh=0, power_cost_per_kw=0, project_cost=installed_cost,
                          upfront_incentive=incentive, fixed_om_per_kw_year=fixed_om,
                          variable_om_per_kwh=variable_om, om_escalation=om_escalation,
                          discount_rate=discount, years=int(years), start_year=int(start_year),
                          replacement_year=replacement_year, replacement_fraction=replacement_fraction,
                          residual_fraction=residual)

with tariff_tab:
    st.markdown("#### Utility rates")
    st.caption("Enter your utility rates. The starting values are illustrative.")
    a,b,c = st.columns(3)
    offpeak = a.number_input("Off-peak energy ($/kWh)",0.,5.,0.08,0.01,format="%.3f",help=HELP['Off-peak energy ($/kWh)'])
    peak = b.number_input("Peak energy ($/kWh)",0.,5.,0.20,0.01,format="%.3f",help=HELP['Peak energy ($/kWh)'])
    demand = c.number_input("Monthly demand charge ($/kW)",0.,200.,15.,1.,help=HELP['Monthly demand charge ($/kW)'])
    a,b = st.columns(2)
    peak_start = a.number_input("Peak window start hour",0,23,16,help=HELP['Peak window start hour'])
    peak_end = b.number_input("Peak window end hour",0,24,21,help=HELP['Peak window end hour'])
    with st.expander("Advanced tariff settings"):
        tariff_escalation = st.number_input("Annual tariff escalation (%)",0.,20.,2.5,0.5,help=HELP['Annual tariff escalation (%)'])/100
        weekend_peak = st.checkbox("Apply peak energy pricing on weekends",False,help=HELP['Apply peak energy pricing on weekends'])
    tariff = Tariff(offpeak,peak,peak_start,peak_end,demand,weekend_peak,tariff_escalation)


st.subheader("2. Configure scenarios")
st.caption("Multipliers apply to the study inputs above. Hover over column headings for definitions and starting values.")
scenario_defaults = pd.DataFrame([asdict(s) for s in DEFAULT_SCENARIOS])
scenario_data = st.data_editor(scenario_defaults[['name','capex_factor','tariff_factor','degradation_factor']],
    hide_index=True, disabled=["name"], width="stretch", key="scenarios", column_config={
        "name":st.column_config.TextColumn("Scenario",help="Conservative uses less favorable assumptions, Base uses the study inputs, and Aggressive uses more favorable assumptions. These are sensitivity cases, not probability forecasts."),
        "capex_factor":st.column_config.NumberColumn("Installed cost ×",min_value=0.1,max_value=3.,step=0.05,required=True,
            help="Multiplies total installed cost before the fixed upfront incentive. Suggested sensitivity values: Conservative 1.15, Base 1.00, Aggressive 0.90. Illustrative, not market averages."),
        "tariff_factor":st.column_config.NumberColumn("Utility rates ×",min_value=0.1,max_value=3.,step=0.05,required=True,
            help="Multiplies energy and demand rates in both baseline and battery bills. Suggested sensitivity values: Conservative 0.85, Base 1.00, Aggressive 1.15. Illustrative, not a price forecast."),
        "degradation_factor":st.column_config.NumberColumn("Degradation ×",min_value=0.1,max_value=3.,step=0.1,required=True,
            help="Multiplies annual capacity degradation. Suggested sensitivity values: Conservative 1.25, Base 1.00, Aggressive 0.75. Illustrative; use vendor-supported bounds.")})
with st.expander("Advanced scenario settings"):
    technical_defaults = scenario_defaults[['name','dispatch_factor','efficiency_delta']].copy()
    technical_defaults['efficiency_delta'] *= 100
    technical_data = st.data_editor(technical_defaults,hide_index=True,disabled=['name'],width="stretch",key="technical_scenarios",column_config={
        "name":st.column_config.TextColumn("Scenario",help="Technical sensitivity assumptions for the same configured nameplate battery."),
        "dispatch_factor":st.column_config.NumberColumn("Available energy & power ×",min_value=0.1,max_value=1.,step=0.05,required=True,
            help="Derates dispatchable energy and power to represent operating headroom, not outage availability. Suggested starting values: Conservative 0.90, Base 1.00, Aggressive 1.00; illustrative."),
        "efficiency_delta":st.column_config.NumberColumn("Efficiency adjustment (pp)",min_value=-15.,max_value=15.,step=1.,format="%.1f",required=True,
            help="Adds percentage points to round-trip efficiency; −2 changes 85% to 83%. Suggested sensitivity values: Conservative −2, Base 0, Aggressive +2; illustrative.")})
scenario_data = scenario_data.merge(technical_data,on='name',validate='one_to_one')
scenario_data['efficiency_delta'] /= 100

snapshot = dict(currency="USD",model_version="2.0",load_source=source,
                equipment=equipment,load_kw=load.tolist(),monthly_factors=factors,weekend_factor=weekend_factor,
                battery=asdict(battery),tariff=asdict(tariff),economics=asdict(economics),scenarios=scenario_data.to_dict('records'))
fingerprint = hashlib.sha256(json.dumps(snapshot,sort_keys=True).encode()).hexdigest()
if input_error:
    st.warning(input_error)
if st.button("Generate technoeconomic report",type="primary",width="stretch",disabled=bool(input_error)):
    try:
        with st.spinner("Optimizing monthly operation and annual cash flows for all three scenarios…"):
            results = calculate(snapshot)
        st.session_state['study'] = dict(results=results,snapshot=snapshot,fingerprint=fingerprint)
    except ValueError as exc:
        st.error(str(exc))

study = st.session_state.get('study')
if study:
    st.divider()
    if study['fingerprint'] != fingerprint:
        st.warning("Inputs have changed. The report below shows the last generated study; generate again to update it.")
    results, saved = study['results'], study['snapshot']
    st.subheader("3. View results")
    st.caption(f"United States · USD · {saved['battery']['capacity_kwh']:,.0f} kWh / {saved['battery']['power_kw']:,.0f} kW · {saved['economics']['years']} years · {saved['economics']['discount_rate']:.1%} nominal discount rate")
    base = next(r for r in results if r['scenario']['name']=='Base')
    first = base['yearly'].iloc[0]
    a,b,c,d = st.columns(4)
    a.metric("Base NPV",money(base['npv']))
    b.metric("Year 1 bill savings",money(first.energy_savings+first.demand_savings))
    c.metric("Achieved peak reduction",f"{(1-first.optimized_peak_kw/first.baseline_peak_kw)*100:.1f}%")
    d.metric("Simple payback",f"{base['payback']:.1f} years" if base['payback'] is not None else "Not reached")
    if base['npv'] >= 0:
        st.success("Base-case modeled returns exceed the selected discount rate. Compare the conservative case and validate tariff and load assumptions before committing capital.")
    else:
        st.info("Base-case bill savings do not recover investment at the selected discount rate. Explore installed cost, tariff spread, demand charges or battery size.")
    comparison, operation, cashflow, method = st.tabs(["Scenario comparison","Battery operation","Cash flows","Methodology & inputs"])
    with comparison:
        table = summary_table(results)
        st.dataframe(table,width="stretch",hide_index=True,column_config={
            k:st.column_config.NumberColumn(format="%.1f" if '%' in k or 'years' in k else "$%.0f") for k in table.columns if k!='Scenario'})
        st.caption("Blank payback means not reached in the selected horizon. Bill savings exclude O&M; operating net subtracts O&M and excludes replacement / residual cash flows.")
        a,b = st.columns(2)
        with a:
            fig = go.Figure(go.Bar(x=table.Scenario,y=table['NPV ($)'],marker_color=[COLORS[n] for n in table.Scenario]))
            fig.add_hline(y=0,line_color="#9ca3af");fig.update_yaxes(title="Discounted net value ($)")
            st.plotly_chart(chart_style(fig,"Investment value by outlook"),width="stretch")
        with b:
            fig = go.Figure()
            for r in results:
                y=r['yearly']
                fig.add_scatter(x=[0]+y.year.tolist(),y=[-r['net_capex']]+y.cumulative_discounted.tolist(),name=r['scenario']['name'],line_color=COLORS[r['scenario']['name']])
            fig.add_hline(y=0,line_color="#9ca3af");fig.update_xaxes(title="Operating year");fig.update_yaxes(title="Cumulative discounted cash ($)")
            st.plotly_chart(chart_style(fig,"Path to discounted break-even"),width="stretch")
    with operation:
        a,b,c=st.columns(3)
        choice=a.selectbox("Scenario",[r['scenario']['name'] for r in results],index=1,help=HELP['Scenario'])
        month=b.selectbox("Representative month",list(calendar.month_abbr)[1:],help=HELP['Representative month'])
        day=c.selectbox("Day type",['Weekday','Weekend'],help=HELP['Day type'])
        r=next(r for r in results if r['scenario']['name']==choice)
        data=r['dispatch'];data=data[(data.month==month)&(data.day_type==day)]
        fig=make_subplots(rows=3,cols=1,shared_xaxes=True,vertical_spacing=.08,subplot_titles=["Facility load and grid import","Battery dispatch (AC)","State of charge / tariff"],specs=[[{}],[{}],[{"secondary_y":True}]])
        fig.add_scatter(x=data.hour,y=data.load_kw,name="Facility load",line_color="#94a3b8",row=1,col=1)
        fig.add_scatter(x=data.hour,y=data.grid_kw,name="Grid import",line_color="#FF4B4B",row=1,col=1)
        fig.add_bar(x=data.hour,y=data.charge_kw,name="Charge",marker_color="#2563eb",row=2,col=1)
        fig.add_bar(x=data.hour,y=-data.discharge_kw,name="Discharge",marker_color="#FF4B4B",row=2,col=1)
        fig.add_scatter(x=list(range(25)),y=data.soc_start_pct.tolist()+[data.soc_end_pct.iloc[-1]],name="SOC",line_color="#d97706",row=3,col=1)
        fig.add_scatter(x=data.hour,y=data.price,name="Energy rate",line=dict(color="#64748b",dash="dot"),row=3,col=1,secondary_y=True)
        fig.update_yaxes(title_text="kW",row=1,col=1);fig.update_yaxes(title_text="kW",row=2,col=1)
        fig.update_yaxes(title_text="SOC (%)",row=3,col=1,secondary_y=False);fig.update_yaxes(title_text="$/kWh",row=3,col=1,secondary_y=True)
        fig.update_xaxes(title_text="Hour boundary / hour of day",row=3,col=1)
        st.plotly_chart(chart_style(fig,height=680),width="stretch")
        y=r['yearly'].iloc[0]
        a,b,c=st.columns(3)
        a.metric("Year 1 energy losses",f"{y.grid_kwh-y.load_kwh:,.0f} kWh")
        b.metric("Year 1 equivalent cycles",f"{y.equivalent_cycles:,.0f}")
        c.metric("Year 1 energy bill saving",money(y.energy_savings))
        st.caption("Grid charging incurs energy losses; storage does not automatically avoid emissions. Cycles are measured against the available SOC window.")
        st.dataframe(r['monthly'].round(2),hide_index=True,width="stretch")
    with cashflow:
        selected=st.selectbox("Cash flow scenario",[r['scenario']['name'] for r in results],index=1,help=HELP['Cash flow scenario'])
        r=next(r for r in results if r['scenario']['name']==selected)
        st.caption(f"Year 0 investment: {money(-r['net_capex'])}. Annual values below are USD; negative cash flow is retained.")
        columns=['year','capacity_remaining_pct','energy_savings','demand_savings','fixed_om','variable_om','replacement','residual','net_cashflow','discounted_cashflow','cumulative_cashflow','cumulative_discounted']
        st.dataframe(r['yearly'][columns].round(2),hide_index=True,width="stretch")
        st.caption("Replacement resets capacity at the start of its selected year. Payback accounts for later cash outflows through the study horizon.")
    with method:
        for paragraph in METHODOLOGY.strip().split('\n\n'):
            st.write(paragraph)
        st.json(saved,expanded=False)
    st.markdown("#### Export the report")
    a,b,c=st.columns(3)
    a.download_button("Download complete HTML report",html_report(results,saved),"zero_whatts_report.html","text/html",width="stretch")
    annual=pd.concat([r['yearly'].assign(scenario=r['scenario']['name']) for r in results],ignore_index=True)
    b.download_button("Download annual cash flows",annual.to_csv(index=False),"zero_whatts_cashflows.csv","text/csv",width="stretch")
    c.download_button("Download study inputs",json.dumps(saved,indent=2),"zero_whatts_inputs.json","application/json",width="stretch")
    dispatch=pd.concat([r['dispatch'].assign(scenario=r['scenario']['name']) for r in results],ignore_index=True)
    st.download_button("Download year-one hourly dispatch",dispatch.to_csv(index=False),"zero_whatts_dispatch.csv","text/csv")
else:
    st.subheader("3. View results")
    st.caption("Generate the report to view results.")

st.divider()
with st.expander("Sources & references", expanded=False):
    st.caption("References checked 15 October 2026. Only the 85% efficiency starting point is adopted as a cited numeric benchmark. Other defaults and scenario factors are illustrative, editable screening assumptions.")
    st.dataframe(pd.DataFrame(REFERENCE_VALUES,columns=['Parameter','Starting value','Basis']),hide_index=True,width="stretch")
    for topic,value,note,url in SOURCES:
        st.markdown(f"**{topic} · {value}**  \n{note} [Reference ↗]({url})")
