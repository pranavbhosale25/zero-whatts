"""Shared report content and portable HTML export."""
import html
import json
from datetime import datetime, timezone
import pandas as pd

SOURCES = [
    ("Round-trip efficiency", "85% default", "NREL 2024 ATB commercial battery storage adopts 85%. Use the vendor's AC-to-AC specification for a real project.", "https://atb.nrel.gov/electricity/2024/commercial_battery_storage"),
    ("Installed cost / O&M", "Illustrative editable inputs", "Energy cost, power cost, installation and O&M defaults are not quoted ATB costs. Use a site-specific installed quote and avoid counting the same cost twice.", "https://atb.nrel.gov/electricity/2024/commercial_battery_storage"),
    ("Energy / demand tariff", "User-entered utility rates", "TOU energy and monthly demand charges must come from the customer's tariff. Wholesale RTO prices are not a retail tariff.", "https://www.eia.gov/energyexplained/electricity/prices-and-factors-affecting-prices.php"),
    ("Demand response", "Excluded from cash flows", "Eligibility, event duration, performance and compensation require a separate dispatch and contract assessment.", "https://www.energy.gov/cmei/femp/demand-response-and-time-variable-pricing-programs"),
    ("Discounted cash flow", "Nominal, unlevered, pre-tax", "Charging energy reduces energy savings. NPV discounts annual cash flows; escalation and discount rates must use a consistent basis.", "https://samrepo.nlr.gov/help/mtf_lcoe.html"),
    ("Emissions", "No avoided-emissions claim", "Storage losses increase grid energy here. Avoided emissions require time-dependent marginal emissions; eGRID provides regional inventory context.", "https://www.epa.gov/egrid"),
    ("Dispatch", "Linear optimization", "SciPy HiGHS minimizes energy, monthly demand and discharge-based variable O&M costs under battery constraints.", "https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html"),
]

REFERENCE_VALUES = [
    ("AC round-trip efficiency", "85%", "NREL 2024 ATB commercial battery storage"),
    ("Nameplate energy / AC power", "2,000 kWh / 500 kW", "Illustrative 4-hour system; choose a vendor configuration"),
    ("SOC minimum / reserve / maximum", "10% / 20% / 90%", "Illustrative operating policy; confirm with vendor and site"),
    ("Annual capacity degradation", "2%", "Illustrative annual fade; use vendor warranty / engineering estimate"),
    ("Daily equivalent cycle limit", "1.0", "Illustrative operating constraint, measured on dispatchable SOC window"),
    ("Total installed cost", "$825,000", "Illustrative all-in budget for the example 2,000 kWh / 500 kW system, not a market quote"),
    ("Fixed / variable O&M", "$10/kW-year / $0.005/kWh discharged", "Illustrative expense allowance"),
    ("Off-peak / peak / monthly demand tariff", "$0.08/kWh / $0.20/kWh / $15/kW", "Illustrative retail tariff; replace with utility rates"),
    ("Tariff / O&M escalation", "2.5% / 2% annually", "Illustrative nominal escalation assumptions"),
    ("Discount rate / study horizon", "8% nominal / 15 years", "Illustrative investor assumptions"),
    ("Incentive / residual value / market revenue", "$0 / 0% / excluded", "No unsupported tax credit, salvage or program payment assumed"),
    ("Scenario multipliers", "Editable in Scenarios tab", "Illustrative sensitivity bounds, not sourced forecasts"),
]

METHODOLOGY = """
This is a behind-the-meter, grid-charged storage screening study in USD. It compares the same
nameplate battery under three editable assumption sets. No battery is automatically enlarged.
Each calendar month contains a representative weekday and weekend with hourly resolution;
calendar day counts weight energy costs. One shared maximum import sets each monthly demand
charge. Load shapes repeat with the entered monthly and weekend multipliers.

Dispatch minimizes electricity cost plus variable O&M. AC charge and discharge respect inverter
power, available energy, reserve floor and upper SOC limit. Charge and discharge efficiencies
each equal the square root of round-trip efficiency. SOC is cyclic within each representative
day, so initial stored energy is never a free source of annual savings. The daily cycle limit
uses internal discharged energy divided by the dispatchable SOC window. Nonnegative prices
and a small throughput tie-break discourage wasteful simultaneous cycling.

Capacity degradation starts in year two; dispatch is reoptimized every year. Scenario dispatch
derating reduces effective energy and power, not nameplate configuration. It represents operating
headroom, not outage availability. Replacement occurs at the start of the selected year, resets
capacity, and is paid at that year's end. Replacement cost is a fraction of scenario gross
installed cost, escalated with O&M. Residual value is a fraction of initial gross installed
cost, received at the end of the final year.

NPV = negative net upfront investment + sum of annual net cash flow / (1 + discount rate)^year.
Cash flow = energy savings + demand savings - fixed O&M - variable O&M - replacement + residual.
ROI is undiscounted lifetime net cash divided by net upfront investment. Payback is interpolated
within the first year after which cumulative cash remains nonnegative through the study horizon;
discounted payback uses discounted cash. NPV uses end-of-year cash flows. Fractional payback
assumes cash accrues uniformly within the crossing year.

Costs, escalation and discount rates are nominal. Taxes, financing, depreciation, PV, export,
ancillary services, DR revenue, demand ratchets, coincident-peak charges, holidays, auxiliary
battery loads, stochastic outages and 15-minute peaks are excluded. Degradation is an annual
capacity assumption, not a cycle-life forecast. Perfect knowledge of representative profiles
is assumed. Measured interval data and the full utility tariff are needed for investment
decisions. Flat emissions factors cannot establish avoided emissions from grid-charged storage.
"""


def summary_table(results):
    rows = []
    for r in results:
        y = r['yearly'].iloc[0]
        rows.append({"Scenario": r['scenario']['name'], "Net investment ($)": r['net_capex'],
                     "Year 1 bill savings ($)": y.energy_savings+y.demand_savings,
                     "Year 1 operating net ($)": y.energy_savings+y.demand_savings-y.fixed_om-y.variable_om,
                     "Achieved peak cut (%)": (1-y.optimized_peak_kw/y.baseline_peak_kw)*100,
                     "NPV ($)": r['npv'], "Lifetime ROI (%)": r['roi']*100,
                     "Payback (years)": r['payback'], "Discounted payback (years)": r['discounted_payback']})
    return pd.DataFrame(rows)


def value_chart(results):
    """Portable SVG investment chart; no remote scripts or image dependencies."""
    values = [r['npv'] for r in results]
    low, high = min(0, min(values)), max(0, max(values))
    span = max(high-low, 1)
    def y(value):
        return 60 + (high-value)/span*170
    zero = y(0)
    parts = [f'<svg viewBox="0 0 720 300" role="img" aria-label="Scenario net present value comparison">',
             f'<line x1="40" y1="{zero:.1f}" x2="680" y2="{zero:.1f}" stroke="#94a3b8"/>']
    for i, (result, color) in enumerate(zip(results,['#94a3b8','#FF4B4B','#2563eb'])):
        value = result['npv']
        top = min(y(value),zero)
        x = 100 + i*210
        label = f'-${abs(value):,.0f}' if value<0 else f'${value:,.0f}'
        parts.append(f'<rect x="{x}" y="{top:.1f}" width="110" height="{max(1,abs(y(value)-zero)):.1f}" rx="4" fill="{color}"/>')
        parts.append(f'<text x="{x+55}" y="35" text-anchor="middle" font-size="16" fill="#31333F">{label}</text>')
        parts.append(f'<text x="{x+55}" y="270" text-anchor="middle" font-size="16" fill="#31333F">{html.escape(result["scenario"]["name"])}</text>')
    return ''.join(parts)+'</svg>'


def html_report(results, snapshot):
    esc = html.escape
    summary = summary_table(results).round(2).to_html(index=False, na_rep="Not reached", border=0)
    sections = []
    for r in results:
        cash_columns = ['year','capacity_remaining_pct','energy_savings','demand_savings','fixed_om',
                        'variable_om','replacement','residual','net_cashflow','cumulative_discounted']
        monthly_columns = ['month','baseline_peak_kw','optimized_peak_kw','baseline_energy',
                           'optimized_energy','baseline_demand','optimized_demand']
        sections.append(f"<h2>{esc(r['scenario']['name'])}</h2><h3>Annual cash flow</h3>" +
                        r['yearly'][cash_columns].rename(columns=lambda c:c.replace('_',' ').title()).round(2).to_html(index=False,border=0) +
                        "<h3>Year-one monthly operation</h3>" +
                        r['monthly'][monthly_columns].rename(columns=lambda c:c.replace('_',' ').title()).round(2).to_html(index=False,border=0))
    refs = ''.join(f'<li><a href="{esc(url)}">{esc(topic)}</a> — {esc(value)}. {esc(note)}</li>' for topic,value,note,url in SOURCES)
    paragraphs = ''.join('<p>'+esc(p.strip())+'</p>' for p in METHODOLOGY.strip().split('\n\n'))
    defaults = pd.DataFrame(REFERENCE_VALUES,columns=['Parameter','Starting value','Basis']).to_html(index=False,border=0)
    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Zero Whatts | Technoeconomic report</title><style>
    body{{font:14px system-ui;color:#31333F;max-width:1400px;margin:40px auto;padding:24px;line-height:1.6}}
    h1,h2{{color:#B4232C}}table{{border-collapse:collapse;width:100%;font-size:12px;display:block;overflow:auto}}
    th,td{{padding:9px;border-bottom:1px solid #dde5e9;text-align:right;white-space:nowrap}}
    th{{background:#F0F2F6}}pre{{white-space:pre-wrap;background:#f5f7f9;padding:20px}}li{{margin-bottom:12px}}
    svg{{max-width:720px;display:block;margin:auto}}
    @page{{size:A4 landscape;margin:12mm}}
    @media print{{body{{margin:0;padding:0}}h2{{break-before:page}}table{{font-size:9px;display:table}}th,td{{padding:5px;white-space:normal}}}}
    </style><h1>⚡️ Zero Whatts</h1><h3>Battery Projects Screening for Industrial Loads</h3>
    <p>Technoeconomic screening report · United States / USD</p>
    <p>Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} · Model v2.0</p>
    <h2>Scenario comparison</h2><p>All scenarios use the same configured nameplate battery. Blank or unreached payback means investment is not recovered within the study horizon. Values are USD unless otherwise labeled.</p>
    {value_chart(results)}{summary}{''.join(sections)}<h2>Study inputs</h2>
    <pre>{esc(json.dumps(snapshot,indent=2))}</pre><h2>Methodology and scope</h2>{paragraphs}
    <h2>Sources and reference values</h2><p>Checked 15 October 2026. These are starting defaults; the actual study inputs above govern this report. Illustrative defaults are not sourced benchmarks.</p>{defaults}<ul>{refs}</ul></html>'''
