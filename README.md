# Zero Whatts

Industrial battery storage configuration and technoeconomic screening in United States dollars.

Configure equipment schedules or upload a representative hourly load, choose battery specifications,
enter an installed project budget and utility tariff, and compare Conservative, Base and Aggressive
assumptions for the **same nameplate battery**.

## Run locally

Python 3.11+ recommended.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open http://localhost:8501. No API keys or external data services are required. Streamlit Community
Cloud can use `streamlit_app.py` as the entry point and `requirements.txt` for dependencies.

## Workflow

1. **Industrial load:** edit rated power, load factors and operating hours, or upload a CSV with
   exactly 24 rows and columns `hour,load_kw` (hours 0–23). Set weekend and monthly multipliers.
2. **Battery:** configure AC power, nameplate energy, AC round-trip efficiency, SOC limits, reserve,
   annual capacity degradation and equivalent daily cycle limit.
3. **Economics:** enter total installed cost, fixed O&M, nominal discount rate and analysis period.
   Incentives, variable O&M, escalation and replacement settings are under Advanced financial settings.
   The installed budget is entered directly; it is not automatically rescaled when battery size changes.
4. **Utility tariff:** enter retail off-peak and peak energy rates, TOU hours, monthly all-hours demand
   charge and nominal escalation. Rates come from the facility's utility tariff; wholesale market
   prices and market revenues are not included.
5. **Scenarios:** adjust capital cost, tariff, degradation, operating headroom and efficiency assumptions.
   Defaults are illustrative sensitivity bounds, not probability forecasts.
6. **Generate report:** compare NPV, savings, achieved peak reduction and payback; inspect hourly
   dispatch, monthly bills and annual cash flow. Download a standalone printable HTML report,
   annual cash-flow CSV, year-one dispatch CSV and complete study inputs as JSON.

Generated reports persist across widget changes. A stale-input notice identifies reports that need
regeneration. Inputs and results live in the current Streamlit session, not a database.

## Calculation model

`model.py` contains the independent calculation engine. SciPy/HiGHS optimizes two cyclic 24-hour
representative days per month with a shared monthly import peak. Energy is weighted by calendar
weekday/weekend counts; demand charges are billed once per month. Dispatch includes charging energy,
round-trip losses, SOC/reserve constraints, power and cycle limits, and no export. Equal beginning/end
SOC prevents counting initial energy as recurring savings. Capacity degrades from year two, dispatch
is reoptimized annually, and optional replacement resets capacity at the start of its year.

NPV discounts end-of-year pre-tax, unlevered cash flows at a nominal rate. Simple and discounted
payback interpolate the first durable cumulative break-even within the selected horizon. Negative
savings and unreached payback remain visible. Report formulas, inputs and references are included
in `reporting.py` and the app's methodology/reference tabs.

The original app's mixed daily/annual costs, double-applied efficiency, free initial battery energy,
automatic battery upsizing, undiscounted "NPV", unsupported DR income and no-payback chart failure
have been replaced. The UI reports achieved dispatch results instead of assumed peak-reduction targets.

## Scope and reference values

This is a representative-day screening model. It does not represent 8,760 measured intervals,
15-minute billing peaks, ratchets, coincident demand, holidays, solar, export, ancillary services,
demand response, taxes, financing, auxiliary battery consumption or outage performance. Dispatch
assumes perfect knowledge of the representative days. Scenario operating headroom is an energy/power
derating, not stochastic outage availability. Annual capacity degradation is an input, not a cycle-life
prediction. Grid-charged storage consumes additional net energy; avoided emissions are not inferred.

The 85% starting efficiency comes from [NREL 2024 ATB commercial battery storage](https://atb.nrel.gov/electricity/2024/commercial_battery_storage).
Other numerical defaults are explicitly illustrative. Replace them with measured load, vendor quotes
and the facility's actual utility tariff before using the study for an investment decision. See the
bottom **Sources & references** collapsible section for EIA, DOE, EPA, SAM and SciPy references.

## Tests

```sh
pip install -r requirements-dev.txt
python -m pytest -q
```

Tests cover dispatch energy conservation, SOC/power/cycle limits, closed-form peak shaving and
arbitrage cases, annual/monthly units, leap years, replacement and discounted cash flows, input
validation, HTML escaping and Streamlit report generation/persistence.

## Layout

- `streamlit_app.py`: configuration workspace, charts and persistent report.
- `model.py`: validated inputs, constrained dispatch and lifecycle economics.
- `reporting.py`: methodology, references, comparison and standalone report export.
- `ui_help.py`: input definitions and clearly labeled reference or illustrative starting values.
- `tests/`: numerical regression and app workflow tests.
