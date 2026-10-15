"""Behind-the-meter battery screening model. All dispatch powers are AC kW.

Each month has a representative weekday and weekend, each with cyclic SOC.
Their calendar weights annualize energy; their shared peak sets monthly demand.
"""
from dataclasses import dataclass, asdict
import calendar
import numpy as np
import pandas as pd
from scipy.optimize import linprog


@dataclass(frozen=True)
class Battery:
    capacity_kwh: float = 2000
    power_kw: float = 500
    round_trip_efficiency: float = 0.85
    min_soc: float = 0.10
    max_soc: float = 0.90
    reserve_soc: float = 0.20
    annual_degradation: float = 0.02
    max_daily_cycles: float = 1.0


@dataclass(frozen=True)
class Tariff:
    offpeak_rate: float = 0.08
    peak_rate: float = 0.20
    peak_start: int = 16
    peak_end: int = 21
    demand_rate: float = 15
    weekend_peak: bool = False
    escalation: float = 0.025


@dataclass(frozen=True)
class Economics:
    energy_cost_per_kwh: float = 350
    power_cost_per_kw: float = 150
    project_cost: float = 50000
    upfront_incentive: float = 0
    fixed_om_per_kw_year: float = 10
    variable_om_per_kwh: float = 0.005
    om_escalation: float = 0.02
    discount_rate: float = 0.08
    years: int = 15
    start_year: int = 2027
    replacement_year: int = 0
    replacement_fraction: float = 0.50
    residual_fraction: float = 0


@dataclass(frozen=True)
class Scenario:
    name: str
    capex_factor: float = 1
    tariff_factor: float = 1
    degradation_factor: float = 1
    dispatch_factor: float = 1
    efficiency_delta: float = 0


DEFAULT_SCENARIOS = [
    Scenario("Conservative", 1.15, 0.85, 1.25, 0.90, -0.02),
    Scenario("Base", 1, 1, 1, 1, 0),
    Scenario("Aggressive", 0.90, 1.15, 0.75, 1, 0.02),
]


def hour_mask(start, end):
    """Half-open operating window; equal endpoints mean 24-hour operation."""
    hours = np.arange(24)
    if start == end:
        return np.ones(24, dtype=bool)
    return (hours >= start) & (hours < end) if start < end else (hours >= start) | (hours < end)


def build_load(components):
    total = np.zeros(24)
    for row in components:
        values = [float(row[k]) for k in ("Rated kW", "Load factor", "Start hour", "End hour")]
        power, factor, start, end = values
        if not np.isfinite(values).all() or power < 0 or not 0 <= factor <= 1:
            raise ValueError("Load power must be nonnegative and load factors must be between 0 and 1.")
        if start != int(start) or end != int(end) or not 0 <= start <= 23 or not 0 <= end <= 24:
            raise ValueError("Load schedule hours must be whole hours: start 0–23, end 0–24.")
        total += power * factor * hour_mask(start, end)
    return total


def read_load_csv(file):
    data = pd.read_csv(file)
    if set(data.columns) != {"hour", "load_kw"} or len(data) != 24:
        raise ValueError("CSV must have exactly 24 rows and columns hour,load_kw.")
    data = data.sort_values("hour")
    if not np.array_equal(data.hour.to_numpy(), np.arange(24)):
        raise ValueError("CSV hours must contain each whole hour from 0 to 23 once.")
    load = pd.to_numeric(data.load_kw, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(load).all() or (load < 0).any():
        raise ValueError("CSV loads must be finite, nonnegative kW values.")
    return load


def validate(load, monthly_factors, weekend_factor, battery, tariff, economics, scenarios):
    load = np.asarray(load, dtype=float)
    if load.shape != (24,) or not np.isfinite(load).all() or (load < 0).any() or load.max() <= 0:
        raise ValueError("Provide 24 finite, nonnegative hourly loads, with at least one positive load.")
    factors = np.asarray(monthly_factors, dtype=float)
    if factors.shape != (12,) or not np.isfinite(factors).all() or (factors <= 0).any():
        raise ValueError("Provide 12 positive monthly load multipliers.")
    if not np.isfinite(weekend_factor) or not 0 <= weekend_factor <= 2:
        raise ValueError("Weekend load factor must be between 0 and 2.")
    for obj in (battery, tariff, economics, *scenarios):
        for key, value in asdict(obj).items():
            if isinstance(value, (int, float)) and not np.isfinite(value):
                raise ValueError(f"{key} must be finite.")
    if battery.capacity_kwh <= 0 or battery.power_kw <= 0 or battery.max_daily_cycles <= 0:
        raise ValueError("Battery energy, power and daily cycle limit must be positive.")
    if not 0 <= battery.min_soc <= battery.reserve_soc < battery.max_soc <= 1:
        raise ValueError("SOC must satisfy 0 ≤ minimum ≤ reserve < maximum ≤ 100%.")
    if not 0 < battery.round_trip_efficiency <= 1 or not 0 <= battery.annual_degradation < 0.2:
        raise ValueError("Check battery efficiency and annual degradation.")
    if min(tariff.offpeak_rate, tariff.peak_rate, tariff.demand_rate) < 0 or tariff.peak_rate < tariff.offpeak_rate:
        raise ValueError("Rates must be nonnegative; peak energy rate must be at least off-peak.")
    if tariff.peak_start not in range(24) or tariff.peak_end not in range(25):
        raise ValueError("Check tariff peak hours.")
    if not 0 <= tariff.escalation <= 0.2 or not 0 <= economics.om_escalation <= 0.2:
        raise ValueError("Escalation rates must be between 0 and 20%.")
    costs = [economics.energy_cost_per_kwh, economics.power_cost_per_kw, economics.project_cost,
             economics.upfront_incentive, economics.fixed_om_per_kw_year, economics.variable_om_per_kwh]
    if min(costs) < 0 or not 1 <= economics.years <= 30 or not 2000 <= economics.start_year <= 2100:
        raise ValueError("Check nonnegative costs, analysis period (1–30) and start year (2000–2100).")
    if not 0 <= economics.discount_rate <= 0.5 or not 0 <= economics.residual_fraction <= 1:
        raise ValueError("Check discount rate and residual value.")
    if not 0 <= economics.replacement_year <= economics.years or not 0 <= economics.replacement_fraction <= 1:
        raise ValueError("Replacement must be disabled (0) or within the analysis period.")
    if len(scenarios) != 3 or len({s.name for s in scenarios}) != 3:
        raise ValueError("Provide three uniquely named scenarios.")
    for scenario in scenarios:
        if min(scenario.capex_factor, scenario.tariff_factor, scenario.degradation_factor) <= 0 or not 0 < scenario.dispatch_factor <= 1:
            raise ValueError("Scenario multipliers must be positive; dispatch factor cannot exceed 1.")
        if not 0 < battery.round_trip_efficiency + scenario.efficiency_delta <= 1:
            raise ValueError("Scenario efficiency must be above 0% and at most 100%.")
        if battery.annual_degradation * scenario.degradation_factor >= 0.2:
            raise ValueError("Scenario annual degradation must be below 20%.")


def optimize_month(loads, prices, day_counts, capacity, power, battery, efficiency, demand_rate, variable_om):
    """Optimize two cyclic 24-hour days against one shared monthly import peak.

    Variables per day: AC charge[24], AC discharge[24], stored energy[25].
    Round trip efficiency is split symmetrically across charge and discharge.
    Positive prices and a small throughput tie-break remove wasteful cycling.
    """
    size, peak_idx = 147, 146
    objective = np.zeros(size)
    objective[peak_idx] = demand_rate
    bounds = []
    equalities, equality_rhs, inequalities, inequality_rhs = [], [], [], []
    eta = np.sqrt(efficiency)
    lower = max(battery.min_soc, battery.reserve_soc) * capacity
    upper = battery.max_soc * capacity
    for day in range(2):
        offset = day * 73
        c, d, e = offset, offset + 24, offset + 48
        objective[c:c+24] = (prices[day] + 1e-8) * day_counts[day]
        objective[d:d+24] = (-prices[day] + variable_om + 1e-8) * day_counts[day]
        bounds.extend([(0, power)] * 48 + [(lower, upper)] * 25)
        for h in range(24):
            row = np.zeros(size)
            row[e+h+1], row[e+h], row[c+h], row[d+h] = 1, -1, -eta, 1/eta
            equalities.append(row); equality_rhs.append(0)
            row = np.zeros(size)
            row[c+h], row[d+h], row[peak_idx] = 1, -1, -1
            inequalities.append(row); inequality_rhs.append(-loads[day, h])
            row = np.zeros(size)
            row[c+h], row[d+h] = -1, 1
            inequalities.append(row); inequality_rhs.append(loads[day, h])  # no export
            row = np.zeros(size)
            row[c+h], row[d+h] = 1, 1
            inequalities.append(row); inequality_rhs.append(power)
        row = np.zeros(size)
        row[e], row[e+24] = 1, -1
        equalities.append(row); equality_rhs.append(0)
        row = np.zeros(size)
        row[d:d+24] = 1/eta
        inequalities.append(row)
        inequality_rhs.append((upper-lower) * battery.max_daily_cycles)
    bounds.append((0, None))
    solution = linprog(objective, A_ub=np.array(inequalities), b_ub=inequality_rhs,
                       A_eq=np.array(equalities), b_eq=equality_rhs, bounds=bounds, method="highs")
    if not solution.success:
        raise ValueError(f"Battery dispatch could not be solved: {solution.message}")
    charge = np.stack([solution.x[i*73:i*73+24] for i in range(2)])
    discharge = np.stack([solution.x[i*73+24:i*73+48] for i in range(2)])
    energy = np.stack([solution.x[i*73+48:i*73+73] for i in range(2)])
    grid = loads + charge - discharge
    return dict(charge=charge, discharge=discharge, energy=energy, grid=grid,
                peak=float(grid.max()), capacity=capacity)


def stable_payback(cashflows):
    """Fractional year after which cumulative cash never becomes negative again."""
    cumulative = np.cumsum(cashflows)
    for year in range(1, len(cumulative)):
        if cumulative[year] >= 0 and np.all(cumulative[year:] >= -1e-8):
            return float(year - 1 + max(0, -cumulative[year-1]) / cashflows[year]) if cashflows[year] > 0 else float(year)
    return None


def analyze(load, monthly_factors, weekend_factor, battery, tariff, economics, scenario):
    gross_capex = (battery.capacity_kwh * economics.energy_cost_per_kwh +
                   battery.power_kw * economics.power_cost_per_kw + economics.project_cost) * scenario.capex_factor
    if economics.upfront_incentive >= gross_capex:
        raise ValueError("Upfront incentive must be less than installed cost in every scenario.")
    net_capex = gross_capex - economics.upfront_incentive
    efficiency = battery.round_trip_efficiency + scenario.efficiency_delta
    degradation = battery.annual_degradation * scenario.degradation_factor
    yearly, monthly, dispatch = [], [], []
    age = 0
    for year in range(1, economics.years+1):
        if year == economics.replacement_year:
            age = 0
        remaining = (1 - degradation) ** age
        capacity = battery.capacity_kwh * remaining * scenario.dispatch_factor
        power = battery.power_kw * scenario.dispatch_factor
        escalation = (1 + tariff.escalation) ** (year-1) * scenario.tariff_factor
        om_factor = (1 + economics.om_escalation) ** (year-1)
        variable_om = economics.variable_om_per_kwh * om_factor
        annual = dict(baseline_energy=0., optimized_energy=0., baseline_demand=0., optimized_demand=0.,
                      discharged_kwh=0., charged_kwh=0., load_kwh=0., grid_kwh=0.)
        annual_peaks = []
        for month in range(1, 13):
            dates = calendar.Calendar().itermonthdays2(economics.start_year + year - 1, month)
            counts = np.zeros(2)
            for date, dow in dates:
                if date:
                    counts[int(dow >= 5)] += 1
            loads = np.stack([load, load * weekend_factor]) * monthly_factors[month-1]
            prices = np.full((2, 24), tariff.offpeak_rate * escalation)
            mask = hour_mask(tariff.peak_start, tariff.peak_end)
            prices[0, mask] = tariff.peak_rate * escalation
            if tariff.weekend_peak:
                prices[1, mask] = tariff.peak_rate * escalation
            result = optimize_month(loads, prices, counts, capacity, power, battery, efficiency,
                                    tariff.demand_rate * escalation, variable_om)
            weights = counts[:, None]
            values = dict(baseline_energy=float(np.sum(loads * prices * weights)),
                          optimized_energy=float(np.sum(result['grid'] * prices * weights)),
                          baseline_demand=float(loads.max() * tariff.demand_rate * escalation),
                          optimized_demand=result['peak'] * tariff.demand_rate * escalation,
                          discharged_kwh=float(np.sum(result['discharge'] * weights)),
                          charged_kwh=float(np.sum(result['charge'] * weights)),
                          load_kwh=float(np.sum(loads * weights)), grid_kwh=float(np.sum(result['grid'] * weights)))
            for key, value in values.items():
                annual[key] += value
            annual_peaks.append((loads.max(), result['peak']))
            if year == 1:
                monthly.append(dict(month=calendar.month_abbr[month], baseline_peak_kw=loads.max(),
                                    optimized_peak_kw=result['peak'], **values))
                for day in range(2):
                    for hour in range(24):
                        dispatch.append(dict(month=calendar.month_abbr[month], day_type=['Weekday', 'Weekend'][day],
                                             hour=hour, load_kw=loads[day, hour], grid_kw=result['grid'][day, hour],
                                             charge_kw=result['charge'][day, hour], discharge_kw=result['discharge'][day, hour],
                                             soc_start_pct=result['energy'][day, hour]/capacity*100,
                                             soc_end_pct=result['energy'][day, hour+1]/capacity*100, price=prices[day, hour]))
        energy_savings = annual['baseline_energy'] - annual['optimized_energy']
        demand_savings = annual['baseline_demand'] - annual['optimized_demand']
        fixed_om = economics.fixed_om_per_kw_year * battery.power_kw * om_factor
        cycling_om = annual['discharged_kwh'] * variable_om
        replacement = gross_capex * economics.replacement_fraction * om_factor if year == economics.replacement_year else 0
        residual = gross_capex * economics.residual_fraction if year == economics.years else 0
        net_cashflow = energy_savings + demand_savings - fixed_om - cycling_om - replacement + residual
        yearly.append(dict(year=year, capacity_remaining_pct=remaining*100, energy_savings=energy_savings,
                           demand_savings=demand_savings, fixed_om=fixed_om, variable_om=cycling_om,
                           replacement=replacement, residual=residual, net_cashflow=net_cashflow,
                           discounted_cashflow=net_cashflow/(1+economics.discount_rate)**year,
                           baseline_peak_kw=max(p[0] for p in annual_peaks),
                           optimized_peak_kw=max(p[1] for p in annual_peaks),
                           equivalent_cycles=annual['discharged_kwh']/np.sqrt(efficiency)/(capacity*(battery.max_soc-battery.reserve_soc)),
                           **annual))
        age += 1
    df = pd.DataFrame(yearly)
    df['cumulative_cashflow'] = df.net_cashflow.cumsum() - net_capex
    df['cumulative_discounted'] = df.discounted_cashflow.cumsum() - net_capex
    cash = [-net_capex] + df.net_cashflow.tolist()
    discounted = [-net_capex] + df.discounted_cashflow.tolist()
    return dict(scenario=asdict(scenario), gross_capex=gross_capex, net_capex=net_capex,
                npv=float(sum(discounted)), lifetime_net=float(sum(cash)),
                payback=stable_payback(cash), discounted_payback=stable_payback(discounted),
                roi=float(sum(cash)/net_capex), yearly=df, monthly=pd.DataFrame(monthly),
                dispatch=pd.DataFrame(dispatch))


def run_study(load, monthly_factors, weekend_factor, battery, tariff, economics, scenarios=DEFAULT_SCENARIOS):
    validate(load, monthly_factors, weekend_factor, battery, tariff, economics, scenarios)
    return [analyze(np.asarray(load), monthly_factors, weekend_factor, battery, tariff, economics, s) for s in scenarios]
