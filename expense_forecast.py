"""Trend-based monthly expense forecasting."""

import pandas as pd


def forecast_monthly_spending(
    monthly_totals: list[dict[str, object]], months: int
) -> pd.DataFrame:
    if not 1 <= months <= 12:
        raise ValueError("Forecast horizon must be between 1 and 12 months.")
    if not monthly_totals:
        raise ValueError("At least one monthly total is required to forecast.")

    totals_by_month: dict[pd.Period, float] = {}
    for row in monthly_totals:
        month = pd.Period(str(row["month"]), freq="M")
        totals_by_month[month] = totals_by_month.get(month, 0.0) + float(
            row["total"]
        )

    history_months = pd.period_range(
        min(totals_by_month), max(totals_by_month), freq="M"
    )
    values = [
        totals_by_month.get(month, 0.0) for month in history_months
    ]
    count = len(values)
    if count == 1:
        slope = 0.0
        intercept = values[0]
    else:
        positions = range(count)
        position_sum = sum(positions)
        value_sum = sum(values)
        position_square_sum = sum(position * position for position in positions)
        position_value_sum = sum(
            position * value for position, value in zip(positions, values)
        )
        denominator = count * position_square_sum - position_sum**2
        slope = (
            count * position_value_sum - position_sum * value_sum
        ) / denominator
        intercept = (value_sum - slope * position_sum) / count

    future_months = pd.period_range(
        start=history_months[-1] + 1, periods=months, freq="M"
    )
    forecast_values = [
        round(max(0.0, intercept + slope * position), 2)
        for position in range(count, count + months)
    ]
    return pd.DataFrame(
        {
            "Month": [str(month) for month in future_months],
            "Predicted Spend": forecast_values,
        }
    )
