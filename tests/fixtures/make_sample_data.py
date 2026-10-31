"""Generates three demo datasets: churn (classification), houses (regression), sales (forecasting)."""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def churn(n=2500, seed=0):
    r = np.random.RandomState(seed)
    tenure = r.randint(1, 72, n)
    charges = r.normal(70, 25, n).clip(20, 150)
    contract = r.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.2])
    calls = r.poisson(1.5, n)
    z = -1.0 - 0.04 * tenure + 0.02 * (charges - 70) + 0.5 * calls + np.where(contract == "Month-to-month", 1.0, -0.6)
    churn = (r.rand(n) < 1 / (1 + np.exp(-z))).astype(int)
    df = pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(n)],
        "signup_date": pd.Timestamp("2020-01-01") + pd.to_timedelta(r.randint(0, 1500, n), unit="D"),
        "tenure_months": tenure, "monthly_charges": charges.round(2), "contract": contract,
        "support_calls": calls, "region": r.choice(["North", "South", "East", "West"], n),
        "account_closed_flag": np.where(r.rand(n) < 0.001, 1 - churn, churn),   # deliberate leak
        "churn": np.where(churn == 1, "Yes", "No"),
    })
    df.loc[r.rand(n) < 0.04, "monthly_charges"] = np.nan
    df.loc[r.rand(n) < 0.02, "contract"] = np.nan
    return df


def houses(n=1500, seed=1):
    r = np.random.RandomState(seed)
    sqft = r.normal(1800, 600, n).clip(500, 5000)
    beds = r.randint(1, 6, n)
    age = r.randint(0, 80, n)
    hood = r.choice(["Downtown", "Suburb", "Rural"], n, p=[0.3, 0.5, 0.2])
    mult = pd.Series(hood).map({"Downtown": 1.5, "Suburb": 1.0, "Rural": 0.7}).to_numpy()
    price = (sqft * 120 * mult + beds * 8000 - age * 600) * np.exp(r.normal(0, 0.15, n))
    return pd.DataFrame({"listing_id": np.arange(100000, 100000 + n), "sqft": sqft.round(0), "bedrooms": beds,
                         "age_years": age, "neighborhood": hood, "has_garage": r.rand(n) < 0.6,
                         "price": price.round(0)})


def sales(days=730, seed=2):
    r = np.random.RandomState(seed)
    d = pd.date_range("2023-01-01", periods=days, freq="D")
    promo = (r.rand(days) < 0.15).astype(int)
    s = 200 + 0.3 * np.arange(days) + 40 * (d.dayofweek >= 5) + 60 * promo + r.normal(0, 12, days)
    return pd.DataFrame({"date": d.strftime("%Y-%m-%d"), "promo": promo,
                         "temperature": (15 + 10 * np.sin(np.arange(days) / 58) + r.normal(0, 2, days)).round(1),
                         "sales": s.round(1)})


if __name__ == "__main__":
    churn().to_csv(os.path.join(HERE, "churn.csv"), index=False)
    houses().to_csv(os.path.join(HERE, "house_prices.csv"), index=False)
    sales().to_csv(os.path.join(HERE, "daily_sales.csv"), index=False)
    print("Wrote churn.csv, house_prices.csv, daily_sales.csv")
