import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
import joblib
import os
import time

def load_and_preprocess_data(india_path, saudi_path):
    """Load daily climate datasets and merge them."""
    print("=" * 60)
    print("LOADING DAILY CLIMATE DATASETS")
    print("=" * 60)

    # Read Excel files - skip the first 2 rows (title + note), use row 3 as header
    df_india = pd.read_excel(india_path, skiprows=2, header=None)
    df_saudi = pd.read_excel(saudi_path, skiprows=2, header=None)

    # Set proper column names from the first row of data
    india_cols = df_india.iloc[0].tolist()
    saudi_cols = df_saudi.iloc[0].tolist()
    df_india.columns = india_cols
    df_saudi.columns = saudi_cols
    df_india = df_india[1:].reset_index(drop=True)
    df_saudi = df_saudi[1:].reset_index(drop=True)

    print(f"India dataset shape: {df_india.shape}")
    print(f"Saudi dataset shape: {df_saudi.shape}")
    print(f"India columns: {list(df_india.columns)}")
    print(f"Saudi columns: {list(df_saudi.columns)}")

    # Rename columns for consistency
    df_india.rename(columns={
        'Date': 'date',
        'Air Temperature (K)': 'india_air_temp',
        'Surface Temperature (K)': 'india_surface_temp',
        'Relative Humidity (%)': 'india_humidity',
        'Wind Speed (m/s)': 'india_wind_speed',
        'Precipitation (mm)': 'india_precipitation'
    }, inplace=True)

    df_saudi.rename(columns={
        'Date': 'date',
        'Air Temperature (K)': 'saudi_air_temp',
        'Surface Temperature (K)': 'saudi_surface_temp',
        'Relative Humidity (%)': 'saudi_humidity',
        'Wind Speed (m/s)': 'saudi_wind_speed',
        'Total Precipitation (m)': 'saudi_precipitation'
    }, inplace=True)

    # Convert date columns to datetime
    df_india['date'] = pd.to_datetime(df_india['date'])
    df_saudi['date'] = pd.to_datetime(df_saudi['date'])

    # Convert all numeric columns
    for col in ['india_air_temp', 'india_surface_temp', 'india_humidity', 'india_wind_speed', 'india_precipitation']:
        df_india[col] = pd.to_numeric(df_india[col], errors='coerce')

    for col in ['saudi_air_temp', 'saudi_surface_temp', 'saudi_humidity', 'saudi_wind_speed', 'saudi_precipitation']:
        df_saudi[col] = pd.to_numeric(df_saudi[col], errors='coerce')

    # Drop NaN rows
    df_india.dropna(inplace=True)
    df_saudi.dropna(inplace=True)

    print(f"\nAfter cleaning - India: {df_india.shape}, Saudi: {df_saudi.shape}")

    # Merge on date
    print("Merging datasets on date column...")
    df = pd.merge(df_saudi, df_india, on='date', how='inner')
    print(f"Merged dataset shape: {df.shape}")

    # Add temporal features
    df['month'] = df['date'].dt.month
    df['day_of_year'] = df['date'].dt.dayofyear
    # Cyclic encoding for month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    # Cyclic encoding for day of year
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

    return df


def train_model(df):
    """Train XGBoost model with continuous loop showing progress month by month."""
    print("\n" + "=" * 60)
    print("TRAINING XGBOOST MODEL (n_estimators=100)")
    print("=" * 60)

    # Define features and targets
    feature_cols = [
        'saudi_air_temp', 'saudi_surface_temp', 'saudi_humidity',
        'saudi_wind_speed', 'saudi_precipitation',
        'month', 'day_of_year', 'month_sin', 'month_cos', 'day_sin', 'day_cos'
    ]
    target_cols = [
        'india_air_temp', 'india_surface_temp', 'india_humidity',
        'india_wind_speed', 'india_precipitation'
    ]

    X = df[feature_cols].values
    y = df[target_cols].values

    print(f"\nFeatures shape: {X.shape}")
    print(f"Targets shape: {y.shape}")

    # 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set: {X_train.shape[0]} samples (80%)")
    print(f"Testing set:  {X_test.shape[0]} samples (20%)")

    # Scale features
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train)

    # Continuous loop training - process data month by month with 1 second gap
    print("\n" + "-" * 60)
    print("CONTINUOUS TRAINING LOOP (month-by-month)")
    print("-" * 60)

    # Get unique year-month combinations from the training data
    train_dates = df.loc[df.index.isin(
        pd.DataFrame(X_train, columns=feature_cols).index
    ), 'date'] if 'date' in df.columns else None

    # Group by month for display purposes
    unique_months = sorted(df['date'].dt.to_period('M').unique())
    print(f"Total months in dataset: {len(unique_months)}")
    print()

    for i, period in enumerate(unique_months):
        month_mask = (df['date'].dt.to_period('M') == period)
        month_count = month_mask.sum()
        print(f"  [{i+1:3d}/{len(unique_months)}] Processing {period} ... {month_count} daily records")
        time.sleep(1)  # 1 second gap between each month

    print("\n  All months processed. Fitting XGBoost model...")

    # Train XGBoost with n_estimators=100
    xgb_model = XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model = MultiOutputRegressor(xgb_model)
    model.fit(X_train_scaled, y_train_scaled)

    print("  Model training complete!")

    # Evaluate
    print("\n" + "-" * 60)
    print("MODEL EVALUATION")
    print("-" * 60)

    pred_scaled = model.predict(X_test_scaled)
    pred = scaler_y.inverse_transform(pred_scaled)
    y_test_actual = y_test

    mse = mean_squared_error(y_test_actual, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_actual, pred)
    accuracy = r2 * 100

    print(f"  MSE:      {mse:.4f}")
    print(f"  RMSE:     {rmse:.4f}")
    print(f"  R² Score: {r2:.4f}")
    print(f"  Accuracy: {accuracy:.2f}%")

    # Feature importance
    print("\n  Feature Importance:")
    importances = {}
    for j, col in enumerate(feature_cols):
        avg_imp = np.mean([est.feature_importances_[j] for est in model.estimators_])
        importances[col] = round(float(avg_imp), 4)
        print(f"    {col:25s} : {avg_imp:.4f}")

    # Save models
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/xgb_model.pkl")
    joblib.dump(scaler_X, "models/scaler_X.pkl")
    joblib.dump(scaler_y, "models/scaler_y.pkl")
    print("\n  Models saved to models/ directory.")

    metrics = {
        'mse': float(mse),
        'rmse': float(rmse),
        'r2': float(r2),
        'accuracy': float(accuracy)
    }

    return metrics, importances


def compute_seasonal_trends(df):
    """Analyze how Summer, Winter, and Rainy seasons are changing year-over-year."""
    import json

    print("\n" + "=" * 60)
    print("SEASONAL TREND ANALYSIS")
    print("=" * 60)

    df = df.copy()
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

    # Convert Kelvin to Celsius for analysis
    df['india_temp_c'] = df['india_air_temp'] - 273.15

    # Define Indian seasons by month
    # Summer: March-June | Rainy/Monsoon: July-October | Winter: November-February
    def get_season(month):
        if month in [3, 4, 5, 6]:
            return 'Summer'
        elif month in [7, 8, 9, 10]:
            return 'Rainy'
        else:
            return 'Winter'

    df['season'] = df['month'].apply(get_season)

    trends = {}

    for season in ['Summer', 'Winter', 'Rainy']:
        print(f"\n  Analyzing {season} season...")
        season_df = df[df['season'] == season]

        # Yearly averages
        yearly = season_df.groupby('year').agg({
            'india_temp_c': 'mean',
            'india_precipitation': 'mean',
            'india_humidity': 'mean',
            'india_wind_speed': 'mean'
        }).reset_index()

        years = yearly['year'].values
        temps = yearly['india_temp_c'].values
        precips = yearly['india_precipitation'].values
        humids = yearly['india_humidity'].values

        # Calculate linear trend (slope per year)
        if len(years) >= 2:
            temp_slope = float(np.polyfit(years, temps, 1)[0])
            precip_slope = float(np.polyfit(years, precips, 1)[0])
            humid_slope = float(np.polyfit(years, humids, 1)[0])
        else:
            temp_slope = 0.0
            precip_slope = 0.0
            humid_slope = 0.0

        # Total change over the full period
        total_years = float(years[-1] - years[0]) if len(years) >= 2 else 1.0
        temp_total_change = temp_slope * total_years
        precip_total_change = precip_slope * total_years
        humid_total_change = humid_slope * total_years

        # Year-by-year data for charts
        yearly_data = []
        for _, row in yearly.iterrows():
            yearly_data.append({
                "year": int(row['year']),
                "avg_temp": round(float(row['india_temp_c']), 2),
                "avg_precipitation": round(float(row['india_precipitation']), 4),
                "avg_humidity": round(float(row['india_humidity']), 4)
            })

        trend_direction = "increasing" if temp_slope > 0 else "decreasing"

        trends[season] = {
            "temp_slope_per_year": round(temp_slope, 4),
            "temp_total_change": round(temp_total_change, 2),
            "temp_trend": trend_direction,
            "precip_slope_per_year": round(precip_slope, 6),
            "precip_total_change": round(precip_total_change, 4),
            "precip_trend": "increasing" if precip_slope > 0 else "decreasing",
            "humid_slope_per_year": round(humid_slope, 6),
            "humid_total_change": round(humid_total_change, 4),
            "humid_trend": "increasing" if humid_slope > 0 else "decreasing",
            "yearly_data": yearly_data
        }

        print(f"    Temperature: {trend_direction} by {abs(temp_total_change):.2f}°C over {total_years:.0f} years")
        print(f"    Precipitation: {'increasing' if precip_slope > 0 else 'decreasing'}")
        print(f"    Humidity: {'increasing' if humid_slope > 0 else 'decreasing'}")

    # Save trends to JSON
    os.makedirs("models", exist_ok=True)
    with open("models/seasonal_trends.json", "w") as f:
        json.dump(trends, f, indent=2)
    print("\n  Seasonal trends saved to models/seasonal_trends.json")

    return trends


if __name__ == "__main__":
    india_path = "../india_daily_climate_2015_2026.xlsx"
    saudi_path = "../saudi_arabia_daily_climate_2015_2026 (1).xlsx"

    df = load_and_preprocess_data(india_path, saudi_path)
    metrics, importances = train_model(df)
    trends = compute_seasonal_trends(df)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print(f"Final Accuracy: {metrics['accuracy']:.2f}%")
    print(f"Final RMSE: {metrics['rmse']:.4f}")
    print("=" * 60)
