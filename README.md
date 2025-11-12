# Airline Delay Prediction

## Overview
A machine learning project that predicts flight delays by merging U.S. Department of Transportation airline delay data with NOAA temperature data. The system analyzes operational and weather factors to forecast delay patterns, helping airlines optimize resource allocation and improve on-time performance.

## Technologies Used
- Python
- Machine Learning (Random Forest Regressor, scikit-learn)
- Data Analysis (pandas, numpy)
- Visualization (matplotlib, seaborn, bokeh)
- Feature Engineering (IQR outlier removal)

## Features
- Data merging of 650,000+ flight records with 29 million temperature records
- Outlier removal using IQR method
- Exploratory data analysis with correlation heatmaps
- Seasonal delay pattern analysis
- Airline performance comparison
- Weather impact assessment
- Random Forest prediction model

## Datasets
- **DOT Airline Delay Cause Data:** ~640,000 rows with flight delay information
  - Source: U.S. Bureau of Transportation Statistics via Kaggle
- **NOAA City Temperature Data:** 29 million rows with monthly average temperatures
  - Source: NOAA Global Surface Summary of Day (GSOD)
- **Final Merged Dataset:** ~310,000 records after inner join on city, month, and year

## Model Performance
- **RMSE:** ~10 minutes
- **R² Score:** 0.62
- **Most Important Feature:** weather_ct (count of flights already delayed by weather)
- **On-Time Performance:** 78% of flights meet DOT's ≤15 minute threshold

## Key Findings
### Airline Performance
- **Best:** Hawaiian Airlines (~5 min average delay)
- **Good:** Alaska, US Airways (~7-8 min)
- **Mid-Range:** Southwest, Delta, American, United (~10-13 min)
- **Highest Delays:** Envoy, SkyWest, Mesa, Frontier, JetBlue (~14-18 min)

### Seasonal Patterns
- **Summer:** Longest delays (~4 minutes) - thunderstorms and peak travel
- **Winter:** ~2.8 minutes - snow, ice, poor visibility
- **Spring:** ~2.1 minutes - clearing weather
- **Fall:** Shortest delays (~1.6 minutes) - calm weather, less crowded

### Temperature Impact
- Arrival delay shows strongest correlation with temperature
- Overall temperature correlation is relatively low (ρ ≈ 0.05)
- Other factors like congestion have much stronger impact

## Installation
```bash
git clone https://github.com/yourusername/airline-delay-prediction.git
cd airline-delay-prediction
pip install -r requirements.txt
```

## Project Structure
```
├── data/                    # Flight and temperature datasets
├── notebooks/               # Jupyter notebooks for analysis
├── src/                     # Python scripts for preprocessing and modeling
├── visualizations/          # Generated plots and charts
└── requirements.txt
```

## Python Libraries Used
- pandas - data loading, cleaning, merging
- numpy - numeric operations
- seaborn - statistical plots and heatmaps
- matplotlib.pyplot - visualization
- bokeh - interactive line charts
- scikit-learn - Random Forest model
- graphviz - optional tree visualization

## Data Preprocessing
1. Removed missing values from both datasets
2. Normalized city names (lowercase, stripped spaces/punctuation)
3. Standardized date columns (month, year)
4. Inner join on city, month, and year keys
5. Applied IQR ± 1.5× rule to remove extreme outliers
6. Feature engineering (seasonal labels, binary flags, congestion aggregation)

## Limitations
- Only includes reported delay data from 2013-2020
- Missing factors: ATC staffing, mechanical issues, gate turnaround times
- Simple train/test split without time-series cross-validation
- Default Random Forest hyperparameters used
- Temperature alone is not a strong predictor

## Future Enhancements
- Integrate real-time weather data (precipitation, wind, visibility)
- Add ATC staffing and gate turnaround time data
- Implement time-series cross-validation
- Hyperparameter tuning (grid search)
- Try XGBoost or neural networks
- Build live dashboard or API for real-time predictions
- City-specific models for better accuracy

## License
MIT License

## Acknowledgments
- U.S. Bureau of Transportation Statistics for flight delay data
- NOAA for temperature data
- IST 652 Scripting for Data Analysis course
