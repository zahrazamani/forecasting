import pandas as pd
import torch
from chronos import ChronosPipeline
import matplotlib.pyplot as plt

# Load time series data
df = pd.read_csv("your_time_series_data.csv")
time_series_data = torch.tensor(df['your_time_series_column'].values)

# Initialize the Chronos model
pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map="cpu",
    torch_dtype=torch.bfloat16
)

# Set context length and prediction length
context_length = len(time_series_data) - 12 # Using last 12 values for testing
prediction_length = 12

# Prepare context
context = time_series_data[:context_length]

# Generate predictions
quantiles = pipeline(
    context,
    num_samples=50,
    prediction_length=prediction_length,
    quantiles=[0.1, 0.5, 0.9]
).cpu().numpy()

# Extract median and confidence intervals
median = quantiles[0, :, 1]
low = quantiles[0, :, 0]
high = quantiles[0, :, 2]

# Create forecast index
forecast_index = range(context_length, len(time_series_data))

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(df['your_time_series_column'], label='Historical Data', color='royalblue')
plt.plot(forecast_index, median, label='Median Forecast', color='tomato')
plt.fill_between(forecast_index, low, high, color='tomato', alpha=0.3, label='80% Prediction Interval')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.title('Time Series Forecasting with Chronos')
plt.legend()
plt.grid(True)
plt.show()
