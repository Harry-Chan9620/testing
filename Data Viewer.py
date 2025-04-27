import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('moga_results.csv')

# Filter valid solutions
valid_df = df[df['valid'] == True]

# Plot all valid solutions
plt.figure(figsize=(10, 6))
plt.scatter(valid_df['total_distance'], valid_df['max_daily_distance'], alpha=0.5, label='All Valid Solutions')
plt.xlabel('Total Distance')
plt.ylabel('Max Daily Distance')
plt.title('Trade-off: Total Distance vs. Max Daily Distance')
plt.legend()
plt.show()