import os
import pandas as pd
import matplotlib.pyplot as plt

# Directory where your CSV files are stored
data_dir = "Data"

# List all CSV files starting with "moga_" in that directory
csv_files = [f for f in os.listdir(data_dir) if f.startswith("moga_") and f.endswith(".csv")]

# Check if there are any matching files
if not csv_files:
    print("No 'moga_' CSV files found in the Data directory.")
else:
    # Show available files
    print("Available 'moga_' CSV files:")
    for idx, file in enumerate(csv_files):
        print(f"{idx}: {file}")
    
    choice = int(input("\nEnter the number of the file you want to open: "))
    
    # Load the selected file
    selected_file = csv_files[choice]
    df = pd.read_csv(os.path.join(data_dir, selected_file))
    print(f"\nLoaded file: {selected_file}")

    # Filter valid solutions
    valid_df = df[df['valid'] == True]

    # Plot all valid solutions
    plt.figure(figsize=(10, 6))
    plt.scatter(valid_df['total_distance'], valid_df['max_daily_distance'], alpha=0.5, label='All Valid Solutions')
    plt.xlabel('Total Distance')
    plt.ylabel('Max Daily Distance')
    plt.title('Trade-off: Total Distance vs. Max Daily Distance')
    plt.legend()
    plt.grid(True)
    plt.show()
