import subprocess
import re
import matplotlib.pyplot as plt

# List of num_workers to test
num_workers_values = list(range(0, 65, 4))

# Initialize lists to store the results
total_data_loading_times = []

# Regular expression to capture data loading time
data_loading_time_regex = re.compile(r"Epoch \d+: Data Loading Time: (\d+\.\d+)s")

# Run the command for each num_workers value
for num_workers in num_workers_values:
    # Run the command
    command = f"python3 main.py --num_workers {num_workers}"
    print(f"Running command: {command}")
    
    # Run the command and capture the output
    result = subprocess.run(command, shell=True, text=True, capture_output=True)
    
    # Extract data loading times from the output
    data_loading_time_matches = data_loading_time_regex.findall(result.stdout)
    
    # If we have data loading times, calculate the total
    if data_loading_time_matches:
        total_time = sum(float(time) for time in data_loading_time_matches)
        total_data_loading_times.append(total_time)
        print(f"num_workers: {num_workers} : {total_time:.4f} s")
    else:
        print(f"No data loading times found for num_workers = {num_workers}")
        total_data_loading_times.append(None)

# Plot the graph of total data loading time vs num_workers
plt.figure(figsize=(10, 6))
plt.plot(num_workers_values, total_data_loading_times, marker='o', linestyle='-', color='b')
plt.xlabel('num_workers')
plt.ylabel('Total Data Loading Time (s)')
plt.title('Total Data Loading Time vs num_workers')
plt.grid(True)
plt.show()
