# this is the script to run the training with multiple number of workers in C3

import subprocess
import re
import matplotlib.pyplot as plt

num_workers_values = list(range(0, 17, 4))

total_data_loading_times = []

# Regular expression to capture data loading time
data_loading_time_regex = re.compile(r"Epoch \d+: Data Loading Time: (\d+\.\d+)s")

previous_time = float('inf') 
stopping_index = None

for i, num_workers in enumerate(num_workers_values):
    command = f"python3 main.py --num_workers {num_workers}"
    print(f"Running command: {command}")

    result = subprocess.run(command, shell=True, text=True, capture_output=True)

    data_loading_time_matches = data_loading_time_regex.findall(result.stdout)

    if data_loading_time_matches:
        total_time = sum(float(time) for time in data_loading_time_matches)
        total_data_loading_times.append(total_time)
        print(f"num_workers: {num_workers} : {total_time:.4f} s")

        if total_time > previous_time:
            stopping_index = i 
            break
        
        previous_time = total_time 
    else:
        print(f"No data loading times found for num_workers = {num_workers}")
        total_data_loading_times.append(None)

if stopping_index is not None:
    num_workers_values = num_workers_values[:stopping_index + 1]
    total_data_loading_times = total_data_loading_times[:stopping_index + 1]

# Plot the graph of total data loading time vs num_workers
plt.figure(figsize=(10, 6))
plt.plot(num_workers_values, total_data_loading_times, marker='o', linestyle='-', color='b')
plt.xlabel('num_workers')
plt.ylabel('Total Data Loading Time (s)')
plt.title('Total Data Loading Time vs num_workers')
plt.grid(True)
plt.show()
