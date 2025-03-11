import subprocess
import sys
import matplotlib.pyplot as plt
import re

def run_c3():
    print(f"\n\nRUNNING EXPERIMENT: C3 - Finding the optimal number of workers\n\n")
    num_workers_values = list(range(0, 17, 4))
    total_data_loading_times = []
    data_loading_time_regex = re.compile(r"- Total Data Loading Time: (\d+\.\d+)s")

    total_epoch_times = []
    total_epoch_time_regex = re.compile(r"- Total Epoch Time: (\d+\.\d+)s")

    previous_data_loading_time = float('inf')
    min_total_epoch_time = float("inf")
    optimal_workers = 0
    stopping_index = None

    print("\nConducting experiment on the most optimal number of workers for the dataloader. "
          "Will automatically stop when performance decreases.\n")

    for i, num_workers in enumerate(num_workers_values):
        command = f"python3 main.py --num_workers {num_workers}"
        print(f"\nRunning command: {command}\n")

        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        stdout_lines = []
        for line in iter(process.stdout.readline, ''):
            sys.stdout.write(line)  # Print to console in real-time
            sys.stdout.flush()
            stdout_lines.append(line)  # Store output for regex matching

        process.stdout.close()
        process.wait()  # Ensure process completes

        # Extract data loading times from captured output
        captured_output = "".join(stdout_lines)
        data_loading_time_matches = data_loading_time_regex.findall(captured_output)
        total_epoch_times_matches = total_epoch_time_regex.findall(captured_output)

        if total_epoch_times_matches:
            total_time = sum(float(time) for time in total_epoch_times_matches)
            total_epoch_times.append(total_time)
            print(f"\nnum_workers: {num_workers} -> Total Runtime: {total_time:.4f} s\n")

            if total_time < min_total_epoch_time:
                min_total_epoch_time = total_time
                optimal_workers = num_workers

        else:
            print(f"\nNo epoch times found for num_workers = {num_workers}\n")
            total_epoch_times.append(None)

        if data_loading_time_matches:
            total_time = sum(float(time) for time in data_loading_time_matches)
            total_data_loading_times.append(total_time)
            print(f"\nnum_workers: {num_workers} -> Total Data Loading Time: {total_time:.4f} s\n")

            if total_time > previous_data_loading_time:
                stopping_index = i
                print("\nPerformance decreased, stopping experiment early.\n")
                break

            previous_data_loading_time = total_time
        else:
            print(f"\nNo data loading times found for num_workers = {num_workers}\n")
            total_data_loading_times.append(None)

    if stopping_index is not None:
        num_workers_values = num_workers_values[:stopping_index + 1]
        total_data_loading_times = total_data_loading_times[:stopping_index + 1]

    print("CONCLUSION OF EXPERIMENT: C3\n\n")
    print("THE OPTIMAL NUMBER OF WORKERS IS: ", num_workers_values[len(num_workers_values) - 2], "\n\n")
    # Plot the graph of total data loading time vs num_workers
    plt.figure(figsize=(10, 6))
    plt.plot(num_workers_values, total_data_loading_times, marker='o', linestyle='-', color='b')
    plt.xlabel('num_workers')
    plt.ylabel('Total Data Loading Time (s)')
    plt.title('Total Data Loading Time vs num_workers')
    plt.grid(True)
    plt.show()
    plt.savefig("num_workers_experiment.png")

    return optimal_workers


def run_experiment(commands: list, experiment: str):
    """
    Runs a list of shell commands as subprocesses and streams their output in real-time.

    Args:
        commands (list): A list of shell commands to execute.
        experiment (str): Name of the experiment.
    """
    print(f"\n\nRUNNING EXPERIMENT: {experiment}\n\n")

    for command in commands:
        print(f"\n{command}\n")

        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        for line in iter(process.stdout.readline, ''):
            sys.stdout.write(line)  # Print output in real-time
            sys.stdout.flush()

        process.stdout.close()
        process.wait()  # Ensure process completes


# Example usage:
experiments_first = [
    (["python3 main.py"], "C2 - Default experiment"),
]

for cmds, experiment in experiments_first:
    run_experiment(cmds, experiment)

# this line is here just in case we want to comment out the next line and not get optimal_workers - we will just assume it is 4 so that future experiements know this value
optimal_workers = 4
optimal_workers = run_c3()

second_experiments = [
    (["python3 main.py --num_workers 1", f"python3 main.py --num_workers {optimal_workers}"], "C4 - 1 worker vs optimal num workers"),
    #([f"python3 main.py --num_workers {optimal_workers}", f"python3 main.py --num_workers {optimal_workers} --disable_cuda"], "C5 - GPU vs CPU"),
    ([f"python3 main.py --num_workers {optimal_workers} --optimizer adam", f"python3 main.py --num_workers {optimal_workers} --optimizer adagrad", f"python3 main.py --num_workers {optimal_workers} --optimizer sgd", f"python3 main.py --num_workers {optimal_workers} --optimizer sgdnesterov", f"python3 main.py --num_workers {optimal_workers} --optimizer adadelta"], "C6 - optimizers"),
    ([f"python3 main.py --num_workers {optimal_workers} --disable_batch_normalization", f"python3 main.py --num_workers {optimal_workers}"], "C7 - Default without Batch Normalization"),
    ([f"python3 main.py --num_workers {optimal_workers} --torch_compile default --num_epochs 10", f"python3 main.py --num_workers {optimal_workers} --torch_compile reduce-overhead --num_epochs 10", f"python3 main.py --num_workers {optimal_workers} --torch_compile max-autotune --num_epochs 10", f"python3 main.py --num_workers {optimal_workers} --torch_compile none --num_epochs 10"], "C8 - torch compile experiments")
]
for cmds, experiment in second_experiments:
    run_experiment(cmds, experiment)