# Input sizes in millions
input_sizes_millions = [1, 5, 10, 50, 100]

# Scenario 1: 1 block, 1 thread (non-unified memory)
non_unified_scenario1_times = [0.0961661, 0.402941, 0.71212, 3.43451, 6.77886]

# Scenario 2: 1 block, 256 threads (non-unified memory)
non_unified_scenario2_times = [0.00175977, 0.00680208, 0.0135579, 0.067533, 0.135784]

# Scenario 3: Multiple blocks, 256 threads per block (non-unified memory)
non_unified_scenario3_times = [6.98566e-05, 0.000255108, 0.000487804, 0.00237107, 0.00472188]

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(input_sizes_millions, non_unified_scenario1_times, 'o-', label='1 block, 1 thread')
plt.plot(input_sizes_millions, non_unified_scenario2_times, 's-', label='1 block, 256 threads')
plt.plot(input_sizes_millions, non_unified_scenario3_times, '^-', label='Multiple blocks, 256 threads')

plt.xscale('log')
plt.yscale('log')

plt.xlabel('Input Size (in millions) [log scale]')
plt.ylabel('Execution Time (seconds) [log scale]')
plt.title('CUDA Array Addition Performance (Non-Unified Memory)')
plt.legend()
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.tight_layout()

# Save the figure
plt.savefig('cuda_array_add_non_unified_memory.png')

