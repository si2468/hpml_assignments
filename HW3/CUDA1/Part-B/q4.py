import matplotlib.pyplot as plt

# X-axis: Input sizes in millions
sizes_millions = [1, 5, 10, 50, 100]

# CPU times (same in both graphs)
cpu_times = [0.00176422, 0.00709667, 0.0150437, 0.0723035, 0.145249]

# GPU Non-Unified Memory times
non_unified_1t = [0.0923722, 0.359234, 0.750198, 3.42735, 6.75108]
non_unified_256t = [0.00169802, 0.006814, 0.0135269, 0.0675011, 0.135606]
non_unified_Nt = [6.29425e-05, 0.000255108, 0.000494003, 0.00236893, 0.00471091]

# GPU Unified Memory times
unified_1t = [0.0646691, 0.41271, 0.619475, 3.1036, 6.08235]
unified_256t = [0.00307393, 0.0142119, 0.027621, 0.148956, 0.318715]
unified_Nt = [0.00256801, 0.012964, 0.0234928, 0.119468, 0.252037]

# ---------- Plot 1: CPU vs Non-Unified Memory ----------
plt.figure(figsize=(10, 6))
plt.plot(sizes_millions, cpu_times, 'o-', label='CPU')
plt.plot(sizes_millions, non_unified_1t, 's-', label='GPU Non-Unified: 1 block, 1 thread')
plt.plot(sizes_millions, non_unified_256t, '^-', label='GPU Non-Unified: 1 block, 256 threads')
plt.plot(sizes_millions, non_unified_Nt, 'd-', label='GPU Non-Unified: N/256 blocks, 256 threads')

plt.xlabel('Input Size (millions)')
plt.ylabel('Time (seconds)')
plt.title('CPU vs GPU Array Add (Non-Unified Memory)')
plt.yscale('log')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('q4_without_unified.jpg', dpi=300)
plt.close()

# ---------- Plot 2: CPU vs Unified Memory ----------
plt.figure(figsize=(10, 6))
plt.plot(sizes_millions, cpu_times, 'o-', label='CPU')
plt.plot(sizes_millions, unified_1t, 's-', label='GPU Unified: 1 block, 1 thread')
plt.plot(sizes_millions, unified_256t, '^-', label='GPU Unified: 1 block, 256 threads')
plt.plot(sizes_millions, unified_Nt, 'd-', label='GPU Unified: N/256 blocks, 256 threads')

plt.xlabel('Input Size (millions)')
plt.ylabel('Time (seconds)')
plt.title('CPU vs GPU Array Add (Unified Memory)')
plt.yscale('log')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('q4_with_unified.jpg', dpi=300)
plt.close()
