import numpy as np
import matplotlib.pyplot as plt

# Given parameters for the roofline model
peak_flops = 200  # GFLOPS
memory_bandwidth = 30  # GB/s

# Define a range of arithmetic intensities (FLOP/Byte)
intensity_range = np.logspace(-2, 3, 100)  # From 0.01 to 1000 FLOP/Byte

# Compute the roofline performance
compute_bound = np.full_like(intensity_range, peak_flops)  # Peak performance line
memory_bound = intensity_range * memory_bandwidth  # Memory bandwidth line

# Find the intersection of the two lines (Roofline Knee)
knee_intensity = peak_flops / memory_bandwidth

# Given benchmark data (Arithmetic Intensity = FLOP/Bandwidth)
benchmark_data = [
    {"N": 1000000, "T": 0.001523, "B": 5.254, "F": 1313473503.627},  # C1
    {"N": 300000000, "T": 0.478714, "B": 5.013, "F": 1253358674.591},

    {"N": 1000000, "T": 0.000469, "B": 17.047, "F": 4261671233.381},  # C2
    {"N": 300000000, "T": 0.228643, "B": 10.497, "F": 2624172581.656},

    {"N": 1000000, "T": 0.000809, "B": 9.231, "F": 2255764182.124},  # C3
    {"N": 300000000, "T": 0.064444, "B": 37.241, "F": 9310364178.697},

    {"N": 1000000, "T": 0.421120, "B": 0.019, "F": 4749236.470},  # C4
    {"N": 300000000, "T": 122.058039, "B": 0.020, "F": 4915694.244},

    {"N": 1000000, "T": 0.001518, "B": 5.271, "F": 1317637660.118},  # C5
    {"N": 300000000, "T": 0.460654, "B": 5.210, "F": 1302496455.643}
]

# Compute arithmetic intensity (AI = FLOPS / Bandwidth)
for entry in benchmark_data:
    entry["AI"] = entry["F"] / (entry["B"] * 1e9)  # Convert to FLOP/Byte

# Extract AI and performance data for plotting
ai_values = np.array([entry["AI"] for entry in benchmark_data])
performance_values = np.array([entry["F"] / 1e9 for entry in benchmark_data])  # Convert FLOP/sec to GFLOP/sec

# C3_3M
# N = 300000000, T = 0.064444, B = 37.241 GB/s, F = 9.31 GFLOPS
# C2_1M
# N = 1000000, T = 0.000469, B = 17.047 GB/s, F = 4.26 GFLOPS
# C2_3M
# N = 300000000, T = 0.228643, B = 10.497 GB/s, F = 2.62 GFLOPS
# C3_1M
# N = 1000000, T = 0.000809, B = 9.231 GB/s, F = 2.26 GFLOPS
# C5_1M
# N = 1000000, T = 0.001518, B = 5.271 GB/s, F = 1.32 GFLOPS
# C1_1M
# N = 1000000, T = 0.001523, B = 5.254 GB/s, F = 1.31 GFLOPS
# C5_3M
# N = 300000000, T = 0.460654, B = 5.21 GB/s, F = 1.30 GFLOPS
# C1_3M
# N = 300000000, T = 0.478714, B = 5.013 GB/s, F = 1.25 GFLOPS
# C4_3M
# N = 300000000, T = 122.058039, B = 0.02 GB/s, F = 0.005 GFLOPS
# C4_1M
# N = 1000000, T = 0.421120, B = 0.019 GB/s, F = 0.005 GFLOPS

print(ai_values)
print(performance_values)

performance_values.sort()
print(performance_values)

# Labels for benchmark points
labels = [
    "C1_1M", "C1_3M", "C2_1M", "C2_3M",
    "C3_1M", "C3_3M", "C4_1M", "C4_3M",
    "C5_1M", "C5_3M"
]

# Plot the Roofline Model with benchmark data points
plt.figure(figsize=(8, 6))

mean_ai = np.mean(ai_values)

plt.axvline(mean_ai, linestyle=":", color="blue", label=f"Dot Product Arithmetic Intensity ≈ {mean_ai:.2f} FLOP/Byte")

# Plot the Roofline Model
plt.loglog(intensity_range, np.minimum(compute_bound, memory_bound), label="Roofline Model", color='red', linewidth=2)
plt.axvline(knee_intensity, linestyle="--", color="gray", label=f"Balance Point ≈ {knee_intensity:.2f} FLOP/Byte")

# Plot benchmark points
for i, (ai, perf, label) in enumerate(zip(ai_values, performance_values, labels)):
    plt.scatter(ai, perf, marker='o', label=label)
    plt.annotate(label, (ai, perf), fontsize=4, color='black', xytext=(-10, 5), textcoords='offset points')

# Labels and formatting
plt.xlabel("Arithmetic Intensity (FLOP/Byte)")
plt.ylabel("Performance (GFLOPS)")
plt.title("Roofline Model with Benchmark Results")
plt.legend()
plt.grid(True, which="major", linestyle="--", linewidth=0.5)

ax = plt.gca()
#ax.set_ylim([0.7, 1.7])

# Display the plot
plt.show()
