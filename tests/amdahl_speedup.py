#!/usr/bin/env python3
"""
Amdahl's Law Speedup Analysis for ZPIC
Authors: Diogo Silva & Tomás Pereira
Date: 2026-01-11
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ============================================================================
# Data (ppc = 500 for all tests)
# ============================================================================

# A64FX (ARM) data
a64fx_threads = np.array([1, 2, 4, 6, 12, 24, 36, 48])
a64fx_times = np.array([13.537415, 6.802112, 3.425256, 2.301500, 1.187615, 0.637273, 0.475179, 0.420524])

# AMD EPYC data
amd_threads = np.array([1, 2, 4, 8, 16, 32, 64, 128])
amd_times = np.array([2.834280, 1.433575, 0.724022, 0.375698, 0.200316, 0.120873, 0.100211, 0.159063])

# ============================================================================
# Calculate Speedup
# ============================================================================

a64fx_speedup = a64fx_times[0] / a64fx_times
amd_speedup = amd_times[0] / amd_times

print("=" * 70)
print("SPEEDUP ANALYSIS - ZPIC (ppc = 500)")
print("=" * 70)

print("\n### A64FX (ARM) ###")
print(f"{'Threads':<10} {'Time (s)':<15} {'Speedup':<15} {'Efficiency (%)':<15}")
print("-" * 55)
for i in range(len(a64fx_threads)):
    eff = (a64fx_speedup[i] / a64fx_threads[i]) * 100
    print(f"{a64fx_threads[i]:<10} {a64fx_times[i]:<15.6f} {a64fx_speedup[i]:<15.2f} {eff:<15.2f}")

print("\n### AMD EPYC ###")
print(f"{'Threads':<10} {'Time (s)':<15} {'Speedup':<15} {'Efficiency (%)':<15}")
print("-" * 55)
for i in range(len(amd_threads)):
    eff = (amd_speedup[i] / amd_threads[i]) * 100
    print(f"{amd_threads[i]:<10} {amd_times[i]:<15.6f} {amd_speedup[i]:<15.2f} {eff:<15.2f}")

# ============================================================================
# Amdahl's Law: Speedup(n) = 1 / ((1-P) + P/n)
# Solving for P: P = (S - 1) * n / (S * (n - 1))
# ============================================================================

def amdahl(n, P):
    """Amdahl's Law speedup model"""
    return 1 / ((1 - P) + P / n)

def calc_parallel_fraction(speedup, n):
    """Calculate parallel fraction P from measured speedup"""
    if n <= 1:
        return None
    return (speedup - 1) * n / (speedup * (n - 1))

# Calculate P for each measurement point
print("\n" + "=" * 70)
print("PARALLEL FRACTION (P) CALCULATION PER DATA POINT")
print("=" * 70)

print("\n### A64FX (ARM) - P values ###")
a64fx_P_values = []
for i in range(1, len(a64fx_threads)):
    P = calc_parallel_fraction(a64fx_speedup[i], a64fx_threads[i])
    a64fx_P_values.append(P)
    print(f"  n={a64fx_threads[i]:2d}, Speedup={a64fx_speedup[i]:.2f}x -> P = {P*100:.4f}%")

print("\n### AMD EPYC - P values ###")
amd_P_values = []
for i in range(1, len(amd_threads)):
    P = calc_parallel_fraction(amd_speedup[i], amd_threads[i])
    amd_P_values.append(P)
    print(f"  n={amd_threads[i]:3d}, Speedup={amd_speedup[i]:.2f}x -> P = {P*100:.4f}%")

# Use MAXIMUM P value to ensure Amdahl curve is always >= measured points
# The curve should be the theoretical upper bound
a64fx_P = max(a64fx_P_values)
amd_P = max([p for p in amd_P_values if amd_threads[amd_P_values.index(p)+1] <= 64])  # Exclude 128 anomaly

print("\n" + "=" * 70)
print("AMDAHL'S LAW ANALYSIS (using minimum P for conservative estimate)")
print("=" * 70)

print(f"\n### A64FX (ARM) ###")
print(f"Parallel fraction:          P = {a64fx_P*100:.2f}%")
print(f"Serial fraction:            S = {(1-a64fx_P)*100:.2f}%")
print(f"Theoretical max speedup:    {1/(1-a64fx_P):.2f}x")

print(f"\n### AMD EPYC ###")
print(f"Parallel fraction:          P = {amd_P*100:.2f}%")
print(f"Serial fraction:            S = {(1-amd_P)*100:.2f}%")
print(f"Theoretical max speedup:    {1/(1-amd_P):.2f}x")
print(f"Note: 128-thread result excluded (performance degradation - NUMA/contention)")

# ============================================================================
# Generate Plots
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- A64FX Plot ---
ax1 = axes[0]
n_range = np.linspace(1, 64, 100)

# Ideal linear speedup
ax1.plot(n_range, n_range, 'k--', label='Ideal (linear)', linewidth=1.5, alpha=0.7)

# Amdahl's Law curve
amdahl_curve_a64fx = amdahl(n_range, a64fx_P)
ax1.plot(n_range, amdahl_curve_a64fx, 'b-', 
         label=f"Amdahl's Law (P={a64fx_P*100:.2f}%)", linewidth=2)

# Measured data
ax1.scatter(a64fx_threads, a64fx_speedup, color='red', s=100, zorder=5, 
            label='Measured', edgecolors='black', linewidths=1)

ax1.set_xlabel('Number of Threads', fontsize=12)
ax1.set_ylabel('Speedup', fontsize=12)
ax1.set_title('A64FX (ARM) - Speedup vs Threads', fontsize=14, fontweight='bold')
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 52)
ax1.set_ylim(0, max(50, 1/(1-a64fx_P) * 1.1))

# Add annotations for key points
for i, (t, s) in enumerate(zip(a64fx_threads, a64fx_speedup)):
    if t in [1, 12, 48]:
        ax1.annotate(f'{s:.1f}x', (t, s), textcoords="offset points", 
                    xytext=(5, 10), fontsize=9)

# --- AMD Plot ---
ax2 = axes[1]
n_range_amd = np.linspace(1, 140, 100)

# Ideal linear speedup
ax2.plot(n_range_amd, n_range_amd, 'k--', label='Ideal (linear)', linewidth=1.5, alpha=0.7)

# Amdahl's Law curve
amdahl_curve_amd = amdahl(n_range_amd, amd_P)
ax2.plot(n_range_amd, amdahl_curve_amd, 'g-', 
         label=f"Amdahl's Law (P={amd_P*100:.2f}%)", linewidth=2)

# Measured data (mark 128 as anomaly)
ax2.scatter(amd_threads[:-1], amd_speedup[:-1], color='orange', s=100, zorder=5, 
            label='Measured', edgecolors='black', linewidths=1)
ax2.scatter(amd_threads[-1], amd_speedup[-1], color='red', s=100, zorder=5, 
            marker='x', linewidths=3, label='128 threads (anomaly)')

ax2.set_xlabel('Number of Threads', fontsize=12)
ax2.set_ylabel('Speedup', fontsize=12)
ax2.set_title('AMD EPYC - Speedup vs Threads', fontsize=14, fontweight='bold')
ax2.legend(loc='upper left', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 140)
ax2.set_ylim(0, max(50, 1/(1-amd_P) * 1.1))

# Add annotations for key points
for i, (t, s) in enumerate(zip(amd_threads, amd_speedup)):
    if t in [1, 64, 128]:
        ax2.annotate(f'{s:.1f}x', (t, s), textcoords="offset points", 
                    xytext=(5, 10), fontsize=9)

plt.tight_layout()
plt.savefig('tests/perf/amdahl_speedup_graph.png', dpi=150, bbox_inches='tight')
plt.savefig('tests/perf/amdahl_speedup_graph.pdf', bbox_inches='tight')
print("\n" + "=" * 70)
print("Graphs saved to:")
print("  - tests/perf/amdahl_speedup_graph.png")
print("  - tests/perf/amdahl_speedup_graph.pdf")
print("=" * 70)

# Show plot
plt.show()
