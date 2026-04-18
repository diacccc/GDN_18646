import matplotlib.pyplot as plt

B = [1, 8, 32, 64, 128, 256]
baseline = [3.1185, 22.1895, 86.8332, 173.4614, 359.9537, 728.528]
wo_unroll = [0.0097, 0.0249, 0.1333, 0.2533, 0.495, 0.9788]
way8_unroll = [0.0088, 0.0259, 0.0927, 0.1777, 0.3458, 0.6832]

plt.figure(figsize=(10, 6))

plt.plot(B, baseline, marker='o', label='baseline', linewidth=2)
plt.plot(B, wo_unroll, marker='s', label='w/o unroll', linewidth=2)
plt.plot(B, way8_unroll, marker='^', label='8-way unroll', linewidth=2)

plt.yscale('log')

plt.xlabel('B (Batch Size)', fontsize=12)
plt.ylabel('Performance / Time (Log Scale)', fontsize=12)
plt.title('Performance Comparison', fontsize=14)
plt.xticks(B) 
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend()

plt.show()