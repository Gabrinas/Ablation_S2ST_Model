import matplotlib.pyplot as plt

# Data
epochs = list(range(1, 27))
VL_a = [5.6731,5.6249,5.6224,5.6123,5.6119,5.6116,5.6053,5.5970,5.5973,5.5973,
        5.5905,5.5677,5.5316,5.4716,5.3855,5.2961,5.2133,5.1201,5.0237,4.9240,
        4.8268,4.7224,4.6239,4.5295,4.4495,4.3768]
VA_a = [0.0111,0.0118,0.0151,0.0151,0.0150,0.0147,0.0153,0.0159,0.0163,0.0158,
        0.0158,0.0163,0.0356,0.0363,0.0408,0.0474,0.0575,0.0674,0.0731,0.0801,
        0.0855,0.0947,0.1086,0.1183,0.1220,0.1354]

VL_g = [5.6703,5.5746,5.4559,5.2911,5.1159,4.9438,4.7887,4.6359,4.5135,4.3919,
        4.2723,4.1513,4.0229,3.9021,3.7616,3.6110,3.4743,3.3318,3.1833,3.0453,
        2.9108,2.7670,2.6307,2.5084,2.3853,2.3991]
VA_g = [0.0111,0.0325,0.0520,0.0569,0.0655,0.0838,0.0998,0.1177,0.1319,0.1472,
        0.1588,0.1744,0.1961,0.2138,0.2424,0.2783,0.3039,0.3382,0.3722,0.3972,
        0.4273,0.4562,0.4948,0.5301,0.5663,0.5704]

# Font configuration
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Subplot 1: Validation Loss
ax1.plot(epochs, VL_a, 'o-', color='blue', label='LSTM Decoder')
ax1.plot(epochs, VL_g, 's-', color='red', label='GRU Decoder')
ax1.set_title('Validation Loss (VL)  of U2UT Model With Joint Head = 4', fontsize=12)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Validation Loss', fontsize=12)
ax1.legend(loc='upper right', fontsize=12)
ax1.grid(True)

# Subplot 2: Validation Accuracy
ax2.plot(epochs, VA_a, 'o-', color='blue', label='LSTM Decoder')
ax2.plot(epochs, VA_g, 's-', color='red', label='GRU Decoder')
ax2.set_title('Validation Accuracy (VA)  of U2UT Model With Joint Head = 4', fontsize=12)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Validation Accuracy', fontsize=12)
ax2.legend(loc='lower right', fontsize=12)
ax2.grid(True)

plt.tight_layout()

# Save as SVG file
plt.savefig('U2UT_VL_VA_comparison.svg', format='svg', dpi=300)
plt.show()
