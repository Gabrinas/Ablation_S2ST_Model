import matplotlib.pyplot as plt

# Data
epochs = list(range(1, 27))
TL_a = [5.8046,5.4051,5.3857,5.3756,5.3697,5.3656,5.3621,5.3577,5.3534,5.3467,5.3309,5.3033,5.2542,5.1811,5.0833,4.9776,4.8835,4.7901,4.6919,4.5886,4.4805,4.3817,4.2745,4.1814,4.0929,4.0177]
TA_a = [0.0323,0.0349,0.0380,0.0390,0.0391,0.0391,0.0393,0.0395,0.0402,0.0404,0.0403,0.0407,0.0461,0.0595,0.0626,0.0727,0.0876,0.0999,0.1137,0.1288,0.1391,0.1478,0.1615,0.1733,0.1841,0.1913]
TL_g = [5.7938,5.3526,5.1877,5.0103,4.8210,4.6349,4.4670,4.3083,4.1697,4.0476,3.9284,3.8079,3.6859,3.5580,3.4305,3.2953,3.1542,3.0217,2.8851,2.7503,2.6181,2.4915,2.3567,2.2391,2.1574,2.5174]
TA_g = [0.0342,0.0392,0.0646,0.0867,0.1033,0.1226,0.1458,0.1669,0.1835,0.2044,0.2206,0.2403,0.2598,0.2821,0.3108,0.3402,0.3728,0.4052,0.4343,0.4630,0.4919,0.5191,0.5486,0.5782,0.6030,0.5484]

# Font configuration
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Subplot 1: Training Loss
ax1.plot(epochs, TL_a, 'o-', color='blue', label='LSTM Decoder')
ax1.plot(epochs, TL_g, 's-', color='red', label='GRU Decoder')
ax1.set_title('Training Loss (TL) of U2UT Model With Joint Head = 4', fontsize=12)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Training Loss', fontsize=12)
ax1.legend(loc='upper right', fontsize=12)
ax1.grid(True)

# Subplot 2: Training Accuracy
ax2.plot(epochs, TA_a, 'o-', color='blue', label='LSTM Decoder')
ax2.plot(epochs, TA_g, 's-', color='red', label='GRU Decoder')
ax2.set_title('Training Accuracy (TA) of U2UT Model With Joint Head = 4', fontsize=12)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Training Accuracy', fontsize=12)
ax2.legend(loc='lower right', fontsize=12)
ax2.grid(True)

plt.tight_layout()

# Save as SVG file
plt.savefig('U2UT_TL_TA_comparison.svg', format='svg', dpi=300)
plt.show()
