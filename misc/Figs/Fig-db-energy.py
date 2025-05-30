import ase.db as db
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.gridspec as gridspec

sns.set(style="whitegrid")
sns.set_context("paper", font_scale=1.8)

eng_min = -9.355
eng_max1 = 1.05
eng_max2 = 0.5

# Create a 2*1 grid of subplots
fig = plt.figure(figsize=(6.2, 8))
gs = gridspec.GridSpec(3, 1, hspace=0.1, wspace=0.1, height_ratios=[1, 0.4, 1])

db0 = db.connect("../../data/source/sp2_sacada.db")
data = [row.mace_energy-eng_min for row in db0.select() if hasattr(row, 'mace_energy') and row.mace_energy < -8.3]
# Convert data to numpy array for easier manipulation
data_array = np.array(data, dtype=float)
nbins = 50
bins = np.linspace(0, eng_max1, nbins+1)
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(data_array, bins=bins, alpha=0.5, label=f'Total: {len(data_array)}', color='rosybrown')
data2 = data_array[data_array < 0.5]
ax1.hist(data2, bins=bins, alpha=0.7, color='r', label=f'Low energy: {len(data2)}')
ax1.legend(loc=2, frameon=False)

ax1.set_title(f'Traing from the known database')
ax1.grid(False)
ax1.set_xlim(-0.005, eng_max1)
ax1.set_ylabel('Count')
ax1.set_xlabel('MACE Energy (eV)')
ax1.set_ylim(0, 15)

# (1, 0): MACE energy distribution by space group symmetry
ax2 = fig.add_subplot(gs[2, 0])
db0 = db.connect("../../data/source/lego-sp2.db")
data = [row.mace_energy-eng_min for row in db0.select() if hasattr(row, 'mace_energy')]
# Convert data to numpy array for easier manipulation
data_array = np.array(data, dtype=float)
#nbins = 100
bins = np.linspace(0, eng_max2, nbins+1)
ax2.hist(data_array, bins=bins, alpha=0.5, color='r', label=f'Low energy: {len(data_array)}')
ax2.set_ylabel('Count')
ax2.set_title(f'Sampling using the LEGO-cryst')
ax2.legend(frameon=False, loc=2)
ax2.grid(False)
ax2.set_xlim(-0.01, eng_max2)
ax2.set_xlabel('MACE Energy (eV)')


plt.tight_layout()
plt.savefig("Fig-energy.pdf")
