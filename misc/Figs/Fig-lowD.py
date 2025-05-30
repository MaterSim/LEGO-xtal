import ase.db as db
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.gridspec as gridspec

sns.set(style="whitegrid")
sns.set_context("paper", font_scale=2.0)
db0 = db.connect("../../data/source/lego-sp2.db")
eng_min = -9.357
eng_max = 0.51
data = []
for row in db0.select():
    data.append((row.mace_energy-eng_min,
                 row.ff_energy,
                 row.natoms,
                 row.space_group_number,
                 row.dimension,
                 row.dof))
# Convert data to numpy array for easier manipulation
data_array = np.array(data, dtype=float)
nbins = 100
bins = np.linspace(0, eng_max, nbins+1)

# Create a 3*2 grid of subplots
fig = plt.figure(figsize=(14, 3.5))
gs = gridspec.GridSpec(1, 1)

ax3 = fig.add_subplot(gs[0, 0])
dim_bins = [-1, 2, np.inf]
dim_labels = ['0-2D', '3D']
col, bin_indices = 4, []
for i in range(len(dim_bins)-1):
    mask = (data_array[:, col] > dim_bins[i]) & (data_array[:, col] <= dim_bins[i+1])
    bin_indices.append(mask)
    dim_labels[i] += f" ({mask.sum()})"

heights = []
bottom = np.zeros(nbins)
for i, mask in enumerate(bin_indices):
    hist, bins = np.histogram(data_array[mask, 0], bins=bins)
    heights.append(hist)

for i, hist in enumerate(heights):
    ax3.bar(bins[:-1], hist, width=np.diff(bins), bottom=bottom,
                  alpha=0.7, label=dim_labels[i])
    bottom += hist
print(bins[0], bottom[0], bins[-1], bottom[-1], sum(bottom))
ax3.set_ylabel('Count')
ax3.set_title('(a) Distribution')
ax3.legend(frameon=False, loc=2)
ax3.grid(False)
ax3.set_xlim(-0.01, eng_max)
ax3.set_xlabel('MACE Energy (eV/atom)')

plt.tight_layout()
plt.savefig("Fig-lowD.pdf")
