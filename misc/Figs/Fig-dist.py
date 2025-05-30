import ase.db as db
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.gridspec as gridspec

sns.set(style="whitegrid")
sns.set_context("paper", font_scale=1.5)
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
    if (row.mace_energy - row.ff_energy) < -1:
        print(row.id, row.mace_energy, row.ff_energy)
# Convert data to numpy array for easier manipulation
data_array = np.array(data, dtype=float)
nbins = 100
bins = np.linspace(0, eng_max, nbins+1)

# Create a 3*2 grid of subplots
fig = plt.figure(figsize=(14, 8))
gs = gridspec.GridSpec(3, 2, hspace=0.25, wspace=0.2, width_ratios=[1.2, 1])

# (0, 0): MACE energy distribution by atom count
ax1 = fig.add_subplot(gs[0, 0])
natom_bins = [0, 40, 80, 160, np.inf]
natom_labels = ['1-40', '41-80', '81-160', '>160']

# Get indices for each bin
col, bin_indices = 2, []
for i in range(len(natom_bins)-1):
    mask = (data_array[:, col] >= natom_bins[i]) & (data_array[:, col] < natom_bins[i+1])
    bin_indices.append(mask)
    natom_labels[i] += f" ({mask.sum()}, {mask.sum()*100/len(data_array):.1f}%)"
    print(natom_labels[i], mask.sum())

heights = []
bottom = np.zeros(nbins)
for i, mask in enumerate(bin_indices):
    hist, bins = np.histogram(data_array[mask, 0], bins=bins)
    heights.append(hist)
for i, hist in enumerate(heights):
    ax1.bar(bins[:-1], hist, width=np.diff(bins), bottom=bottom,
                 alpha=0.7, label=natom_labels[i])
    bottom += hist
print(bins[0], bottom[0], bins[-1], bottom[-1], sum(bottom))

ax1.set_title('(a) Number of Atoms')
ax1.legend(frameon=False, loc=2, bbox_to_anchor=(-0.02, 0.95))
ax1.grid(False)
ax1.set_xlim(-0.01, eng_max)
ax1.set_xticks([])
ax1.set_ylabel('Count')

# (1, 0): MACE energy distribution by space group symmetry
ax2 = fig.add_subplot(gs[1, 0])
spg_bins = [0, 74, 142, 194, np.inf]
spg_labels = ['1-74', '75-142', '143-194', '195-230']
col, bin_indices = 3, []
for i in range(len(spg_bins)-1):
    mask = (data_array[:, col] > spg_bins[i]) & (data_array[:, col] <= spg_bins[i+1])
    bin_indices.append(mask)
    spg_labels[i] += f" ({mask.sum()}, {mask.sum()*100/len(data_array):.1f}%)"

heights = []
bottom = np.zeros(nbins)
for i, mask in enumerate(bin_indices):
    hist, bins = np.histogram(data_array[mask, 0], bins=bins)
    heights.append(hist)

for i, hist in enumerate(heights):
    ax2.bar(bins[:-1], hist, width=np.diff(bins), bottom=bottom,
                  alpha=0.7, label=spg_labels[i])
    bottom += hist
print(bins[0], bottom[0], bins[-1], bottom[-1], sum(bottom))

ax2.set_ylabel('Count')
ax2.set_title('(b) Space Group')
ax2.legend(frameon=False, loc=2, bbox_to_anchor=(-0.02, 0.95))
ax2.grid(False)
ax2.set_xlim(-0.01, eng_max)
ax2.set_xticks([])

# (2, 0): MACE energy distribution by dimension
ax3 = fig.add_subplot(gs[0, 1])
dim_bins = [-1, 2, np.inf]
dim_labels = ['0-2D', '3D']
col, bin_indices = 4, []
for i in range(len(dim_bins)-1):
    mask = (data_array[:, col] > dim_bins[i]) & (data_array[:, col] <= dim_bins[i+1])
    bin_indices.append(mask)
    dim_labels[i] += f" ({mask.sum()}, {mask.sum()*100/len(data_array):.1f}%)"

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
ax3.set_title('(d) Dimension')
ax3.legend(frameon=False, loc=2, bbox_to_anchor=(-0.02, 0.9))
ax3.grid(False)
ax3.set_xlim(-0.01, eng_max)
ax3.set_xticks([])


# (0, 1): MACE energy distribution by dimension
ax4 = fig.add_subplot(gs[2, 0])
dof_bins = [0, 10, 15, 20, np.inf]
dof_labels = ['1-10', '11-15', '16-20', '21-30']
col, bin_indices = 5, []
for i in range(len(dof_bins)-1):
    mask = (data_array[:, col] > dof_bins[i]) & (data_array[:, col] <= dof_bins[i+1])
    bin_indices.append(mask)
    dof_labels[i] += f" ({mask.sum()}, {mask.sum()*100/len(data_array):.1f}%)"

heights = []
bottom = np.zeros(nbins)
for i, mask in enumerate(bin_indices):
    hist, bins = np.histogram(data_array[mask, 0], bins=bins)
    heights.append(hist)

for i, hist in enumerate(heights):
    ax4.bar(bins[:-1], hist, width=np.diff(bins), bottom=bottom,
                  alpha=0.7, label=dof_labels[i])
    bottom += hist
print(bins[0], bottom[0], bins[-1], bottom[-1], sum(bottom))
#ax4.set_xlabel('MACE Energy (eV)')
ax4.set_ylabel('Count')
ax4.set_title('(c) Number of reduced variables')
ax4.legend(frameon=False, loc=2, bbox_to_anchor=(-0.02, 0.9))
ax4.grid(False)
ax4.set_xlim(-0.01, eng_max)
ax4.set_xlabel('MACE Energy (eV)')

# Energy correlation scatter plot in the right column
ax = fig.add_subplot(gs[1:, 1])
ax.scatter(data_array[:,0], data_array[:,1])
ax.set_xlabel('MACE Energy (eV)')
ax.set_ylabel('FF Energy (eV)')
ax.set_xlim(-0.01, eng_max)
ax.set_ylim(-8.8, -8.1)
ax.set_title('(e) MACE vs FF Energy')

plt.tight_layout()
plt.grid(False)
plt.savefig("lego-sp2.pdf")
