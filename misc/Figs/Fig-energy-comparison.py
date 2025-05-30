import ase.db as db
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.gridspec as gridspec

sns.set(style="whitegrid")
sns.set_context("paper", font_scale=1.8)

# Create a 2*1 grid of subplots
fig = plt.figure(figsize=(6.2, 8))
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])

db0 = db.connect("../../data/source/sp2_sacada.db")

data = []
for row in db0.select():
    if hasattr(row, 'mace_energy'):
        data.append([row.mace_energy, row.vasp_energy, row.ff_energy])

data_array = np.array(data, dtype=float)
mace_eng = data_array[:, 0] - data_array[:, 0].min()
vasp_eng = data_array[:, 1] - data_array[:, 1].min()
ff_eng = data_array[:, 2] - data_array[:, 2].min()
ref = np.linspace(vasp_eng.min(), vasp_eng.max(), 100)

ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(vasp_eng, mace_eng, s=10, color='blue')
ax1.plot(ref, ref, color='black', linestyle='--', linewidth=1.5)
ax1.set_title(f'(a) VASP vs. MACE\n')
ax1.set_ylabel('MACE Energy (eV/atom)')
# Set same limits for both axes
max_val = max(vasp_eng.max(), mace_eng.max())
min_val = min(vasp_eng.min(), mace_eng.min())
ax1.set_xlim(min_val, 1.0)
ax1.set_ylim(min_val, 1.0)
# Make the x and y axes have the same ticklabels
ax1.set_xlabel('')  # Remove x-label for top plot
ticks = np.linspace(0, 1.0, 6)
ax1.set_xticks(ticks)
ax1.set_yticks(ticks)

ax2 = fig.add_subplot(gs[1, 0])
ax2.scatter(vasp_eng, ff_eng, s=10, color='orange')
ax2.set_title(f'(b) VASP vs. ReaxFF\n')
ax2.set_ylabel('Reaxff Energy (eV/atom)')
ax2.set_xlabel('VASP Energy (eV/atom)')
ax2.plot(ref, ref, color='black', linestyle='--', linewidth=1.5)
# Set same limits for both axes
max_val = max(vasp_eng.max(), ff_eng.max())
min_val = min(vasp_eng.min(), ff_eng.min())
ax2.set_xlim(min_val, 1.0)
ax2.set_ylim(min_val, 1.0)
# Make the x and y axes have the same ticklabels
ax2.set_xticks(ticks)
ax2.set_yticks(ticks)

plt.tight_layout()
plt.savefig("Fig-energy-comp.pdf")
