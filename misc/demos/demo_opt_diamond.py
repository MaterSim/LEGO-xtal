from pyxtal.lego.builder import builder
from pyxtal import pyxtal

# Get the graphite reference environment and set up optimizer
xtal = pyxtal()
xtal.from_prototype('graphite')
cif_file = xtal.to_pymatgen()
bu = builder(['C'], [1])
bu.set_descriptor_calculator(mykwargs={'rcut': 2.0})
bu.set_reference_enviroments(cif_file)
print(bu)

# Get the diamond crystal
xtal = pyxtal()
xtal.from_prototype('diamond')
print(xtal)
print(xtal.get_1d_rep_x())
_, _, _ = bu.optimize_xtal(xtal)

# Subgroup conversion
sub_xtal = xtal.subgroup_once(H=166, eps=1e-4)
sub_xtal.to_file('sp3.cif')
print(sub_xtal)
print(sub_xtal.get_1d_rep_x())

# Optimization
sub_xtal, loss, _ = bu.optimize_xtal(sub_xtal)
print(sub_xtal)
print(sub_xtal.get_1d_rep_x())
sub_xtal.to_file('sp2.cif')
