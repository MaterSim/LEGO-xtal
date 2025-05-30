from pyxtal import pyxtal
from pyxtal.lego.builder import builder
xtal = pyxtal()
xtal.from_spg_wps_rep(194, ['2c', '2b'], [2.46, 6.70])
cif_file = xtal.to_pymatgen()
bu = builder(['C'], [1], verbose=False)
bu.set_descriptor_calculator(mykwargs={'rcut': 2.0})
bu.set_reference_enviroments(cif_file)

c1 = pyxtal()
c1.from_spg_wps_rep(53, ['4h', '4h', '4h'], [2.365503,  3.005858, 12.789526,  0.003843,  0.414543,  0.009659, 0.050974,  0.0076,  0.815])
print(c1)
c1.to_file('init.cif')
c2, sim, sta = bu.optimize_xtal(c1, opt_type='global')
c2.to_file('opt.cif')
print(c2, sim)
