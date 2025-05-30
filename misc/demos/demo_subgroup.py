from pyxtal import pyxtal
xtal = pyxtal()
xtal.from_prototype('diamond')
for H in [216, 166, 141]:
    c1 = xtal.subgroup_once(H=H, eps=1e-5)
    c1.to_file(f'diamond-{H}.cif')
    print(c1)
