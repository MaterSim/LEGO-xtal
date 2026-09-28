from copy import deepcopy

import pytest

from lego.builder import builder


class FakeWP:
    def __init__(self, label):
        self.label = label

    def get_label(self):
        return self.label


class FakeSite:
    def __init__(self, specie, wp, properties=None):
        self.specie = specie
        self.wp = FakeWP(wp)
        self.property = deepcopy(properties or {})


class FakeXtal:
    def __init__(self, sites):
        self.atom_sites = sites


def test_copy_site_properties_preserves_nested_metadata():
    source = FakeXtal(
        [
            FakeSite(
                "C",
                "2a",
                {
                    "target_coordination": 3,
                    "lego": {"environment": "sp2"},
                },
            ),
            FakeSite(
                "C",
                "2b",
                {
                    "target_coordination": 4,
                    "lego": {"environment": "sp3"},
                },
            ),
        ]
    )

    target = FakeXtal(
        [
            FakeSite("C", "2a"),
            FakeSite("C", "2b"),
        ]
    )

    builder._copy_site_properties(source, target)

    assert target.atom_sites[0].property["target_coordination"] == 3
    assert target.atom_sites[1].property["target_coordination"] == 4

    target.atom_sites[0].property["lego"]["environment"] = "changed"

    assert (
        source.atom_sites[0].property["lego"]["environment"]
        == "sp2"
    )


def test_copy_site_properties_rejects_count_change():
    source = FakeXtal(
        [
            FakeSite(
                "C",
                "2a",
                {"target_coordination": 3},
            )
        ]
    )

    target = FakeXtal(
        [
            FakeSite("C", "2a"),
            FakeSite("C", "2b"),
        ]
    )

    with pytest.raises(
        ValueError,
        match="atom-site count changed",
    ):
        builder._copy_site_properties(source, target)


def test_copy_site_properties_rejects_reordering():
    source = FakeXtal(
        [
            FakeSite(
                "C",
                "2a",
                {"target_coordination": 3},
            ),
            FakeSite(
                "C",
                "2b",
                {"target_coordination": 4},
            ),
        ]
    )

    target = FakeXtal(
        [
            FakeSite("C", "2b"),
            FakeSite("C", "2a"),
        ]
    )

    with pytest.raises(
        ValueError,
        match="atom-site ordering changed",
    ):
        builder._copy_site_properties(source, target)
