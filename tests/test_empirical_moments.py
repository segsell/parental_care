"""Tests for creation of empirical moments."""

import numpy as np

from caregiving.moments.task_create_empirical_moments import (
    get_employment_transitions_soep,
)


def test_employment_transitions_soep():
    """Tests that employment transitions sum to one."""
    trans = get_employment_transitions_soep()

    from_not_working = trans[[idx.startswith("not_working") for idx in trans.index]]
    from_part_time = trans[[idx.startswith("part_time") for idx in trans.index]]
    from_full_time = trans[[idx.startswith("full_time") for idx in trans.index]]

    assert np.allclose(from_not_working.sum(), 1)
    assert np.allclose(from_part_time.sum(), 1)
    assert np.allclose(from_full_time.sum(), 1)
