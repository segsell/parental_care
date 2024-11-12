import pandas as pd
import pytest

from caregiving.config import BLD


def is_weakly_increasing(series):
    return (series.diff().dropna() >= 0).all()


@pytest.fixture()
def data():
    return pd.read_csv(BLD / "data" / "estimation_data.csv")


def test_work_experience_weakly_increasing(data):
    """Checks if work experience is weakly increasing for each individual."""

    # Sort the data by mergeid and int_year
    dat_sorted = data.sort_values(by=["mergeid", "int_year"])

    assert (data["work_exp"] >= 0).all()

    # Group the data by mergeid and apply the is_weakly_increasing function
    result = dat_sorted.groupby("mergeid")["work_exp"].apply(is_weakly_increasing)

    # Assert that all groups have weakly increasing work experience
    assert result.all()

    mask = ~data["mergeid"].isin(result.index)
    # Assert that 'mask' is False everywhere
    assert not mask.all()


def test_work_experience_non_negative(data):
    """Checks if all elements are non-negative."""
    assert (data["work_exp"] >= 0).all()
