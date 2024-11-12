"""Tests for data mergeing and preparation of the estimation data set."""

# def test_merge_waves();
# def is_weakly_increasing(series):
# # Group the data by mergeid and apply the is_weakly_increasing function
# # Assert that all groups have weakly increasing work experience
# def is_weakly_increasing(series):
# # Group the data by mergeid and apply the is_weakly_increasing function
# # Assert that all groups have weakly increasing work experience
import pandas as pd

from caregiving.config import BLD


def test_data_merged():
    """Tests shape of estimation data set."""
    data = pd.read_csv(BLD / "data" / "data_merged.csv")

    n_cols = 294

    assert data.shape == (26593, n_cols)


def test_parent_child_data_merged():
    """Tests shape of parent child data set."""
    data = pd.read_csv(BLD / "data" / "data_parent_child_merged.csv")

    n_cols = 405

    assert data.shape == (24607, n_cols)
