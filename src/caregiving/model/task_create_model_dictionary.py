"""Assemble model dictionaries."""

from pathlib import Path
from typing import Annotated

from pytask import Product

from caregiving.config import BLD
from caregiving.utils import load_dict_from_pickle, save_dict_to_pickle


def task_assemble_exog_processes(
    path_to_survival_prob_female: Path = BLD
    / "model"
    / "exog_survival_prob_female.pkl",
    path_to_health_transition_female: Path = BLD
    / "model"
    / "exog_health_transition_female.pkl",
    path_to_save: Annotated[Path, Product] = BLD / "model" / "exog_processes.pkl",
):
    """Assemble exogenous processes dictionaries.

    Exogenous processes:
    - survival probabilities
    - health transition

    No care demand atm! Implicit by medium and bad health of mother.

    # path_to_care_single_mother: Path = BLD / "model" / "exog_care_single_mother.pkl",
    # path_to_care_single_father: Path = BLD / "model" / "exog_care_single_father.pkl",
    # path_to_care_couple: Path = BLD / "model" / "exog_care_couple.pkl",

    exog_care_single_mother = load_dict_from_pickle(path_to_care_single_mother)
    exog_care_single_father = load_dict_from_pickle(path_to_care_single_father)
    exog_care_couple = load_dict_from_pickle(path_to_care_couple)
    _exog_care = exog_care_single_mother | exog_care_single_father | exog_care_couple

    """
    exog_survival_prob_female = load_dict_from_pickle(path_to_survival_prob_female)
    _exog_survival_prob = exog_survival_prob_female

    exog_health_transition_female = load_dict_from_pickle(
        path_to_health_transition_female,
    )
    _exog_health_transition = exog_health_transition_female

    exog_all = _exog_survival_prob | _exog_health_transition

    save_dict_to_pickle(exog_all, path_to_save)

    return exog_all
