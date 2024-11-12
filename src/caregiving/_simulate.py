"""Module to create simulated moments."""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

MIN_AGE = 40
MAX_AGE = 75

AGE_40 = 40
AGE_45 = 45
AGE_50 = 50
AGE_55 = 55
AGE_60 = 60
AGE_65 = 65
AGE_70 = 70
AGE_75 = 75

AGE_BINS = [
    (AGE_40 - MIN_AGE, AGE_45 - MIN_AGE),
    (AGE_45 - MIN_AGE, AGE_50 - MIN_AGE),
    (AGE_50 - MIN_AGE, AGE_55 - MIN_AGE),
    (AGE_55 - MIN_AGE, AGE_60 - MIN_AGE),
    (AGE_60 - MIN_AGE, AGE_65 - MIN_AGE),
    (AGE_65 - MIN_AGE, AGE_70 - MIN_AGE),
    (AGE_70 - MIN_AGE, AGE_75 - MIN_AGE),
]

PARENT_MIN_AGE = 68
PARENT_MAX_AGE = 98

RETIREMENT_AGE = 62

GOOD_HEALTH = 0
MEDIUM_HEALTH = 1
BAD_HEALTH = 2

ALL = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]


TOTAL_WEEKLY_HOURS = 80
WEEKLY_HOURS_PART_TIME = 20
WEEKLY_HOURS_FULL_TIME = 40
WEEKLY_INTENSIVE_INFORMAL_HOURS = 14  # (21 + 7) / 2

N_MONTHS = 12
N_WEEKS = 4.33

PART_TIME_HOURS = 20 * N_WEEKS * N_MONTHS
FULL_TIME_HOURS = 40 * N_WEEKS * N_MONTHS


# ==============================================================================
# JAX Arrays
# ==============================================================================


ALL = jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

NO_WORK = jnp.array([0, 1, 2, 3])
PART_TIME = jnp.array([4, 5, 6, 7])
FULL_TIME = jnp.array([8, 9, 10, 11])
WORK = jnp.concatenate([PART_TIME, FULL_TIME])

NO_CARE = jnp.array([0, 4, 8])
FORMAL_CARE = jnp.array([1, 3, 5, 7, 9, 11])
INFORMAL_CARE = jnp.array([2, 3, 6, 7, 10, 11])
CARE = jnp.concatenate([FORMAL_CARE, INFORMAL_CARE])

COMBINATION_CARE = jnp.array([3, 7, 11])

# For NO_INFORMAL_CARE and NO_FORMAL_CARE, we need to perform set operations before
# converting to JAX arrays.
# This is because JAX doesn't support direct set operations.
# Convert the results of set operations to lists, then to JAX arrays.
NO_INFORMAL_CARE = jnp.array(list(set(ALL.tolist()) - set(INFORMAL_CARE.tolist())))
NO_FORMAL_CARE = jnp.array(list(set(ALL.tolist()) - set(FORMAL_CARE.tolist())))


# ==============================================================================
# Model
# ==============================================================================


def is_not_working(lagged_choice):
    return jnp.any(lagged_choice == NO_WORK)


def is_part_time(lagged_choice):
    return jnp.any(lagged_choice == PART_TIME)


def is_full_time(lagged_choice):
    return jnp.any(lagged_choice == FULL_TIME)


def is_informal_care(lagged_choice):
    # intensive only here
    return jnp.any(lagged_choice == INFORMAL_CARE)


def is_no_informal_care(lagged_choice):
    # intensive only here
    return jnp.all(lagged_choice != INFORMAL_CARE)


def is_formal_care(lagged_choice):
    return jnp.any(lagged_choice == FORMAL_CARE)


def is_no_formal_care(lagged_choice):
    return jnp.all(lagged_choice != FORMAL_CARE)


def draw_initial_states(initial_conditions, initial_wealth, n_agents, seed):
    """Draw initial resources and states for the simulation.

    mother_health = get_initial_share_three(     initial_conditions,
    ["mother_good_health", "mother_medium_health", "mother_bad_health"], ) father_health
    = get_initial_share_three(     initial_conditions,     ["father_good_health",
    "father_medium_health", "father_bad_health"], )

    informal_care = get_initial_share_two(initial_conditions, "share_informal_care")

    """
    n_choices = 12

    # choices
    employment = get_initial_share_three(
        initial_conditions,
        ["share_not_working", "share_part_time", "share_full_time"],
    )

    informal_care = jnp.array([1, 0])  # agents start with no informal care provided
    formal_care = jnp.array([1, 0])  # agents start with no formal care provided

    informal_formal = jnp.outer(informal_care, formal_care)
    lagged_choice_probs = jnp.outer(employment, informal_formal).flatten()

    # endogenous states
    married = get_initial_share_two(initial_conditions, "share_married")
    has_sibling = get_initial_share_two(initial_conditions, "share_has_sibling")

    # exogenous states
    mother_alive = get_initial_share_two(initial_conditions, "share_mother_alive")
    father_alive = get_initial_share_two(initial_conditions, "share_father_alive")

    initial_resources = draw_random_sequence_from_array(
        seed,
        initial_wealth,
        n_agents=n_agents,
    )

    initial_states = {
        "period": jnp.zeros(n_agents, dtype=np.int16),
        "lagged_choice": draw_random_array(
            seed=seed - 1,
            n_agents=n_agents,
            values=jnp.arange(n_choices),
            probabilities=lagged_choice_probs,
        ).astype(np.int16),
        "married": draw_random_array(
            seed=seed - 2,
            n_agents=n_agents,
            values=jnp.array([0, 1]),
            probabilities=married,
        ).astype(np.int16),
        "has_sibling": draw_random_array(
            seed=seed - 3,
            n_agents=n_agents,
            values=jnp.array([0, 1]),
            probabilities=has_sibling,
        ).astype(np.int16),
        # exogenous states
        "part_time_offer": jnp.ones(n_agents, dtype=np.int16),
        "full_time_offer": jnp.ones(n_agents, dtype=np.int16),
        "care_demand": jnp.zeros(n_agents, dtype=np.int16),
        "mother_alive": draw_random_array(
            seed=seed - 6,
            n_agents=n_agents,
            values=jnp.array([0, 1]),
            probabilities=mother_alive,
        ).astype(np.int16),
        "father_alive": draw_random_array(
            seed=seed - 7,
            n_agents=n_agents,
            values=jnp.array([0, 1]),
            probabilities=father_alive,
        ).astype(np.int16),
    }

    return initial_resources, initial_states


def simulate_moments(sim):
    """Df has multiindex ["period", "agent"] necessary?

    or "agent", "period" as columns. "age" is also a column

    .loc needed below?!


    share_not_working_no_informal_care_by_age_bin = get_share_by_type_by_age_bin(
        arr,
        ind=idx,
        lagged_choice=NO_WORK,
        care_type=NO_INFORMAL_CARE,
    )
    share_part_time_no_informal_care_by_age_bin = get_share_by_type_by_age_bin(
        arr,
        ind=idx,
        lagged_choice=PART_TIME,
        care_type=NO_INFORMAL_CARE,
    )
    share_full_time_no_informal_care_by_age_bin = get_share_by_type_by_age_bin(
        arr,
        ind=idx,
        lagged_choice=FULL_TIME,
        care_type=NO_INFORMAL_CARE,
    )

    share_not_working_informal_care_by_age_bin = get_share_by_type_by_age_bin(
        arr,
        ind=idx,
        lagged_choice=NO_WORK,
        care_type=INFORMAL_CARE,
    )
    share_part_time_informal_care_by_age_bin = get_share_by_type_by_age_bin(
        arr,
        ind=idx,
        lagged_choice=PART_TIME,
        care_type=INFORMAL_CARE,
    )
    share_full_time_informal_care_by_age_bin = get_share_by_type_by_age_bin(
        arr,
        ind=idx,
        lagged_choice=FULL_TIME,
        care_type=INFORMAL_CARE,
    )

    """
    column_indices = {col: idx for idx, col in enumerate(sim.columns)}
    idx = column_indices.copy()
    arr = jnp.asarray(sim)

    # share working by age
    share_not_working_by_age = get_share_by_age(
        arr,
        ind=idx,
        lagged_choice=NO_WORK,
    )  # 15
    share_working_part_time_by_age = get_share_by_age(
        arr,
        ind=idx,
        lagged_choice=PART_TIME,
    )  # 15
    share_working_full_time_by_age = get_share_by_age(
        arr,
        ind=idx,
        lagged_choice=FULL_TIME,
    )  # 15
    # do timeit of jnp.array versus jnp.asarray?

    share_informal_care_by_age_bin = get_share_by_type_by_age_bin(
        arr,
        ind=idx,
        care_type=ALL,
        lagged_choice=INFORMAL_CARE,
    )

    # yearly net income
    # Caution: Some bug here!! Zero income, althouh working (full-time),
    # so people have to live off of their initial wealth
    income_part_time_by_age_bin = get_mean_by_age_bin_for_lagged_choice(
        arr,
        ind=idx,
        var="income",
        lagged_choice=PART_TIME,
    )
    income_full_time_by_age_bin = get_mean_by_age_bin_for_lagged_choice(
        arr,
        ind=idx,
        var="income",
        lagged_choice=FULL_TIME,
    )

    # wealth
    wealth_by_age_bin = get_mean_by_age_bin_for_lagged_choice(
        arr,
        ind=idx,
        var="wealth",
        lagged_choice=ALL,
    )

    # share working by caregiving type (and age bin) --> to be checked

    share_not_working_no_informal_care = get_share_by_type(
        arr,
        ind=idx,
        lagged_choice=NO_WORK,
        care_type=NO_INFORMAL_CARE,
    )
    share_part_time_no_informal_care = get_share_by_type(
        arr,
        ind=idx,
        lagged_choice=PART_TIME,
        care_type=NO_INFORMAL_CARE,
    )
    share_full_time_no_informal_care = get_share_by_type(
        arr,
        ind=idx,
        lagged_choice=FULL_TIME,
        care_type=NO_INFORMAL_CARE,
    )

    share_not_working_informal_care = get_share_by_type(
        arr,
        ind=idx,
        lagged_choice=NO_WORK,
        care_type=INFORMAL_CARE,
    )
    share_part_time_informal_care = get_share_by_type(
        arr,
        ind=idx,
        lagged_choice=PART_TIME,
        care_type=INFORMAL_CARE,
    )
    share_full_time_informal_care = get_share_by_type(
        arr,
        ind=idx,
        lagged_choice=FULL_TIME,
        care_type=INFORMAL_CARE,
    )

    # work transitions
    no_work_to_no_work = get_transition(
        arr,
        ind=idx,
        lagged_choice=NO_WORK,
        current_choice=NO_WORK,
    )
    no_work_to_part_time = get_transition(
        arr,
        ind=idx,
        lagged_choice=NO_WORK,
        current_choice=PART_TIME,
    )
    no_work_to_full_time = get_transition(
        arr,
        ind=idx,
        lagged_choice=NO_WORK,
        current_choice=FULL_TIME,
    )

    part_time_to_no_work = get_transition(
        arr,
        ind=idx,
        lagged_choice=PART_TIME,
        current_choice=NO_WORK,
    )
    part_time_to_part_time = get_transition(
        arr,
        ind=idx,
        lagged_choice=PART_TIME,
        current_choice=PART_TIME,
    )
    part_time_to_full_time = get_transition(
        arr,
        ind=idx,
        lagged_choice=PART_TIME,
        current_choice=FULL_TIME,
    )

    full_time_to_no_work = get_transition(
        arr,
        ind=idx,
        lagged_choice=FULL_TIME,
        current_choice=NO_WORK,
    )
    full_time_to_part_time = get_transition(
        arr,
        ind=idx,
        lagged_choice=FULL_TIME,
        current_choice=PART_TIME,
    )
    full_time_to_full_time = get_transition(
        arr,
        ind=idx,
        lagged_choice=FULL_TIME,
        current_choice=FULL_TIME,
    )

    # caregiving transitions
    no_informal_care_to_no_informal_care = get_transition(
        arr,
        ind=idx,
        lagged_choice=NO_INFORMAL_CARE,
        current_choice=NO_INFORMAL_CARE,
    )
    no_informal_care_to_informal_care = get_transition(
        arr,
        ind=idx,
        lagged_choice=NO_INFORMAL_CARE,
        current_choice=INFORMAL_CARE,
    )

    informal_care_to_no_informal_care = get_transition(
        arr,
        ind=idx,
        lagged_choice=INFORMAL_CARE,
        current_choice=NO_INFORMAL_CARE,
    )
    informal_care_to_informal_care = get_transition(
        arr,
        ind=idx,
        lagged_choice=INFORMAL_CARE,
        current_choice=INFORMAL_CARE,
    )

    no_informal_care_to_no_formal_care = get_transition(
        arr,
        ind=idx,
        lagged_choice=NO_INFORMAL_CARE,
        current_choice=NO_FORMAL_CARE,
    )
    no_informal_care_to_formal_care = get_transition(
        arr,
        ind=idx,
        lagged_choice=NO_INFORMAL_CARE,
        current_choice=FORMAL_CARE,
    )

    informal_care_to_no_formal_care = get_transition(
        arr,
        ind=idx,
        lagged_choice=INFORMAL_CARE,
        current_choice=NO_FORMAL_CARE,
    )
    informal_care_to_formal_care = get_transition(
        arr,
        ind=idx,
        lagged_choice=INFORMAL_CARE,
        current_choice=FORMAL_CARE,
    )

    no_formal_care_to_no_informal_care = get_transition(
        arr,
        ind=idx,
        lagged_choice=NO_FORMAL_CARE,
        current_choice=NO_INFORMAL_CARE,
    )
    no_formal_care_to_informal_care = get_transition(
        arr,
        ind=idx,
        lagged_choice=NO_FORMAL_CARE,
        current_choice=INFORMAL_CARE,
    )

    formal_care_to_no_informal_care = get_transition(
        arr,
        ind=idx,
        lagged_choice=FORMAL_CARE,
        current_choice=NO_INFORMAL_CARE,
    )
    formal_care_to_informal_care = get_transition(
        arr,
        ind=idx,
        lagged_choice=FORMAL_CARE,
        current_choice=INFORMAL_CARE,
    )

    no_formal_care_to_no_formal_care = get_transition(
        arr,
        ind=idx,
        lagged_choice=NO_FORMAL_CARE,
        current_choice=NO_FORMAL_CARE,
    )
    no_formal_care_to_formal_care = get_transition(
        arr,
        ind=idx,
        lagged_choice=NO_FORMAL_CARE,
        current_choice=FORMAL_CARE,
    )

    formal_care_to_no_formal_care = get_transition(
        arr,
        ind=idx,
        lagged_choice=FORMAL_CARE,
        current_choice=NO_FORMAL_CARE,
    )
    formal_care_to_formal_care = get_transition(
        arr,
        ind=idx,
        lagged_choice=FORMAL_CARE,
        current_choice=FORMAL_CARE,
    )

    return jnp.asarray(
        share_not_working_by_age
        + share_working_part_time_by_age
        + share_working_full_time_by_age
        + share_informal_care_by_age_bin
        + income_part_time_by_age_bin
        + income_full_time_by_age_bin
        + wealth_by_age_bin
        + [share_not_working_no_informal_care]
        + [share_part_time_no_informal_care]
        + [share_full_time_no_informal_care]
        + [share_not_working_informal_care]
        + [share_part_time_informal_care]
        + [share_full_time_informal_care]
        + no_work_to_no_work
        + no_work_to_part_time
        + no_work_to_full_time
        + part_time_to_no_work
        + part_time_to_part_time
        + part_time_to_full_time
        + full_time_to_no_work
        + full_time_to_part_time
        + full_time_to_full_time
        +
        # caregiving transitions
        no_informal_care_to_no_informal_care
        + no_informal_care_to_informal_care
        + informal_care_to_no_informal_care
        + informal_care_to_informal_care
        + no_informal_care_to_no_formal_care
        + no_informal_care_to_formal_care
        + informal_care_to_no_formal_care
        + informal_care_to_formal_care
        + no_formal_care_to_no_informal_care
        + no_formal_care_to_informal_care
        + formal_care_to_no_informal_care
        + formal_care_to_informal_care
        + no_formal_care_to_no_formal_care
        + no_formal_care_to_formal_care
        + formal_care_to_no_formal_care
        + formal_care_to_formal_care,
    )


def create_simulation_df(sim_dict, options, params):
    n_periods, n_agents, n_choices = sim_dict["taste_shocks"].shape

    keys_to_drop = ["taste_shocks"]
    dict_to_df = {key: sim_dict[key] for key in sim_dict if key not in keys_to_drop}

    dat = pd.DataFrame(
        {key: val.ravel() for key, val in dict_to_df.items()},
        index=pd.MultiIndex.from_product(
            [np.arange(n_periods), np.arange(n_agents)],
            names=["period", "agent"],
        ),
    )

    dat["wealth"] = dat["savings"] + dat["consumption"]
    dat["age"] = dat["period"] + options["model_params"]["min_age"]

    dat["log_wage"] = (
        params["wage_constant"]
        + params["wage_age"] * dat["age"]
        + params["wage_age_squared"] * dat["age"] ** 2
        + params["wage_part_time"] * (dat["lagged_choice"].isin(PART_TIME))
        + params["wage_not_working"] * (dat["lagged_choice"].isin(FULL_TIME))
    )

    dat["wage"] = np.exp(dat["log_wage"] + dat["income_shock"])
    dat["working_hours"] = dat["lagged_choice"].apply(_assign_working_hours)
    dat["income"] = dat["working_hours"] * dat["wage"]

    return dat


def _assign_working_hours(choice):
    if choice in NO_WORK:
        hours = 0
    elif choice in PART_TIME:
        hours = 20
    elif choice in FULL_TIME:
        hours = 40
    else:
        hours = None

    return hours


# ==============================================================================


def get_share_by_age(df_arr, ind, lagged_choice):
    """Get share of agents choosing lagged choice by age bin."""
    lagged_choice_mask = jnp.isin(df_arr[:, ind["lagged_choice"]], lagged_choice)
    shares = []
    for period in range(MAX_AGE - MIN_AGE + 1):
        period_mask = df_arr[:, ind["period"]] == period

        share = jnp.sum(period_mask & lagged_choice_mask) / jnp.sum(period_mask)
        # period count is always larger than 0! otherwise error
        shares.append(share)

    return shares


def get_share_by_age_bin(df_arr, ind, lagged_choice):
    """Get share of agents choosing lagged choice by age bin."""
    return [
        jnp.mean(
            jnp.isin(df_arr[:, ind["lagged_choice"]], lagged_choice)
            & (df_arr[:, ind["period"]] > age_bin[0])
            & (df_arr[:, ind["period"]] <= age_bin[1]),
        )
        for age_bin in AGE_BINS
    ]


def get_mean_by_age_bin_for_lagged_choice(df_arr, ind, var, lagged_choice):
    """Get mean of agents choosing lagged choice by age bin."""
    lagged_choice_mask = jnp.isin(df_arr[:, ind["lagged_choice"]], lagged_choice)
    means = []

    for age_bin in AGE_BINS:
        age_bin_mask = (df_arr[:, ind["period"]] > age_bin[0]) & (
            df_arr[:, ind["period"]] <= age_bin[1]
        )
        means += [jnp.mean(df_arr[lagged_choice_mask & age_bin_mask, ind[var]])]

    return means


def get_share_by_type_by_age_bin(df_arr, ind, lagged_choice, care_type):
    """Get share of agents of given care type choosing lagged choice by age bin."""
    lagged_choice_mask = jnp.isin(df_arr[:, ind["lagged_choice"]], lagged_choice)
    care_type_mask = jnp.isin(df_arr[:, ind["lagged_choice"]], care_type)

    shares = []
    for age_bin in AGE_BINS:
        age_bin_mask = (df_arr[:, ind["period"]] > age_bin[0]) & (
            df_arr[:, ind["period"]] <= age_bin[1]
        )
        share = jnp.sum(lagged_choice_mask & care_type_mask & age_bin_mask) / jnp.sum(
            care_type_mask & age_bin_mask,
        )
        shares.append(share)

    return shares


def get_share_by_type(df_arr, ind, lagged_choice, care_type):
    """Get share of agents of given care type choosing lagged choice by age bin."""
    lagged_choice_mask = jnp.isin(df_arr[:, ind["lagged_choice"]], lagged_choice)
    care_type_mask = jnp.isin(df_arr[:, ind["lagged_choice"]], care_type)

    return jnp.sum(lagged_choice_mask & care_type_mask) / jnp.sum(care_type_mask)


def get_transition(df_arr, ind, lagged_choice, current_choice):
    """Get transition probability from lagged choice to current choice."""
    return [
        jnp.sum(
            jnp.isin(df_arr[:, ind["lagged_choice"]], lagged_choice)
            & jnp.isin(df_arr[:, ind["choice"]], current_choice),
        )
        / jnp.sum(jnp.isin(df_arr[:, ind["lagged_choice"]], lagged_choice)),
    ]


def get_share_care_by_parental_health(
    df_arr,
    ind,
    care_choice,
    parent="mother",
):
    """Get share of agents choosing given care choice by parental health."""
    return [
        jnp.mean(
            jnp.isin(df_arr[:, ind["lagged_choice"]], care_choice)
            & (df_arr[:, ind[f"{parent}_health"]], health),
        )
        for health in (GOOD_HEALTH, MEDIUM_HEALTH, BAD_HEALTH)
    ]


def get_share_care_by_parental_health_and_presence_of_sibling(
    df_arr,
    ind,
    care_choice,
    has_sibling,
    parent,
):
    """Get share of agents choosing given care choice by parental health."""
    return [
        jnp.mean(
            jnp.isin(df_arr[:, ind["lagged_choice"]], care_choice)
            & (df_arr[:, ind["has_sibling"]] == has_sibling)
            & (df_arr[:, ind[f"{parent}_health"]], health),
        )
        for health in (GOOD_HEALTH, MEDIUM_HEALTH, BAD_HEALTH)
    ]


def _get_share_care_type_by_parental_health_and_other_parent_alive(
    df_arr,
    ind,
    care_choice,
    parent,
    is_other_parent_alive,
):
    """Get share of agents choosing given care choice by parental health."""
    other_parent = ("father") * (parent == "mother") + ("mother") * (parent == "father")

    return [
        jnp.mean(
            jnp.isin(df_arr[:, ind["lagged_choice"]], care_choice)
            & (df_arr[:, ind[f"{other_parent}_alive"]] == is_other_parent_alive)
            & (df_arr[:, ind[f"{parent}_health"]], health),
        )
        for health in (GOOD_HEALTH, MEDIUM_HEALTH, BAD_HEALTH)
    ]


# ==============================================================================
# Initial conditions
# ==============================================================================


def get_initial_share_three(initial_conditions, shares):
    return jnp.asarray(initial_conditions.loc[shares]).ravel()


def get_initial_share_two(initial_conditions, var):
    share_yes = initial_conditions.loc[var]
    return jnp.asarray([1 - share_yes, share_yes]).ravel()


def draw_random_sequence_from_array(seed, arr, n_agents):
    """Draw a random sequence from an array.

    rand = draw_random_sequence_from_array(     seed=2024,     n_agents=10_000,
    arr=jnp.array(initial_wealth), )

    """
    key = jax.random.PRNGKey(seed)
    return jax.random.choice(key, arr, shape=(n_agents,), replace=True)


def draw_random_array(seed, n_agents, values, probabilities):
    """Draw a random array with given probabilities.

    Usage:

    seed = 2024
    n_agents = 10_000

    # Parameters
    values = jnp.array([-1, 0, 1, 2])  # Values to choose from
    probabilities = jnp.array([0.3, 0.3, 0.2, 0.2])  # Corresponding probabilities

    table(pd.DataFrame(random_array)[0]) / 1000

    """
    key = jax.random.PRNGKey(seed)
    return jax.random.choice(key, values, shape=(n_agents,), p=probabilities)


def draw_parental_age(seed, n_agents, mean, std_dev):
    """Draw discrete parental age."""
    key = jax.random.PRNGKey(seed)

    sample_standard_normal = jax.random.normal(key, (n_agents,))

    # Scaling and shifting to get the desired mean and standard deviation, then rounding
    return jnp.round(mean + std_dev * sample_standard_normal).astype(jnp.int32)
