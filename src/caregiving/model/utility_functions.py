"""Utility functions for the model."""

from typing import Any

import jax.numpy as jnp

from caregiving.model.shared import (
    is_bad_health,
    is_combination_care,
    is_formal_care,
    is_full_time,
    is_informal_care,
    is_no_care,
    is_no_informal_care,
    is_part_time,
    is_pure_informal_care,
)

CONSUMPTION_EQUIVALENCE_SCALE = 0.5

# =====================================================================================


def create_utility_functions():
    """Create dict of utility functions."""
    return {
        "utility": utility_func,
        "marginal_utility": marginal_utility,
        "inverse_marginal_utility": inverse_marginal_utility,
    }


def create_final_period_utility_functions():
    """Create dict of utility functions for the final period."""
    return {
        "utility": utility_final_consume_all,
        "marginal_utility": marginal_utility_final_consume_all,
    }


# =====================================================================================
# Per-period utility
# =====================================================================================


def utility_func(
    consumption: jnp.array,
    choice: int,
    period: int,
    mother_health: int,
    # has_sibling: int,
    # options: dict,
    params: dict,
) -> jnp.array:
    """Computes the agent's current utility based on a CRRA utility function.

    working_hours_weekly = (
        part_time * WEEKLY_HOURS_PART_TIME + full_time * WEEKLY_HOURS_FULL_TIME
    )
    # From SOEP data we know that the 25% and 75% percentile in the care hours
    # distribution are 7 and 21 hours per week in a comparative sample.
    # We use these discrete mass-points as discrete choices of non-intensive and
    # intensive informal care.
    # In SHARE, respondents inform about the frequency with which they provide
    # informal care. We use this information to proxy the care provision in the data.
    caregiving_hours_weekly = informal_care * WEEKLY_INTENSIVE_INFORMAL_HOURS
    leisure_hours = (
        (TOTAL_WEEKLY_HOURS - working_hours_weekly - caregiving_hours_weekly)
        * 4.33  # month
        * 12  # year
    )

    # age is a proxy for health impacting the taste for free-time.
    utility_leisure = (
        (
            params["utility_leisure_constant"]
            + params["utility_leisure_age"] * (age - MIN_AGE)
        )
        * jnp.log(leisure_hours)
        / 4_000
    )

    total_yearly_hours = TOTAL_WEEKLY_HOURS * N_WEEKS * N_MONTHS

    Args:
        period (int): Current period.
        consumption (jnp.array): Level of the agent's consumption.
            Array of shape (i) (n_quad_stochastic * n_grid_wealth,)
            when called by :func:`~dcgm.call_egm_step.map_exog_to_endog_grid`
            and :func:`~dcgm.call_egm_step.get_next_period_value`, or
            (ii) of shape (n_grid_wealth,) when called by
            :func:`~dcgm.call_egm_step.get_current_period_value`.
        choice (int): Choice of the agent, e.g. 0 = "retirement", 1 = "working".
        mother_alive (int): Indicator for whether the mother is alive.
            0 = mother is not alive, 1 = mother is alive.
        father_alive (int): Indicator for whether the father is alive.
            0 = father is not alive, 1 = father is alive.
        mother_health (int): Health status of the mother. One of 0, 1, 2.
            0 = good health, 1 = medium health, 2 = bad health.
        father_health (int): Health status of the father. One of 0, 1, 2.
            0 = good health, 1 = medium health, 2 = bad health.
        has_sibling (int): Indicator for whether the agent has a sibling.
            0 = no sibling, 1 = has sibling.
        params (dict): Dictionary containing model parameters.
            Relevant here is the CRRA coefficient theta.
        options (dict): Dictionary containing model options.

    Returns:
        utility (jnp.array): Agent's utility. Array of shape
            (n_quad_stochastic * n_grid_wealth,) or (n_grid_wealth,).

    """
    rho = params["rho"]

    # no_care = is_no_care(choice)
    # no_informal_care = is_no_informal_care(choice)
    # combination_care = is_combination_care(choice)

    # pure_informal_care = is_pure_informal_care(choice)
    # pure_formal_care = is_pure_formal_care(choice)

    informal_care = is_informal_care(choice)
    part_time = is_part_time(choice)
    full_time = is_full_time(choice)

    mother_bad_health = is_bad_health(mother_health)

    # working_hours_weekly = (
    #     part_time * WEEKLY_HOURS_PART_TIME + full_time * WEEKLY_HOURS_FULL_TIME
    # )
    # From SOEP data we know that the 25% and 75% percentile in the care hours
    # distribution are 7 and 21 hours per week in a comparative sample.
    # We use these discrete mass-points as discrete choices of non-intensive and
    # intensive informal care.
    # In SHARE, respondents inform about the frequency with which they provide
    # informal care. We use this information to proxy the care provision in the data.
    # caregiving_hours_weekly = informal_care * WEEKLY_INTENSIVE_INFORMAL_HOURS
    # leisure_hours = (
    #     (TOTAL_WEEKLY_HOURS - working_hours_weekly)
    #     * N_WEEKS  # month
    #     * N_MONTHS  # year
    # )

    # age is a proxy for health impacting the taste for free-time.
    # utility_leisure = (
    #     params["utility_leisure_constant"]
    #     + params["utility_leisure_age"] * period
    #     + params["utility_leisure_age_squared"] * (period**2)
    # ) * jnp.log(leisure_hours)

    utility_consumption = (
        (consumption * CONSUMPTION_EQUIVALENCE_SCALE) ** (1 - rho) - 1
    ) / (1 - rho)

    # disutility_working = (
    #     params["disutility_part_time_constant"] * part_time
    #     + params["disutility_part_time_age"] * period * part_time
    #     + params["disutility_part_time_age_squared"] * (period**2) * part_time
    #     + params["disutility_full_time_constant"] * full_time
    #     + params["disutility_full_time_age"] * period * full_time
    #     + params["disutility_full_time_age_squared"] * (period**2) * full_time
    # )

    disutility_working = (
        params["disutility_part_time_constant"] * part_time
        + params["disutility_full_time_constant"] * full_time
        #
        # + params["disutility_part_time_age_40_50"]
        # * period
        # * part_time
        # * is_in_age_range(period, 0, 10)
        # + params["disutility_full_time_age_40_50"]
        # * period
        # * full_time
        # * is_in_age_range(period, 0, 10)
        # + params["disutility_part_time_age_50_plus"]
        # * period
        # * part_time
        # * is_in_age_range(period, 10, 40)
        # + params["disutility_full_time_age_50_plus"]
        # * period
        # * full_time
        # * is_in_age_range(period, 10, 40)
        # + params["disutility_part_time_age_squared_50_plus"]
        # * (period**2)
        # * part_time
        # * is_in_age_range(period, 10, 40)
        # + params["disutility_full_time_age_squared_50_plus"]
        # * (period**2)
        # * full_time
        # * is_in_age_range(period, 10, 40)
        + params["disutility_part_time_age"] * period * part_time
        + params["disutility_full_time_age"] * period * full_time
        + params["disutility_part_time_age_squared"] * (period**2) * part_time
        + params["disutility_full_time_age_squared"] * (period**2) * full_time
    ) * (1 - informal_care)

    disutility_working_informal_care = (
        params["disutility_part_time_informal_care_constant"] * part_time
        + params["disutility_full_time_informal_care_constant"] * full_time
        # + params["disutility_part_time_informal_care_age"] * period * part_time
        # + params["disutility_full_time_informal_care_age"] * period * full_time
        # + params["disutility_part_time_informal_care_age_squared"]
        # * (period**2)
        # * part_time
        # + params["disutility_full_time_informal_care_age_squared"]
        # * (period**2)
        # * full_time
    ) * informal_care

    # disutility_working_no_informal_care = (
    #     params["disutility_part_time_constant"] * part_time * no_informal_care
    #     + params["disutility_part_time_age"] * period * part_time * no_informal_care
    #     + params["disutility_part_time_age_squared"]
    #     * (period) ** 2
    #     * part_time
    #     * no_informal_care
    #     + params["disutility_full_time_constant"] * full_time * no_informal_care
    #     + params["disutility_full_time_age"] * period * full_time * no_informal_care
    #     + params["disutility_full_time_age_squared"]
    #     * (period**2)
    #     * full_time
    #     * no_informal_care
    # )

    # disutility_working_informal_care = (
    #     params["disutility_part_time_informal_care_constant"]
    #     * part_time
    #     * informal_care
    #     + params["disutility_part_time_informal_care_age"]
    #     * period
    #     * part_time
    #     * informal_care
    #     + params["disutility_part_time_informal_care_age_squared"]
    #     * (period) ** 2
    #     * part_time
    #     * informal_care
    #     + params["disutility_full_time_informal_care_constant"]
    #     * full_time
    #     * informal_care
    #     + params["disutility_full_time_informal_care_age"]
    #     * period
    #     * full_time
    #     * informal_care
    #     + params["disutility_full_time_informal_care_age_squared"]
    #     * (period**2)
    #     * full_time
    #     * informal_care
    # )

    utility_caregiving = (
        params["utility_informal_care_parent_bad_health"]
        * is_no_care(choice)
        * mother_bad_health
        + params["utility_informal_care_parent_bad_health"]
        * is_pure_informal_care(choice)
        * mother_bad_health
        + params["utility_combination_care_parent_bad_health"]
        * is_combination_care(choice)
        * mother_bad_health
        + params["utility_formal_care_parent_bad_health"]
        * is_formal_care(choice)
        * mother_bad_health
    )

    return (
        utility_consumption
        + disutility_working
        + disutility_working_informal_care
        + utility_caregiving
    )


def marginal_utility(consumption, params):
    """Compute marginal utility of CRRA utility function."""
    return consumption ** (-params["rho"])


def inverse_marginal_utility(marginal_utility, params):
    """Compute inverse of marginal utility of CRRA utility function."""
    return marginal_utility ** (-1 / params["rho"])


# =====================================================================================
# Final period utility
# =====================================================================================


def utility_final_consume_all(
    resources: jnp.array,
    params: dict[str, float],
    options: dict[str, Any],
):
    """Compute the agent's utility in the final period including bequest."""
    rho = params["rho"]
    bequest_scale = options["bequest_scale"]

    return bequest_scale * (resources ** (1 - rho) - 1) / (1 - rho)


def marginal_utility_final_consume_all(
    resources: jnp.array,
    params: dict[str, float],
    options,
) -> jnp.array:
    """Computes marginal utility of CRRA utility function in final period.

    Args:
        choice (int): Choice of the agent, e.g. 0 = "retirement", 1 = "working".
        resources (jnp.array): The agent's financial resources.
            Array of shape (n_quad_stochastic * n_grid_wealth,).
        consumption (jnp.array): Level of the agent's consumption.
            Array of shape (n_quad_stochastic * n_grid_wealth,).
        params (dict): Dictionary containing model parameters.
            Relevant here is the CRRA coefficient theta.
        options (dict): Dictionary containing model options.

    Returns:
        marginal_utility (jnp.array): Marginal utility of CRRA consumption
            function. Array of shape (n_quad_stochastic * n_grid_wealth,).

    """
    bequest_scale = options["bequest_scale"]
    return bequest_scale * (resources ** (-params["rho"]))
