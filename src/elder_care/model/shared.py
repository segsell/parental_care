import jax.numpy as jnp

GOOD_HEALTH = 0
MEDIUM_HEALTH = 1
BAD_HEALTH = 2


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
# Choices
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


def is_combination_care(lagged_choice):
    return jnp.any(lagged_choice == COMBINATION_CARE)


# ==============================================================================
# Parental health
# ==============================================================================


def is_good_health(parental_health):
    return jnp.any(parental_health == GOOD_HEALTH)


def is_medium_health(parental_health):
    return jnp.any(parental_health == MEDIUM_HEALTH)


def is_bad_health(parental_health):
    return jnp.any(parental_health == BAD_HEALTH)