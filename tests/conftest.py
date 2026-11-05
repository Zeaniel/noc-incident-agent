import pytest

from noc_agent.synth import generate_dataset


@pytest.fixture(scope="session")
def ds():
    return generate_dataset(seed=7)


def incident(ds, cause):
    return next(i for i in ds.incidents if i.root_cause.value == cause)
