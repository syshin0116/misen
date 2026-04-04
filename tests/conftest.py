import pytest

from misen import tool


@pytest.fixture
def add_one():
    @tool(name="add_one")
    def _add_one(input: dict) -> dict:
        return {"value": input["value"] + 1}

    return _add_one


@pytest.fixture
def double():
    @tool(name="double")
    def _double(input: dict) -> dict:
        return {"value": input["value"] * 2}

    return _double


@pytest.fixture
def extract_length():
    """Block that reads 'text' and outputs 'length'."""

    @tool(name="extract_length")
    def _extract_length(input: dict) -> dict:
        return {"length": len(input["text"])}

    return _extract_length


@pytest.fixture
def to_upper():
    """Block that uppercases 'text'."""

    @tool(name="to_upper")
    def _to_upper(input: dict) -> dict:
        return {"text": input["text"].upper()}

    return _to_upper
