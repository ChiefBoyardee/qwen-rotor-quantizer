from rotorquant.schema import ROTORQUANT_PACKED_SCHEMA_VERSION


def test_schema_semver():
    assert ROTORQUANT_PACKED_SCHEMA_VERSION == "0.1.0"
