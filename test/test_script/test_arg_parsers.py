import json
from argparse import ArgumentTypeError

import pytest

from sgs_tools.scripts.arg_parsers import parse_json_or_file


def test_none_input():
    assert parse_json_or_file(None) is None


def test_valid_json_string_list():
    data = [{"label": "sim1", "color": "k"}, {"label": "sim2", "color": "C1"}]
    result = parse_json_or_file(json.dumps(data))
    assert result == data


def test_valid_json_string_dict():
    data = {"key": "value", "n": 42}
    result = parse_json_or_file(json.dumps(data))
    assert result == data


def test_valid_json_file(tmp_path):
    data = [{"label": "sim1"}, {"label": "sim2"}]
    json_file = tmp_path / "styles.json"
    json_file.write_text(json.dumps(data))
    result = parse_json_or_file(str(json_file))
    assert result == data


def test_invalid_raises(tmp_path):
    with pytest.raises(ArgumentTypeError, match="Invalid JSON or file path"):
        parse_json_or_file("not_json_and_not_a_file.json")


def test_json_string_takes_priority_over_file(tmp_path):
    """A valid JSON string is parsed directly even if it could look like a path."""
    data = {"a": 1}
    # write a file that would match if treated as a path — should never be read
    (tmp_path / '{"a": 1}').write_text("should not be read")
    result = parse_json_or_file(json.dumps(data))
    assert result == data
