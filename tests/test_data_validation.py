import pytest
from helpers import csv_bytes
from src.data_validation import ValidationError, load_csv_bytes, validate_target


def test_valid_csv_and_missing_tokens():
    lr = load_csv_bytes(csv_bytes("a,b,c\n1, x ,?\n2,y,\n3,  ,5\n"), "t.csv")
    assert lr.clean.shape == (3, 3)
    assert lr.clean["a"].dtype.kind in "if"
    assert lr.clean["b"].isna().sum() == 1 and lr.clean["b"].iloc[0] == "x"
    assert lr.clean["c"].isna().sum() == 2 and lr.clean["c"].dtype.kind == "f"
    assert lr.raw["c"].iloc[0] == "?"          # raw kept separate from cleaned


def test_mixed_column_stays_text_not_coerced():
    lr = load_csv_bytes(csv_bytes("a,b\n1,x\nfoo,y\n3,z\n"), "t.csv")
    assert list(lr.clean["a"]) == ["1", "foo", "3"]


@pytest.mark.parametrize("data,name", [
    (b"", "a.csv"), (b"   \n", "a.csv"), (b"a,b\n", "a.csv"),
    (b"a,b\n1,2\n", "a.txt"), (b"a,b\n1,2\n", "a.exe"), (b"a\n1\n2\n", "a.csv"),
    (b"a,b\n1,2,3,4\n1,2\n", "a.csv"), (b"PK\x00\x03binary", "a.csv"),
])
def test_bad_files_rejected(data, name):
    with pytest.raises(ValidationError):
        load_csv_bytes(data, name)


def test_oversized_rejected():
    with pytest.raises(ValidationError):
        load_csv_bytes(b"a,b\n" + b"1,2\n" * 100, "a.csv", max_bytes=50)


def test_odd_column_names_and_duplicates():
    lr = load_csv_bytes(csv_bytes("a,a,,  spaced  \n1,2,3,4\n"), "t.csv")
    assert list(lr.clean.columns) == ["a", "a_1", "unnamed_2", "spaced"]


def test_latin1_fallback():
    lr = load_csv_bytes("a,b\ncaf\xe9,1\n".encode("latin-1"), "t.csv")
    assert lr.clean["a"].iloc[0] == "café" and any("latin-1" in n for n in lr.notes)


def test_target_validation():
    lr = load_csv_bytes(csv_bytes("a,b\n1,\n2,\n"), "t.csv")
    with pytest.raises(ValidationError):
        validate_target(lr.clean, "zzz")
    with pytest.raises(ValidationError):
        validate_target(lr.clean, "b")
    validate_target(lr.clean, "a")


def test_ragged_rows_rejected_both_directions_and_blank_lines_skipped():
    for bad in (b"a,b\n1,2,3\n", b"a,b\n1\n", b"a,b\n1,2,3,4\n1,2\n"):
        with pytest.raises(ValidationError):
            load_csv_bytes(bad, "t.csv")
    assert len(load_csv_bytes(b"a,b\n1,2\n\n3,4\n\n", "t.csv").clean) == 2


def test_quoted_fields_with_commas_and_newlines():
    lr = load_csv_bytes(b'a,b\n"x, y",1\n"line1\nline2",2\n', "t.csv")
    assert lr.clean["a"].iloc[0] == "x, y" and "\n" in lr.clean["a"].iloc[1]


def test_whitespace_duplicate_names_deduplicated():
    assert list(load_csv_bytes(b"a, a\n1,2\n", "t.csv").clean.columns) == ["a", "a_1"]
