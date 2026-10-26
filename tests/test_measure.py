from lermolieff.hierarchy import ancestors, most_specific, terms
from lermolieff.measure import branch_f1, hier_f1


def test_ancestors_simple():
    assert ancestors("25F23") == {"2", "25", "25F", "25F2", "25F23"}


def test_ancestors_bracket_and_key():
    assert {"1", "11", "11H", "11H(JEROME)"} == ancestors("11H(JEROME)")
    assert "31A" in ancestors("31A23(+1)")


def test_compound_keeps_parts():
    a = ancestors("25F23(LION):46A12")
    assert {"25F", "46A", "25F23(LION):46A12"} <= a


def test_unreadable_is_itself():
    assert ancestors("not a code") == {"not a code"}


def test_most_specific():
    assert most_specific(["25F", "25F23", "46A"]) == ["25F23", "46A"]


def test_terms_split_outside_brackets():
    assert terms("25F23(A:B):46A") == ["25F23(A:B)", "46A"]


def test_hier_f1():
    assert hier_f1(["25F23"], ["25F23"]) == 1.0
    assert hier_f1(["25F23"], []) == 0.0
    assert 0 < hier_f1(["25F23"], ["25F2"]) < 1


def test_branch_f1():
    assert branch_f1(["31A23", "25F"], ["31A2"], "31A") is not None
    assert branch_f1(["25F"], ["31A2"], "31A") is None
