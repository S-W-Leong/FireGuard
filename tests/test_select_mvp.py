import pytest


@pytest.mark.parametrize("new_ap,old_ap,expected", [(0.4, 0.48, "previous_no_coordinates"), (0.5, 0.48, "risk_score")])
def test_mvp_selection_uses_development_year_even_when_2024_disagrees(new_ap, old_ap, expected):
    # Break caught: picking the winner after seeing exploratory outcomes.
    from fireguard.select_mvp import select_method
    data = {"2023": {"methods": {"risk_score": {"average_precision": new_ap},
                                "previous_no_coordinates": {"average_precision": old_ap}}},
            "2024": {"methods": {"risk_score": {"average_precision": 1 - new_ap},
                                "previous_no_coordinates": {"average_precision": 1 - old_ap}}}}
    assert select_method(data) == expected
