from lib.dsutils import Rect, is_rect_contains_rect


def test_rect_oob():
    r_big = Rect(2, 2, 9, 7)

    # outside completely off the top left corner
    ro1 = Rect(0, 0, 2, 2)

    # partially outside off the top left corner
    ro2 = Rect(1, 1, 3, 3)

    # touching upper left corner
    ri1 = Rect(2, 2, 3, 3)

    # all inside
    ri2 = Rect(5, 4, 2, 2)

    # touching bottom right corner
    ri3 = Rect(7, 6, 4, 3)

    # partially outside off the bottom right corner
    ro3 = Rect(9, 7, 3, 3)

    # outside off the bottom right corner
    ro4 = Rect(11, 9, 2, 2)

    assert is_rect_contains_rect(r_big, ri1)
    assert is_rect_contains_rect(r_big, ri2)
    assert is_rect_contains_rect(r_big, ri3)

    assert not is_rect_contains_rect(r_big, ro1)
    assert not is_rect_contains_rect(r_big, ro2)
    assert not is_rect_contains_rect(r_big, ro3)
    assert not is_rect_contains_rect(r_big, ro4)


