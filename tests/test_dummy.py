from recommender_universal.utils.hello import greet


def test_greet():
    assert greet("World") == "Hello, World!"
