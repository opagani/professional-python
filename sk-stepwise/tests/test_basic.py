import sk_stepwise.sw as sw
import pytest


def test_initialization():
    print("test_initialization")
    print(dir(sw))
    model = None
    rounds = []
    optimizer = sw.StepwiseHyperoptOptimizer(model, rounds)
    assert optimizer is not None


# import sk_stepwise as sw


# def test_hello():
#     assert sw.hello() == "Hello from sk-stepwise!"


def test_that_fails():
    ages = [1, 2, 3, 4, 5]
    name = "suzzie"
    rounds = []
    assert "matt" == "fred"


def test_with_exception(one):
    assert one == 1


@pytest.mark.xfail(raises=TypeError)
def test_logistic():
    from sklearn import linear_model

    model = linear_model.LinearRegression()
    rounds = []
    opt = sw.StepwiseHyperoptOptimizer(model, rounds)
    X = [[0, 1], [0, 2]]
    y = [1, 0]
    opt.fit(X, y)


@pytest.mark.matt
def test_matt():
    assert 1 == 1
