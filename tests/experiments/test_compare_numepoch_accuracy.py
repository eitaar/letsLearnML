import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[2] / "experiments" / "mnist" / "compare_numepoch_accuracy.py"
SPEC = importlib.util.spec_from_file_location("compare_numepoch_accuracy", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
compare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare)


def test_iterations_for_epochs_converts_one_to_five_epochs_to_mnist_updates():
    assert compare.iterations_for_epochs([1, 2, 3, 4, 5], train_size=60_000, batch_size=100) == [
        600,
        1_200,
        1_800,
        2_400,
        3_000,
    ]


def test_create_network_uses_three_hidden_layers_of_64_neurons():
    network = compare.create_network()

    assert network.params["W1"].shape == (784, 64)
    assert network.params["W2"].shape == (64, 64)
    assert network.params["W3"].shape == (64, 64)
    assert network.params["W4"].shape == (64, 10)
