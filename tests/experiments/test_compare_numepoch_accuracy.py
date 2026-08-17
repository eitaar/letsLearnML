import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
MODULE_PATH = PROJECT_ROOT / "experiments" / "mnist" / "compare_numepoch_accuracy.py"
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


def test_create_network_uses_five_requested_hidden_layers():
    network = compare.create_network()

    assert network.params["W1"].shape == (784, 512)
    assert network.params["W2"].shape == (512, 256)
    assert network.params["W3"].shape == (256, 128)
    assert network.params["W4"].shape == (128, 64)
    assert network.params["W5"].shape == (64, 32)
    assert network.params["W6"].shape == (32, 10)


def test_create_network_can_disable_batch_normalization():
    network = compare.create_network(use_batch_norm=False)

    assert list(network.layers) == [
        "Affine1",
        "Relu1",
        "Affine2",
        "Relu2",
        "Affine3",
        "Relu3",
        "Affine4",
        "Relu4",
        "Affine5",
        "Relu5",
        "Affine6",
    ]
    assert "gamma1" not in network.params
    assert "beta1" not in network.params
