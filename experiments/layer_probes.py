"""Reproduce probes before and after every frozen SmolLM2 layer."""

from har_challenge.probes import run_layer_probes


def main():
    run_layer_probes(
        "outputs/reproduction/context",
        "outputs/reproduction/diagnostics/layer_probes",
        seeds=(42, 43, 44, 45),
        extract_batch_size=64,
        probe_batch_size=512,
        probe_epochs=100,
        probe_learning_rate=1e-2,
    )


if __name__ == "__main__":
    main()
