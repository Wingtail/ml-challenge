"""Run one training batch through every required workflow."""

from har_challenge.evaluation import evaluate_saved_context
from har_challenge.train import pretrain_encoder, train_context, train_direct


def main():
    pretrain_encoder(smoke=True)
    train_direct(smoke=True)
    train_context(smoke=True)
    evaluate_saved_context(
        "outputs/code_smoke/context/seed_42/model.pt",
        "outputs/code_smoke/context/seed_42/shuffle.json",
        repeats=1,
    )


if __name__ == "__main__":
    main()
