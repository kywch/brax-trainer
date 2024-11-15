import numpy as np

from brax_trainer.train import parse_args
from brax_trainer.clean_pufferl import seed_everything
from brax_trainer.carbs_sweep import init_carbs, carbs_runner_fn

RANDOM_SEED = 42
NUM_SUGGESTIONS = 20
NUM_RANDOM_SAMPLES = 10


def print_pareto_front(carbs, is_conservative=False):
    pareto_front = carbs._get_pareto_groups(is_conservative=is_conservative)
    print(f"\n\nPareto front (conservative {is_conservative}):")
    for i, obs_group in enumerate(pareto_front):
        mean_cost = np.mean([o.cost for o in obs_group])
        mean_output = np.mean([o.output for o in obs_group])
        print(
            f"Obs group {i+1}, {len(obs_group)} samples - cost: {mean_cost:.2f}, mean output: {mean_output:.2f}"
        )
    print("\n\n")


if __name__ == "__main__":
    # import sys
    # from loguru import logger
    # logger.remove()
    # logger.add(sink=sys.stderr, level="ERROR")

    # Use the default config to test CARBS
    (
        args,
        _,
    ) = parse_args()
    carbs = init_carbs(args, num_random_samples=NUM_RANDOM_SAMPLES)

    rng = np.random.default_rng(RANDOM_SEED)

    def dummy_train_fn(args, wandb):
        # seeding like clean pufferl
        seed_everything(args["train"]["seed"], args["train"]["torch_deterministic"])

        stats = [{"episode_return": rng.integers(1, 100)}]
        cost = args["train"]["total_timesteps"] // 10000

        return stats, cost

    carbs_runner = carbs_runner_fn(
        args, "test", carbs, "test", train_fn=dummy_train_fn, disable_wandb=True, debug=True
    )

    for i in range(NUM_SUGGESTIONS):
        carbs_runner()
        print("CARBS state:", carbs.get_state_dict(), "\n")
        print_pareto_front(carbs)

    # # Test the wandb sweep
    # import wandb
    # sweep_id = wandb.sweep(
    #     sweep=args["sweep"],
    #     project="carbs",
    # )

    # # Run sweep
    # wandb.agent(
    #     sweep_id,
    #     carbs_runner_fn(args, "test", carbs, sweep_id, dummy_train_fn),
    #     count=NUM_SUGGESTIONS,
    # )
