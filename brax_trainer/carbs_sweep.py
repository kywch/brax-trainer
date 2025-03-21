import json
import time

import numpy as np

from carbs import LinearSpace
from carbs import LogSpace
from carbs import LogitSpace
from carbs import Param
from carbs import CARBS
from carbs import CARBSParams
from carbs import ObservationInParam

from brax_trainer.utils import init_wandb


def carbs_param(group, name, carbs_kwargs, rounding_factor=1):
    # carbs_kwargs should have either ("min", "max") or "values"
    if "values" in carbs_kwargs:
        values = carbs_kwargs["values"]
        mmin = min(values)
        mmax = max(values)
    else:
        mmin = carbs_kwargs["min"]
        mmax = carbs_kwargs["max"] if "max" in carbs_kwargs else float("+inf")

    space = carbs_kwargs["space"]
    search_center = carbs_kwargs["search_center"] if "search_center" in carbs_kwargs else None
    is_integer = carbs_kwargs["is_integer"] if "is_integer" in carbs_kwargs else False

    if space == "log":
        Space = LogSpace
    elif space == "linear":
        Space = LinearSpace
    elif space == "logit":
        Space = LogitSpace
        assert mmin == 0
        assert mmax == 1
        assert search_center is not None
    else:
        raise ValueError(f"Invalid CARBS space: {space} (log/linear)")

    return Param(
        name=f"{group}-{name}",
        space=Space(min=mmin, max=mmax, is_integer=is_integer, rounding_factor=rounding_factor),
        search_center=search_center,
    )


def init_carbs(args, resample_frequency=5, num_random_samples=2, max_suggestion_cost=600):
    assert "sweep" in args, "No wandb sweep config found in args"
    assert "carbs" in args, "No carbs config found in args"

    carbs_param_spaces = []
    wandb_sweep_params = args["sweep"]["parameters"]
    carbs_config = args["carbs"]

    for group in wandb_sweep_params:
        for name in wandb_sweep_params[group]["parameters"]:
            assert name in carbs_config, f"Invalid name {name} in {group}"

            # Handle special cases: total timesteps, batch size, num_minibatch, num_envs
            if name in [
                "total_timesteps",
                "batch_size",
                "num_minibatches",
                "bptt_horizon",
                "num_envs",
            ]:
                assert (
                    "min" in carbs_config[name]
                ), f"Special param {name} must have min in carbs config"

            # Others: append min/max from wandb param to carbs param
            else:
                carbs_config[name].update(wandb_sweep_params[group]["parameters"][name])

            carbs_param_spaces.append(
                carbs_param(group, name, carbs_config[name], rounding_factor=1)
            )

    carbs_params = CARBSParams(
        better_direction_sign=1,
        is_wandb_logging_enabled=False,
        resample_frequency=resample_frequency,
        num_random_samples=num_random_samples,
        max_suggestion_cost=max_suggestion_cost,
    )

    return CARBS(carbs_params, carbs_param_spaces)


def carbs_runner_fn(args, env_name, carbs, sweep_id, train_fn, disable_wandb=False, debug=False):
    carbs_file = "carbs_" + sweep_id + ".txt"

    def run_sweep_session():
        print("--------------------------------------------------------------------------------")
        print("Starting a new session...")
        print("--------------------------------------------------------------------------------")
        wandb = init_wandb(args, env_name, id=args["exp_id"], disable=disable_wandb)
        wandb.config.__dict__["_locked"] = {}

        print(f"Getting suggestion based on CARBS's {len(carbs.success_observations)} obs...")

        if debug is True:
            carbs._set_seed(int(time.time()))  # To get random "random samples"

        orig_suggestion = carbs.suggest().suggestion
        suggestion = orig_suggestion.copy()
        print("\nCARBS suggestion:", suggestion)
        train_suggestion = {
            k.split("-")[1]: v for k, v in suggestion.items() if k.startswith("train-")
        }

        # Correcting critical parameters before updating
        # train_suggestion["total_timesteps"] = int(train_suggestion["total_timesteps"] * 10**6)
        for key in ["batch_size", "bptt_horizon", "num_envs"]:
            if key in train_suggestion:
                train_suggestion[key] = 2 ** round(train_suggestion[key])
        train_suggestion["update_epochs"] = round(train_suggestion["update_epochs"])

        # Adjust the eval timesteps based on the num_envs
        train_suggestion["eval_timesteps"] = train_suggestion["num_envs"] * 1024

        # CARBS minibatch_size is actually the number of minibatches
        train_suggestion["num_minibatches"] = 2 ** round(train_suggestion["num_minibatches"])
        train_suggestion["minibatch_size"] = (
            train_suggestion["batch_size"] // train_suggestion["num_minibatches"]
        )

        # args["train"]["num_envs"] = closest_power(train_suggestion["num_envs"])  # 16, 32, 64
        args["train"].update(train_suggestion)

        # These might be needed later
        env_suggestion = {k.split("-")[1]: v for k, v in suggestion.items() if k.startswith("env-")}
        args["env"].update(env_suggestion)

        args["track"] = True
        wandb.config.update({"train": args["train"], "env": args["env"]}, allow_val_change=True)

        print("\nTrain config:", wandb.config.train)
        print("\nEnv config:", wandb.config.env, "\n")
        # print(wandb.config.policy)

        outputs, uptime, is_success = {}, 0, False
        try:
            outputs, uptime = train_fn(args, wandb)
            is_success = len(outputs) > 0
        except Exception as e:  # noqa
            import traceback

            traceback.print_exc()

        # NOTE: What happens if training fails?
        """
        A run should be reported as a failure if the hyperparameters suggested by CARBS 
        caused the failure, for example a batch size that is too large that caused an OOM failure. 
        If a failure occurs that is not related to the hyperparameters, it is better to forget 
        the suggestion or retry it. Report a failure by making an ObservationInParam with is_failure=True
        """
        output = np.mean(outputs) if outputs else 0

        print(f"\n\nTrain success: {is_success}, Time: {uptime:.0f} s, Output: {output:.0f}\n\n")
        obs_out = carbs.observe(  # noqa
            ObservationInParam(
                input=orig_suggestion,
                output=output,
                cost=uptime,
                is_failure=not is_success,
            )
        )

        # Save CARBS suggestions and results
        with open(carbs_file, "a") as f:
            train_suggestion.update({"output": output, "cost": uptime})
            results_txt = json.dumps(str(train_suggestion))
            f.write(results_txt + "\n")
            f.flush()

    return run_sweep_session
