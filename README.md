# brax-trainer
Brax + Pufferlib + CARBS for gpu-accelerated robotics RL

## Getting Started
1. Clone this repository.
    ```
    git clone https://github.com/kywch/brax-trainer.git
    ```

    Then, go to the directory.
    ```
    cd brax-trainer
    ```

2. Using pixi, setup the virutal environment and install the dependencies.
    Install pixi, if you haven't already. See [pixi documentation](https://pixi.sh/latest/#installation) for more details. The following command is for linux.
    ```
    curl -fsSL https://pixi.sh/install.sh | bash
    ```

    The following command sets up the virtual environment and installs the dependencies.
    ```
    pixi install
    ```

    Then, activate the virtual environment.
    ```
    pixi shell
    ```

    To check if both pytorch and jax are installed correctly using cuda, run the following command.
    ```
    pixi run test_torch
    pixi run test_jax
    ```

3. Train a policy.
    In the virtual environment, run:
    ```
    python brax_trainer/train.py -m train
    ```
    
    Or, run:
    ```
    pixi run train -m train
    ```

4. Evaluate the trained policy.
    ```
    python brax_trainer/train.py -m eval -p <path_to_model>
    ```

    To make a video of the trained policy, run:
    ```
    python brax_trainer/train.py -m video -p <path_to_model>
    ```

    Try these with the pre-trained policy for the Ant env: `brax_ant_policy.pt`.

6. Sweep the hyperparameters using [CARBS](https://arxiv.org/abs/2306.08055).

    ```
    python brax_trainer/train.py -m sweep
    ```
