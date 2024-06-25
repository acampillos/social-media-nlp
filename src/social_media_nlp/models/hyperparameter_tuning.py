import mlflow
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.base import ClassifierMixin
from typing import Any, Tuple, Union, Dict


def objective(
    trial: optuna.Trial,
    model: Union[ClassifierMixin, Any],
    X: Any,
    y: Any,
    scoring: str = "f1_weighted",
    cv: int = 3,
) -> float:
    """
    Objective function for Optuna optimization.

    Args:
        trial (optuna.Trial): A single optimization trial.
        model (Union[ClassifierMixin, Any]): Model configuration dictionary or object.
        X (Any): Input features for training.
        y (Any): Target labels for training.
        scoring (str, optional): Scoring metric for cross-validation. Defaults to "f1_weighted".
        cv (int, optional): Number of cross-validation folds. Defaults to 3.

    Returns:
        float: Mean score of cross-validation using the model with suggested hyperparameters.
    """
    params = {}
    for param_name, param_distribution in model["hyperparams"].items():
        param_type = param_distribution["type"]
        param_values = param_distribution["values"]

        if param_type == "int":
            params[param_name] = trial.suggest_int(param_name, *param_values)
        elif param_type == "float":
            params[param_name] = trial.suggest_float(param_name, *param_values)
        elif param_type == "categorical":
            params[param_name] = trial.suggest_categorical(param_name, param_values)
        else:
            raise ValueError(f"Invalid distribution for parameter {param_name}")

    model_instance = model["model_class"](**params)
    mean_score = cross_val_score(model_instance, X, y, cv=cv, scoring=scoring).mean()

    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
        mlflow.log_metric(f"cv_{scoring}", mean_score)

    return mean_score


def optimize_model(
    model_config: Dict[str, Any],
    objective_function: Any,
    X: Any,
    y: Any,
    n_trials: int = 100,
    study_name: str = None,
    sampler: optuna.samplers.BaseSampler = None,
    pruner: optuna.pruners.BasePruner = None,
    storage: str = "sqlite:///mlflow.db",
    direction: str = "maximize",
    scoring: str = "f1_weighted",
) -> Tuple[Dict[str, Any], float]:
    """
    Optimizes a model using Optuna's hyperparameter optimization.

    Args:
        model_config (Dict[str, Any]): Configuration dictionary for the model and hyperparameters.
        objective_function (Any): Objective function that Optuna will optimize.
        X (Any): Input features for training.
        y (Any): Target labels for training.
        n_trials (int, optional): Number of trials for optimization. Defaults to 100.
        study_name (str, optional): Name of the Optuna study. Defaults to None.
        sampler (optuna.samplers.BaseSampler, optional): Sampler for Optuna. Defaults to None.
        pruner (optuna.pruners.BasePruner, optional): Pruner for Optuna. Defaults to None.
        storage (str, optional): Storage URL for Optuna study. Defaults to "sqlite:///mlflow.db".
        direction (str, optional): Optimization direction (maximize, minimize).
            Defaults to "maximize".
        scoring (str, optional): Scoring metric for optimization. Defaults to "f1_weighted".

    Returns:
        Tuple[Dict[str, Any], float]: Best hyperparameters and the best objective value found.
    """
    study = optuna.create_study(
        load_if_exists=True,
        direction=direction,
        study_name=study_name,
        sampler=sampler,
        pruner=pruner,
        storage=storage,
    )
    objective_fn = lambda trial: objective_function(trial, model_config, X, y, scoring)
    study.optimize(objective_fn, n_trials=n_trials, n_jobs=-1, show_progress_bar=True)
    return study.best_params, study.best_value
