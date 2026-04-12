import threading
from queue import Queue
from itertools import product
import random
import numpy as np
import time
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import optuna
from graphsage import *

def run_graphsage_bo(train_data, test_data, n_init=5, n_iter=15):
    start_time = time.time()
    search_space = {
        "hidden_channels": list(range(32, 513, 16)),
        "lr": np.logspace(-5, -1, 30),
        "num_layers": [2, 3, 4, 5, 6],
        "dropout": np.linspace(0.0, 0.8, 20),
        "weight_decay": np.logspace(-7, -2, 12),
    }
    keys = list(search_space.keys())
    space_sizes = [len(search_space[k]) for k in keys]

    def idxs_to_params(idxs):
        return {k: search_space[k][i] for k, i in zip(keys, idxs)}

    X_samples = []
    y_samples = []

    def evaluate_params(params):
        model = GraphSAGELinkPredictor(
            in_channels=5,
            hidden_channels=int(params["hidden_channels"]),
            num_layers=int(params["num_layers"]),
            dropout=float(params["dropout"]),
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(params["lr"]),
            weight_decay=float(params["weight_decay"]),
        )

        total_loss = 0
        for _ in range(10):
            loss = GraphSAGEtrain(model, optimizer, train_data)
            total_loss += loss.item() if hasattr(loss, 'item') else loss
        avg_loss = total_loss / 10

        test_probs = GraphSAGEtest(model, test_data)
        auc, f1, ndcg = evaluate_model(test_probs, test_data.edge_label)
        return f1, auc, avg_loss, ndcg

    for _ in range(n_init):
        sample = [random.randint(0, size - 1) for size in space_sizes]
        params = idxs_to_params(sample)
        f1, auc, loss, ndcg = evaluate_params(params)
        X_samples.append(sample)
        y_samples.append(f1)

    X_samples = np.array(X_samples)
    y_samples = np.array(y_samples)

    gp = GaussianProcessRegressor(kernel=Matern(), normalize_y=True)

    for _ in range(n_iter):
        gp.fit(X_samples, y_samples)

        candidates = np.array([
            [random.randint(0, size - 1) for size in space_sizes]
            for _ in range(100)
        ])

        preds, stds = gp.predict(candidates, return_std=True)
        acquisition = preds + 0.1 * stds

        best_idx = np.argmax(acquisition)
        best_candidate = candidates[best_idx]

        params = idxs_to_params(best_candidate)
        f1, auc, avg_loss, ndcg = evaluate_params(params)

        X_samples = np.vstack([X_samples, best_candidate])
        y_samples = np.append(y_samples, f1)

    best_idx = np.argmax(y_samples)
    best_indices = X_samples[best_idx]
    best_params = idxs_to_params(best_indices)
    best_f1, best_auc, best_loss, best_ndcg = evaluate_params(best_params)
    end_time = time.time()

    return {
        "best_params": {
            "hidden_channels": int(best_params["hidden_channels"]),
            "lr": best_params["lr"],
            "num_layers": int(best_params["num_layers"]),
            "dropout": best_params["dropout"],
            "weight_decay": best_params["weight_decay"],
        },
        'f1': best_f1,
        'auc': best_auc,
        'loss': best_loss,
        "ndcg": best_ndcg,
        "time_taken": end_time - start_time
    }


def objective(trial, train_data, test_data):
    hidden_channels = trial.suggest_int("hidden_channels", 32, 512)
    lr = trial.suggest_loguniform("lr", 1e-5, 0.1)
    num_layers = trial.suggest_int("num_layers", 2, 6)
    dropout = trial.suggest_uniform("dropout", 0.0, 0.8)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-7, 1e-2)

    model = GraphSAGELinkPredictor(
        in_channels=5,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    total_loss = 0
    for epoch in range(10):
        loss = GraphSAGEtrain(model, optimizer, train_data)
        total_loss += loss

    avg_loss = total_loss / 10

    test_probs = GraphSAGEtest(model, test_data)
    auc, f1, ndcg = evaluate_model(test_probs, test_data.edge_label)

    trial.set_user_attr("f1", f1)
    trial.set_user_attr("auc", auc)
    trial.set_user_attr("loss", avg_loss)
    trial.set_user_attr("ndcg", ndcg)

    return f1

def run_graphsage_optuna(train_data, test_data, n_trials=30):
    start_time = time.time()
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, train_data, test_data), n_trials=n_trials)

    best_trial = study.best_trial
    end_time = time.time()

    return {
        "best_params": {
            "hidden_channels": int(best_trial.params["hidden_channels"]),
            "lr": best_trial.params["lr"],
            "num_layers": int(best_trial.params["num_layers"]),
            "dropout": best_trial.params["dropout"],
            "weight_decay": best_trial.params["weight_decay"]
        },
        "f1": best_trial.user_attrs["f1"],
        "auc": best_trial.user_attrs["auc"],
        "loss": best_trial.user_attrs["loss"],
        "ndcg": best_trial.user_attrs["ndcg"],
        "time_taken": end_time - start_time
    }

def run_graphsage_aco(train_data, test_data, n_ants=5, n_gen=5, alpha=1, beta=2, evap=0.5):
    start_time = time.time()
    search_space = {
        "hidden_channels": list(range(32, 513, 32)),
        "lr": np.logspace(-5, -1, 15).tolist(),
        "num_layers": [2, 3, 4, 5, 6],
        "dropout": np.round(np.linspace(0.0, 0.8, 10), 2).tolist(),
        "weight_decay": np.round(np.logspace(-7, -2, 10), 9).tolist()
    }

    keys = list(search_space.keys())
    values = list(search_space.values())
    n_params = len(keys)
    pheromones = [np.ones(len(v), dtype=np.float64) for v in values]

    best_idx = None
    best_score = 0
    best_auc = None
    best_loss = None
    best_ndcg = None

    for _ in range(n_gen):
        solutions = np.zeros((n_ants, n_params), dtype=int)

        for i in range(n_params):
            probs = pheromones[i] ** alpha
            probs /= probs.sum()
            solutions[:, i] = np.random.choice(len(probs), size=n_ants, p=probs)

        scores = np.zeros(n_ants)
        aucs = np.zeros(n_ants)
        losses = np.zeros(n_ants)
        ndcgs = np.zeros(n_ants)

        for ant in range(n_ants):
            idxs = solutions[ant]
            try:
                params = {keys[i]: values[i][idxs[i]] for i in range(n_params)}

                model = GraphSAGELinkPredictor(
                    in_channels=5,
                    hidden_channels=params["hidden_channels"],
                    num_layers=params["num_layers"],
                    dropout=params["dropout"]
                ).to(device)
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=params["lr"],
                    weight_decay=params["weight_decay"],
                )

                total_loss = 0
                for epoch in range(10):
                    loss = GraphSAGEtrain(model, optimizer, train_data)
                    total_loss += loss
                avg_loss = total_loss / 10

                test_probs = GraphSAGEtest(model, test_data)
                auc, f1, ndcg = evaluate_model(test_probs, test_data.edge_label)

                scores[ant] = f1
                aucs[ant] = auc
                losses[ant] = avg_loss
                ndcgs[ant] = ndcg

            except Exception as e:
                scores[ant] = 0
                aucs[ant] = 0
                losses[ant] = float("inf")

        max_idx = np.argmax(scores)
        if scores[max_idx] > best_score:
            best_score = scores[max_idx]
            best_idx = solutions[max_idx]
            best_auc = aucs[max_idx]
            best_loss = losses[max_idx]
            best_ndcg = ndcgs[max_idx]

        for i in range(n_params):
            pheromones[i] *= (1 - evap)
            update = np.bincount(solutions[:, i], weights=scores, minlength=len(pheromones[i]))
            pheromones[i][:len(update)] += update

    best_params = {keys[i]: values[i][best_idx[i]] for i in range(n_params)}
    end_time = time.time()

    return {
        "best_params": {
            "hidden_channels": int(best_params["hidden_channels"]),
            "lr": best_params["lr"],
            "num_layers": int(best_params["num_layers"]),
            "dropout": best_params["dropout"],
            "weight_decay": best_params["weight_decay"]
        },
        "f1": best_score,
        "auc": best_auc,
        "loss": best_loss,
        "ndcg": best_ndcg,
        "time_taken": end_time - start_time
    }


def run_graphsage_gs(train_data, test_data):
    start_time = time.time()
    param_grid = {
        "hidden_channels": [64, 128, 256, 512],
        "lr": [0.01, 0.001, 0.0001],
        "num_layers": [2, 3, 4, 5, 6],
        "dropout": [0.0, 0.3, 0.5, 0.7],
        "weight_decay": [1e-5, 1e-4, 1e-3]
    }

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    grid = list(product(*values))

    best_score = 0
    best_params = None
    best_auc = None
    best_loss = None

    for combination in grid:
        params = dict(zip(keys, combination))

        try:
            model = GraphSAGELinkPredictor(
                in_channels=5,
                hidden_channels=params["hidden_channels"],
                num_layers=params["num_layers"],
                dropout=params["dropout"]
            ).to(device)
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=params["lr"],
                weight_decay=params["weight_decay"],
            )

            total_loss = 0
            for _ in range(10):
                loss = GraphSAGEtrain(model, optimizer, train_data)
                total_loss += loss

            avg_loss = total_loss / 10

            test_probs = GraphSAGEtest(model, test_data)
            auc, f1, ndcg = evaluate_model(test_probs, test_data.edge_label)

            if f1 > best_score:
                best_score = f1
                best_params = params
                best_auc = auc
                best_loss = avg_loss
                best_ndcg=ndcg

        except Exception as e:
            continue

    duration = time.time() - start_time
    return {
        "best_params": {
            "hidden_channels": int(best_params["hidden_channels"]),
            "lr": best_params["lr"],
            "num_layers": int(best_params["num_layers"]),
            "dropout": best_params["dropout"],
            "weight_decay": best_params["weight_decay"]
        },
        "f1": best_score,
        "auc": best_auc,
        "loss": best_loss,
        "ndcg": best_ndcg,
        "time_taken": duration
    }
