import warnings
warnings.filterwarnings('ignore')
from optimizers_niapy_gcn import *
from other_optimizers_gcn import *
from other_optimizers_gan import *
from optimizers_niapy_gan import *

import json
import os
import time
import shutil

all_results = []
RESULTS_FILE = "optimization_results.json"

def load_existing_results():
    global all_results
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, 'r') as f:
                all_results = json.load(f)
            print(f"[INFO] Loaded {len(all_results)} existing results from {RESULTS_FILE}")
        except Exception as e:
            print(f"\n[CRITICAL ERROR] Could not load {RESULTS_FILE}. It might be corrupted!")
            print(f"To protect your existing data, the script will EXIT now.")
            print(f"Error: {e}")
            print("Please fix the JSON file or restore from the backup (.bak) file.")
            import sys
            sys.exit(1)

def save_all_results():
    # Performance safety: create a backup before overwriting
    if os.path.exists(RESULTS_FILE):
        shutil.copy2(RESULTS_FILE, RESULTS_FILE + ".bak")
    
    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=4)
    # print(f"[INFO] Results updated in {RESULTS_FILE}")

def print_result(result, model_name):
    print(f"\nBest Hyperparameters Found for {model_name}:")
    params = result['best_params']
    print(f"  Hidden Channels : {params.get('hidden_channels')}")
    print(f"  Learning Rate   : {params.get('lr'):.5f}")
    if 'num_layers' in params:
        print(f"  Num Layers      : {params.get('num_layers')}")
    print(f"  Dropout         : {params.get('dropout'):.2f}")
    if 'weight_decay' in params:
        print(f"  Weight Decay    : {params.get('weight_decay'):.6f}")
    if 'beta1' in params:
        print(f"  Beta1   : {params.get('beta1'):.2f}")

    print("\nPerformance Metrics:")
    print(f"  F1 Score        : {result.get('f1'):.4f}")
    print(f"  AUC             : {result.get('auc'):.4f}")
    print(f"  Average Loss    : {result.get('loss'):.4f}")
    print(f"  NDCG            : {result.get('ndcg'):.4f}")
    print(f"  Time Taken      : {result.get('time_taken', 0):.2f} seconds")
    
    # Store result and save immediately
    result['model_name'] = model_name
    all_results.append(result)
    save_all_results()

def gcn_none():
    if any(res['model_name'] == "GCN (None)" for res in all_results):
        print("\nSkipping GCN (None) - already exists.")
        return
    print("\nNo optimizing (GCN)")
    start_time = time.time()
    # Aggressively small baseline: 32 channels, 0.1 lr, and 5 layers (over-smoothing threshold)
    model = GCNLinkPredictor(in_channels=5, hidden_channels=32, num_layers=5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    epoch, loss, auc_gcn, f1_gcn, ndcg = 0,0,0,0,0
    for _ in range(10):
      loss = GCNtrain(model, optimizer, train_data)
      test_probs = GCNtest(model, test_data)
      auc_gcn, f1_gcn, ndcg = evaluate_model(test_probs, test_data.edge_label)
      epoch=_
    
    duration = time.time() - start_time
    result = {
        "model_name": "GCN (None)",
        "best_params": {"hidden_channels": 32, "lr": 0.1, "num_layers": 5},
        "f1": f1_gcn,
        "auc": auc_gcn,
        "loss": loss,
        "ndcg": ndcg,
        "time_taken": duration
    }
    all_results.append(result)
    save_all_results()
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}, AUC score: {auc_gcn:.4f}, f1 score: {f1_gcn:.4f}, NDCG: {ndcg:.4f}, Time: {duration:.2f}s")

def gan_none():
    if any(res['model_name'] == "GAN (None)" for res in all_results):
        print("\nSkipping GAN (None) - already exists.")
        return
    print("\nNo optimizing (GAN)")
    start_time = time.time()
    # Aggressively small GAN baseline: 64 channels and high 0.01 lr
    generator = Generator(in_channels=5, hidden_channels=64).to(device)
    discriminator = Discriminator(hidden_channels=64).to(device)

    generator.apply(init_weights)
    discriminator.apply(init_weights)

    optimizer_G = optim.Adam(generator.parameters(), lr=0.01, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.01, betas=(0.5, 0.999))

    epoch, d_loss, g_loss, auc_gan, f1_gan, ndcg = 0, 0, 0, 0, 0, 0
    for _ in range(10):
        d_loss, g_loss = GANtrain(generator, discriminator, optimizer_G, optimizer_D, train_data)
        test_probs = GANtest(generator, discriminator, test_data)
        auc_gan, f1_gan, ndcg = evaluate_model(test_probs, test_data.edge_label)
        epoch = _
    
    duration = time.time() - start_time
    result = {
        "model_name": "GAN (None)",
        "best_params": {"hidden_channels": 64, "lr": 0.01, "betas": (0.5, 0.999)},
        "f1": f1_gan,
        "auc": auc_gan,
        "loss": d_loss - g_loss,
        "ndcg": ndcg,
        "time_taken": duration
    }
    all_results.append(result)
    save_all_results()
    print(f"Epoch {epoch + 1}, Average Loss: {(d_loss - g_loss):.4f}, AUC score: {auc_gan:.4f}, f1 score: {f1_gan:.4f}, NDCG: {ndcg:.4f}, Time: {duration:.2f}s")

def gcn_ga():
    print_result(run_gcn_ga(), "GCN (GA)")

def gan_ga():
    print_result(run_gan_ga(), "GAN (GA)")

def gcn_pso():
    print_result(run_gcn_pso(), "GCN (PSO)")

def gan_pso():
    print_result(run_gan_pso(), "GAN (PSO)")

def gcn_sa():
    print_result(run_gcn_sa(), "GCN (SA)")

def gan_sa():
    print_result(run_gan_sa(), "GAN (SA)")

def gcn_abc():
    print_result(run_gcn_abc(), "GCN (ABC)")

def gan_abc():
    print_result(run_gan_abc(), "GAN (ABC)")

def gcn_aco():
    print_result(run_gcn_aco(train_data, test_data), "GCN (ACO)")

def gan_aco():
    print_result(run_gan_aco(train_data, test_data), "GAN (ACO)")

def gcn_hc():
    print_result(run_gcn_hc(), "GCN (HC)")

def gan_hc():
    print_result(run_gan_hc(), "GAN (HC)")

def gcn_rs():
    print_result(run_gcn_ra(), "GCN (RS)")

def gan_rs():
    print_result(run_gan_ra(), "GAN (RS)")

def gcn_bo():
    print_result(run_gcn_bo(train_data, test_data), "GCN (BO)")

def gan_bo():
    print_result(run_gan_bo(train_data, test_data), "GAN (BO)")

def gcn_optuna():
    print_result(run_gcn_optuna(train_data, test_data), "GCN (Optuna)")

def gan_optuna():
    print_result(run_gan_optuna(train_data, test_data), "GAN (Optuna)")

def gcn_gs():
    print_result(run_gcn_gs(train_data, test_data), "GCN (GS)")

def gan_gs():
    print_result(run_gan_gs(train_data, test_data), "GAN (GS)")

def run_step(func, name):
    if any(res['model_name'] == name for res in all_results):
        print(f"\nSkipping {name} - already exists.")
        return
    func()

if __name__ == "__main__":
    load_existing_results()
    try:
        run_step(gcn_none, "GCN (None)")
        run_step(gan_none, "GAN (None)")
        run_step(gcn_ga, "GCN (GA)")
        run_step(gan_ga, "GAN (GA)")
        run_step(gcn_pso, "GCN (PSO)")
        run_step(gan_pso, "GAN (PSO)")
        run_step(gcn_sa, "GCN (SA)")
        run_step(gan_sa, "GAN (SA)")
        run_step(gcn_abc, "GCN (ABC)")
        run_step(gan_abc, "GAN (ABC)")
        run_step(gcn_aco, "GCN (ACO)")
        run_step(gan_aco, "GAN (ACO)")
        run_step(gcn_hc, "GCN (HC)")
        run_step(gan_hc, "GAN (HC)")
        run_step(gcn_rs, "GCN (RS)")
        run_step(gan_rs, "GAN (RS)")
        run_step(gcn_bo, "GCN (BO)")
        run_step(gan_bo, "GAN (BO)")
        run_step(gcn_optuna, "GCN (Optuna)")
        run_step(gan_optuna, "GAN (Optuna)")
        run_step(gcn_gs, "GCN (GS)")
        run_step(gan_gs, "GAN (GS)")
    except KeyboardInterrupt:
        print("\n[STOP] Interrupted by user.")
    finally:
        save_all_results()

