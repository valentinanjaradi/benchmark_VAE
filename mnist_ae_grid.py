"""
mnist_ae_grid.py
================
Grid experiment: train pythae autoencoders on MNIST subsets, then do
unregularized linear regression on frozen latent representations.

Grid axes
---------
- n_b : number of MNIST training samples used to train the AE
- n_d : number of downstream samples used for linear regression
- m   : latent dimension (passed to pythae as latent_dim)

Metrics (matching compare_ae_pr.py conventions)
------------------------------------------------
- est_error   : ||w - w_star||^2  where w_star is the OLS solution on the
                full test set (proxy for the Bayes-optimal linear predictor)
- train_error : per-sample MSE on the n_d downstream training samples
- gen_error   : per-sample MSE on the full MNIST test set
- recon_train : per-pixel MSE of AE reconstruction on the AE training set
- recon_test  : per-pixel MSE of AE reconstruction on the MNIST test set

Results are saved as pickle files:
    <output_dir>/result_nb_{n_b}_nd_{n_d}_m_{m}.pkl

Usage
-----
    python mnist_ae_grid.py --mode local   # quick sanity check
    python mnist_ae_grid.py --mode slurm   # submit SLURM array
"""

import os
import pickle
import argparse
import datetime
import json
import time
from contextlib import contextmanager
import submitit
import numpy as np
import torch
from torchvision import datasets


@contextmanager
def _timer(label):
    t0 = time.perf_counter()
    yield
    print(f'  [timing] {label}: {time.perf_counter() - t0:.2f}s', flush=True)


from pythae.models.nn.benchmarks.mnist import (
                Encoder_Conv_VAE_MNIST as Encoder_VAE,
                Decoder_Conv_AE_MNIST as Decoder_AE
            )
from pythae.pipelines import TrainingPipeline
from pythae.trainers import BaseTrainerConfig
from pythae.trainers.training_callbacks import TrainingCallback
from pythae.models import VAE, VAEConfig, AutoModel

# ---------------------------------------------------------------------------
# MNIST loading
# ---------------------------------------------------------------------------

def load_mnist(n_base, n_downstream, data_dir='examples/scripts/data/mnist/', seed=None):
    """Returns (X_train, y_train, X_test, y_test) as tensors.
    Images are flattened to 784-dim vectors, pixel values in [0, 1].
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    train_ds = datasets.MNIST(data_dir, train=True,  download=True, transform=None)
    test_ds  = datasets.MNIST(data_dir, train=False, download=True, transform=None)

    # get random subspace of training set for AE and downstream regression, ensuring disjointness
    indices = np.random.permutation(len(train_ds))
    train_ds.data = train_ds.data[indices]
    train_ds.targets = train_ds.targets[indices]
    
    X_base = train_ds.data[:n_base,:] / 255
    X_downstream = train_ds.data[n_base:n_base+n_downstream, :] / 255
    y_downstream = train_ds.targets[n_base:n_base+n_downstream]
    X_test = test_ds.data / 255
    y_test = test_ds.targets
    return X_base, X_downstream, y_downstream, X_test, y_test

def batched_reconstruct(model, X, batch_size=512):
    device = next(model.parameters()).device
    parts = []
    for i in range(0, len(X), batch_size):
        batch = X[i:i+batch_size].reshape(-1, 1, 28, 28).to(device)
        parts.append(model.reconstruct(batch).cpu().reshape(len(batch), -1))
    return torch.cat(parts, dim=0)

class LossLogger(TrainingCallback):
    """Captures per-epoch train/eval losses from the pythae pipeline."""
    def __init__(self):
        self.train_losses = []
        self.eval_losses  = []

    def on_log(self, _training_config, logs, **_kwargs):
        if "train_epoch_loss" in logs:
            self.train_losses.append(logs["train_epoch_loss"])
        if "eval_epoch_loss" in logs:
            self.eval_losses.append(logs["eval_epoch_loss"])


# ---------------------------------------------------------------------------
# Pythae AE training
# ---------------------------------------------------------------------------

def train_pythae_ae(X_ae, X_test, latent_dim, output_dir=None):
    """Train a pythae autoencoder on X_ae (numpy float32, shape [n, d]).

    Returns (model, ae_train_losses, ae_eval_losses).
    """

    X_ae   = X_ae.reshape(-1, 1, 28, 28)
    X_test = X_test.reshape(-1, 1, 28, 28)
    input_dim = tuple(X_ae.shape[1:])

    model_config = VAEConfig.from_json_file('examples/scripts/configs/mnist/vae_config.json')
    model_config.input_dim  = input_dim
    model_config.latent_dim = int(latent_dim)
    model = VAE(
        model_config=model_config,
        encoder=Encoder_VAE(model_config),
        decoder=Decoder_AE(model_config),
    )
    training_config = BaseTrainerConfig.from_json_file('examples/scripts/configs/mnist/base_training_config.json')
    training_config.output_dir = output_dir
    training_config.num_epochs = 100 if output_dir and 'local' in output_dir else 500
    pipeline = TrainingPipeline(training_config=training_config, model=model)

    loss_logger = LossLogger()
    pipeline(train_data=X_ae, eval_data=X_test, callbacks=[loss_logger])

    print(f'  [LossLogger] captured {len(loss_logger.train_losses)} train / '
          f'{len(loss_logger.eval_losses)} eval epochs', flush=True)
    return pipeline.model, loss_logger.train_losses, loss_logger.eval_losses

#----------------------------------------------------------------------------
# Train linear probe
#----------------------------------------------------------------------------

def train_linear_probe(Xhat_downstream, y_downstream, Xhat_test, y_test, num_epochs=20000):
    """OLS regression of y_downstream on X_hat_downstream."""
    # linear probe
    probe = torch.nn.Linear(Xhat_downstream.shape[1], 10)  # 10 classes

    # train only the probe
    optimizer = torch.optim.Adam(probe.parameters(), lr=5e-3)
    criterion = torch.nn.CrossEntropyLoss()

    # train probe
    losses = []
    for _ in range(num_epochs):
        logits = probe(Xhat_downstream)
        loss = criterion(logits, y_downstream)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # evaluate
    with torch.no_grad():
        preds_training = probe(Xhat_downstream)
        preds_test = probe(Xhat_test)
        err_training = (preds_training.argmax(dim=1) != y_downstream).float().mean()
        err_test = (preds_test.argmax(dim=1) != y_test).float().mean()
        crossentropy_err_training = criterion(preds_training, y_downstream)
        crossentropy_err_test = criterion(preds_test, y_test)

        return err_training.item(), err_test.item(), crossentropy_err_training.item(), crossentropy_err_test.item(), losses



# ---------------------------------------------------------------------------
# Single grid column, single seed
# ---------------------------------------------------------------------------

def run_one_seed(n_b, n_d_list, m,
                 model_name='vae',
                 base_output_dir='results/mnist_ae_grid',
                 seed=0):
    """Train AE on n_b samples, run linear probes for all n_d values.

    Returns a list of result dicts, one per n_d value.
    """
    print(f'Loading MNIST (seed={seed})...', flush=True)
    X_base, X_downstream, y_downstream, X_test, y_test = load_mnist(
        n_b, int(np.max(n_d_list)), seed=seed
    )

    X_ae = X_base[:n_b]

    # --- train or load AE ---
    ae_output_dir = os.path.join(base_output_dir, f'ae_nb{n_b}_m{m}_seed{seed}')
    os.makedirs(ae_output_dir, exist_ok=True)

    runs = sorted(os.listdir(ae_output_dir))
    last_model_dir = os.path.join(ae_output_dir, runs[-1], "final_model") if runs else None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if last_model_dir and os.path.isdir(last_model_dir):
        print("Found existing model, loading from:", last_model_dir)
        with _timer('load model'):
            model = VAE.load_from_folder(last_model_dir).to(device)
        ae_losses_path = os.path.join(last_model_dir, 'ae_losses.json')
        if os.path.exists(ae_losses_path):
            with open(ae_losses_path) as f:
                saved = json.load(f)
            ae_train_losses = saved['ae_train_losses']
            ae_eval_losses  = saved['ae_eval_losses']
        else:
            ae_train_losses, ae_eval_losses = None, None
    else:
        print("No existing model found, training new model.")
        with _timer('train AE'):
            model, ae_train_losses, ae_eval_losses = train_pythae_ae(
                X_ae, X_test, latent_dim=m, output_dir=ae_output_dir)
        model = model.to(device)
        runs = sorted(os.listdir(ae_output_dir))
        last_model_dir = os.path.join(ae_output_dir, runs[-1], "final_model")
        ae_losses_path = os.path.join(last_model_dir, 'ae_losses.json')
        with open(ae_losses_path, 'w') as f:
            json.dump({'ae_train_losses': ae_train_losses, 'ae_eval_losses': ae_eval_losses}, f)

    # --- shared reconstructions ---
    model.eval()
    with torch.no_grad():
        with _timer('reconstruct test'):
            Xhat_test = batched_reconstruct(model, X_test)
        with _timer('reconstruct ae train'):
            Xhat_ae = batched_reconstruct(model, X_ae)
        recon_train = float(torch.mean((X_ae.reshape(len(X_ae), -1) - Xhat_ae) ** 2))
        recon_test  = float(torch.mean((X_test.reshape(len(X_test), -1) - Xhat_test) ** 2))

    results = []
    for n_d in n_d_list:
        X_down_nd = X_downstream[:n_d]
        y_down_nd = y_downstream[:n_d]

        model.eval()
        with torch.no_grad():
            with _timer(f'reconstruct downstream n_d={n_d}'):
                Xhat_downstream = batched_reconstruct(model, X_down_nd)

        print(f'Training linear probe on {n_d} samples...', flush=True)
        with _timer('train linear probe'):
            probe_err_train, probe_err_test, crossentropy_err_train, crossentropy_err_test, probe_losses = train_linear_probe(
                Xhat_downstream, y_down_nd, Xhat_test, y_test)

        result = dict(
            n_b=int(n_b), n_d=int(n_d), m=int(m), seed=int(seed), model_name=model_name,
            recon_train=recon_train, recon_test=recon_test,
            train_error=probe_err_train, gen_error=probe_err_test,
            crossentropy_train=crossentropy_err_train, crossentropy_test=crossentropy_err_test,
            probe_losses=probe_losses,
            ae_train_losses=ae_train_losses, ae_eval_losses=ae_eval_losses,
        )
        print(f'[n_b={n_b:5d}, n_d={n_d:5d}, m={m:3d}, seed={seed}] '
              f'train={probe_err_train:.4f}  gen={probe_err_test:.4f}', flush=True)
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Grid job: multiple seeds, save aggregate only
# ---------------------------------------------------------------------------

def process_grid_job(n_b, n_d_list, m,
                     model_name='vae',
                     base_output_dir='results/mnist_ae_grid',
                     seed=0,
                     num_seeds=1):
    """Run num_seeds seeds for one (n_b, m) point.

    Saves one aggregate pkl per n_d with mean/std over seeds.
    """
    os.makedirs(base_output_dir, exist_ok=True)
    scalar_metrics = ['recon_train', 'recon_test', 'train_error', 'gen_error']

    # collect results across seeds
    all_results = {int(n_d): [] for n_d in n_d_list}
    for s in range(seed, seed + num_seeds):
        for r in run_one_seed(n_b, n_d_list, m, model_name, base_output_dir, seed=s):
            all_results[r['n_d']].append(r)

    # save aggregate (mean + std over seeds) for each n_d
    for n_d in n_d_list:
        rs = all_results[n_d]
        agg = dict(n_b=int(n_b), n_d=int(n_d), m=int(m),
                   seeds=list(range(seed, seed + num_seeds)), model_name=model_name)
        for key in scalar_metrics:
            vals = [r[key] for r in rs]
            agg[f'{key}_mean'] = float(np.mean(vals))
            agg[f'{key}_std']  = float(np.std(vals))
        # keep trajectory from first seed for inspection
        agg['probe_losses']    = rs[0]['probe_losses']
        agg['ae_train_losses'] = rs[0]['ae_train_losses']
        agg['ae_eval_losses']  = rs[0]['ae_eval_losses']

        fname = os.path.join(base_output_dir, f'result_nb_{n_b}_nd_{n_d}_m_{m}_agg.pkl')
        with open(fname, 'wb') as f:
            pickle.dump(agg, f)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def run_local():
    """Small sanity-check run."""
    seed      = 0
    num_seeds = 2
    n_b_values = np.array([200])
    n_d_values = np.array([50, 100, 200, 300])
    m_values   = np.array([14])

    for n_b in n_b_values:
        for m in m_values:
            process_grid_job(
                n_b=n_b, n_d_list=n_d_values, m=m,
                model_name='vae',
                base_output_dir='results/mnist_ae_local',
                seed=seed,
                num_seeds=num_seeds,
            )
    print('Local test done. Results in results/mnist_ae_local/')


def main():
    """Submit a SLURM job array over (n_b, n_d, m)."""
    n_b_values = np.linspace(1, 1000, num=6).astype(int)
    n_d_values = np.linspace(1, 1000,  num=6).astype(int)
    m_values   = np.append(np.logspace(1, 9, num=14, base=2).astype(int), 28*28)
    model_name    = 'vae'
    seed          = 0
    num_seeds     = 3
    base_output_dir = 'results/mnist_ae_grid_log_bottleneck'

    executor = submitit.AutoExecutor(folder='submitit_logs_mnist_ae')
    executor.update_parameters(
        timeout_min=120,
        slurm_partition='gpu_lowp',
        slurm_tasks_per_node=1,
        slurm_cpus_per_task=8,
        slurm_mem_gb=8,
        slurm_gres='gpu:1',
        slurm_job_name='mnist_ae_grid'
        #slurm_nodelist="gpu-xd670-30,gpu-sr675-31,gpu-sr675-33,gpu-sr675-34,gpu-sr675-35"
    )

    all_args = [
        (n_b, n_d_values, m, model_name, base_output_dir, seed, num_seeds)
        for n_b in n_b_values
        for m   in m_values
    ]

    print(f'Submitting {len(all_args)} jobs...', flush=True)
    jobs = executor.map_array(process_grid_job, *zip(*all_args))

    log_entry = {
        'timestamp':       datetime.datetime.now().isoformat(),
        'array_job_id':    jobs[0].job_id.split('_')[0],
        'n_tasks':         len(jobs),
        'base_output_dir': base_output_dir,
        'model_name':      model_name,
        'n_b_values':      n_b_values.tolist(),
        'n_d_values':      n_d_values.tolist(),
        'm_values':        m_values.tolist(),
        'seed':            int(seed),
    }
    log_path = 'submitit_job_log_mnist.json'
    log = json.load(open(log_path)) if os.path.exists(log_path) else []
    log.append(log_entry)
    json.dump(log, open(log_path, 'w'), indent=2)
    print(f'Submitted. Array job ID: {log_entry["array_job_id"]}. Logged to {log_path}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['local', 'slurm'], default='local')
    args = parser.parse_args()

    if args.mode == 'local':
        run_local()
    else:
        main()
