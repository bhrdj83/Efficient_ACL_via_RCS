import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from diffusers import DDPMScheduler, UNet2DModel
import torchvision.transforms as transforms

from src.attacks.attack_factory import AttackFactory

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

class SubsetAnalyzer:
    """
    A comprehensive toolkit to analyze and log data subset metrics across training epochs.
    """
    def __init__(self, experiment_root, device="cuda", attack_config=None):
        self.experiment_root = Path(experiment_root)
        self.experiment_root.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Analyzer initialized. Device: {self.device}")
        print(f"Analysis reports will be saved to: {self.experiment_root}")

        # --- Data stores for epoch-wise logging ---
        # This will store the average metric value for each epoch
        # Format: {'clean_loss': {0: 0.5, 1: 0.45, ...}, 'robust_loss': {0: 1.2, ...}}
        self.epoch_metrics_history = {}
        
        self.run_dir = None
        self.run_seed = None
        self.epoch_metrics_history = {}
        self.history_log = {}
        self.epochs_log = []

        # --- DDPM Model components ---
        self.ddpm_model = None
        self.ddpm_scheduler = None
        self.ddpm_transform = None
        self.input_normalization = None
        self.register_input_normalization(CIFAR10_MEAN, CIFAR10_STD)
        self.setup_ddpm_model()
        
        # --- Setup a default attack for robustness metrics ---
        self.attack_config = attack_config or \
            {'name': 'pgd', 'params' : {'epsilon': 8, 'alpha': 1, 'steps': 20}}
            
            
    def register_input_normalization(self, mean, std):
        """
        Tell the analyzer which per-channel mean/std were used to normalize the training data.
        Args:
            mean (tuple/list): e.g., (0.4914, 0.4822, 0.4465) for CIFAR-10
            std (tuple/list):  e.g., (0.2470, 0.2435, 0.2616)
        """
        mean_tensor = torch.tensor(mean, dtype=torch.float32, device=self.device).view(1, -1, 1, 1)
        std_tensor = torch.tensor(std, dtype=torch.float32, device=self.device).view(1, -1, 1, 1)
        self.input_normalization = {'mean': mean_tensor, 'std': std_tensor}
        print(f"Registered input normalization: mean={mean}, std={std}")


    def setup_ddpm_model(self, model_id="google/ddpm-cifar10-32"):
        """
        Initializes the pretrained DDPM model, scheduler, and necessary transforms.
        This must be called before requesting DDPM-based metrics.
        """
        print(f"Setting up pre-trained DDPM model from '{model_id}'...")
        try:
            self.ddpm_model = UNet2DModel.from_pretrained(model_id).to(self.device)
            self.ddpm_scheduler = DDPMScheduler.from_pretrained(model_id)
            self.ddpm_model.eval()
            # This transform normalizes images from [0, 1] to [-1, 1], as expected by the DDPM
            self.ddpm_transform = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            print("DDPM setup successful.")
        except Exception as e:
            print(f"Error setting up DDPM model: {e}")
            print("DDPM analysis will be disabled.")
            self.ddpm_model = None


    def _undo_input_normalization(self, data):
        """
        Converts normalized tensors back to the [0,1] range expected by the DDPM.
        """
        if self.input_normalization is None:
            # Assume data is already roughly in [0,1]; clamp to be safe.
            return torch.clamp(data, 0.0, 1.0)
        mean = self.input_normalization['mean']
        std = self.input_normalization['std']
        return torch.clamp(data * std + mean, 0.0, 1.0)


    def _get_ddpm_denoising_loss(self, data_loader):
        """
        Calculates the DDPM denoising loss (proxy for negative log-likelihood) for a given subset.
        A higher loss means the sample is more 'surprising' or 'anomalous'.

        Args:
            data_loader (DataLoader): DataLoader for the subset of data to analyze.
            timestep (int): The diffusion timestep for adding noise.

        Returns:
            np.ndarray: An array of per-sample denoising losses.
        """
        if not self.ddpm_model:
            print("Warning: DDPM model not set up. Skipping denoising loss calculation.")
            return None

        all_losses = []
        with torch.no_grad():
            for data, _ in tqdm(data_loader, desc="DDPM Loss", leave=False):
                data = data.to(self.device)
                
                data_01 = self._undo_input_normalization(data)
                
                # Normalize data from [0, 1] to [-1, 1] for the DDPM model
                data_normalized = self.ddpm_transform(data_01)

                noise = torch.randn_like(data_normalized)
                timesteps = torch.randint(0, self.ddpm_scheduler.config.num_train_timesteps, (data.shape[0],), device=self.device).long()
                # timesteps = torch.full((data.shape[0],), timestep, device=self.device, dtype=torch.long)
                noisy_images = self.ddpm_scheduler.add_noise(data_normalized, noise, timesteps)
                
                noise_pred = self.ddpm_model(noisy_images, timesteps).sample
                
                loss_per_sample = F.mse_loss(noise_pred, noise, reduction='none').mean(dim=[1, 2, 3])
                all_losses.extend(loss_per_sample.cpu().numpy())
                
        return np.array(all_losses)


    def start_run(self, seed):
        """
        Prepare analyzer for a specific seed run. Call once per seed before training.
        """
        self.run_seed = seed
        self.run_dir = self.experiment_root / f"seed_{seed}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.epoch_metrics_history = {}
        self.history_log = {}
        self.epochs_log = []

        print(f"[Analyzer] Starting run for seed {seed}. Output dir: {self.run_dir}")


    def _log_epoch_metrics(self, epoch, metrics):
        """
        Save epoch metrics to in-memory structures for this run.
        """
        self.epoch_metrics_history[epoch] = metrics
        self.epochs_log.append(epoch)
        for key, value in metrics.items():
            if key not in self.history_log:
                self.history_log[key] = []
            self.history_log[key].append(float(value))


    def finalize_run(self):
        """
        Persist epoch-wise metrics/history for the current seed run.
        """
        if self.run_dir is None:
            raise RuntimeError("Call start_run(seed) before finalizing.")

        history_path = self.run_dir / "history.json"
        with open(history_path, "w") as f:
            json.dump(
                {
                    "epoch": self.epochs_log,
                    **{metric: values for metric, values in self.history_log.items()},
                },
                f,
                indent=4,
            )

        print(f"[Analyzer] Saved history log for seed {self.run_seed} to {history_path}")

        self.run_seed = None
        self.run_dir = None
        self.epoch_metrics_history = {}
        self.history_log = {}
        self.epochs_log = []


    def update_and_log_metrics_for_epoch(self, epoch, model, subset_dataset, batch_size=256):
        """
        Run all metric computations for the current epoch/seed and store them.
        """
        if self.run_dir is None:
            raise RuntimeError("start_run(seed) must be called before logging metrics.")
        
        print(f"\n--- Analyzing metrics for Epoch {epoch} ---")
        model.eval()  # Ensure model is in evaluation mode

        # Create a temporary attack instance for the current model state
        attack = AttackFactory.create_attack(self.attack_config['name'], model, self.attack_config)

        subset_loader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        # This dict will hold lists of all per-sample metrics for this epoch
        current_epoch_metrics = {
            'clean_loss': [], 'robust_loss': [], 'entropy': [], 'margin': [],
            'grad_magnitude': [], 'grad_variance': [], 'ddpm_loss': [], 
        }

        grads = []
        # --- Pass 1: Discriminative and Gradient Metrics ---
        for data, target in tqdm(subset_loader, desc=f"Epoch {epoch} Metrics", leave=False):
            data, target = data.to(self.device), target.to(self.device)
            data.requires_grad = True # For gradient magnitude

            # Clean metrics
            model.zero_grad()
            output_clean = model(data)
            loss_clean = F.cross_entropy(output_clean, target, reduction='none')
            
            # Gradient Magnitude
            loss_clean.sum().backward(retain_graph=True)
            grad = data.grad.view(data.shape[0], -1)
            grad_mag = torch.norm(grad, p=2, dim=1)
            grads.append(grad)

            # Adversarial metrics
            adv_data = attack.attack(data.detach(), target)
            output_robust = model(adv_data)
            loss_robust = F.cross_entropy(output_robust, target, reduction='none')

            # Softmax-based metrics
            probs = torch.softmax(output_clean.detach(), dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=1)
            top2_probs, _ = torch.topk(probs, 2, dim=1)
            margin = top2_probs[:, 0] - top2_probs[:, 1]
            
            # Append batch results
            current_epoch_metrics['clean_loss'].extend(loss_clean.cpu().detach().numpy())
            current_epoch_metrics['robust_loss'].extend(loss_robust.cpu().detach().numpy())
            current_epoch_metrics['entropy'].extend(entropy.cpu().numpy())
            current_epoch_metrics['margin'].extend(margin.cpu().numpy())
            current_epoch_metrics['grad_magnitude'].extend(grad_mag.cpu().detach().numpy())
            
        
        # Gradient Variance
        if grads:
            all_grads = torch.cat(grads, dim=0)  # Shape: (N, D)
            grad_var = torch.var(all_grads, dim=0, unbiased=True)  # Variance per feature
            grad_var_mean = torch.mean(grad_var).item()  # Mean variance across features
            current_epoch_metrics['grad_variance'] = [grad_var_mean]


        # --- Pass 2: Generative Metric (DDPM Loss) ---
        if self.ddpm_model:
            ddpm_losses = self._get_ddpm_denoising_loss(subset_loader)
            current_epoch_metrics['ddpm_loss'] = ddpm_losses

        # --- Log the average of each metric for this epoch ---
        print(f"--- Epoch {epoch} Summary ---")
        for name, values in current_epoch_metrics.items():
            if values is not None and len(values) > 0:
                avg_value = np.mean(values)
                # Store the average value for the current epoch
                current_epoch_metrics[name] = float(avg_value)
                print(f"  - Avg {name.replace('_', ' ').title()}: {avg_value:.4f}")

        model.train() # Return model to training mode

        # After computing:
        self._log_epoch_metrics(epoch, current_epoch_metrics)


    def aggregate_runs(self, generate_plots=True):
        """
        Aggregate all per-seed analyzer histories into mean/variance summary files.
        """
        experiment_root = Path(self.experiment_root)
        seed_dirs = sorted((self.experiment_root).glob("seed_*"))

        per_seed_histories = []

        for seed_dir in seed_dirs:
            history_file = seed_dir / "history.json"

            if history_file.exists():
                with open(history_file, "r") as f:
                    per_seed_histories.append(json.load(f))

        if not per_seed_histories:
            raise ValueError("No per-seed histories found.")

        aggregate_dir = experiment_root / "aggregate"
        aggregate_dir.mkdir(exist_ok=True, parents=True)

        history_summary = self._aggregate_history(per_seed_histories)
        with open(aggregate_dir / "aggregate_history.json", "w") as f:
            json.dump(history_summary, f, indent=4)

        print(f"[Analyzer] Aggregated results saved to {aggregate_dir}")
        
        if generate_plots:
            self._plot_aggregate_history(history_summary, aggregate_dir)


    def _aggregate_history(self, histories):
        sample = next(iter(histories))
        epochs = sample.get("epoch", [])
        metrics = sorted(k for k in sample.keys() if k != "epoch")

        summary = {"epoch": epochs}

        for metric in metrics:
            stacked = []
            for history in histories:
                if metric not in history:
                    continue
                stacked.append(np.array(history[metric], dtype=np.float32))

            if not stacked:
                continue

            stacked_arr = np.stack(stacked, axis=0)
            summary[f"{metric}_mean"] = np.mean(stacked_arr, axis=0).tolist()
            variance = np.var(stacked_arr, axis=0, ddof=1) if stacked_arr.shape[0] > 1 else np.zeros_like(stacked_arr[0])
            summary[f"{metric}_variance"] = variance.tolist()

        return summary


    def _plot_aggregate_history(
        self,
        history_summary,
        aggregate_dir,
        plot_metrics=None,
    ):
        epochs = history_summary.get("epoch", [])
        if not epochs:
            print("[Analyzer] aggregate_history.json missing 'epoch' field; skipping plots.")
            return

        # Collect metric base names (i.e., strip "_mean" suffix)
        base_metrics = set()
        for key in history_summary.keys():
            if key.endswith("_mean"):
                metric = key[: -len("_mean")]
                base_metrics.add(metric)

        if plot_metrics:
            base_metrics = base_metrics.intersection(set(plot_metrics))

        plots_dir = aggregate_dir / "plots"
        plots_dir.mkdir(exist_ok=True, parents=True)

        for metric in sorted(base_metrics):
            mean_key = f"{metric}_mean"
            var_key = f"{metric}_variance"
            mean_values = history_summary.get(mean_key)
            var_values = history_summary.get(var_key)

            if mean_values is None:
                continue

            mean = np.array(mean_values, dtype=np.float32)
            std = (
                np.sqrt(np.array(var_values, dtype=np.float32))
                if var_values is not None
                else None
            )

            plt.figure(figsize=(6, 4))
            plt.plot(epochs, mean, label="Mean", color="steelblue", linewidth=2)

            if std is not None:
                plt.fill_between(
                    epochs,
                    mean - std,
                    mean + std,
                    color="steelblue",
                    alpha=0.25,
                )

            plt.xlabel("Epoch")
            plt.ylabel(metric.replace("_", " ").title())
            plt.title(f"{metric.replace('_', ' ').title()} (Aggregate)")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.legend()
            plt.tight_layout()

            out_path = plots_dir / f"aggregate_{metric}.png"
            plt.savefig(out_path, dpi=120)
            plt.close()
            print(f"[Analyzer] Saved aggregate plot: {out_path}")

