import os
import sys
import time
import csv
import threading
import multiprocessing
from datetime import datetime
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import timm
from jtop import jtop

# Note: Ensure VisionTransformerDiffPruning is imported from your DynamicViT directory if needed
# e.g., from models.dyvit import VisionTransformerDiffPruning

TIMESTAMP_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

# --- GLOBAL FRAMEWORK CONFIGURATION ---
CONFIG = {
    'IMAGE_SIZE': 224,
    'BATCH_SIZE': 64,      
    'DATASET_DIR': '/media/jetson/0118dae2-db05-4ad8-b86a-2bad3355fc8e/helal_eccv/hdatap/imagenet_full_val',
    'SUMMARY_CSV': f'./results/benchmark_summary_{TIMESTAMP_ID}.csv',  
    'NUM_RUNS': 1,          
    'JTOP_INTERVAL': 0.015,
    'MAX_EVAL_IMAGES': 5000 # Caps evaluation to first 5000 images to save execution time
}

# --- EXTENSIBLE LOADERS ---
def load_deit():
    print("-> Instantiating Pretrained DeiT-S (1000 Classes)...")
    model = timm.create_model('deit_small_patch16_224', pretrained=True)
    return model.cuda()


def load_dynamicvit():
    DYNAMICVIT_ROOT = '/home/jetson/reu2026/hdatap/models/DynamicViT'
    if DYNAMICVIT_ROOT not in sys.path:
            sys.path.insert(0, DYNAMICVIT_ROOT)
            
    from models.dyvit import VisionTransformerDiffPruning 

    print("-> Instantiating DynamicViT...")
    if DYNAMICVIT_ROOT not in sys.path:
        sys.path.insert(0, DYNAMICVIT_ROOT)
       
    base_rate = 0.7
    keep_rate = [
        base_rate,
        base_rate**2,
        base_rate**3
    ]

    # Uses the class name from original configuration
    model = VisionTransformerDiffPruning(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        pruning_loc=[3,6,9],
        token_ratio=keep_rate
    )

    checkpoint = torch.load(
        "./checkpoints/dynamic-vit_384_r0.7.pth",
        map_location="cpu"
    )
    model.load_state_dict(checkpoint["model"])
    model.cuda()
    model.eval()
    return model

def load_aphq_deit_small():
    print("-> Instantiating APHQ-ViT Quantized DeiT-S (W4/A4)...")
   
    APHQ_ROOT = '/home/jetson/reu2026/hdatap/models/APHQ-ViT'
   
    if not os.path.exists(APHQ_ROOT):
        raise FileNotFoundError(f"[FATAL] The path {APHQ_ROOT} does not exist.")
       
    if APHQ_ROOT not in sys.path:
        sys.path.insert(0, APHQ_ROOT)
       
    import utils.wrap_net as awn
    from utils.wrap_net import wrap_modules_in_net
    import importlib.util
   
    # Timm Version Compatibility Monkeypatch
    if hasattr(awn, 'vit_attn_forward'):
        orig_vit_attn_forward = awn.vit_attn_forward
       
        def patched_vit_attn_forward(self, x, *args, **kwargs):
            kwargs.pop('attn_mask', None)
            kwargs.pop('is_causal', None)
            return orig_vit_attn_forward(self, x, *args, **kwargs)
           
        awn.vit_attn_forward = patched_vit_attn_forward

    # Load Configuration Object
    config_path = os.path.join(APHQ_ROOT, 'configs/4bit/best.py')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"[FATAL] APHQ config file missing at: {config_path}")
       
    spec = importlib.util.spec_from_file_location("quant_config_mod", config_path)
    cfg_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_mod)
    quant_cfg = cfg_mod.Config()

    # Base Model Initialization
    base_model = timm.create_model('deit_small_patch16_224', pretrained=False)
   
    # Activation Function Swap for MLP Reconstruction Compatibility
    def replace_gelu_with_relu(model):
        for name, child in model.named_children():
            if isinstance(child, nn.GELU):
                setattr(model, name, nn.ReLU(inplace=True))
            else:
                replace_gelu_with_relu(child)
               
    replace_gelu_with_relu(base_model)
   
    # Graph Wrapping and Checkpoint Weights Binding
    quantized_model = wrap_modules_in_net(base_model, quant_cfg)
   
    checkpoint_path = "./checkpoints/deit_small_w4_a4_optimsize_1024_hessian_perturb_qdrop_recon.pth"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"[FATAL] APHQ checkpoint missing at: {checkpoint_path}")
       
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
   
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    else:
        state_dict = checkpoint
       
    quantized_model.load_state_dict(state_dict, strict=False)
       
    quantized_model.cuda()
    quantized_model.eval()
   
    print("APHQ-ViT Model compiled and frozen for benchmarking successfully.")
    return quantized_model

def load_vitkd():
    print("-> Instantiating DeiT-S trained via ViTKD...")
    # 1. Create stock model template
    model = timm.create_model('deit_small_patch16_224', pretrained=False)
   
    # 2. Load the copied file
    checkpoint_path = "./checkpoints/vitkd_deit_small_converted.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
   
    # 3. Unpack state dict
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # 4. Translate the custom key layout into timm names
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("backbone.", "")
       
        # Map Patch Embedding layer names
        new_key = new_key.replace("patch_embed.projection.", "patch_embed.proj.")
       
        # Map Transformer Block sequence components
        new_key = new_key.replace("layers.", "blocks.")
        new_key = new_key.replace(".ln1.", ".norm1.")
        new_key = new_key.replace(".ln2.", ".norm2.")
       
        # Fix the specific FFN/MLP mapping mismatch found in your file
        new_key = new_key.replace(".ffn.blocks.0.0.", ".mlp.fc1.")
        new_key = new_key.replace(".ffn.blocks.1.", ".mlp.fc2.")
       
        # Map Final Global Norm Layer
        if new_key == "ln1.weight":    new_key = "norm.weight"
        elif new_key == "ln1.bias":    new_key = "norm.bias"
       
        # Fix the head mapping mismatch found in your file
        new_key = new_key.replace("head.blocks.head.", "head.")
       
        cleaned_state_dict[new_key] = v

    # Load the correctly mapped weights
    msg = model.load_state_dict(cleaned_state_dict, strict=False)
    print(f"   └─ Checkpoint loaded with status: {msg}")
   
    model.cuda()
    model.eval()
    return model

MODELS_REGISTRY = [
    {
        "name": "DeiT-S",
        "loader": load_deit
    },
    {
        "name": "DynamicViT",
        "loader": load_dynamicvit
    },
    {
        "name": "APHQ-ViT",
        "loader": load_aphq_deit_small
    },
    {
        "name": "ViTKD",
        "loader": load_vitkd
    }
]

# --- BACKGROUND JETSON HARDWARE METRICS MONITOR ---
class JtopMonitorThread(threading.Thread):
    def __init__(self, model_name, run_idx, interval=0.015):
        super().__init__()
        self.model_name = model_name.lower().replace("-", "_")
        self.run_idx = run_idx
        self.interval = interval
        self.stop_event = threading.Event()
       
        os.makedirs("./results/raw_time_series", exist_ok=True)
        self.power_file = f"./results/raw_time_series/{self.model_name}_run{run_idx}_power.csv"
        self.gpu_file = f"./results/raw_time_series/{self.model_name}_run{run_idx}_gpu.csv"
       
        self.power_readings = []
        self.gpu_util_readings = []

    def run(self):
        with open(self.power_file, 'w', newline='') as pf, open(self.gpu_file, 'w', newline='') as gf:
            p_writer = csv.writer(pf)
            g_writer = csv.writer(gf)
            p_writer.writerow(["elapsed_seconds", "power_mW"])
            g_writer.writerow(["elapsed_seconds", "gpu_utilization_pct"])
           
            start_time = time.perf_counter()
            with jtop() as jetson:
                while not self.stop_event.is_set():
                    if jetson.ok():
                        elapsed = time.perf_counter() - start_time
                        current_power = jetson.power['tot']['power']
                        current_gpu = jetson.stats['GPU']            
                       
                        self.power_readings.append(current_power)
                        self.gpu_util_readings.append(current_gpu)
                       
                        p_writer.writerow([elapsed, current_power])
                        g_writer.writerow([elapsed, current_gpu])
                    time.sleep(self.interval)

    def stop(self):
        self.stop_event.set()

# --- ISOLATED WORK ENVIRONMENT SUBPROCESS EXECUTION ---
def run_isolated_benchmark_subprocess(model_entry, run_idx, return_dict):
    print(f"      [Subprocess {os.getpid()}] Initializing environment for Run #{run_idx}...")
   
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(CONFIG['IMAGE_SIZE']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
   
    val_dataset = datasets.ImageFolder(root=CONFIG['DATASET_DIR'], transform=transform_val)
   
    if CONFIG['MAX_EVAL_IMAGES'] is not None and CONFIG['MAX_EVAL_IMAGES'] < len(val_dataset):
        print(f"      [Subprocess] Slicing dataset to evaluation cap of first {CONFIG['MAX_EVAL_IMAGES']} images.")
        val_dataset = Subset(val_dataset, list(range(CONFIG['MAX_EVAL_IMAGES'])))
    else:
        print(f"      [Subprocess] Processing full dataset folder ({len(val_dataset)} images)...")
       
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, num_workers=4, pin_memory=True)
    criterion = nn.CrossEntropyLoss()
   
    model = model_entry['loader']()
    model.eval()
   
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    time.sleep(2)
   
    monitor = JtopMonitorThread(model_entry['name'], run_idx, CONFIG['JTOP_INTERVAL'])
    monitor.start()
   
    running_loss = 0.0
    correct = 0
    total = 0
    latencies = []

    print(f"      [Subprocess] Streaming evaluation inference across real batches...")
    total_batches = len(val_loader)
   
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
       
            start_batch = time.perf_counter()
            outputs = model(inputs)
            torch.cuda.synchronize()
            end_batch = time.perf_counter()
       
            batch_time = (end_batch - start_batch) * 1000.0
            latencies.append(end_batch - start_batch)
       
            if (batch_idx + 1) % 5 == 0 or batch_idx == 0:
                print(f"        └─ Processing Batch {batch_idx+1}/{total_batches} | Latency: {batch_time:.2f} ms")
       
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
           
    monitor.stop()
    monitor.join()
   
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    accuracy = (correct / total) * 100
    avg_loss = running_loss / len(val_loader)
   
    return_dict['accuracy'] = accuracy
    return_dict['loss'] = avg_loss
    return_dict['latencies'] = latencies
    return_dict['power_readings'] = monitor.power_readings
    return_dict['gpu_readings'] = monitor.gpu_util_readings
    return_dict['peak_vram'] = peak_vram_mb
    print(f"      [Subprocess] Target evaluation execution completed successfully.")

# --- MASTER COORDINATOR ORCHESTRATOR ---
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
   
    print("==================================================================")
    os.makedirs(os.path.dirname(CONFIG['SUMMARY_CSV']), exist_ok=True)
    if not os.path.isfile(CONFIG['SUMMARY_CSV']):
        with open(CONFIG['SUMMARY_CSV'], 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Timestamp", "Model", "Run_Index", "Accuracy", "Loss",
                "Mean_Latency_ms", "Median_Latency_ms", "Std_Latency_ms",
                "Mean_Power_mW", "Median_Power_mW", "Max_Power_mW", "Min_Power_mW", "Std_Power_mW",
                "Mean_GPU_Util", "Median_GPU_Util", "Max_GPU_Util", "Std_GPU_Util", "Peak_VRAM_MB", "EDP_WattSec2"
            ])

    for model_entry in MODELS_REGISTRY:
        print(f"\n==================================================================")
        print(f"STARTING COMPREHENSIVE BENCHMARK: {model_entry['name']}")
        print(f"==================================================================")
       
        for run_idx in range(1, CONFIG['NUM_RUNS'] + 1):
            print(f"\n---> Spawning Trial {run_idx}/{CONFIG['NUM_RUNS']} for {model_entry['name']}...")
           
            manager = multiprocessing.Manager()
            result_payload = manager.dict()
           
            p = multiprocessing.Process(
                target=run_isolated_benchmark_subprocess,
                args=(model_entry, run_idx, result_payload)
            )
            p.start()
            p.join()
           
            if p.exitcode != 0:
                print(f"[FATAL ERROR] Run #{run_idx} crashed with exitcode {p.exitcode}. Terminating system loop.")
                sys.exit(p.exitcode)
               
            acc = result_payload['accuracy']
            loss = result_payload['loss']
            lats_ms = np.array(result_payload['latencies']) * 1000.0  
            powers = np.array(result_payload['power_readings'])      
            gpus = np.array(result_payload['gpu_readings'])          
            vram = result_payload['peak_vram']

            total_delay_sec = np.sum(lats_ms) / 1000.0
            mean_power_watts = np.mean(powers) / 1000.0
            total_energy_joules = mean_power_watts * total_delay_sec
            edp = total_energy_joules * total_delay_sec
           
            with open(CONFIG['SUMMARY_CSV'], 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    model_entry['name'],
                    run_idx,
                    acc,
                    loss,
                    np.mean(lats_ms),
                    np.median(lats_ms),
                    np.std(lats_ms),
                    np.mean(powers),
                    np.median(powers),
                    np.max(powers),
                    np.min(powers),
                    np.std(powers),
                    np.mean(gpus),
                    np.median(gpus),
                    np.max(gpus),
                    np.std(gpus),
                    vram,
                    total_delay_sec,    
                    total_energy_joules,
                    edp
                ])
               
            print(f" Trial {run_idx} finished cleanly. Summary logged to {CONFIG['SUMMARY_CSV']}")
            print(f"  └─ Accuracy: {acc:.4f}% | Median Latency: {np.median(lats_ms):.4f} ms | Median Power: {np.median(powers):.2f} mW")
           
    print("\nALL EXPERIMENTAL BENCHMARK COMPILATIONS COMPLETED SUCCESSFULLY.")
