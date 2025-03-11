# README: Data Loader Optimization Experiments

This repository contains Python code for running experiments to optimize data loading performance in PyTorch models. The experiments evaluate different configurations to determine the most efficient number of workers for the `DataLoader`, as well as other performance improvements like Torch.compile optimizations.

---

## Getting Started

### Prerequisites

Ensure you have the following dependencies installed:

- Python 3.x
- Required Python packages (listed in `requirements.txt`)

Install the required packages using:
```bash
pip install -r requirements.txt
```

---

## Usage

### Step 1: Running the Experiments

1. **Run All Experiments Sequentially**

To run all experiments sequentially, execute:
```bash
python3 driver.py
```

2. **Experiment Commands**

Below are the commands for each experiment outlined in `driver.py`:

**C2 - Default experiment:**
```bash
python3 main.py
```

**C3 - Optimal number of workers experiment:**
```bash
python3 main.py --num_workers 0
python3 main.py --num_workers 4
python3 main.py --num_workers 8
python3 main.py --num_workers 12
python3 main.py --num_workers 16
```

**C4 - 1 worker vs optimal number of workers:**
```bash
python3 main.py --num_workers 1
python3 main.py --num_workers <optimal_workers>
```

**C5 - GPU vs CPU:**
```bash
python3 main.py --num_workers <optimal_workers>
python3 main.py --num_workers <optimal_workers> --disable_cuda
```

**C6 - Optimizers comparison:**
```bash
python3 main.py --num_workers <optimal_workers> --optimizer adam
python3 main.py --num_workers <optimal_workers> --optimizer adagrad
python3 main.py --num_workers <optimal_workers> --optimizer sgd
python3 main.py --num_workers <optimal_workers> --optimizer sgdnesterov
python3 main.py --num_workers <optimal_workers> --optimizer adadelta
```

**C7 - Default with vs without Batch Normalization:**
```bash
python3 main.py --num_workers <optimal_workers> --disable_batch_normalization
python3 main.py --num_workers <optimal_workers>
```

**C8 - Torch.compile experiments:**
```bash
python3 main.py --num_workers <optimal_workers> --torch_compile default --num_epochs 10
python3 main.py --num_workers <optimal_workers> --torch_compile reduce-overhead --num_epochs 10
python3 main.py --num_workers <optimal_workers> --torch_compile max-autotune --num_epochs 10
python3 main.py --num_workers <optimal_workers> --torch_compile none --num_epochs 10
```

---

## Visualizing Results

The `run_c3()` function generates a plot illustrating total data loading time versus the number of workers. The plot is saved as:

```
num_workers_experiment.png
```

---

## Code Structure

- `driver.py` - Main entry point for running the experiments.
- `main.py` - The core training script for testing data loading configurations.
- `num_workers_experiment.png` - Graph showing the effect of different `num_workers` values.
- `Resnet18.py` - Defines the model for classification
- `param_count.py` - Extra file to count parameters and gradients in the model
- `train.py` - Defines the training loop for the model
- `Dataset.py` - Creates the dataloader
- `requirements.txt` - Required packages
- `Resnet18.py` - Defines the model for classification

---

## Example Output
```
RUNNING EXPERIMENT: C3 - Finding the optimal number of workers

num_workers: 0 -> Total Runtime: 30.45 s
num_workers: 4 -> Total Runtime: 24.12 s
num_workers: 8 -> Total Runtime: 22.87 s
num_workers: 12 -> Total Runtime: 23.90 s

Performance decreased, stopping experiment early.

THE OPTIMAL NUMBER OF WORKERS IS: 8
```

---

## Notes
- If no optimal worker count is identified, the script defaults to `num_workers = 4`.
- For best results, ensure your hardware supports parallel data loading efficiently.

---

