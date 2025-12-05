# CIFAR-10 Model Benchmark: Performance & Energy Efficiency

This project is a benchmarking tool for various deep learning architectures (CNN, Vision Transformers, Hybrid models) on the CIFAR-10 dataset. The script measures not only Accuracy and computational complexity (FLOPs/Params) but also real-time GPU energy consumption during training.

## 📋 Features

* **Multi-Architecture:** Compare custom implementations (SimpleCNN, PureViT, HybridConvViT) against SOTA baselines (ResNet, CCT).
* **Energy Consumption:** Integration with `pynvml` to measure energy usage (in Joules) in real-time.
* **Complexity Metrics:** Automatic calculation of parameters and FLOPs using `ptflops`.
* **Flexible Configuration:** Filter models, adjust epochs, batch sizes, and other hyperparameters via CLI.
* **Logging:** Results are saved to CSV files (Summary table + Training curves).

## 🛠 Requirements

The script requires `torch` and `torchvision`. For additional metrics, `pynvml` (energy) and `ptflops` (complexity) are recommended.

```bash
pip install -r requirements.txt
```

## 🛠 Execution

```bash
python benchmark_cifar10.py --filter ours_
```