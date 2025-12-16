# Adversarial Attack Simulator for AI Models

This project is a Python-based GUI application that allows users to test machine learning models against various adversarial attacks used in AI security.

## What are Adversarial Attacks?

Adversarial attacks are techniques used to fool machine learning models by introducing small, often imperceptible perturbations to input data. These perturbations can cause the model to make incorrect predictions, highlighting vulnerabilities in AI systems.

## Supported Attacks

### 1. FGSM (Fast Gradient Sign Method)
FGSM is a one-step attack that computes the gradient of the loss with respect to the input and adds a perturbation in the direction of the gradient sign.

### 2. PGD (Projected Gradient Descent)
PGD is an iterative attack that performs multiple steps of gradient descent, projecting the perturbed image back onto the allowed perturbation range after each step.

### 3. Data Poisoning Attack
Data poisoning involves modifying the training data to introduce backdoors or biases that affect the model's behavior.

### 4. Evasion Attack
Evasion attacks modify input data at test time to evade detection or classification.

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Train the model (or create a dummy model for testing):
   ```bash
   python models/train_model.py
   ```
   Or for quick testing:
   ```bash
   python models/create_dummy_model.py
   ```

3. Run the GUI application:
   ```bash
   python gui/main_gui.py
   ```

## GUI Features

- Select attack type from dropdown
- Choose pretrained model
- Adjust parameters (epsilon, iterations, step size)
- Load model and run attack
- View original and adversarial images
- See prediction changes and attack success rate

## Ethical Usage Disclaimer

This tool is for educational and research purposes only. Adversarial attacks can be used maliciously to compromise AI systems. Use responsibly and only on models you own or have permission to test.

## Project Structure

```
adversarial_attack_simulator/
├── gui/
│   └── main_gui.py
├── attacks/
│   ├── fgsm.py
│   ├── pgd.py
│   ├── data_poisoning.py
│   └── evasion.py
├── models/
│   ├── cnn_model.py
│   └── pretrained_model.pth
├── utils/
│   ├── visualization.py
│   └── dataset_loader.py
├── requirements.txt
└── README.md
