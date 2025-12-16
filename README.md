🔐 Adversarial Attack Simulator for AI Models

This project is a GUI-based adversarial attack simulation tool designed to test the robustness and security of machine learning models against common adversarial attacks. It allows users to visually analyze how small, malicious perturbations can mislead AI models.

The simulator is built using Python, PyTorch, and a graphical user interface (GUI), making it beginner-friendly while still being powerful enough for AI security experimentation.

🚀 Features

📊 Interactive GUI interface

🧠 Pretrained CNN model (MNIST)

⚔️ Supports multiple adversarial attacks:

FGSM (Fast Gradient Sign Method)

PGD (Projected Gradient Descent)

Data Poisoning Attack

Evasion Attack

🖼️ Visual comparison of:

Original image

Adversarial image

📉 Model predictions before and after attack

🎛️ Adjustable attack parameters (epsilon, steps, learning rate)

📦 Modular & scalable project structure

🛠️ Tech Stack

Python

PyTorch & Torchvision

Tkinter / PyQt5

NumPy

Matplotlib

Pillow

🧪 Attacks Implemented
🔹 FGSM (Fast Gradient Sign Method)

Generates adversarial examples using a single-step gradient-based perturbation.

🔹 PGD (Projected Gradient Descent)

An iterative version of FGSM that creates stronger adversarial attacks.

🔹 Data Poisoning

Introduces malicious samples into training data to compromise model learning.

🔹 Evasion Attack

Modifies inputs at inference time to evade correct classification.

🖥️ GUI Preview

The application provides a user-friendly interface to:

Select attack type

Load models

Tune parameters

Run attacks

Visualize results

(Screenshots can be added here)

📂 Project Structure
adversarial_attack_simulator/
├── gui/
├── attacks/
├── models/
├── utils/
├── requirements.txt
└── README.md

⚠️ Ethical Disclaimer

This project is intended strictly for educational and research purposes.
It must not be used to harm systems, violate privacy, or exploit real-world AI deployments.

🎯 Use Cases

AI Security Research

Adversarial Machine Learning Learning

College / Final Year Project

ML Robustness Testing

Resume & Portfolio Project

📌 Future Enhancements

Support for CIFAR-10 and custom datasets

Defense mechanisms (Adversarial Training)

Model robustness metrics

Report export (PDF)

Web-based version

🤝 Contributions

Contributions, issues, and feature requests are welcome!
Feel free to fork the repository and submit a pull request.

⭐ If You Like This Project

Give it a ⭐ on GitHub to support the project!
