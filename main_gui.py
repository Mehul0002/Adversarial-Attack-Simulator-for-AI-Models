import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import torch
from models.cnn_model import SimpleCNN
from utils.dataset_loader import load_mnist_data, get_sample_image
from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack
from attacks.data_poisoning import data_poisoning_attack
from attacks.evasion import evasion_attack
from utils.visualization import tensor_to_image, plot_images, plot_perturbation, plot_confidences
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class AdversarialAttackSimulator:
    def __init__(self, root):
        self.root = root
        self.root.title("Adversarial Attack Simulator")
        self.root.geometry("1200x800")

        self.model = None
        self.original_image = None
        self.adversarial_image = None
        self.original_label = None
        self.adversarial_label = None

        self.setup_gui()

    def setup_gui(self):
        # Control Frame
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        ttk.Label(control_frame, text="Attack Type:").grid(row=0, column=0, sticky=tk.W)
        self.attack_var = tk.StringVar(value="FGSM")
        attack_combo = ttk.Combobox(control_frame, textvariable=self.attack_var, values=["FGSM", "PGD", "Data Poisoning", "Evasion"])
        attack_combo.grid(row=0, column=1, padx=5)

        ttk.Label(control_frame, text="Model:").grid(row=1, column=0, sticky=tk.W)
        self.model_var = tk.StringVar(value="SimpleCNN")
        model_combo = ttk.Combobox(control_frame, textvariable=self.model_var, values=["SimpleCNN"])
        model_combo.grid(row=1, column=1, padx=5)

        # Parameters
        ttk.Label(control_frame, text="Epsilon:").grid(row=2, column=0, sticky=tk.W)
        self.epsilon_var = tk.DoubleVar(value=0.1)
        epsilon_scale = ttk.Scale(control_frame, from_=0.0, to=1.0, variable=self.epsilon_var, orient=tk.HORIZONTAL)
        epsilon_scale.grid(row=2, column=1, sticky=(tk.W, tk.E))

        ttk.Label(control_frame, text="Iterations:").grid(row=3, column=0, sticky=tk.W)
        self.iter_var = tk.IntVar(value=10)
        iter_scale = ttk.Scale(control_frame, from_=1, to=100, variable=self.iter_var, orient=tk.HORIZONTAL)
        iter_scale.grid(row=3, column=1, sticky=(tk.W, tk.E))

        ttk.Label(control_frame, text="Step Size:").grid(row=4, column=0, sticky=tk.W)
        self.step_var = tk.DoubleVar(value=0.01)
        step_scale = ttk.Scale(control_frame, from_=0.001, to=0.1, variable=self.step_var, orient=tk.HORIZONTAL)
        step_scale.grid(row=4, column=1, sticky=(tk.W, tk.E))

        # Buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=5, column=0, columnspan=2, pady=10)

        ttk.Button(button_frame, text="Load Model", command=self.load_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Run Attack", command=self.run_attack).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Reset", command=self.reset).pack(side=tk.LEFT, padx=5)

        # Output Frame
        output_frame = ttk.Frame(self.root)
        output_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Images
        image_frame = ttk.Frame(output_frame)
        image_frame.pack(side=tk.TOP, fill=tk.X)

        self.original_label = ttk.Label(image_frame, text="Original Image")
        self.original_label.pack(side=tk.LEFT, padx=10)

        self.adversarial_label = ttk.Label(image_frame, text="Adversarial Image")
        self.adversarial_label.pack(side=tk.RIGHT, padx=10)

        # Results
        self.results_text = tk.Text(output_frame, height=10, width=60)
        self.results_text.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

    def load_model(self):
        try:
            self.model = SimpleCNN()
            self.model.load_state_dict(torch.load('models/pretrained_model.pth'))
            self.model.eval()
            messagebox.showinfo("Success", "Model loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")

    def run_attack(self):
        if self.model is None:
            messagebox.showerror("Error", "Please load the model first!")
            return

        try:
            # Get sample image
            dataloader = load_mnist_data(batch_size=1, train=False)
            image, label = get_sample_image(dataloader)
            self.original_image = image

            attack_type = self.attack_var.get()
            epsilon = self.epsilon_var.get()
            iterations = self.iter_var.get()
            step_size = self.step_var.get()

            if attack_type == "FGSM":
                adv_image = fgsm_attack(self.model, image, label.unsqueeze(0), epsilon)
            elif attack_type == "PGD":
                adv_image = pgd_attack(self.model, image, label.unsqueeze(0), epsilon, iterations, step_size)
            elif attack_type == "Data Poisoning":
                adv_image, _ = data_poisoning_attack(image, label.unsqueeze(0), poison_rate=0.5, noise_level=epsilon)
            elif attack_type == "Evasion":
                adv_image = evasion_attack(self.model, image, label.unsqueeze(0), epsilon, iterations)
            else:
                raise ValueError("Unknown attack type")

            self.adversarial_image = adv_image

            # Get predictions
            with torch.no_grad():
                orig_pred = self.model(image).argmax().item()
                adv_pred = self.model(adv_image).argmax().item()

            # Update GUI
            self.update_images()
            self.update_results(orig_pred, adv_pred, label.item())

        except Exception as e:
            messagebox.showerror("Error", f"Attack failed: {str(e)}")

    def update_images(self):
        if self.original_image is not None:
            orig_img = tensor_to_image(self.original_image)
            orig_img = orig_img.resize((200, 200))
            orig_photo = ImageTk.PhotoImage(orig_img)
            self.original_label.config(image=orig_photo)
            self.original_label.image = orig_photo

        if self.adversarial_image is not None:
            adv_img = tensor_to_image(self.adversarial_image)
            adv_img = adv_img.resize((200, 200))
            adv_photo = ImageTk.PhotoImage(adv_img)
            self.adversarial_label.config(image=adv_photo)
            self.adversarial_label.image = adv_photo

    def update_results(self, orig_pred, adv_pred, true_label):
        success = "Yes" if orig_pred == true_label and adv_pred != true_label else "No"
        results = f"True Label: {true_label}\nOriginal Prediction: {orig_pred}\nAdversarial Prediction: {adv_pred}\nAttack Success: {success}"
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, results)

    def reset(self):
        self.model = None
        self.original_image = None
        self.adversarial_image = None
        self.original_label.config(image='')
        self.adversarial_label.config(image='')
        self.results_text.delete(1.0, tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = AdversarialAttackSimulator(root)
    root.mainloop()
