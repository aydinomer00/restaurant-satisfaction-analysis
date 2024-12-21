import tkinter as tk
from tkinter import ttk
from fuzzy_logic import satisfaction_simulator


class InteractivePredictionGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Müşteri Memnuniyeti Tahmin Sistemi")
        self.root.geometry("400x300")

        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        ttk.Label(main_frame, text="Hizmet Hızı (0-10):").grid(row=0, column=0, padx=5, pady=5)
        self.service_speed = tk.Scale(main_frame, from_=0, to=10, orient=tk.HORIZONTAL, resolution=0.1)
        self.service_speed.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(main_frame, text="Yemek Kalitesi (0-10):").grid(row=1, column=0, padx=5, pady=5)
        self.food_quality = tk.Scale(main_frame, from_=0, to=10, orient=tk.HORIZONTAL, resolution=0.1)
        self.food_quality.grid(row=1, column=1, padx=5, pady=5)

        ttk.Button(main_frame, text="Memnuniyet Tahmin Et",
                   command=self.predict).grid(row=2, column=0, columnspan=2, pady=10)

        self.result_label = ttk.Label(main_frame, text="")
        self.result_label.grid(row=3, column=0, columnspan=2, pady=5)

    def predict(self):
        try:
            satisfaction_simulator.input['service_speed'] = self.service_speed.get()
            satisfaction_simulator.input['food_quality'] = self.food_quality.get()
            satisfaction_simulator.compute()

            satisfaction = satisfaction_simulator.output['customer_satisfaction']

            if satisfaction < 3.33:
                category = "DÜŞÜK"
                color = "red"
            elif satisfaction < 6.66:
                category = "ORTA"
                color = "orange"
            else:
                category = "YÜKSEK"
                color = "green"

            result_text = f"Tahmini Müşteri Memnuniyeti: {satisfaction:.2f}/10 ({category})"
            self.result_label.configure(text=result_text, foreground=color)
        except Exception as e:
            self.result_label.configure(text=f"Hata oluştu: {str(e)}", foreground="red")

    def run(self):
        self.root.mainloop()
