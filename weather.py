import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog, messagebox

class WeatherGUIPredictor:
    def __init__(self, root):
        self.root = root
        self.root.title("Non-Parametric Localized Weather Predictor")
        self.df = None
        
        # --- UI Layout ---
        self.frame = tk.Frame(self.root)
        self.frame.pack(side=tk.TOP, pady=10)
        
        self.btn_load = tk.Button(self.frame, text="1. Load Kaggle CSV", command=self.load_csv)
        self.btn_load.grid(row=0, column=0, padx=5)
        
        tk.Label(self.frame, text="Query Time (0-24):").grid(row=0, column=1)
        self.entry_time = tk.Entry(self.frame, width=10)
        self.entry_time.insert(0, "14.5")
        self.entry_time.grid(row=0, column=2, padx=5)
        
        tk.Label(self.frame, text="Bandwidth (tau):").grid(row=0, column=3)
        self.entry_tau = tk.Entry(self.frame, width=10)
        self.entry_tau.insert(0, "0.5")
        self.entry_tau.grid(row=0, column=4, padx=5)
        
        self.btn_predict = tk.Button(self.frame, text="2. Generate Forecast", command=self.update_plot)
        self.btn_predict.grid(row=0, column=5, padx=5)

        # Plot Area
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                # Load Kaggle Data (Assuming standard weather.csv format)
                self.df = pd.read_csv(file_path).head(200) # Limit for speed
                messagebox.showinfo("Success", "Dataset loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load: {e}")

    def get_weights(self, query_point, X, tau):
        m = X.shape[0]
        weights = np.eye(m)
        for i in range(m):
            diff = query_point - X[i]
            weights[i, i] = np.exp(np.dot(diff, diff.T) / (-2.0 * tau**2))
        return weights

    def predict_lwr(self, X, y, query_point, tau):
        m = X.shape[0]
        X_bias = np.append(np.ones((m, 1)), X, axis=1)
        query_bias = np.array([1, query_point])
        W = self.get_weights(query_bias, X_bias, tau)
        XTWX = X_bias.T @ W @ X_bias
        XTWy = X_bias.T @ W @ y
        theta = np.linalg.pinv(XTWX) @ XTWy
        return query_bias @ theta

    def update_plot(self):
        if self.df is None:
            messagebox.showwarning("Warning", "Please load a CSV dataset first!")
            return
            
        try:
            tau = float(self.entry_tau.get())
            query_time = float(self.entry_time.get())
            
            # Map Dataset: Assuming 'Formatted Date' contains hour or using index as time
            # For simplicity, we convert index to a 24-hour scale
            X = (np.arange(len(self.df)) % 24).reshape(-1, 1)
            y = self.df.iloc[:, 3].values # Assuming Temperature is 4th column
            
            # Generate local curve
            X_test = np.linspace(0, 24, 50)
            y_pred = [self.predict_lwr(X, y, x_p, tau) for x_p in X_test]
            
            # Update Plot
            self.ax.clear()
            self.ax.scatter(X, y, color='lightblue', label='Actual Data', alpha=0.5)
            self.ax.plot(X_test, y_pred, color='red', label='LWR Local Fit')
            self.ax.axvline(query_time, color='green', linestyle='--', label='Query Point')
            self.ax.set_title(f"Localized Forecast (tau={tau})")
            self.ax.set_xlabel("Hour of Day")
            self.ax.set_ylabel("Temperature (°C)")
            self.ax.legend()
            self.canvas.draw()
            
        except ValueError:
            messagebox.showerror("Error", "Enter valid numeric values for Time and Tau.")

if __name__ == "__main__":
    root = tk.Tk()
    # Ensure the window stays on top and is visible
    root.attributes('-topmost', True) 
    app = WeatherGUIPredictor(root)
    root.mainloop()