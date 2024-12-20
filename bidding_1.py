import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import tkinter as tk
from tkinter import filedialog, messagebox

class BiddingStrategyApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Bidding Strategy Predictor")

        # Button to load file
        self.load_button = tk.Button(master, text="Load Excel File", command=self.load_file)
        self.load_button.pack(pady=10)

        # Button to predict bidding action
        self.predict_button = tk.Button(master, text="Predict Bidding Action", command=self.predict_action)
        self.predict_button.pack(pady=10)

        self.model = None
        self.scaler = None
        self.data = None

    def load_file(self):
        choice = input("Choose an option:\n1. Browse a file\n2. Enter file path\nEnter 1 or 2: ")
        
        if choice == '1':
            # Use tkinter to browse a file
            root = tk.Tk()
            root.withdraw()  # Hide the root window
            file_path = filedialog.askopenfilename(title="Select the Excel file", filetypes=[("Excel files", "*.xlsx")])
            root.destroy()
            if file_path:
                print(f"\nFile selected: {file_path}")
                self.load_data(file_path)
            else:
                messagebox.showerror("Error", "No file selected. Exiting...")
                exit()
        elif choice == '2':
            # Prompt for file path input
            file_path = input("Please enter the full file path of the Excel file: ").strip('"')  # Remove any quotes
            if not file_path:
                messagebox.showerror("Error", "No file path entered. Exiting...")
                exit()
            self.load_data(file_path)
        else:
            messagebox.showerror("Error", "Invalid choice. Exiting...")
            exit()

    def load_data(self, file_path):
        try:
            self.data = pd.read_excel(file_path)
            messagebox.showinfo("Success", f"File successfully loaded from: {file_path}")
            self.preprocess_data()
        except Exception as e:
            messagebox.showerror("Error", f"Error loading file: {e}")

    def preprocess_data(self):
        self.data['Clicks'] = pd.to_numeric(self.data['Clicks'], errors='coerce')
        self.data['Spend'] = pd.to_numeric(self.data['Spend'], errors='coerce')
        self.data['Orders'] = pd.to_numeric(self.data['Orders'], errors='coerce')
        self.data.fillna(0, inplace=True)
        self.data['Cost_per_Conversion'] = self.data['Spend'] / self.data['Orders'].replace(0, 1)

        def classify_bidding_action(row):
            if row['Clicks'] < 10 and row['Cost_per_Conversion'] > 10:
                return 0  # Pause the bid
            elif row['Clicks'] >= 10 and row['Cost_per_Conversion'] <= 10:
                return 1  # Increase the bid
            else:
                return 2  # Reduce the bid

        self.data['Bidding_Action'] = self.data.apply(classify_bidding_action, axis=1)

        # Step 4: Feature Selection and Splitting the Data
        X = self.data[['Clicks', 'Spend', 'Orders', 'Cost_per_Conversion']]
        y = self.data['Bidding_Action']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Step 5: Feature Scaling
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        # Step 6: Train the Model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        # Save predictions for the entire dataset
        self.save_predictions()

        messagebox.showinfo("Success", "Data processed and model trained.")

    def save_predictions(self):
        # Predict Bidding Actions for the entire dataset
        self.data['Predicted_Bidding_Action'] = self.model.predict(self.scaler.transform(self.data[['Clicks', 'Spend', 'Orders', 'Cost_per_Conversion']]))
        action_map = {0: "Pause the bid", 1: "Increase the bid", 2: "Reduce the bid"}
        self.data['Bidding_Action_Description'] = self.data['Predicted_Bidding_Action'].map(action_map)

        # Save predictions to Excel
        output_file = "Predicted_Bidding_Strategy.xlsx"
        self.data.to_excel(output_file, index=False)
        print(f"\nPredictions saved to: {output_file}")

        # Summary of actions
        action_counts = self.data['Bidding_Action_Description'].value_counts()
        print("\nSummary of Recommended Bidding Actions:")
        print(action_counts)

        # Save summary to a separate Excel file
        summary_file = "Bidding_Action_Summary.xlsx"
        action_counts.to_excel(summary_file, header=True)
        print(f"\nBidding action summary saved to: {summary_file}")

    def predict_action(self):
        if self.data is None or self.model is None or self.scaler is None:
            messagebox.showerror("Error", "Please load data and train the model first.")
            return

        try:
            name=str(input("enter the keyword name:"))
            new_clicks = float(input("Enter Clicks for the new keyword: "))
            new_spend = float(input("Enter Spend for the new keyword: "))
            new_orders = float(input("Enter Orders for the new keyword: "))

            new_data = pd.DataFrame({
                'Clicks': [new_clicks],
                'Spend': [new_spend],
                'Orders': [new_orders],
                'Cost_per_Conversion': [new_spend / new_orders if new_orders != 0 else new_spend]
            })

            new_data_scaled = self.scaler.transform(new_data)
            new_prediction = self.model.predict(new_data_scaled)

            action_map = {0: "Pause the bid", 1: "Increase the bid", 2: "Reduce the bid"}
            messagebox.showinfo("Prediction Result", f"Prediction: {action_map[new_prediction[0]]}")

        except Exception as e:
            messagebox.showerror("Error", f"Error with new data input: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = BiddingStrategyApp(root)
    root.mainloop()
