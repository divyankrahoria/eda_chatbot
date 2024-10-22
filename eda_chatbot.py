# Save this as eda_chatbot.py
import pandas as pd
import numpy as np
from typing import Dict, Any
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_iris
import os
import sys
from datetime import datetime

class EDABot:
    def __init__(self):
        self.df = None
        self.chat_history = []
        
    def load_sample_data(self) -> str:
        """Load iris dataset as a sample."""
        try:
            iris = load_iris()
            self.df = pd.DataFrame(iris.data, columns=iris.feature_names)
            return "✓ Sample iris dataset loaded successfully!"
        except Exception as e:
            return f"Error loading sample data: {str(e)}"
    
    def load_data(self, file_path: str) -> str:
        """Load user's dataset."""
        try:
            if file_path.endswith('.csv'):
                self.df = pd.read_csv(file_path)
            elif file_path.endswith(('.xls', '.xlsx')):
                self.df = pd.read_excel(file_path)
            else:
                return "⚠ Unsupported file format. Please use CSV or Excel files."
            
            return f"✓ Data loaded successfully! Shape: {self.df.shape}"
        except Exception as e:
            return f"⚠ Error loading data: {str(e)}"
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get basic information about the dataset."""
        if self.df is None:
            return "⚠ No data loaded. Please load a dataset first."
        
        info = {
            "Shape": self.df.shape,
            "Columns": list(self.df.columns),
            "Data Types": self.df.dtypes.to_dict(),
            "Missing Values": self.df.isnull().sum().to_dict(),
            "Sample": self.df.head(3)
        }
        return info
    
    def analyze_column(self, column_name: str) -> Dict[str, Any]:
        """Analyze a specific column."""
        if self.df is None:
            return "⚠ No data loaded. Please load a dataset first."
        
        if column_name not in self.df.columns:
            return f"⚠ Column '{column_name}' not found in dataset."
        
        column_data = self.df[column_name]
        analysis = {
            "dtype": str(column_data.dtype),
            "unique_values": len(column_data.unique()),
            "missing_values": column_data.isnull().sum(),
            "missing_percentage": (column_data.isnull().sum() / len(column_data)) * 100
        }
        
        if np.issubdtype(column_data.dtype, np.number):
            analysis.update({
                "mean": column_data.mean(),
                "median": column_data.median(),
                "std": column_data.std(),
                "min": column_data.min(),
                "max": column_data.max()
            })
            
            # Create histogram
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=column_data, nbinsx=30))
            fig.update_layout(title=f'Distribution of {column_name}')
            analysis["plot"] = fig
            
        return analysis
    
    def get_correlations(self) -> Dict[str, Any]:
        """Get correlation matrix for numerical columns."""
        if self.df is None:
            return "⚠ No data loaded. Please load a dataset first."
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return "⚠ Not enough numerical columns for correlation analysis."
        
        corr_matrix = self.df[numeric_cols].corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu'
        ))
        fig.update_layout(title='Correlation Heatmap')
        
        return {
            "correlation_matrix": corr_matrix,
            "plot": fig
        }
    
    def save_analysis(self, analysis_type: str, content: Any):
        """Save analysis results to a file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analysis_{analysis_type}_{timestamp}.txt"
        
        with open(filename, 'w') as f:
            f.write(str(content))
        
        return f"✓ Analysis saved to {filename}"

def print_menu():
    """Print the main menu."""
    print("\n=== EDA Chatbot Menu ===")
    print("1. Load sample data (iris dataset)")
    print("2. Load your own data")
    print("3. Get data information")
    print("4. Analyze a specific column")
    print("5. Get correlations")
    print("6. Save last analysis")
    print("7. Exit")
    print("=======================")

def main():
    bot = EDABot()
    last_analysis = None
    
    print("Welcome to EDA Chatbot! 🤖")
    
    while True:
        print_menu()
        choice = input("\nEnter your choice (1-7): ").strip()
        
        if choice == "1":
            print(bot.load_sample_data())
        
        elif choice == "2":
            file_path = input("Enter the path to your data file (CSV or Excel): ").strip()
            print(bot.load_data(file_path))
        
        elif choice == "3":
            info = bot.get_data_info()
            if isinstance(info, dict):
                print("\nDataset Information:")
                print(f"Shape: {info['Shape']}")
                print(f"Columns: {', '.join(info['Columns'])}")
                print("\nData Types:")
                for col, dtype in info['Data Types'].items():
                    print(f"  {col}: {dtype}")
                print("\nMissing Values:")
                for col, missing in info['Missing Values'].items():
                    print(f"  {col}: {missing}")
                print("\nSample Data:")
                print(info['Sample'])
                last_analysis = info
            else:
                print(info)
        
        elif choice == "4":
            if bot.df is None:
                print("⚠ Please load a dataset first!")
                continue
                
            print("\nAvailable columns:")
            for i, col in enumerate(bot.df.columns, 1):
                print(f"{i}. {col}")
            
            col_choice = input("\nEnter column number: ").strip()
            try:
                column_name = bot.df.columns[int(col_choice) - 1]
                analysis = bot.analyze_column(column_name)
                if isinstance(analysis, dict):
                    print(f"\nAnalysis of column '{column_name}':")
                    for key, value in analysis.items():
                        if key != "plot":
                            print(f"{key}: {value}")
                    if "plot" in analysis:
                        analysis["plot"].show()
                    last_analysis = analysis
                else:
                    print(analysis)
            except (ValueError, IndexError):
                print("⚠ Invalid column number!")
        
        elif choice == "5":
            result = bot.get_correlations()
            if isinstance(result, dict):
                print("\nCorrelation Matrix:")
                print(result["correlation_matrix"])
                result["plot"].show()
                last_analysis = result
            else:
                print(result)
        
        elif choice == "6":
            if last_analysis is None:
                print("⚠ No analysis to save!")
                continue
            
            analysis_type = input("Enter a name for this analysis: ").strip()
            print(bot.save_analysis(analysis_type, last_analysis))
        
        elif choice == "7":
            print("Thank you for using EDA Chatbot! Goodbye! 👋")
            break
        
        else:
            print("⚠ Invalid choice! Please try again.")

if __name__ == "__main__":
    main()