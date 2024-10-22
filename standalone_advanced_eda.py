# Save as standalone_advanced_eda.py
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class AdvancedEDABot:
    def __init__(self):
        self.df = None
        self.analysis_history = []
        
    def load_sample_data(self) -> str:
        """Load iris dataset as a sample."""
        try:
            iris = load_iris()
            self.df = pd.DataFrame(
                iris.data, 
                columns=iris.feature_names
            )
            self.df['target'] = iris.target
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

    def get_smart_summary(self) -> Dict[str, Any]:
        """Get comprehensive data summary with insights."""
        if self.df is None:
            return "⚠ No data loaded. Please load a dataset first."
        
        summary = {
            "basic_info": {
                "rows": self.df.shape[0],
                "columns": self.df.shape[1],
                "total_cells": self.df.size,
                "memory_usage": f"{self.df.memory_usage(deep=True).sum() / 1024:.2f} KB"
            },
            "column_types": {
                "numeric": list(self.df.select_dtypes(include=[np.number]).columns),
                "categorical": list(self.df.select_dtypes(include=['object', 'category']).columns),
                "datetime": list(self.df.select_dtypes(include=['datetime64']).columns)
            },
            "missing_data": {
                "total_missing": self.df.isnull().sum().sum(),
                "missing_by_column": self.df.isnull().sum().to_dict()
            },
            "numeric_summary": {}
        }
        
        # Analyze numeric columns
        for col in summary["column_types"]["numeric"]:
            col_data = self.df[col].dropna()
            summary["numeric_summary"][col] = {
                "mean": col_data.mean(),
                "median": col_data.median(),
                "std": col_data.std(),
                "skew": col_data.skew(),
                "unique_values": len(col_data.unique())
            }
        
        return summary

    def analyze_correlations(self) -> Dict[str, Any]:
        """Analyze correlations with insights."""
        if self.df is None:
            return "⚠ No data loaded. Please load a dataset first."
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return "⚠ Need at least 2 numerical columns for correlation analysis."
        
        corr_matrix = self.df[numeric_cols].corr()
        
        # Find strongest correlations
        strong_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                coef = corr_matrix.iloc[i, j]
                if abs(coef) > 0.5:  # Threshold for strong correlation
                    strong_corr.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': coef
                    })
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmin=-1,
            zmax=1
        ))
        fig.update_layout(
            title='Correlation Heatmap',
            height=600,
            width=800
        )
        
        return {
            "correlation_matrix": corr_matrix,
            "strong_correlations": strong_corr,
            "plot": fig
        }

    def analyze_distributions(self) -> Dict[str, Any]:
        """Analyze distributions with statistical tests."""
        if self.df is None:
            return "⚠ No data loaded. Please load a dataset first."
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        results = {}
        
        for col in numeric_cols:
            data = self.df[col].dropna()
            
            # Statistical tests and metrics
            normality_test = stats.normaltest(data)
            iqr = data.quantile(0.75) - data.quantile(0.25)
            
            stats_data = {
                "mean": data.mean(),
                "median": data.median(),
                "std": data.std(),
                "skew": data.skew(),
                "kurtosis": data.kurtosis(),
                "iqr": iqr,
                "normality_test_p_value": normality_test.pvalue
            }
            
            # Create distribution plot
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=data,
                name='Histogram',
                showlegend=True
            ))
            
            # Add KDE
            kde_x = np.linspace(data.min(), data.max(), 100)
            kde = stats.gaussian_kde(data)
            fig.add_trace(go.Scatter(
                x=kde_x,
                y=kde(kde_x) * len(data) * (data.max() - data.min()) / 30,
                name='KDE',
                line=dict(color='red')
            ))
            
            fig.update_layout(
                title=f'Distribution of {col}',
                xaxis_title=col,
                yaxis_title='Count'
            )
            
            results[col] = {
                "stats": stats_data,
                "plot": fig,
                "insights": self._generate_distribution_insights(stats_data)
            }
        
        return results
    
    def _generate_distribution_insights(self, stats: Dict) -> List[str]:
        """Generate insights about the distribution."""
        insights = []
        
        # Check normality
        if stats['normality_test_p_value'] < 0.05:
            insights.append("Distribution is significantly non-normal")
        else:
            insights.append("Distribution appears approximately normal")
        
        # Check skewness
        if abs(stats['skew']) > 1:
            direction = "right" if stats['skew'] > 0 else "left"
            insights.append(f"Distribution is heavily skewed to the {direction}")
        
        # Check kurtosis
        if abs(stats['kurtosis']) > 2:
            type_kurt = "heavy tails" if stats['kurtosis'] > 0 else "light tails"
            insights.append(f"Distribution has {type_kurt}")
        
        return insights

    def detect_outliers(self, method='both') -> Dict[str, Any]:
        """Detect outliers using multiple methods."""
        if self.df is None:
            return "⚠ No data loaded. Please load a dataset first."
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        results = {}
        
        for col in numeric_cols:
            data = self.df[col].dropna()
            
            # Z-score method
            z_scores = np.abs(stats.zscore(data))
            z_outliers = z_scores > 3
            
            # IQR method
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            iqr_outliers = (data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))
            
            # Combined results
            outlier_summary = {
                "zscore_outliers": sum(z_outliers),
                "iqr_outliers": sum(iqr_outliers),
                "zscore_percentage": (sum(z_outliers) / len(data)) * 100,
                "iqr_percentage": (sum(iqr_outliers) / len(data)) * 100
            }
            
            # Create box plot with points
            fig = go.Figure()
            fig.add_trace(go.Box(
                y=data,
                name=col,
                boxpoints='outliers',
                jitter=0.3,
                pointpos=-1.8
            ))
            
            fig.update_layout(
                title=f'Box Plot with Outliers - {col}',
                yaxis_title=col,
                showlegend=False
            )
            
            results[col] = {
                "summary": outlier_summary,
                "plot": fig
            }
        
        return results

def print_menu():
    """Print the main menu."""
    print("\n=== Advanced EDA Bot Menu ===")
    print("1. Load sample data (iris dataset)")
    print("2. Load your own data")
    print("3. Get smart data summary")
    print("4. Analyze correlations")
    print("5. Analyze distributions")
    print("6. Detect outliers")
    print("7. Save analysis")
    print("8. Exit")
    print("===========================")

def main():
    bot = AdvancedEDABot()
    last_analysis = None
    
    print("Welcome to Advanced EDA Bot! 🤖")
    print("This version includes advanced analytics without requiring API keys.")
    
    while True:
        print_menu()
        choice = input("\nEnter your choice (1-8): ").strip()
        
        if choice == "1":
            print(bot.load_sample_data())
            
        elif choice == "2":
            file_path = input("Enter the path to your data file (CSV or Excel): ").strip()
            print(bot.load_data(file_path))
            
        elif choice == "3":
            summary = bot.get_smart_summary()
            if isinstance(summary, dict):
                print("\n=== Smart Data Summary ===")
                print(f"\nBasic Information:")
                for k, v in summary["basic_info"].items():
                    print(f"  {k}: {v}")
                
                print(f"\nColumn Types:")
                for type_name, cols in summary["column_types"].items():
                    print(f"  {type_name}: {', '.join(cols)}")
                
                print(f"\nMissing Data:")
                print(f"  Total missing values: {summary['missing_data']['total_missing']}")
                
                print("\nNumeric Column Statistics:")
                for col, stats in summary["numeric_summary"].items():
                    print(f"\n  {col}:")
                    for stat, value in stats.items():
                        print(f"    {stat}: {value:.2f}")
                
                last_analysis = summary
                
        elif choice == "4":
            result = bot.analyze_correlations()
            if isinstance(result, dict):
                print("\n=== Correlation Analysis ===")
                print("\nStrong Correlations:")
                for corr in result["strong_correlations"]:
                    print(f"  {corr['var1']} & {corr['var2']}: {corr['correlation']:.3f}")
                result["plot"].show()
                last_analysis = result
                
        elif choice == "5":
            results = bot.analyze_distributions()
            if isinstance(results, dict):
                for col, result in results.items():
                    print(f"\n=== Distribution Analysis: {col} ===")
                    print("\nStatistics:")
                    for stat, value in result["stats"].items():
                        print(f"  {stat}: {value:.3f}")
                    print("\nInsights:")
                    for insight in result["insights"]:
                        print(f"  • {insight}")
                    result["plot"].show()
                last_analysis = results
                
        elif choice == "6":
            results = bot.detect_outliers()
            if isinstance(results, dict):
                for col, result in results.items():
                    print(f"\n=== Outlier Analysis: {col} ===")
                    print("\nSummary:")
                    for method, value in result["summary"].items():
                        print(f"  {method}: {value:.2f}")
                    result["plot"].show()
                last_analysis = results
                
        elif choice == "7":
            if last_analysis is None:
                print("⚠ No analysis to save!")
                continue
            
            filename = input("Enter filename to save analysis (will save as .txt): ").strip()
            if not filename.endswith('.txt'):
                filename += '.txt'
            
            with open(filename, 'w') as f:
                f.write(str(last_analysis))
            print(f"✓ Analysis saved to {filename}")
            
        elif choice == "8":
            print("Thank you for using Advanced EDA Bot! Goodbye! 👋")
            break
            
        else:
            print("⚠ Invalid choice! Please try again.")

if __name__ == "__main__":
    main()