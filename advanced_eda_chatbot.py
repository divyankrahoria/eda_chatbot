# Save as advanced_eda_chatbot.py
import pandas as pd
import numpy as np
from typing import Dict, Any
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
import seaborn as sns
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS

class AdvancedEDABot:
    def __init__(self, openai_api_key=None):
        self.df = None
        self.chat_history = []
        self.openai_api_key = openai_api_key
        
        if openai_api_key:
            self.setup_rag()
    
    def setup_rag(self):
        """Set up RAG components with statistical analysis knowledge."""
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        self.llm = ChatOpenAI(
            temperature=0.7,
            model="gpt-3.5-turbo",
            openai_api_key=self.openai_api_key
        )
        
        # Load statistical analysis knowledge
        knowledge = """
        Statistical Analysis Guide:
        
        1. Descriptive Statistics:
        - Mean: Central tendency measure
        - Median: Middle value
        - Mode: Most frequent value
        - Standard Deviation: Spread measure
        - Variance: Squared deviation
        
        2. Distribution Analysis:
        - Normal distribution checks
        - Skewness: Asymmetry measure
        - Kurtosis: Tail heaviness
        - Q-Q plots: Normality check
        
        3. Hypothesis Testing:
        - T-tests: Compare means
        - Chi-square: Independence test
        - ANOVA: Multiple group comparison
        
        4. Advanced Analytics:
        - PCA: Dimensionality reduction
        - Clustering: Group similar data
        - Outlier detection: Find anomalies
        """
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_text(knowledge)
        
        self.vector_store = FAISS.from_texts(texts, self.embeddings)
        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(),
            memory=ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
        )

    def ask_analysis_question(self, question: str) -> str:
        """Ask a question about the analysis."""
        if not self.openai_api_key:
            return "⚠ OpenAI API key not provided. Running in basic mode."
        
        try:
            response = self.conversation_chain({"question": question})
            return response["answer"]
        except Exception as e:
            return f"⚠ Error processing question: {str(e)}"

    def analyze_distributions(self) -> Dict[str, Any]:
        """Analyze distributions of numerical columns."""
        if self.df is None:
            return "⚠ No data loaded. Please load a dataset first."
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        results = {}
        
        for col in numeric_cols:
            data = self.df[col].dropna()
            
            # Basic statistics
            stats_data = {
                "mean": data.mean(),
                "median": data.median(),
                "std": data.std(),
                "skew": data.skew(),
                "kurtosis": data.kurtosis()
            }
            
            # Normality test
            _, p_value = stats.normaltest(data)
            stats_data["normal_dist_p_value"] = p_value
            
            # Create distribution plot
            fig = px.histogram(
                self.df,
                x=col,
                marginal="box",
                title=f'Distribution of {col}'
            )
            
            results[col] = {
                "stats": stats_data,
                "plot": fig
            }
        
        return results

    def perform_pca(self) -> Dict[str, Any]:
        """Perform PCA on numerical columns."""
        if self.df is None:
            return "⚠ No data loaded. Please load a dataset first."
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return "⚠ Need at least 2 numerical columns for PCA."
        
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.df[numeric_cols])
        
        # Perform PCA
        pca = PCA()
        pca_result = pca.fit_transform(scaled_data)
        
        # Create scree plot
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
        
        fig_scree = go.Figure()
        fig_scree.add_trace(go.Scatter(
            y=explained_variance_ratio,
            mode='lines+markers',
            name='Individual'
        ))
        fig_scree.add_trace(go.Scatter(
            y=cumulative_variance_ratio,
            mode='lines+markers',
            name='Cumulative'
        ))
        fig_scree.update_layout(title='Scree Plot')
        
        # Create biplot
        if pca_result.shape[1] >= 2:
            fig_biplot = px.scatter(
                x=pca_result[:, 0],
                y=pca_result[:, 1],
                title='PCA Biplot'
            )
        
        return {
            "explained_variance_ratio": explained_variance_ratio.tolist(),
            "cumulative_variance_ratio": cumulative_variance_ratio.tolist(),
            "scree_plot": fig_scree,
            "biplot": fig_biplot if pca_result.shape[1] >= 2 else None
        }

    def detect_outliers(self, method='zscore') -> Dict[str, Any]:
        """Detect outliers in numerical columns."""
        if self.df is None:
            return "⚠ No data loaded. Please load a dataset first."
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        results = {}
        
        for col in numeric_cols:
            data = self.df[col].dropna()
            
            if method == 'zscore':
                z_scores = np.abs(stats.zscore(data))
                outliers = z_scores > 3
            else:  # IQR method
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                outliers = (data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))
            
            # Create box plot
            fig = go.Figure()
            fig.add_trace(go.Box(y=data, name=col))
            fig.update_layout(title=f'Box Plot with Outliers - {col}')
            
            results[col] = {
                "n_outliers": sum(outliers),
                "outlier_percentage": (sum(outliers) / len(data)) * 100,
                "plot": fig
            }
        
        return results

def print_menu():
    """Print the enhanced menu."""
    print("\n=== Advanced EDA Chatbot Menu ===")
    print("1. Load sample data (iris dataset)")
    print("2. Load your own data")
    print("3. Basic data information")
    print("4. Analyze distributions")
    print("5. Perform PCA")
    print("6. Detect outliers")
    print("7. Ask analysis question (requires OpenAI API key)")
    print("8. Save analysis results")
    print("9. Exit")
    print("===============================")

def main():
    api_key = input("Enter OpenAI API key (or press Enter to skip): ").strip()
    bot = AdvancedEDABot(api_key if api_key else None)
    
    print("Welcome to Advanced EDA Chatbot! 🤖")
    
    while True:
        print_menu()
        choice = input("\nEnter your choice (1-9): ").strip()
        
        # ... (implement menu choices similar to previous version but with new methods)
        
        if choice == "9":
            print("Thank you for using Advanced EDA Chatbot! Goodbye! 👋")
            break

if __name__ == "__main__":
    main()