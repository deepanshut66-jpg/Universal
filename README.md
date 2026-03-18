# 🏦 Universal Bank — Personal Loan Campaign Intelligence Dashboard

A comprehensive Streamlit dashboard for predicting personal loan acceptance using classification algorithms (Decision Tree, Random Forest, Gradient Boosted Tree). Built for Universal Bank's marketing team to optimise campaigns with a reduced budget.

## 🚀 Live Demo
Deploy on [Streamlit Community Cloud](https://streamlit.io/cloud) by connecting this GitHub repository.

## 📊 Features

### 1. Descriptive Analytics
- Customer demographics & financial profiles
- Loan acceptance distribution and patterns
- Education, family, income, and banking relationship analysis

### 2. Diagnostic Analytics
- Correlation heatmap across all features
- Income × CC Spend scatter analysis
- Multi-factor analysis (Education × Income)
- CD Account deep-dive and mortgage analysis

### 3. Predictive Analytics
- Decision Tree, Random Forest, and Gradient Boosted Tree models
- Performance metrics table (Accuracy, Precision, Recall, F1, ROC-AUC)
- Combined ROC curve for all models
- Confusion matrices with counts and percentages
- Feature importance visualization

### 4. Prescriptive Analytics
- Customer priority segmentation (Low → Very High)
- Budget-optimised campaign strategy
- Strategic recommendations with priority scoring
- Expected ROI and budget impact estimation

### 5. Predict New Customers
- Upload new customer CSV data
- Get predictions with acceptance probability
- Download results with priority tiers

## 📁 Files
| File | Description |
|------|-------------|
| `app.py` | Main Streamlit dashboard application |
| `UniversalBank.csv` | Training dataset (5,000 customers) |
| `test_data_for_prediction.csv` | Sample test file for upload & prediction |
| `requirements.txt` | Python dependencies |

## ⚙️ Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd universal_bank_dashboard

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

## 🛠 Tech Stack
- **Python 3.9+**
- **Streamlit** — Dashboard framework
- **Scikit-Learn** — ML models
- **Plotly** — Interactive charts
- **Pandas / NumPy** — Data processing

## 📋 Column Descriptions
| Column | Description |
|--------|-------------|
| ID | Customer ID |
| Age | Customer's age in years |
| Experience | Years of professional experience |
| Income | Annual income ($000) |
| ZIP Code | Home address ZIP code |
| Family | Family size |
| CCAvg | Avg monthly credit card spending ($000) |
| Education | 1: Undergrad, 2: Graduate, 3: Advanced |
| Mortgage | House mortgage value ($000) |
| Personal Loan | Target — accepted loan? (0/1) |
| Securities Account | Has securities account? (0/1) |
| CD Account | Has CD account? (0/1) |
| Online | Uses internet banking? (0/1) |
| CreditCard | Uses bank credit card? (0/1) |
