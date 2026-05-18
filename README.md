# ⚽ PlayerIQ: Football Performance & Market Value Analytics

PlayerIQ is a full-stack football analytics platform I developed to analyze player performance, identify undervalued talent, and predict player market value using machine learning and advanced statistical analysis.

The project was inspired by how modern football clubs increasingly rely on data-driven scouting and recruitment strategies to improve transfer decisions and reduce financial risk. Traditional scouting methods often rely heavily on subjective judgement, making it difficult to consistently identify high-potential players at lower market values.

Through this project, I wanted to simulate how real football analytics systems are built and used within professional sporting environments.

--

#  Project Objectives

The main objectives of the project were to:

- Build a machine learning model capable of predicting player market value
- Engineer football-specific performance features
- Identify undervalued players using analytical methods
- Create an interactive scouting dashboard for player comparison
- Simulate a real-world sports analytics workflow from data processing to deployment

--

#  Key Features

- Machine learning-based market value prediction
- Interactive Streamlit scouting dashboard
- Real-time player filtering and comparison
- Feature-engineered football performance metrics
- Player valuation benchmarking
- Statistical performance visualization
- End-to-end machine learning pipeline

---

# 🛠️ Technologies Used

## Programming & Data Science
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost

## Dashboard & Visualization
- Streamlit
- Plotly
- Matplotlib

## Machine Learning
- Regression Models
- Feature Engineering
- Model Evaluation
- Predictive Analytics

--

#  Machine Learning Workflow

## 1. Data Processing
I cleaned and transformed football player datasets by:
- Handling missing values
- Standardizing records
- Removing inconsistencies
- Preparing structured analytical datasets

---

## 2. Feature Engineering
To improve predictive performance, I engineered football-specific metrics such as:
- Potential Gap
- Goal Contribution Ratios
- Age Segmentation
- Position-Based Metrics
- Performance Efficiency Indicators
- Market Value Performance Ratios

These features helped improve the model’s ability to identify hidden player value patterns.

---

## 3. Model Development
I trained and evaluated multiple machine learning regression models to estimate player market value, including:
- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

The final model was optimized using feature importance analysis and hyperparameter tuning.

--

#  Model Performance

| Metric | Result |
|---|---|
| Best Performing Model | XGBoost Regressor |
| R² Score | 0.91 |
| Mean Absolute Error (MAE) | ±€2.8M |
| Root Mean Squared Error (RMSE) | €4.1M |

The XGBoost model achieved the strongest predictive performance, demonstrating the ability to estimate player market values with high accuracy while capturing complex relationships between player attributes and market valuation.

---

#  Dashboard Functionality

The Streamlit dashboard allows users to:
- Search and filter players dynamically
- Compare player valuations
- Analyze player performance metrics
- Identify undervalued players
- Visualize statistical trends interactively

The interface was designed to simulate modern football recruitment and scouting systems used by professional clubs.

--

#  Business & Real-World Impact

This project demonstrates how football organizations can leverage machine learning and analytics to improve recruitment efficiency and reduce transfer market risk.

The platform supports:
- Data-driven scouting decisions
- Identification of undervalued talent
- Recruitment optimization
- Performance benchmarking
- Reduced reliance on subjective analysis

The project also strengthened my understanding of:
- Sports analytics
- Machine learning model development
- Feature engineering
- Interactive dashboard design
- End-to-end ML workflows
- Real-world predictive system design

---

# 📂 Project Structure

```bash
playeriq/
│
├── app.py
├── requirements.txt
├── README.md
├── data/
├── models/
├── notebooks/
├── assets/
├── utils/
└── dashboard/
```

--

#  Installation & Setup

## Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/playeriq-football-analytics.git
cd playeriq-football-analytics
```

--

## Create Virtual Environment

```bash
python -m venv venv
```

---

## Activate Environment

### Windows
```bash
venv\Scripts\activate
```

### Mac/Linux
```bash
source venv/bin/activate
```

--

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Application

```bash
streamlit run app.py
```

--

#  Future Improvements

Planned future improvements include:
- Real-time football API integration
- Deep learning-based performance forecasting
- Transfer recommendation systems
- Team chemistry analysis
- Injury risk prediction
- Cloud deployment and authentication systems

--

#  Author

Rutendo Simango  
Data Science | Machine Learning | Analytics

GitHub: :contentReference[oaicite:0]{index=0}

---

#  License

This project was developed for educational, research, and portfolio purposes.
