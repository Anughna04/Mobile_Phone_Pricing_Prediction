# 📱 Mobile Phone Price Range Prediction

A machine learning project to predict the **price range** of mobile phones based on hardware specifications. This application uses a supervised ML model trained on market data and is deployed using **Streamlit** for a user-friendly interface.

---

## 📌 Objective

To develop a predictive system that classifies a mobile phone into one of four price categories:

- `0` → Low Cost  
- `1` → Medium Cost  
- `2` → High Cost  
- `3` → Very High Cost

This helps manufacturers and online platforms evaluate how a new mobile might be priced based on its technical features.

---

## 🧠 Machine Learning Approach

The model used is:

### ✅ **XGBoost Classifier (Extreme Gradient Boosting)**

- Optimized for speed and performance
- Handles categorical and numerical features well
- Supports feature importance visualization

The model is trained on labeled data and saved using `joblib` for fast loading in a production environment.

---

## 🧰 Tools & Technologies Used

| Purpose             | Stack                                    |
|---------------------|------------------------------------------|
| Language            | Python 3.x                               |
| Data Processing     | pandas, numpy                            |
| Modeling            | scikit-learn (RandomForest)              |
| Model Serialization | joblib                                   |
| Deployment          | Streamlit                                |
| Visualization       | seaborn, matplotlib (optional)           |

---

## 🚀 Streamlit App Features

✅ Predict price range via form-based input  
✅ CSV Upload for batch predictions  
✅ Visual display of predictions and model confidence  
✅ Responsive error handling for invalid inputs  

---

## 📂 Project Structure

    mobile_phone_pricing/
    ├── app.py # Streamlit Web App
    ├── train_model.py # Model training script using XGBoost
    ├── model.pkl # Trained XGBoost model
    ├── requirements.txt # Dependencies
    ├── dataset.csv # Mobile pricing dataset
    └── README.md # Project documentation

🧪 How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/mobile-phone-pricing.git
cd mobile-phone-pricing
```
2. Install Dependencies

       pip install -r requirements.txt

3. Train the Model

        python train_model.py
4. Launch the Streamlit App

        streamlit run app.py

###👩‍💻 Author

Anughna Kandimalla
