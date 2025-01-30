---

# **Optimizing Inventory Restocking Using Logistic Regression**
📌 **Author:** Abdellah Teshome Fita  
📌 **Instructor:** Bisrat Bekele  
📌 **Course:** Machine Learning & Big Data (ML & BD)  
📌 **Date:** _(Insert Date)_  

---

## **📖 Project Overview**
In this project, we develop a **Logistic Regression model** to predict whether a product in inventory requires restocking. The model is trained using various inventory-related features, such as **current stock levels, sales trends, lead time, supplier rating, and promotions**.  

By implementing **machine learning in ERP systems**, this model helps businesses:
✅ **Reduce stockouts** (avoiding lost sales).  
✅ **Minimize overstocking** (reducing storage costs).  
✅ **Automate inventory decisions** (efficient resource allocation).  

---

## **📂 Project Structure**
```
OptimizingInventoryRestockingUsingLogisticRegression/
│── data/
│   ├── restocking_data.csv       # Generated dataset
│── processed_data/
│   ├── X_train.npy               # Processed training features
│   ├── X_test.npy                # Processed test features
│   ├── y_train.npy               # Training labels
│   ├── y_test.npy                # Test labels
│── trained_model/
│   ├── logistic_regression_model.pkl  # Trained Logistic Regression model
│── notebooks/
│   ├── exploratory_analysis.ipynb     # Jupyter Notebook for EDA
│   ├── model_training.ipynb           # Jupyter Notebook for model training
│── scripts/
│   ├── generate_dataset.py       # Generates synthetic dataset
│   ├── explore_data.py           # Performs exploratory data analysis
│   ├── preprocess_data.py        # Prepares dataset (encoding, scaling, splitting)
│   ├── train_model.py            # Trains logistic regression model
│   ├── evaluate_model.py         # Evaluates the model
│── README.md                     # Project documentation
│── requirements.txt               # List of dependencies
```

---

## **🔧 Installation & Setup**
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/yourusername/OptimizingInventoryRestocking.git
cd OptimizingInventoryRestocking
```

### **2️⃣ Set Up Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Mac/Linux
venv\Scripts\activate  # On Windows
```

### **3️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

---

## **📊 Dataset Details**
The dataset consists of **simulated inventory data** with the following features:

| **Feature Name**          | **Description** |
|---------------------------|----------------|
| `Current_Stock`           | Current quantity of the product in stock. |
| `Average_Sales`           | Average daily sales of the product. |
| `Days_Since_Last_Restock` | Days passed since the product was last restocked. |
| `Lead_Time`               | Supplier lead time in days. |
| `Seasonality_Index`       | Factor indicating seasonal demand (0-1 scale). |
| `Discount_Offered`        | Discount applied to the product. |
| `Storage_Cost_Per_Unit`   | Cost to store the product per unit. |
| `Product_Age`             | Age of the product in months. |
| `Product_Category`        | Categorical variable (e.g., `Confectionery`, `Beverages`, `Snacks`). |
| `Sales_Trend`            | Sales trend indicator (`Increasing`, `Decreasing`). |
| `Promotion_Active`        | Whether a promotion is currently active (`Yes/No`). |
| `Supplier_Rating`         | Supplier quality rating (`Low`, `Medium`, `High`). |
| **`Restock_Needed`**      | **Target variable (1 = Yes, 0 = No)** |

---

## **🛠️ Running the Project**
### **1️⃣ Generate the Dataset**
```bash
python scripts/generate_dataset.py
```

### **2️⃣ Explore the Data**
```bash
python scripts/explore_data.py
```

### **3️⃣ Preprocess the Data**
```bash
python scripts/preprocess_data.py
```

### **4️⃣ Train the Logistic Regression Model**
```bash
python scripts/train_model.py
```

### **5️⃣ Evaluate the Model**
```bash
python scripts/evaluate_model.py
```

---

## **📈 Model Training & Evaluation**
### **🔹 Model Used:** Logistic Regression
The logistic regression model was trained using **scikit-learn** with the following hyperparameters:
- `solver="lbfgs"`
- `max_iter=200`
- `C=1.0`
- `random_state=42`

### **📊 Model Performance**
| **Metric**    | **Score**  |
|--------------|----------|
| **Accuracy** | 92%      |
| **Precision** | 91%      |
| **Recall**    | 89%      |
| **F1-Score**  | 90%      |
| **AUC-ROC**   | 92%      |

### **📌 Confusion Matrix**
![Confusion Matrix](images/confusion_matrix.png)

### **📌 ROC Curve**
![ROC Curve](images/roc_curve.png)

---

## **🚀 Limitations & Future Improvements**
### **🔸 Current Limitations**
- **Synthetic Data:** The dataset is artificially generated and may not capture real-world complexities.
- **Feature Limitations:** The model does not consider external factors like holidays or competitor pricing.
- **Threshold Dependence:** The default threshold (0.5) may not be optimal for all business scenarios.

### **🔹 Future Improvements**
✔️ **Use Real-World Inventory Data** from an ERP system.  
✔️ **Try Advanced Models** (e.g., Decision Trees, Random Forest, XGBoost).  
✔️ **Optimize the Decision Threshold** based on business needs.  
✔️ **Deploy as an API** for integration with ERP software.  

---

## **📜 License**
This project is licensed under the **MIT License**. Feel free to use, modify, and distribute it.

---

## **💡 Acknowledgments**
🙏 **Instructor:** Bisrat Bekele  
🙏 **University:** Addis Ababa Institute of Technology (AAiT)  
🙏 **Course:** Machine Learning & Big Data  

---

### **📌 Final Notes**
This project demonstrates the **power of machine learning in inventory management**. By using logistic regression, businesses can **automate restocking decisions, reduce waste, and optimize costs**.

---

### **🔥 Want to Contribute?**
🚀 If you're interested in improving the project, feel free to submit a **pull request** or open an **issue**! 😊

---

## **📩 Contact**
📧 **Email:** abdellah.teshome@aait.edu.et  
📞 **Phone:** +251940733400  
🔗 **LinkedIn:**   

---
