import pandas as pd
import numpy as np
import random

product_categories = ["Confectionery", "Beverages", "Snacks"]

sales_trends = ["Increasing", "Decreasing", "Stable"]

num_samples = 1000

data = {
    "Current_Stock": np.random.randint(0, 500, num_samples),
    "Average_Sales": np.random.randint(10, 200, num_samples),
    "Days_Since_Last_Restock": np.random.randint(1, 30, num_samples),
    "Product_Category": [random.choice(product_categories) for _ in range(num_samples)],
    "Sales_Trend": [random.choice(sales_trends) for _ in range(num_samples)],
    "Lead_Time": np.random.randint(1, 30, num_samples),
    "Supplier_Rating": np.round(np.random.uniform(1.0, 5.0, num_samples), 1),
    "Seasonality_Index": np.round(np.random.uniform(1.0, 2.0, num_samples), 1),
    "Discount_Offered": np.random.randint(0, 2, num_samples),
    "Storage_Cost_Per_Unit": np.round(np.random.uniform(0.1, 1.0, num_samples), 2),
    "Product_Age": np.random.randint(1, 365, num_samples),
    "Promotion_Active": np.random.randint(0, 2, num_samples),
}

df = pd.DataFrame(data)

df["Restock_Needed"] = np.where(
    (df["Current_Stock"] < df["Average_Sales"]) | (df["Days_Since_Last_Restock"] > 14), 1, 0
)

df.to_csv("restocking_data.csv", index=False)

print("Dataset generated and saved as 'restocking_data.csv'.")