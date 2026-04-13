import pandas as pd

# Step 1: Load dataset
data = pd.read_csv("cybersecurity_intrusion_data.csv")

# Step 2: Fill missing values safely
data = data.fillna("missing")

# Step 3: Remove duplicates
data = data.drop_duplicates()

# Step 4: Remove useless column
data = data.drop(['session_id'], axis=1, errors='ignore')

# Step 5: Convert ALL columns to string first (fix mixed types)
for col in data.columns:
    data[col] = data[col].astype(str)

# Step 6: Encode ALL columns using factorize (best fix)
for col in data.columns:
    data[col] = pd.factorize(data[col])[0]

# Step 7: Save cleaned dataset
data.to_csv("cleaned_dataset.csv", index=False)

print("✅ FINAL CLEANING DONE")