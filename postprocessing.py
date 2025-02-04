import pandas as pd

df = pd.read_csv("synthetic_data.csv")

# Enforce constraints programmatically
df["age"] = df["age"].clip(0.1, 80.0)
df["bmi"] = df["bmi"].clip(13.0, 60.0)
df["diabetes"] = df.apply(
    lambda x: 1 if (x["HbA1c_level"] > 6.5) or (x["blood_glucose_level"] > 200) else 0,
    axis=1,
)

df.to_csv("final_synthetic_data.csv", index=False)
