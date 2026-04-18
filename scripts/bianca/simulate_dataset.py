import pandas as pd
import numpy as np

# Step 1: Load your dataset
df = pd.read_csv(
    "database/dataset/simulated/cardio_train.csv", sep=";"
)  # assuming typical format
df = df.drop(columns=["id"])

# Step 2: Add BMI column (BMI = weight (kg) / height (m)^2)
df["bmi"] = df["weight"] / (df["height"] / 100) ** 2

# Step 3: Create class imbalance (~8% positive class)
df_pos = df[df["cardio"] == 1].sample(frac=0.25, random_state=42)
df_neg = df[df["cardio"] == 0]
df_imbal = (
    pd.concat([df_neg, df_pos]).sample(frac=1, random_state=42).reset_index(drop=True)
)

# Step X: Add accessdate
start_date = pd.to_datetime("1980-01-01")
end_date = pd.to_datetime("2000-12-31")

# Randomly generate access dates
access_dates = pd.to_datetime(
    np.random.randint(
        start_date.value // 10**9,  # convert to seconds
        end_date.value // 10**9,
        size=len(df_imbal),
    ),
    unit="s",
)
df_imbal["accessdate"] = access_dates

# Step Y: Add birthdate based on age
# Remember, "age" is in days, so subtract it from accessdate
df_imbal["birthdate"] = df_imbal["accessdate"] - pd.to_timedelta(
    df_imbal["age"], unit="d"
)

# (Optional) Format both dates as yyyy-mm-dd string
df_imbal["accessdate"] = df_imbal["accessdate"].dt.strftime("%Y-%m-%d")
df_imbal["birthdate"] = df_imbal["birthdate"].dt.strftime("%Y-%m-%d")

original_cols = list(df_imbal.columns)
original_cols = [
    col
    for col in original_cols
    if col not in ["cardio", "age", "accessdate", "birthdate"]
]

# Step 5: Add synthetic columns to reach 100 total
current_cols = df_imbal.shape[1]  # already includes target and bmi
cols_to_add = 100 - current_cols

for i in range(cols_to_add):
    if i % 4 == 0:
        # Randomly decide how many normal distributions (1 to 3)
        n_distributions = np.random.randint(1, 4)
        samples_per_distribution = len(df_imbal) // n_distributions
        values = []

        for _ in range(n_distributions):
            loc = np.random.uniform(0, 100)
            scale = np.random.uniform(1, 2)
            vals = np.random.normal(loc=loc, scale=scale, size=samples_per_distribution)
            values.append(vals)

        # Concatenate and shuffle values to make sure they are mixed
        values = np.concatenate(values)

        # If there is a small mismatch in total length, pad with extra samples
        if len(values) < len(df_imbal):
            extra = np.random.normal(
                loc=np.random.uniform(0, 100),
                scale=np.random.uniform(1, 2),
                size=(len(df_imbal) - len(values)),
            )
            values = np.concatenate([values, extra])

        np.random.shuffle(values)
        df_imbal[f"cont_feat_{i}"] = values

    elif i % 4 == 1:
        n_classes = np.random.randint(4, 11)
        classes = [chr(65 + j) for j in range(n_classes)]
        df_imbal[f"cat_feat_{i}"] = np.random.choice(classes, size=len(df_imbal))

    elif i % 4 == 2:
        n_classes = np.random.randint(7, 13)
        df_imbal[f"ordinal_feat_{i}"] = np.random.randint(
            0, n_classes, size=len(df_imbal)
        )

    else:
        df_imbal[f"bin_feat_{i}"] = np.random.choice([0, 1], size=len(df_imbal))


# Step 6: Add missing values differently for original and synthetic columns
def add_missing_differential(df, original_cols, synthetic_cols):
    df_missing = df.copy()

    # Original columns: 8-15% missing
    for col in original_cols:
        missing_fraction = np.random.uniform(0.08, 0.15)
        mask = np.random.rand(len(df_missing)) < missing_fraction
        df_missing.loc[mask, col] = np.nan

    # Synthetic columns: 20-90% missing
    for col in synthetic_cols:
        missing_fraction = np.random.uniform(0.20, 0.90)
        mask = np.random.rand(len(df_missing)) < missing_fraction
        df_missing.loc[mask, col] = np.nan

    return df_missing


# Separate columns
synthetic_cols = [
    col
    for col in df_imbal.columns
    if col not in original_cols
    and col not in ["cardio", "age", "accessdate", "birthdate"]
]

# Apply missingness
df_final = add_missing_differential(df_imbal, original_cols, synthetic_cols)

# Step 7: Move 'cardio' column to the last
cardio = df_final["cardio"]
df_final = df_final.drop(columns=["cardio"])
df_final["cardio"] = cardio

# Done!
print(df_final.shape)
print(df_final["cardio"].value_counts(normalize=True))
print(df_final)

df_final.to_csv(
    "database/dataset/daniel/dummy.csv",
    sep="\t",
    encoding="utf-8",
    index=False,
)
