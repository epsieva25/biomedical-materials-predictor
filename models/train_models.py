import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("data/sustainable_materials.csv")
X = df.drop("bio_compatibility_score", axis=1)
y = df["bio_compatibility_score"]

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train SVR
svr = SVR(kernel='rbf')
svr.fit(X_train, y_train)
joblib.dump(svr, "models/svr_model.pkl")

# Train Decision Tree
dtr = DecisionTreeRegressor(max_depth=6)
dtr.fit(X_train, y_train)
joblib.dump(dtr, "models/dtr_model.pkl")

# Save scaler
joblib.dump(scaler, "models/scaler.pkl")

print("Models trained and saved.")
