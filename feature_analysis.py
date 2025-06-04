from main import build_rankings
import pandas as pd
import numpy as np 
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

target = 'Pts_pg'
features = [
    'Poss_pg', 'GF_pg', 'GA_pg', 'SoG_pg', 'SoGA_pg', 'ShAtt_pg', 
    'ShAttA_pg', 'Corners_pg', 'CornersA_pg', 'Saves_pg',
    'SavesA_pg'
]

print("--- Loading and Preparing Data for Analysis ---")

rank_df, home_rank, away_rank, clean = build_rankings()

model_df = rank_df.copy()
model_df.replace([np.inf, -np.inf], np.nan, inplace=True)
print(f"Shape of data afer loading: {model_df.shape}")

columns_for_model = [target] + features
original_rows = model_df.shape[0]
model_df.dropna(subset=columns_for_model, inplace=True)
rows_after_dropna = model_df.shape[0]

print(f"Original number of teams loaded: {original_rows}")
print(f"Number of teams after removing rows with NaN in target/features: {rows_after_dropna}")
print(f"Number of teams dropped: {original_rows - rows_after_dropna}")

if rows_after_dropna < len(features) + 1:
    print(f"\nWarning: Not enought data ({rows_after_dropna} teams) to build a reliable linear regression model.")
    print(f"You have {len(features)} features, and ideally need more than {len(features)} with complete data.")
    if rows_after_dropna == 0:
        print("Error No data left after cleaning. Cannot proceed with linear regrfession.")

X = model_df[features]
y = model_df[target]

# --- Model Evaluation Setup ---
print("\n--- Model Evaluation ---")

# 1. Train-Test Split (as a first step)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training dta shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# Train the model on the training data
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rf_model.predict(X_test)

# Calculate and print evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\n--- Performance on Test Set ---")
print(f"MEan Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R2): {r2:.4f}")

print("\n--- K-Fold Cross Validation ---")
kf = KFold(n_splits=5, shuffle=True, random_state=42)

mae_scores = []
mse_scores = []
rmse_scores = []
r2_scores = []

for train_index, test_index in kf.split(X):
    X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
    y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

    rf_model_fold = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model_fold.fit(X_train_fold, y_train_fold)
    y_pred_fold = rf_model_fold.predict(X_test_fold)

    mae_scores.append(mean_absolute_error(y_test_fold, y_pred_fold))
    mse_scores.append(mean_squared_error(y_test_fold, y_pred_fold))
    rmse_scores.append(np.sqrt(mean_squared_error(y_test_fold, y_pred_fold)))
    r2_scores.append(r2_score(y_test_fold, y_pred_fold))

print(f"Average MAE (K-Fold): {np.mean(mae_scores):.4f} +/- {np.std(mae_scores):.4f}")
print(f"Average MSE (K-Fold): {np.mean(mse_scores):.4f} +/- {np.std(mse_scores):.4f}")
print(f"Average RMSE (K-Fold): {np.mean(rmse_scores):.4f} +/- {np.std(rmse_scores):.4f}")
print(f"Average R-squared (K-Fold): {np.mean(r2_scores):.4f} +/- {np.std(r2_scores):.4f}")


print("\n--- Feature Importance with RandomForestRegressor ---")

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)

importances = rf_model.feature_importances_
feature_importances = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

print(feature_importances)

feature_importances.to_csv('feature_importances.csv', index=False)


plt.figure(figsize=(10, 6))
plt.barh(feature_importances['Feature'], feature_importances['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance for Predicting Pts_pg')
plt.show()