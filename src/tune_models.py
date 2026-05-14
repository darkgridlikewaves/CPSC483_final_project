import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from src.config import TRAIN_CSV

def tune_random_forest():
    print("Loading data for tuning...")
    df = pd.read_csv(TRAIN_CSV)
    
    features = ['GrLivArea', 'YearBuilt', 'OverallQual', 'TotalBsmtSF', 'GarageArea']
    
    # Simple fillna for any missing numerical values
    X = df[features].fillna(0) 
    
    # Using log transformation for the target variable (standard for Ames dataset)
    y = np.log1p(df['SalePrice']) 

    print("Setting up GridSearchCV...")
    rf = RandomForestRegressor(random_state=42)

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=2
    )

    print("Running simulations... (this might take 30-60 seconds)")
    grid_search.fit(X, y)

    print("\n" + "="*40)
    print("!!!TUNING COMPLETE!!!")
    print("="*40)
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best CV RMSE Score: {-grid_search.best_score_:.4f}")
    print("="*40)

if __name__ == "__main__":
    tune_random_forest()