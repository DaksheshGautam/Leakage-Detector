import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def run_leakguard(file_path, target_column):
    """
    Main function to run the LeakGuard tool.
    """
    report = {}
    
    # 1. Load Data
    df = pd.read_csv(file_path)
    report['dataset_shape'] = df.shape
    
    # 2. Separate Target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # 3. Run Detectors
    report['identifier_risk'] = detect_identifiers(X)
    report['duplicates'] = detect_duplicates(df)
    report['high_correlation'] = detect_high_correlation(X, y)
    report['high_importance'] = detect_feature_importance_leakage(X, y)
    report['temporal_leakage'] = detect_temporal_leakage(df)
    
    # 4. Generate Report
    print_report(report)

def detect_identifiers(df, threshold=0.95):
    """
    Detects columns that are likely identifiers based on uniqueness.
    """
    identifier_cols = []
    for col in df.columns:
        if df[col].nunique() / len(df) > threshold:
            identifier_cols.append(col)
    return identifier_cols

def detect_duplicates(df):
    """
    Detects duplicate rows in the dataframe.
    """
    return df.duplicated().sum()

def detect_high_correlation(X, y, threshold=0.8):
    """
    Detects features with high correlation to the target.
    """
    # Ensure y is numeric for correlation calculation
    y_numeric = y
    if y.dtype == 'object':
        le = LabelEncoder()
        y_numeric = le.fit_transform(y)

    # Combine features and target for correlation matrix
    df_corr = X.copy()
    # Preprocess data: handle categorical features and NaNs
    for col in df_corr.columns:
        if df_corr[col].dtype == 'object':
            df_corr[col] = df_corr[col].astype('category').cat.codes
        if df_corr[col].isnull().any():
            df_corr[col] = df_corr[col].fillna(df_corr[col].median())

    df_corr['target'] = y_numeric

    # Select only numeric columns for correlation calculation
    numeric_df_corr = df_corr.select_dtypes(include=np.number)

    # Calculate correlations
    correlations = numeric_df_corr.corr()['target'].abs().sort_values(ascending=False)

    # Filter high correlations (excluding target itself)
    high_corr_features = correlations[correlations > threshold]
    high_corr_features = high_corr_features.drop('target', errors='ignore')

    return high_corr_features.index.tolist()


def detect_feature_importance_leakage(X, y, threshold=0.30):
    """
    Detects leakage using feature importance from a RandomForest model.
    """
    # Preprocess data: handle categorical features
    X_processed = X.copy()
    for col in X_processed.columns:
        if X_processed[col].dtype == 'object':
            X_processed[col] = X_processed[col].astype('category').cat.codes
        # Fill NaNs for model training
        if X_processed[col].isnull().any():
            X_processed[col] = X_processed[col].fillna(X_processed[col].median())

    # Handle categorical target
    y_processed = y
    if y_processed.dtype == 'object':
        le = LabelEncoder()
        y_processed = le.fit_transform(y_processed)
            
    # Train a lightweight RandomForest model
    # Use a small portion of data to keep it fast
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y_processed, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Get feature importances
    importances = pd.Series(model.feature_importances_, index=X_processed.columns)
    
    # Flag features with unusually high importance
    high_importance_features = importances[importances > threshold].index.tolist()
    
    return high_importance_features

def detect_temporal_leakage(df):
    """
    Detects potential temporal leakage risks by checking if the data 
    is sorted by any datetime columns.
    """
    temporal_warnings = []
    
    # Identify potential datetime columns
    # 1. Existing datetime dtypes
    date_cols = df.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()
    
    # 2. Object columns that look like dates (heuristic based on name)
    candidate_cols = [c for c in df.select_dtypes(include=['object']).columns 
                      if any(x in c.lower() for x in ['date', 'time', 'year', 'month', 'day'])]
    
    for col in candidate_cols:
        try:
            # Check first few non-null values to see if they parse
            sample = df[col].dropna().head(10)
            if len(sample) > 0:
                pd.to_datetime(sample, errors='raise')
                date_cols.append(col)
        except:
            pass
            
    # Check monotonicity
    for col in set(date_cols):
        try:
            series = pd.to_datetime(df[col], errors='coerce').dropna()
            if len(series) < 2:
                continue
            if series.is_monotonic_increasing:
                temporal_warnings.append(f"Dataset is sorted by '{col}' (Ascending) - Random splits may leak future info")
            elif series.is_monotonic_decreasing:
                temporal_warnings.append(f"Dataset is sorted by '{col}' (Descending) - Random splits may leak future info")
        except:
            continue
            
    return temporal_warnings

def print_report(report):
    """
    Prints the final LeakGuard report.
    """
    print("========== LeakGuard Report ==========")
    print(f"Dataset shape: {report['dataset_shape']}")
    
    if report['identifier_risk']:
        print("Identifier Risk:")
        print(report['identifier_risk'])
        
    if report['duplicates'] > 0:
        print(f"Duplicates:{report['duplicates']}")
        
    if report['high_correlation']:
        print("High Correlation with Target:")
        print(report['high_correlation'])
        
    if report['high_importance']:
        print("High Importance Features (Potential Leakage):")
        print(report['high_importance'])
        
    if report.get('temporal_leakage'):
        print("Temporal Leakage Risks:")
        for warning in report['temporal_leakage']:
            print(f"- {warning}")
        
    print("========== End of Report ==========")
