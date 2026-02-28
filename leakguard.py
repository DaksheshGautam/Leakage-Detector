__version__ = "0.6.0"

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from dataclasses import dataclass
from typing import List, Any, Optional

@dataclass
class Finding:
    title: str
    severity: str
    description: str
    evidence: Any
    recommendation: List[str]

def run_leakguard(file_path, target_column):        #Main function to run the LeakGuard tool.
    
    findings = []
    
    # 1. Load Data
    df = pd.read_csv(file_path, parse_dates=['timestamp'])
    
    # 2. Separate Target
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Defensive check: Ensure target is not in X (e.g. due to duplicate columns)
    while target_column in X.columns:
        X = X.drop(columns=[target_column])
    
    # 3. Run Detectors
    if f := detect_identifiers(X): findings.append(f)
    if f := detect_duplicates(df): findings.append(f)
    if f := detect_group_leakage(df, target_column): findings.append(f)
    if f := detect_high_correlation(X, y, target_column): findings.append(f)
    if f := detect_feature_importance_leakage(X, y, target_column): findings.append(f)
    if f := detect_temporal_leakage(df, target_column): findings.append(f)
    
    # 4. Generate Report
    print_report(findings, df.shape)

    # 5. Get overall advice
    advice = advisory_logic(findings)
    print_advice(advice)

def advisory_logic(findings):
    
    severity_weights = {
        "LOW": 1,
        "MEDIUM": 3,
        "HIGH": 5
    }

    total_score = 0
    for finding in findings:
        total_score += severity_weights.get(finding.severity, 0)

    advice = {
        "splitting_strategy": "Standard (e.g., StratifiedKFold)",
        "dataset_tips": [],
        "leakage_score": total_score
    }

    has_temporal = any(f.title == "Temporal leakage risk" for f in findings)
    has_group = any(f.title == "Group leakage risk detected" for f in findings)

    if has_temporal:
        advice["splitting_strategy"] = "TimeSeriesSplit"
    elif has_group:
        advice["splitting_strategy"] = "GroupKFold"
    
    if total_score > 10:
        advice["dataset_tips"].append("High risk of data leakage. Manual inspection of features is highly recommended.")
    elif total_score >= 5:
        advice["dataset_tips"].append("Moderate risk of data leakage. Review the findings and apply recommended actions.")
    else:
        advice["dataset_tips"].append("Low risk of data leakage, but it's good practice to review the findings.")

    if not findings:
        advice["dataset_tips"] = ["No leakage risks detected. Dataset looks safe for standard modeling."]

    return advice


def detect_identifiers(df, threshold=None):         #Detects columns that are likely identifiers based on uniqueness.
    
    if len(df) == 0:
        return None

    if threshold is None:
        # Adaptive threshold: 1 - (1 / sqrt(n))
        # As n increases, threshold approaches 1.0 (stricter).
        threshold = 1 - (1 / np.sqrt(len(df)))

    identifier_cols = []
    for col in df.columns:
        if df[col].nunique() / len(df) > threshold:
            # New check: Exclude continuous float columns from being flagged as identifiers.
            # True identifiers are typically integers or strings.
            if pd.api.types.is_float_dtype(df[col]):
                continue

            identifier_cols.append(col)
            
    if identifier_cols:
        return Finding(
            title="Identifier columns detected",
            severity="MEDIUM",
            description="Columns with near-unique values allow row memorization.",
            evidence=identifier_cols,
            recommendation=[
                "Drop identifier columns before training",
                "Keep only if needed for joins"
            ]
        )
    return None

def detect_duplicates(df):                  #Detects duplicate rows in the dataframe.
    count = df.duplicated().sum()
    if count > 0:
        return Finding(
            title="Duplicate rows detected",
            severity="LOW",
            description="Duplicate rows can artificially inflate model performance if they appear in both train and test sets.",
            evidence=f"{count} rows",
            recommendation=[
                "Remove duplicate rows",
                "Check data collection pipeline"
            ]
        )
    return None

def detect_group_leakage(df, target_column, uniqueness_min=0.01, uniqueness_max=0.8, group_purity_threshold=0.9, feature_constancy_threshold=0.9):
    """
    Detects group leakage where a single entity (e.g., user, patient) appears multiple times,
    This can cause two types of leakage:
    1. Target Leakage: The target is highly consistent for an entity.
    2. Entity Feature Leakage: Other features are constant for an entity, acting as a proxy for its ID.
    """
    leakage_evidence = {}  # Dict to store evidence: {group_col: [reason1, reason2]}
    feature_cols = [col for col in df.columns if col != target_column]

    for col in feature_cols:
        # Exclude float columns (likely continuous features, not identifiers)
        if pd.api.types.is_float_dtype(df[col]):            # Exclude float columns since they typically represent continuous measurements
            continue                                        # rather than entity identifiers

        # --- Step 1: Identify candidate grouping columns based on cardinality ---
        n_unique = df[col].nunique()
        n_total = len(df)

        if n_unique <= 1 or n_total == 0:
            continue

        uniqueness_ratio = n_unique / n_total

        # A grouping column should not be a constant, but also not a unique identifier
        if not (uniqueness_min < uniqueness_ratio < uniqueness_max):
            continue

        # 'col' is a candidate grouping column.

        # --- Step 2: Prepare data for group analysis ---
        # We only care about groups that appear more than once.
        group_sizes = df.groupby(col).size()
        multi_member_group_ids = group_sizes[group_sizes > 1].index

        if len(multi_member_group_ids) == 0:
            continue

        # Filter the dataframe to only include rows from multi-member groups
        multi_member_df = df[df[col].isin(multi_member_group_ids)].copy()

        # --- Signal A: Check for Target Consistency within groups ---
        is_categorical_target = df[target_column].dtype == 'object' or df[target_column].nunique() < 20

        if is_categorical_target:
            def get_purity(group):
                if group.empty or group.value_counts().empty: return 0.0
                return group.value_counts().iloc[0] / len(group)

            group_purity = multi_member_df.groupby(col)[target_column].apply(get_purity)

            if not group_purity.empty and (avg_purity := group_purity.mean()) > group_purity_threshold:
                leakage_evidence.setdefault(col, []).append(f"High target consistency (avg. purity: {avg_purity:.2f})")
        else: # Numerical Target
            intra_group_std = multi_member_df.groupby(col)[target_column].std().fillna(0)
            avg_intra_group_std = intra_group_std.mean()
            overall_std = multi_member_df[target_column].std()

            if overall_std > 1e-6 and (avg_intra_group_std / overall_std) < 0.1:
                leakage_evidence.setdefault(col, []).append("Low target variance within groups")

        # --- Signal B: Check for Entity Feature Constancy ---
        constant_features = []
        other_features = [c for c in feature_cols if c != col]

        if other_features:
            grouped_by_col = multi_member_df.groupby(col)

            for feature in other_features:
                feature_uniqueness_per_group = grouped_by_col[feature].nunique()
                num_constant_groups = (feature_uniqueness_per_group == 1).sum()
                proportion_constant = num_constant_groups / len(multi_member_group_ids)

                if proportion_constant > feature_constancy_threshold:
                    constant_features.append(feature)

        if constant_features:
            leakage_evidence.setdefault(col, []).append(f"Constant entity features: {', '.join(constant_features)}")

    if not leakage_evidence:
        return None

    # --- Step 3: Format findings ---
    final_evidence = []
    for group_col, reasons in leakage_evidence.items():
        final_evidence.append(f"Grouping column '{group_col}': {'; '.join(reasons)}")

    if final_evidence:
        return Finding(
            title="Group leakage risk detected",
            severity="HIGH",
            description="Columns were found that group data points (e.g., user_id, session_id). This can cause leakage if groups are split across train/test sets.",
            evidence=final_evidence,
            recommendation=[
                "Use GroupKFold or a similar group-aware splitting strategy, using the identified grouping column(s).",
                "Ensure that all data for a given group ID is in the same split (train or test)."
            ]
        )
    return None

def detect_high_correlation(X, y, target_name=None, threshold=None):           #Detects features with high correlation to the target.

    y_numeric = y                               # Ensure y is numeric for correlation calculation
    if y.dtype == 'object':
        le = LabelEncoder()
        y_numeric = le.fit_transform(y)

    
    df_corr = X.copy()                          # Combine features and target for correlation matrix  

    # Defensive: Ensure target column is not in feature matrix
    cols_to_drop = set()
    if target_name: cols_to_drop.add(target_name)
    if hasattr(y, 'name') and y.name: cols_to_drop.add(y.name)
    
    df_corr = df_corr.drop(columns=[c for c in cols_to_drop if c in df_corr.columns], errors='ignore')

    for col in df_corr.columns:                 # Preprocess data: handle categorical features and NaNs
        if df_corr[col].dtype == 'object':
            df_corr[col] = df_corr[col].astype('category').cat.codes
        if df_corr[col].isnull().any():
            df_corr[col] = df_corr[col].fillna(df_corr[col].median())

    df_corr['target'] = y_numeric

    
    numeric_df_corr = df_corr.select_dtypes(include=np.number)          # Select only numeric columns for correlation calculation

    
    correlations = numeric_df_corr.corr()['target'].abs().sort_values(ascending=False)      # Calculate correlations

    if threshold is None:
        # Adaptive threshold: Mean + 3*Std
        # We enforce a minimum floor (0.75) to avoid flagging noise in low-correlation datasets.
        corr_mean = correlations.mean()
        corr_std = correlations.std() if len(correlations) > 1 else 0
        
        threshold = max(corr_mean + 3 * corr_std, 0.75)
    
    high_corr_features = correlations[correlations > threshold]          # Filter high correlations
    
    evidence = high_corr_features.index.tolist()
    
    # Final safety filter: Remove target-related names from evidence list
    evidence = [f for f in evidence if f != 'target' and f not in cols_to_drop]

    if evidence:
        return Finding(
            title="High correlation with target",
            severity="HIGH",
            description="Features with extremely high correlation to the target may be proxies for the target itself (leakage).",
            evidence=evidence,
            recommendation=[
                "Inspect these features manually",
                "Remove if they are post-outcome variables"
            ]
        )
    return None


def detect_feature_importance_leakage(X, y, target_name=None, threshold=None):            #Detects leakage using feature importance from a RandomForest model.
    
    
    X_processed = X.copy()                      # Preprocess data: handle categorical features and NaNs

    # Defensive: Ensure target column is not in feature matrix
    cols_to_drop = set()
    if target_name: cols_to_drop.add(target_name)
    if hasattr(y, 'name') and y.name: cols_to_drop.add(y.name)

    X_processed = X_processed.drop(columns=[c for c in cols_to_drop if c in X_processed.columns], errors='ignore')

    for col in X_processed.columns:
        if pd.api.types.is_datetime64_any_dtype(X_processed[col]):
            X_processed[col] = X_processed[col].astype(np.int64) // 10**9
        elif X_processed[col].dtype == 'object':
            X_processed[col] = X_processed[col].astype('category').cat.codes
        
        if X_processed[col].isnull().any():                 # Handle missing values
            X_processed[col] = X_processed[col].fillna(X_processed[col].median())

    
    y_processed = y                             
    if y_processed.dtype == 'object':
        le = LabelEncoder()                             # Handle categorical target
        y_processed = le.fit_transform(y_processed)
            
    # Train a lightweight RandomForest model
    # Use a small portion of data to keep it fast
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y_processed, test_size=0.2, random_state=42)
    
    # Determine if it's a classification or regression problem to choose the right model
    is_classification = y.dtype == 'object' or y.nunique() < 20

    if is_classification:
        model = RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=-1)
    else:
        model = RandomForestRegressor(n_estimators=20, random_state=42, n_jobs=-1)

    model.fit(X_train, y_train)
    
    
    importances = pd.Series(model.feature_importances_, index=X_processed.columns)          # Get feature importances
    
    if threshold is None:
        # Adaptive logic: > 2x Median OR > 40% total importance
        # Added safety: > 2x Median is only valid if importance is also > 0.10 (to ignore flat distributions)
        median_imp = importances.median()
        mask = ((importances > 2 * median_imp) & (importances > 0.10)) | (importances > 0.40)
        high_importance_features = importances[mask].index.tolist()
    else:
        high_importance_features = importances[importances > threshold].index.tolist()          # Filter high importance features
    
    # Final safety filter: Remove target-related names from evidence list
    high_importance_features = [f for f in high_importance_features if f not in cols_to_drop]

    if high_importance_features:
        return Finding(
            title="High feature importance detected",
            severity="MEDIUM",
            description="A simple model found these features to be overwhelmingly predictive, suggesting potential leakage.",
            evidence=high_importance_features,
            recommendation=[
                "Verify if these features are available at prediction time",
                "Check for target leakage"
            ]
        )
    return None

def detect_temporal_leakage(df, target_col, threshold=0.4):
    
    #Detects potential temporal leakage risks by checking multiple signals:
    #1. Target Autocorrelation (when sorted by date)
    #2. Regular Time Spacing (e.g., hourly, daily)
    #3. Timestamp Uniqueness

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
            sample = df[col].dropna().head(100)
            if len(sample) > 0:
                # Use coerce to handle potential mixed formats or noise
                parsed = pd.to_datetime(sample, errors='coerce')
                # If more than 50% parse successfully, treat as date
                if parsed.notna().sum() > len(sample) * 0.5:
                    date_cols.append(col)
        except:
            pass
            
    # Check Autocorrelation
    for col in set(date_cols):
        # Create a temporary dataframe to sort without affecting original
        temp_df = df[[col, target_col]].copy()
        
        # Drop NaNs
        temp_df = temp_df.dropna()

        # Parse date column to ensure it's datetime
        try:
            temp_df[col] = pd.to_datetime(temp_df[col], errors='coerce')
            temp_df = temp_df.dropna(subset=[col])
        except:
            continue

        if len(temp_df) < 10: 
            continue

        # Ensure target is numeric for correlation
        if temp_df[target_col].dtype == 'object':
            try:
                temp_df[target_col] = pd.to_numeric(temp_df[target_col])
            except:
                le = LabelEncoder()
                temp_df[target_col] = le.fit_transform(temp_df[target_col].astype(str))
        
        # Sort by date
        temp_df = temp_df.sort_values(by=col)
        
        # --- Signal 1: Autocorrelation ---
        autocorr = temp_df[target_col].autocorr(lag=1)
        if pd.isna(autocorr):
            autocorr = 0.0
            
        # --- Signal 2: Regular Spacing ---
        is_regular = False
        time_diffs = temp_df[col].diff().dropna()
        if len(time_diffs) > 0:
            # Check if the most frequent time delta constitutes > 80% of the data
            mode_freq = time_diffs.value_counts(normalize=True).iloc[0]
            if mode_freq > 0.8:
                is_regular = True

        # --- Signal 3: Uniqueness ---
        n_unique = temp_df[col].nunique()
        uniqueness_ratio = n_unique / len(temp_df)
        is_unique = uniqueness_ratio > 0.95

        # --- Decision Logic ---
        detected_signals = []
        
        if abs(autocorr) > threshold:
            detected_signals.append(f"High Target Autocorrelation ({autocorr:.2f})")
        elif abs(autocorr) > 0.1:
            detected_signals.append(f"Moderate Target Autocorrelation ({autocorr:.2f})")
            
        if is_regular:
            detected_signals.append("Regular Time Spacing")
            
        if is_unique:
            detected_signals.append("High Timestamp Uniqueness")
            
        # Flag if we have strong autocorrelation OR multiple temporal signals
        if (abs(autocorr) > threshold) or (len(detected_signals) >= 2):
            temporal_warnings.append(
                f"Temporal Leakage Risk in '{col}': {', '.join(detected_signals)}. "
                "Data appears to be a time-series; use TimeSeriesSplit."
            )
            
    if temporal_warnings:
        return Finding(
            title="Temporal leakage risk",
            severity="HIGH",
            description="Data exhibits strong time-dependence. Random splits will cause future-to-past leakage.",
            evidence=temporal_warnings,
            recommendation=[
                "Use TimeSeriesSplit for validation",
                "Do not use random K-Fold or train_test_split"
            ]
        )
    return None


def print_advice(advice):
    print("\nAdvisory")
    print("="*50)
    print(f"Leakage Risk Score: {advice['leakage_score']}")
    print(f"Recommended Splitting Strategy: {advice['splitting_strategy']}")
    for tip in advice['dataset_tips']:
        print(f"• {tip}")
    print("="*50)


def print_report(findings, shape):

    print(f"\nLeakGuard Report (v{__version__})")
    print("="*50)
    print(f"Dataset shape: {shape}\n")
    
    if not findings:
        print("✅ No leakage risks detected. Dataset looks safe.")
        return

    severity_icon = {
        "HIGH": "HIGH",
        "MEDIUM": "MEDIUM",
        "LOW": "LOW"
    }

    for finding in findings:
        icon = severity_icon.get(finding.severity, "")

        print(f"{icon} {finding.title}")
        print(f"Severity: {finding.severity}")
        print(f"{finding.description}")

        if finding.evidence:
            print("Evidence:")
            if isinstance(finding.evidence, list):
                for e in finding.evidence:
                    print(f"  • {e}")
            else:
                print(f"  • {finding.evidence}")

        if finding.recommendation:
            print("Recommended actions:")
            for rec in finding.recommendation:
                print(f"  -> {rec}")

        print("-"*50)

    print("End of Report.\n")