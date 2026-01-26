
# Data Type Optimization Script
# Generated from 03_data_types.ipynb

def optimize_dtypes(df):
    """Apply optimized data types to dataframe"""

    # Date columns: ['issue_d', 'earliest_cr_line', 'last_pymnt_d', 'next_pymnt_d', 'last_credit_pull_d', 'hardship_start_date', 'hardship_end_date', 'payment_plan_start_date', 'debt_settlement_flag_date', 'settlement_date']
    date_cols = ['issue_d', 'earliest_cr_line', 'last_pymnt_d', 'next_pymnt_d', 'last_credit_pull_d', 'hardship_start_date', 'hardship_end_date', 'payment_plan_start_date', 'debt_settlement_flag_date', 'settlement_date']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Category columns (low cardinality)
    category_cols = ['term', 'grade', 'sub_grade', 'emp_length', 'home_ownership', 'verification_status', 'loan_status', 'pymnt_plan', 'desc', 'purpose', 'title', 'addr_state', 'initial_list_status', 'application_type', 'verification_status_joint', 'hardship_flag', 'hardship_type', 'hardship_reason', 'hardship_status', 'hardship_loan_status', 'disbursement_method', 'debt_settlement_flag', 'settlement_status']
    for col in category_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')

    # Numeric optimization
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')

    return df
