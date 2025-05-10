# utils.py

from sklearn.preprocessing import LabelEncoder

def preparar_X_y(df):
    features = ['Price per Unit', 'Region', 'Product', 'Sales Method', 'State', 'City']
    target = 'Units Sold'

    df_proc = df.copy()
    
    # Codificar categóricas
    label_cols = ['Region', 'Product', 'Sales Method', 'State', 'City']
    le_dict = {}
    for col in label_cols:
        le = LabelEncoder()
        df_proc[col] = le.fit_transform(df_proc[col])
        le_dict[col] = le

    X = df_proc[features]
    y = df_proc[target]

    return X, y
