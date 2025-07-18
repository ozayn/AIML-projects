
from sklearn.impute import SimpleImputer

def impute_data(df):
  cols_to_impute = df.columns[df.isnull().sum()>0].tolist()
  cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

  df_imputed = df.copy()
  for col in cols_to_impute:
    strategy = 'most_frequent' if col in cat_cols else 'mean'
    imputer = SimpleImputer(strategy=strategy)
    df_imputed[f'{col} (missing)'] = df_imputed[col]
    df_imputed[[col]] = imputer.fit_transform(df_imputed[[f'{col} (missing)']])
  return df_imputed
