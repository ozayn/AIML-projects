
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

def drop_outliers(data, col):
  has_outliers = False
  name = data[col].name

  # Calculate Q1 (25th percentile) and Q3 (75th percentile)
  Q1 = data[col].quantile(0.25)
  Q3 = data[col].quantile(0.75)
  IQR = Q3 - Q1

  # Define outlier thresholds
  lower_bound = Q1 - 1.5 * IQR
  upper_bound = Q3 + 1.5 * IQR

  # Identify outliers
  outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]

  if not outliers.empty:
    has_outliers = True

    print(f'{outliers.shape[0]} {col} outliers')

    # Calculate percentage of outliers
    percentage_outliers = (len(outliers) / len(data[col])) * 100
    print(f"Percentage of {name} outliers using IQR: {percentage_outliers:.2f}%")

  # out = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]

  return lower_bound, upper_bound, has_outliers
