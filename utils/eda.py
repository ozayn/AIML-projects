# Libraries to help with reading and manipulating data


image_dir = '../images'

import math

from scipy.stats import chisquare
from collections import Counter
from pprint import pprint

import statsmodels.api as sm # for qqplot
from scipy.stats import zscore # outliers
import scipy.stats as stats

import pandas as pd
import numpy as np



# libaries to help with data visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("talk")

# Library to split data
from sklearn.model_selection import train_test_split

# Libraries to build decision tree classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# To tune different models
from sklearn.model_selection import GridSearchCV

# To perform statistical analysis
import scipy.stats as stats

# To get diferent metric scores
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    make_scorer,
)

# Library to suppress warnings or deprecation notes
import warnings
warnings.filterwarnings("ignore")



def camel_to_spaces(s):
    return re.sub(r'([a-z])([A-Z])', r'\1 \2', s)



def make_data_dictionary(df, ddict_str=""):
  dtypes = df.copy().dtypes
  description_dict = {D[0]: ': '.join(D[1:]) for D in [d.split(": ") for d in ddict_str.split('\n')]}
  dtypedf = pd.DataFrame(dtypes, columns=['Data Type'])
  if ddict_str!='':
    dtypedf['Description'] = dtypedf.index.map(description_dict)
  else:
    # dtypedf['Description'] = dtypedf.index.to_series().apply(lambda x: x.replace('_', ' ').title())
    dtypedf['Description'] = dtypedf.index.to_series().apply(lambda x: camel_to_spaces(x))
  # dtypedf['Description'] = dtypedf['Description'].fillna(dtypedf.Series(dtypedf.index, index=dtypedf.index))
  dtypedf['# unique'] = df.nunique()
  # if df.isnull().sum().sum()!=0:
  #   dtypedf['# null'] = df.isnull().sum()
  dtypedf.index.name='Column'
  return dtypedf


def get_anomalous_values(df, positive_cols=[]):
  L = ['Anamolous Values']
  data_num = df.select_dtypes(include="number")
  count_neg = (data_num < 0).sum()
  neg_dict_count = count_neg[count_neg>0].to_dict()
  for f, count in neg_dict_count.items():
    L.append(f'There are {count:,} negative values in {f} column.')
  return L

def feature_engineering(df, feature_info={}):
  """
  feature_info = {'formula': {'ZIPCode': ('SCF', lambda x: x[0:2])},
                'dtype': {'SCF': 'category'},
                'description': {'SCF': '(sectional center facility) is the first two digits of the ZIPCode.'}
                }
  """
  if feature_info == {}:
    return df, 'No feature engineering is done.'
  formula = feature_info['formula']
  dtype = feature_info['dtype']
  description = feature_info['description']
  print(description)

  L = []
  for originalfield, v in formula.items():
    df[v[0]] = df[originalfield].apply(v[1])
    if v[0]!=originalfield:
      L.append(f'The {v[0]} column is created from the {originalfield} column.')
    else:
      L.append(f'The {originalfield} column is modified.')
  for f, t in dtype.items():
    df[f] = df[f].astype(t)
  for f, desc in description.items():
    L.append(f'{f}: {desc}')

  t = '\n'.join(L)
  return df, t


def data_preprocessing(df, positive_cols=[], drop_cols = [], type_conv={},
                        feature_info={}):
  df, L = correct_anomalous_values(df, positive_cols=positive_cols)
  df, t_type_conv = convert_dtypes(df, type_conv=type_conv)
  df, t_feature = feature_engineering(df, feature_info=feature_info)
  L.append(t_feature)
  if drop_cols:
    df = df.drop(drop_cols, axis = 1)
    L.append(f"Column(s) {', '.join(drop_cols)} is/are dropped.")


  L.append(t_type_conv)

  text = '\n\n'.join(L)
  return df, text


def correct_anomalous_values(df, positive_cols=[]):
  data_num = df.select_dtypes(include="number")
  count_neg = (data_num < 0).sum()
  L = []
  neg_dict_count = count_neg[count_neg>0].to_dict()
  for f, count in neg_dict_count.items():
    if f in positive_cols:
      df[f] = df[f].map(lambda x: abs(x))
      L.append(f'{count:,} values in {f} column are converted to positive values.')
  return df, L

def show_null_columns(df):
  if df.isnull().sum().sum()==0:
    return pd.DataFrame()
  # nc = df.isnull().sum().sort_values(ascending=False).to_frame('# Null')
  nc = df.isnull().sum()[df.isnull().sum()!=0].sort_values(ascending=False).to_frame('# Null')
  return nc

def get_memory_usage(df):
  return df.memory_usage(deep=False).sum()/1024

def get_summary_info(df):
  out = {}
  out['Memory Usage'] = f"{get_memory_usage(df): .1f} KB"
  out[''] = ''
  out['#'] = ''
  out['Rows'] = df.shape[0]
  out['Columns'] = df.shape[1]
  out['Null Values'] = df.isnull().sum().sum()
  out['Duplicated Rows'] = df.duplicated().sum()
  out[' '] = ' '
  p1 = pd.DataFrame(out, index=['']).T
  p2 = df.dtypes.astype(str).value_counts().to_frame('')
  p = pd.concat([p1, p2])
  return p


def get_describe_tables(df):
  data_obj_cat = df.select_dtypes(include=['object', 'category'])
  data_num = df.select_dtypes(include='number')

  descr_o_cat = pd.DataFrame()
  descr_n = pd.DataFrame()

  number_df = df.select_dtypes(include=['number'])
  if data_obj_cat.shape[1]> 0:
    descr_o_cat = df.describe(include=['object', 'category']).T
    descr_o_cat.drop(['count'], axis=1, inplace=True)
    descr_o_cat.index.name = 'Object/Categorical Column'

  # if data_num.shape[1]> 0:
  #   descr_n = df.describe(include='number').T
  #   descr_n['count']= descr_n['count'].astype('int')
  #   descr_n['IQR'] = descr_n['75%'] - descr_n['25%']
  #   descr_n['Lower Bound'] = descr_n['25%'] - 1.5 * descr_n['IQR']
  #   descr_n['Upper Bound'] = descr_n['75%'] + 1.5 * descr_n['IQR']
  #   descr_n['Outlier count (Upper)'] = (data_num >= descr_n['Upper Bound']).sum()
  #   descr_n['Outlier count (Lower)'] = (data_num <= descr_n['Lower Bound']).sum()
  #   descr_n['Outlier count'] = descr_n['Outlier count (Upper)'] + descr_n['Outlier count (Lower)']
  #   descr_n.drop(['count'], axis=1, inplace=True)
  #   descr_n.index.name = 'Numerical Column'

  if data_num.shape[1]> 0:
    descr_n = df.describe(include='number').T
    descr_n['mean'] = descr_n['mean'].round(1)
    descr_n['std'] = descr_n['std'].round(1)
    descr_n['count']= descr_n['count'].astype('int')
    descr_n['IQR'] = descr_n['75%'] - descr_n['25%']
    descr_n['Lower Bound'] = descr_n['25%'] - 1.5 * descr_n['IQR']
    descr_n['Upper Bound'] = descr_n['75%'] + 1.5 * descr_n['IQR']
    descr_n['# Outliers (Upper)'] = (data_num >= descr_n['Upper Bound']).sum()
    descr_n['# Outliers (Lower)'] = (data_num <= descr_n['Lower Bound']).sum()
    descr_n['# Outliers'] = descr_n['# Outliers (Upper)'] + descr_n['# Outliers (Lower)']
    descr_n['Outliers %'] = (100 * descr_n['# Outliers']/data_num.shape[0]).round(1)
    descr_n.drop(['count', 'Upper Bound', 'Lower Bound'], axis=1, inplace=True)
    descr_n.index.name = 'Numerical Column'
  return [descr_o_cat, descr_n]


def round_num(n):
  if n > 1e6:
    return f' (~{n/ 1e6:.0f}M)'
  elif n>1e3:
    return f' (~{n/ 1e3:.0f}K)'
  else:
    return ''


def null_text_info(df):
  L = []
  n = df.isnull().sum().sum()
  if n > 0:
    null_pct = (df.isnull().sum()/df.shape[0]*100).sort_values(ascending=False)
    return [f"There are _{n:,} null values_ in the dataset. {null_pct.iloc[0]}% of the missing data is in {null_pct.index[0]}."]
  return ['There are __no missing__ values in the data.']


def convert_dtypes(df, type_conv=None):
  t = ''
  if type_conv!= {}:
    if type(list(type_conv.values())[0])==list:
      dtypes = {col: k for k, v in type_conv.items() for col in v}
      df = df.astype(dtypes)
      t = '\n'.join([f"{', '.join(v)} are/is converted to {k} type." for k, v in type_conv.items()])
    else:
      df = df.astype(type_conv)
      for k, v in type_conv.items():
        t += f"'{k}' is converted to {v} type.\n"
  return df, t


def duplicated_text_info(df):
  n = df.duplicated().sum()
  if n > 0:
    return [f'There are _{n:,} duplicated rows_ in the dataset.']
  return ['There are __no duplicated__ rows in the data.']

def get_text_info(df):
  L = []
  L.append(f'There are {df.shape[0]:,}{round_num(df.shape[0])} _rows_ and {df.shape[1]} _columns_ in the dataset.')
  L.append(f"The __memory usage__ is approximately {get_memory_usage(df): .1f} KB.")
  L.extend(null_text_info(df))
  L.extend(duplicated_text_info(df))
  L.extend(get_anomalous_values(df))
  return '\n\n'.join(L)


def recommendations(data, ddict_str):
  recommend_category = data.columns[data.select_dtypes(exclude=['category']).nunique()<=10].tolist()

  print('Recommendations:\n'+'-'*120)
  print('Convert to category: ')
  print('type_conv =')
  pprint({r: 'category' for r in recommend_category})

  cat_orders = {}
  convert_to_int = lambda x: int(x) if type(x) == str and x.isdigit() else x
  import re
  pattern = r'\b\d+-[A-Za-z]+\b'
  ddict = {k: v for k, v in [a.split(": ") for a in ddict_str.split('\n')]}
  for k, v in ddict.items():
    if v.count('-') > 1:
      matches = re.findall(pattern, v)
      values = [convert_to_int(m.split('-')[0].strip()) for m in matches]
      cat_orders[k] = values
  print('cat_orders = ')
  pprint(cat_orders)

  single_value_columns = data.nunique()[data.nunique()==1].index.tolist()
  possible_id_columns = data.nunique()[data.nunique()==data.shape[0]].index.tolist()
  print('drop_cols =', single_value_columns,'+', possible_id_columns)
  print('-'*120)


def modify_data(data, modify_dict ={}):
  if modify_dict!={}:

    for field, conv in modify_dict.items():
      data[field].replace(conv[0], conv[1], inplace=True)

  return data



def create_info(data, positive_cols=[], drop_cols=[], type_conv={},
                  feature_info={}, modify_dict={}, ddict_str="", cat_orders={}):
  mkdown_name = "info_dataframes.md"
  recommendations(data, ddict_str)
  modify_data(data, modify_dict = modify_dict)
  text = get_text_info(data)
  mkdf = []
  mkdf.append(make_data_dictionary(data, ddict_str))
  mkdf.append(get_summary_info(data))
  mkdf.append(show_null_columns(data))
  data, t_preprocessing = data_preprocessing(data, positive_cols=positive_cols,
                                            type_conv=type_conv,
                                             feature_info=feature_info,
                                             drop_cols = drop_cols)
  for k, v in cat_orders.items():
    data[k] = pd.Categorical(data[k], categories=v, ordered=True)
  mkdf.append(make_data_dictionary(data, ddict_str))
  mkdf.append(get_summary_info(data))
  mkdf.append(show_null_columns(data))
  text += '\n\n' + t_preprocessing
  print(text)
  mkdf.extend(get_describe_tables(data))
  with open(mkdown_name, "w") as f:
    f.write(text)
    f.write("\n\n")
    for i, d in enumerate(mkdf, start=1):
      display(d)
      f.write(d.to_markdown())
      f.write("\n\n")
  print(f'Information is written to {mkdown_name}')
  return data



def is_categorical_data_uniform(data, name = None, alpha = .05):
    """
    Analyzes the distribution of categorical data to see if it resembles
    a uniform distribution.
    """

    if name is None:
        name = data.name

    # Frequency distribution
    counts = data.value_counts().to_frame()
    counts.columns=['observed']
    counts.index.name = 'categories'
    counts['expected'] = counts['observed'].mean()

    # Perform Chi-Square Goodness-of-Fit Test
    chi2_stat, p_value = chisquare(counts['observed'], counts['expected'])


    H0 = f"'{name}' APPEARS to follow a balanced/uniform-like distribution (p: {p_value:.3f})."
    Ha = f"'{name}' does NOT follow a balanced/uniform-like distribution (p: {p_value:.3f})."

    is_balanced = True
    H = H0

    if p_value < alpha:
        is_balanced = False
        H = Ha

    return is_balanced, H


def categorical_countplot(data, figsize = None, palette = 'Paired',
                          perc = True, rotation=0, nmax = None, to_sort = False,
                          flip_axes = False):

  thresh = 9
  get_top = False

  name = data.name
  if nmax == None:
    nmax = data.nunique()

  if nmax > thresh:
    flip_axes = True


  if nmax < data.nunique():
    get_top = True

  if not flip_axes:
    if type(data.unique()[0]) == str:
      if rotation == 0:
        rotation = 60

  total = len(data)

  if figsize == None:
    figsize = (min(nmax+1, 10), 3)
    if flip_axes:
      figize = figsize = (4, min(nmax-1, 10))

  plt.figure(figsize=figsize)

  name = modify_varname(name)


  order = None
  if to_sort or get_top:
    order = data.value_counts().index[:nmax]
    if get_top:
      data = data[data.isin(order)]


  if flip_axes:
    ax = sns.countplot(
        y=data,
        hue=data,
        palette=palette,
        order=order,
        legend=False
    )
  else:
    ax = sns.countplot(
      x=data,
      hue=data,
      palette=palette,
      order=order,
      legend=False
    )

  if flip_axes:

    for p in ax.patches:
        if perc == True:
            label = "{:.1f}%".format(100 * p.get_width() / total)
        else:
            label = p.get_width()  # count of each level of the category

        x = p.get_x() + p.get_width()
        y = p.get_y() + p.get_height()/2  # height of the plot

        xytext=(30, 0)

        # annotate the percentage
        ax.annotate(label,(x, y),
                    ha="center",
                    va="center",
                    xytext=xytext,
                    textcoords="offset points")
    plt.ylabel(name)
  else:
    for p in ax.patches:
        if perc == True:
            label = "{:.1f}%".format(100 * p.get_height() / total)
        else:
            label = p.get_height()  # count of each level of the category

        x = p.get_x() + p.get_width() / 2  # width of the plot
        y = p.get_height()  # height of the plot

        xytext=(0, 11)

        # annotate the percentage
        ax.annotate(label,(x, y),
                    ha="center",
                    va="center",
                    xytext=xytext,
                    textcoords="offset points")
    plt.xlabel(name)
    plt.xticks(rotation=rotation)


  title = f"Count Plot of '{name}'"
  if get_top:
    title += f' (Top {nmax})'
  plt.title(title, pad = 25)
  sns.despine(ax=ax)
  plt.savefig(f"{image_dir}/countplot_{name}.png", bbox_inches='tight')

def univariate_categorical_info(data):
  print(f'\n{data.name}')
  n = data.nunique()
  values = data.unique().tolist()
  proportions = 100 * data.value_counts(normalize=True)
  majority_index = proportions.index[0]
  majority_proportion = proportions.iloc[0]
  if n == 2:
    if majority_index in [1, 'Yes', 'yes', True]:
      return f'The majority are {data.name} ({majority_proportion:.1f}%).'
    elif majority_index in [0, 'No', 'no', False]:
      return f'The majority are not {data.name} ({majority_proportion:.1f}%).'
  if n > 3:
    return f'The most common value in {data.name} is {majority_index} ({majority_proportion:.1f}%) followed by {proportions.index[1]} ({proportions.iloc[1]:.1f}%).'


  return f'The most common value in {data.name} is {majority_index} ({majority_proportion:.1f}%).'


def univariate_categorical(data, name = None, figsize = None, palette = 'Paired',
                          perc = True, rotation=0, nmax = None):
    """
    Analyzes the distribution of categorical data to see if it resembles
    a balanced or uniform-like distribution.
    """

    if name is None:
        name = data.name

    # print(f"\nAnalyzing categorical data: {name}")

    is_balanced, H = is_categorical_data_uniform(data)
    if is_balanced:
      print(H)

    categorical_countplot(data, figsize = figsize, palette = palette,
                          perc = perc, rotation=rotation, nmax = nmax)

    print(univariate_categorical_info(data))


def modify_varname(name):
  # name = name.replace('_', ' ')
  name = camel_to_spaces(name)
  conv = {' Amt': ' Amount',
          ' Chng': ' Change',
          ' Bal': ' Balance',
          ' mon': ' Months',
          ' Trans': ' Transaction',
          'Q4 Q1': 'Q4-Q1',
          ' To ': ' to ',
          ' Ct': ' Count'}
  conv.update({'Cr ': 'Credit ',
          'Of ': 'of '})
  for k, v in conv.items():
    if k in name:
      name = name.replace(k, v)
  return name

def histplot_boxplot(data, figsize = (9, 5), height_ratios = [3, 1],
             round_by=2,
             kde = True,
             mean_color='blue',
             median_color='gray',
             color='skyblue',
             alpha_hist = .3,
             alpha_box = .3,
             showmeans=True):
  name = data.name

  name = modify_varname(name)

  # Create subplots with shared x-axis
  fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True,
                           figsize=figsize,
                           gridspec_kw={'height_ratios': height_ratios })

  # Plot the histogram
  sns.histplot(data, kde=kde, ax=axes[0], color=color,
               alpha=alpha_hist,
               stat = 'density',
               edgecolor='black', linewidth=.5)
  axes[0].set_title(f"{name} - Histogram & Boxplot")
  axes[0].set_ylabel("Frequency")

  # Plot the boxplot
  sns.boxplot(x=data, ax=axes[1], color=color, width=0.5,
              showmeans = showmeans,
              boxprops=dict(alpha=alpha_box),
              meanprops={"marker": "^", "markerfacecolor": mean_color, "markeredgecolor": "black"},
              flierprops={"marker": "x", "markersize": 3, "markerfacecolor": "gray"})
  axes[1].set_xlabel(name)

  sns.despine(ax=axes[0])
  sns.despine(ax=axes[1])


  if showmeans:
    mean = np.mean(data)
    median = np.median(data)

    # Plot mean and median lines
    axes[0].axvline(x = mean, color=mean_color, linestyle='--', linewidth=.5, label=f'Mean: {round(mean, round_by):,}')
    axes[0].axvline(x = median, color=median_color, linestyle='-', linewidth=.5, label=f'Median: {round(median, round_by):,}')
    axes[0].legend(frameon=False, loc='upper right', bbox_to_anchor=(1.25, 1))


  # Adjust layout and show
  plt.tight_layout()
  plt.savefig(f"{image_dir}/histplot_boxplot_{name}.png", bbox_inches='tight')


def qqplot(data, color='skyblue', figsize=(4,3.50)):

  name = data.name

  name = modify_varname(name)

  data_cleaned = data[~np.isnan(data)]

  plt.figure(figsize=figsize)
  # Create Q-Q plot
  fig = sm.qqplot(data_cleaned, line='s', ax=plt.gca());

  # Customize the color of the points
  plt.gca().get_lines()[1].set_color(color)  # Set points to purple


  plt.title(f"Q-Q Plot of {name}")
  sns.despine(top=True, right=True)
  plt.tight_layout();
  plt.savefig(f"{image_dir}/qqplot_{name}.png", bbox_inches='tight')

def is_numerical_data_normal(data):
  name = data.name

  from scipy.stats import skew, kurtosis

  data_skewness = skew(data)
  data_kurtosis = kurtosis(data, fisher=False)  # Fisher=False returns kurtosis with normal = 3

  # Calculate skewness and specify its level and direction
  if data_skewness > 0:
      if data_skewness < 0.5:
          skew_description = "slightly positively skewed (right-skewed)"
      else:
          skew_description = "highly positively skewed (right-skewed)"
  elif data_skewness < 0:
      if data_skewness > -0.5:
          skew_description = "slightly negatively skewed (left-skewed)"
      else:
          skew_description = "highly negatively skewed (left-skewed)"
  else:
      skew_description = "approximately symmetric"

  # print(f"{name} - Skewness: {data_skewness:.4f} ({skew_description})")
  # print(f"{name} - Kurtosis: {data_kurtosis:.4f}")

  t = f'{name} is {skew_description} (skewness: {data_skewness: .2f}) with average of {data.mean():.2f}.'
  print(t)

  if abs(data_skewness) < 0.5 and abs(data_kurtosis - 3) < 0.5:
    print(f"The data {name} is LIKELY CLOSE to normal distribution.")
    return True
  else:
    print(f"The data {name} may NOT be normally distributed.")
    return False

def get_outlier_percentage(data):
  name = data.name

  # Calculate Q1 (25th percentile) and Q3 (75th percentile)
  Q1 = data.quantile(0.25)
  Q3 = data.quantile(0.75)
  IQR = Q3 - Q1

  # Define outlier thresholds
  lower_bound = Q1 - 1.5 * IQR
  upper_bound = Q3 + 1.5 * IQR

  # Identify outliers
  outliers = data[(data < lower_bound) | (data > upper_bound)]

  # Calculate percentage of outliers
  percentage_outliers = (len(outliers) / len(data)) * 100
  print(f"Percentage of {name} outliers using IQR: {percentage_outliers:.2f}%")




  # Calculate z-scores
  dfzscore = zscore(data)

  # Define outliers (e.g., z-score > 3 or < -3)
  outliers_z = data[(dfzscore > 3) | (dfzscore < -3)]

  # Calculate percentage of outliers
  percentage_outliers_z = (len(outliers_z) / len(data)) * 100
  print(f"Percentage of {name} outliers using Z-score: {percentage_outliers_z:.2f}%")


def univariate_numerical(data, color, proportion_thresh = .6):
    print(f'\n{data.name}')
    is_normal = is_numerical_data_normal(data)
    get_outlier_percentage(data)
    histplot_boxplot(data, color=color, kde=is_normal)
    qqplot(data, color=color)

def pairplot(data, vars = [], hue = None, palette = None,
             marker = 'o', alpha = 0.5,
             figsize = None,
             rotation = 0):
  if figsize:
    plt.figure(figsize=figsize)
  else:
    plt.figure()

  if vars == []:
    vars = data.select_dtypes(include=['number']).columns.tolist()

  g = sns.pairplot(data, vars = vars,
            corner = True,
            hue = hue,
            palette = palette,
            plot_kws=dict(alpha=alpha, marker=marker, linewidth=1),
            diag_kws=dict(fill=False),
            );

  for ax in g.axes.flatten():
    if ax:
        # rotate x axis labels
        ax.set_xlabel(ax.get_xlabel(), rotation = rotation);
        # rotate y axis labels
        ax.set_ylabel(ax.get_ylabel(), rotation = 0);
        # set y labels alignment
        ax.yaxis.get_label().set_horizontalalignment('right');

  fname = 'pairplot'
  if hue:
    fname += f'_hue-{hue}'

  if hue:

    hue_name = hue.replace('_', '\n')


    sns.move_legend(g, "upper right", bbox_to_anchor=(0.8, 0.95), title = hue_name)

  plt.savefig(f"{image_dir}/{fname}.png", bbox_inches='tight');


def correlation_numeric_analysis(corr):
  upper_triangle_indices = np.triu_indices_from(corr, k=1)
  corr_pairs = [
      (corr.index[i], corr.columns[j], corr.iloc[i, j])
      for i, j in zip(*upper_triangle_indices)
  ]

  # Convert to DataFrame for sorting
  corr_pairs_df = pd.DataFrame(corr_pairs, columns=["Variable 1", "Variable 2", "Correlation"])

  # Sort the pairs by correlation value
  sorted_corr_pairs = corr_pairs_df.sort_values(by="Correlation", ascending=False).reset_index(drop=True)

  return sorted_corr_pairs





def heatmap(matrix,figsize=(15, 7), vmin = -1 , vmax = 1, corner = True,
            rotation_x = 45, rotation_y = 45, annot=True, to_rename = True,
            tick_fontsize = None,
            annot_fontsize = 12,
            name=''):

  if to_rename:
    matrix_columns = matrix.columns.tolist()
    matrix_index = matrix.index.tolist()

    #rename_var = lambda x: x.replace('_', '\n')
    # rename_var = lambda x: x.replace('_', ' ')
    rename_var = modify_varname

    matrix.columns = list(map(rename_var, matrix_columns))
    matrix.index = list(map(rename_var, matrix_index))


  mask = None
  plt.figure(figsize = figsize)

  shrink = 1

  corner_or_square = 'corner' if corner else 'square'

  if corner:
    mask = np.triu(np.ones_like(matrix, dtype=bool))
    matrix_rm_diag = matrix.where(np.triu(np.ones(matrix.shape), k=1).astype(bool))


    round_num_up = lambda x: -math.floor(abs(x) * 10) / (10) if x<0 else math.ceil(x * 10) / (10)
    round_num_down = lambda x: -math.ceil(abs(x) * 10) / (10) if x<0 else math.floor(x * 10) / (10)
    min_value = matrix_rm_diag.min().min()
    max_value = matrix_rm_diag.max().max()

    vmin = round_num_down(min_value)
    vmax = round_num_up(max_value)

    max_abs = max(abs(vmin), abs(vmax))
    vmin = -max_abs
    vmax = max_abs

  if corner:
    shrink = (matrix.shape[0]-2)/matrix.shape[0]


  ax = sns.heatmap(matrix, mask=mask, annot=annot, vmin=vmin, vmax=vmax, fmt=".2f",
              cbar_kws={"shrink": shrink, 'pad': 0},
              annot_kws={"size": annot_fontsize},
              linewidth=2,
              square = True,
              cmap="coolwarm")

  ax.set(xlabel="", ylabel="")
  # ax.xaxis.tick_top()
  if tick_fontsize:
    ax.tick_params(axis='both', labelsize=tick_fontsize)
  ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation_x, ha='right')
  ax.set_yticklabels(ax.get_yticklabels(), rotation=rotation_y)

  plt.tick_params(left=False, bottom=False)


  plt.savefig(f"{image_dir}/heatmap_{name}_{corner_or_square}.png", bbox_inches='tight');


def bivariate_numerical(df, numerical_cols, figsize=(4,4), color = 'skyblue',
                        alpha = .3):



  if len(numerical_cols)>1:
    pairplot(df, figsize=(13, 13), rotation=30)
    # for col in categorical_cols:
    #   if df[col].nunique() < 5:
    #     pairplot(df, hue = col, palette = palette_dict_cat[col])
    corr = df.corr(numeric_only=True)
    table = correlation_numeric_analysis(corr)
    display(table)
    heatmap(corr,figsize=(18, 10), corner = True, name='corr_numeric',
            rotation_y=0, rotation_x=30, tick_fontsize=12, annot_fontsize=12)
    heatmap(corr,figsize=(18, 10), corner = False, name='corr_numeric',
            rotation_y=0, rotation_x=30, tick_fontsize=12, annot_fontsize=12)


def conditional_entropy(x, y):
    """Calculate the conditional entropy of x given y."""
    y_counter = Counter(y)
    xy_counter = Counter(list(zip(x, y)))
    total_occurrences = sum(y_counter.values())
    entropy = 0.0
    for (x_val, y_val), xy_count in xy_counter.items():
        p_xy = xy_count / total_occurrences
        p_y = y_counter[y_val] / total_occurrences
        entropy += p_xy * np.log(p_y / p_xy)
    return entropy

def theils_u(x, y):
    """Calculate Theil's U (Uncertainty Coefficient)."""
    s_xy = conditional_entropy(x, y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    s_x = -sum((count / total_occurrences) * np.log(count / total_occurrences) for count in x_counter.values())
    if s_x == 0:
        return 1  # If x has no entropy, then it's perfectly predictable
    return (s_x - s_xy) / s_x

def corr_categorical(df, vars = []):
  if vars == []:
    vars = df.select_dtypes(include=['object', 'category']).columns.tolist()

  dfcat = df[vars]

  correlations = {}

  for i in range(len(vars)):
    for j in range(i+1, len(vars)):
      x = dfcat[vars[i]]
      y = dfcat[vars[j]]
      theils_u_value = theils_u(x, y)
      correlations.update({(x.name, y.name): theils_u_value})

  # Extract unique variable names
  variables = set()
  for pair in correlations.keys():
      variables.update(pair)
  variables = sorted(variables)  # Sort for consistent ordering

  # Create an empty DataFrame
  matrix = pd.DataFrame(np.nan, index=variables, columns=variables)

  # Fill the DataFrame with the correlation values
  for (var1, var2), corr_value in correlations.items():
    if var1!=var2:
      matrix.loc[var1, var2] = corr_value
      matrix.loc[var2, var1] = corr_value  # Ensure symmetry

  # Fill diagonal with 1.0 (if not already provided)
  np.fill_diagonal(matrix.values, 1.0)

  display(matrix)
  return matrix


def stacked_barplot(predictor_data, target_data, to_normalize=True,
                    alpha = 0.5,
                    rotation_x_tick = 0,
                    height= 3):
    """
    Print the category counts and plot a stacked bar chart

    predictor: independent variable
    target: target variable
    """
    count = predictor_data.nunique()
    sorter = target_data.value_counts().index[-1]
    tab1 = pd.crosstab(predictor_data, target_data, margins=True).sort_values(
        by=sorter, ascending=False
    )
    # display(tab1)

    if to_normalize:
      tab = pd.crosstab(predictor_data, target_data, normalize="index").sort_values(
          by=sorter, ascending=False
      )
    else:
      tab = pd.crosstab(predictor_data, target_data).sort_values(
          by=sorter, ascending=False
      )

    tab.plot(kind="bar", stacked=True, alpha=alpha, edgecolor='gray', figsize=(count + height, height))
    # plt.xlabel(predictor_data.name.replace('_', '\n'))
    plt.xlabel(modify_varname(predictor_data.name))
    plt.ylabel('Normalized\nFrequency')
    # plt.legend(loc="upper left", bbox_to_anchor=(1, 1), frameon=False, title=target_data.name.replace('_', '\n'))
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), frameon=False, title=modify_varname(target_data.name))
    plt.xticks(rotation=rotation_x_tick)
    sns.despine(top = True, right = True)
    extra = '_normalized' if to_normalize else ''
    plt.savefig(f"{image_dir}/stacked_barplot_{target_data.name}_vs_{predictor_data.name}{extra}.png", bbox_inches='tight');

def bivariate_categorical_plots(data, vars = []):
  if vars == []:
    vars = data.select_dtypes(include=['object', 'category']).columns.tolist()

  vars_limited = [c for c in vars if data[c].nunique()< 10]

  for i in range(len(vars_limited)):
    for j in range(i, len(vars_limited)):
      var1 = vars_limited[i]
      var2 = vars_limited[j]
      if var1!=var2:
        stacked_barplot(data[var1], data[var2])


def pairplot_categorical(data, vars = [], hue = None, marker = 'o', alpha = 0.5,
                         to_normalize=True,
                         figsize=(10, 10),
                         rotation_x = 30,
                         rotation_y = 90,
                         rotation_x_tick = 0,
                         rotation_y_tick = 0,
                         ):

  if vars == []:
    vars = data.select_dtypes(include=['object', 'category']).columns.tolist()

  vars_limited = [c for c in vars if data[c].nunique()<5]


  modify_name = lambda x: x.replace('_', '\n')

  n_cat = len(vars_limited)
  fig, axs = plt.subplots(n_cat, n_cat, layout="constrained",
                          figsize=figsize, sharex='col', sharey='row')

  for i in range(n_cat):
    for j in range(n_cat):
      x = data[vars_limited[i]]
      y = data[vars_limited[j]]
      if j >= i:
        if i == j:
          tab = x.value_counts(normalize=to_normalize)
          stacked = False
          tab.plot(kind="bar", stacked=stacked, ax = axs[j, i], facecolor='none', edgecolor='gray')
        else:
          stacked = True
          if to_normalize:
            tab = pd.crosstab(x,y, normalize="index")
          else:
            tab = pd.crosstab(x,y)
          tab.plot(kind="bar", stacked=stacked, ax = axs[j, i], alpha=0.5, edgecolor='gray')

        if j!=i:
          if i == 0:
            axs[j, i].legend(frameon = False, bbox_to_anchor=(-1.4, .5), loc='center')
          else:
            axs[j, i].get_legend().remove()
        axs[j, i].set_xlabel(modify_varname(x.name), rotation=rotation_x)
        axs[j, i].set_ylabel(modify_varname(y.name), rotation=rotation_y)
        axs[j, i].tick_params(axis='x', rotation=rotation_x_tick)
        axs[j, i].tick_params(axis='y', rotation=rotation_y_tick)
        if i!=0:
          axs[j, i].set_ylabel('')
        if j != n_cat-1:
          axs[j, i].set_xlabel('')
        sns.despine(ax=axs[j, i], top=True, right=True)
      else:
        sns.despine(ax=axs[j, i], top=True, right=True, left=True, bottom=True)
        axs[j, i].axis('off')

  plt.savefig(f"{image_dir}/pairplot_categorical.png", bbox_inches='tight');


def bivariate_category_effect(cat1, cat2, alpha = 0.5):
  crosstab = pd.crosstab(cat1, cat2)
  chi, p_value, dof, expected =  stats.chi2_contingency(crosstab)
  Ho = f"{cat1.name} has NO effect on {cat2.name}"
  Ha = f"{cat1.name} HAS AN EFFECT on {cat2.name}"

  has_effect = False

  if p_value < alpha:  # Setting our significance level at 5%
      t = f'{Ha} as the p_value {p_value:.1e} < {alpha}.'
      has_effect = True
  else:
      t = f'{Ho} as the p_value {p_value:.1e} > {alpha}.'

  return has_effect, p_value, t

def get_chi_contingency(df, categorical_cols, alpha = 0.05):
  chi_dict = {}
  for i in range(len(categorical_cols)):
    for j in range(i+1, len(categorical_cols)):
      cat1 = df[categorical_cols[i]]
      cat2 = df[categorical_cols[j]]
      has_effect, p_value, t = bivariate_category_effect(cat1, cat2, alpha=alpha)
      chi_dict.update({(cat1.name, cat2.name): {'p-value': p_value, 'Explanation': t}})

  chi_df = (
                              pd.DataFrame(chi_dict)
                              .T.reset_index()
                              .sort_values(by='p-value', ascending=True)
                              .reset_index(drop=True)
                              )
  chi_df.columns = ['Category 1', 'Category 2', 'p-value', 'Explanation']
  chi_df = chi_df[chi_df['p-value']< alpha]
  chi_df['p-value'] = chi_df['p-value'].apply(lambda x: f'{x:.1e}')
  for i, row in chi_df.iterrows():
    print(row['Explanation'])
  display(chi_df.drop('Explanation', axis=1))


def bivariate_categorical(df, categorical_cols, figsize=(4,4), color = 'skyblue',
                        alpha = .3):
  if len(categorical_cols) > 1:
    pairplot_categorical(df, rotation_x_tick=90, rotation_x=0, figsize=(13, 13))
    bivariate_categorical_plots(df)
    corr = corr_categorical(df)
    get_chi_contingency(df, categorical_cols)
    heatmap(corr, figsize=(18, 10), corner = True, name='corr_categorical', rotation_y=0)


def histplot_boxplot_hue(data, var,
             figsize = (9, 5), height_ratios = [3, 2],
             round_by=2,
             kde = True,
             hue = None,
             mean_color='blue',
             median_color='gray',
             palette='Blues',
             alpha_hist = .3,
             alpha_box = .3,
             showmeans=True):

  if hue:
    height_ratios = [3, data[hue].nunique()-1]

  name = var.replace('_', '\n')

  # Create subplots with shared x-axis
  fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True,
                           figsize=figsize,
                           gridspec_kw={'height_ratios': height_ratios })

  # Plot the histogram
  sns.histplot(data = data,
               x = var,
               kde=kde, ax=axes[0],
               palette = palette,
               alpha=alpha_hist,
               hue = hue,
               stat = 'density',
               edgecolor='black', linewidth=.5)

  title = f"{var.replace('_', ' ')} & {hue.replace('_', ' ')}" if hue else f"{var.replace('_', ' ')}"
  # plt.suptitle("Histogram & Boxplot")
  axes[0].set_title(title)
  axes[0].set_ylabel("Frequency")

  # Plot the boxplot
  sns.boxplot(data = data, x=var, ax=axes[1], width=0.5, palette=palette,
              showmeans = showmeans,
              hue = hue,
              boxprops=dict(alpha=alpha_box),
              meanprops={"marker": "^", "markerfacecolor": mean_color, "markeredgecolor": "black"},
              flierprops={"marker": "x", "markersize": 1, "markerfacecolor": "gray"})
  axes[1].set_xlabel(var.replace('_', '\n'))

  sns.despine(ax=axes[0])
  sns.despine(ax=axes[1])



  axes[0].get_legend().remove()
  if hue:
    axes[1].legend(frameon=False, loc='upper right', bbox_to_anchor=(1.5, 1),
                   fontsize = 12,
                   title=hue.replace('_', '\n'))


  # Adjust layout and show
  plt.tight_layout()
  extra = f'_vs_{hue}' if hue else ''
  plt.savefig(f"{image_dir}/histplot_boxplot_{name}{extra}.png", bbox_inches='tight')


def bivariate_categorical_numerical_effect(num_data, cat_data, alpha = 0.05):
  cat_values = cat_data.unique().tolist()

  groups = [num_data[cat_data == value] for value in cat_values]


  Ho = f"{cat_data.name} has no effect on {num_data.name}"
  Ha = f"{cat_data.name} HAS AN EFFECT on {num_data.name}"

  if len(groups) == 2:
    test_stat, p_value  = stats.ttest_ind(*groups)
    method = 'Two-Sample T-Test'
  else:
    test_stat, p_value = stats.f_oneway(*groups)
    method = 'One-Way ANOVA F-test'

  has_effect = False

  if p_value < alpha:
      has_effect = True
      t = f'{Ha} as the {method} p_value {p_value:.1e} < {alpha}.'
  else:
      t = f'{Ho} as the {method} p_value {p_value:.1e} > {alpha}.'

  return has_effect, p_value, t


def get_pvalue_table_categorical_numerical(df, categorical_cols, numerical_cols, alpha = 0.05):
  pdict = {}
  for cat_col in categorical_cols:
    for num_col in numerical_cols:
      cat_data = df[cat_col]
      num_data = df[num_col]
      has_effect, p_value, t = bivariate_categorical_numerical_effect(num_data, cat_data, alpha=alpha)
      pdict.update({(cat_col, num_col): {'p-value': p_value, 'Explanation': t}})

  p_df = (
          pd.DataFrame(pdict)
          .T.reset_index()
          .sort_values(by='p-value', ascending=True)
          .reset_index(drop=True)
  )
  p_df.columns = ['Category', 'Numerical', 'p-value', 'Explanation']
  p_df = p_df[p_df['p-value']< alpha]
  p_df['p-value'] = p_df['p-value'].apply(lambda x: f'{x:.1e}')
  for i, row in p_df.iterrows():
    print(row['Explanation'])
  display(p_df.drop('Explanation', axis=1))

def bivariate_numerical_categorical(df, categorical_cols, numerical_cols, palette_dict_cat):
  get_pvalue_table_categorical_numerical(df, categorical_cols, numerical_cols)
  for cat in categorical_cols:
    for num in numerical_cols:
      palette = palette_dict_cat[cat]
      histplot_boxplot_hue(df, var=num, hue=cat, palette=palette)


def clustered_heatmap(corr):
  from scipy.cluster.hierarchy import linkage, leaves_list


  # Compute correlation matrix
  correlation_matrix = corr

  # Use hierarchical clustering to reorder rows and columns
  linkage_matrix = linkage(correlation_matrix, method='average')
  ordered_indices = leaves_list(linkage_matrix)

  # Reorder the correlation matrix
  correlation_matrix_sorted = correlation_matrix.iloc[ordered_indices, ordered_indices]


  heatmap(correlation_matrix_sorted,figsize=(18, 10), vmin = -1 , vmax = 1, corner = True,
              rotation_x = 45, rotation_y = 0, annot=False, to_rename=False,
              tick_fontsize = 11,
              name='all_columns_clustered')


# sort_columns
def get_numerical_categorical_info(df):
  df = df[df.columns.sort_values().tolist()]
  categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
  numerical_cols = df.select_dtypes(include=['number']).columns.tolist()

  palettes_cat = [
    "Blues", "Greens", "Reds", "Purples", "Oranges", "Greys",
    "YlGn", "YlGnBu", "GnBu", "BuGn", "PuBu", "PuBuGn", "BuPu",
    "OrRd", "PuRd", "RdPu", "YlOrBr", "YlOrRd", 'rocket', 'mako', 'flare', 'crest'
  ]
  colors_num = ['blue', 'green', 'red', 'orange', 'purple', 'dodgerblue', 'gray'] + ['steelblue', 'skyblue', 'seagreen', 'limegreen', 'lightgreen', 'firebrick', 'tomato', 'indianred', 'darkorange', 'gold', 'mediumpurple',
                                            'orchid', 'yellow', 'gold', 'khaki', 'lightgray', 'darkgray', 'silver']

  if len(colors_num) < len(numerical_cols):
    print(f'Add {len(numerical_cols) - len(colors_num)} colors.')

  if len(palettes_cat) < len(categorical_cols):
    print(f'Add {len(categorical_cols) - len(palettes_cat)} palettes.')

  categorical_cols_missing = [c for c in categorical_cols if f'{c} (missing)' in categorical_cols]
  numerical_cols_missing = [c for c in numerical_cols if f'{c} (missing)' in numerical_cols]

  palette_dict_cat = {k: v for k, v in zip(categorical_cols, palettes_cat)}
  palette_dict_cat.update({k: palette_dict_cat[k.split(' (missing)')[0]] for k in palette_dict_cat if '(missing)' in k})
  colors_dict_num = {k: v for k, v in zip(numerical_cols, colors_num) if '> 0' not in k}
  colors_dict_num.update({k: colors_dict_num[k.split(' > 0')[0]] for k in numerical_cols if '> 0' in k})
  colors_dict_num.update({k: colors_dict_num[k.split(' (missing)')[0]] for k in numerical_cols if '(missing)' in k})
  return df, categorical_cols, numerical_cols, palette_dict_cat, colors_dict_num


def plot_all_univariate(df, categorical_cols, numerical_cols, palette_dict_cat, colors_dict_num, thresh = 10):
  for col in categorical_cols:
    palette = palette_dict_cat[col]
    nmax = None
    if df[col].nunique() > thresh:
      nmax = 9
    univariate_categorical(df[col], palette = palette, nmax = nmax)

  for col in numerical_cols:
    color = colors_dict_num[col]
    univariate_numerical(df[col], color)


def plot_all_bivariate(df, categorical_cols, numerical_cols, palette_dict_cat):
  bivariate_numerical(df, numerical_cols)
  bivariate_categorical(df, categorical_cols)
  bivariate_numerical_categorical(df, categorical_cols, numerical_cols, palette_dict_cat)
  get_pvalue_table_categorical_numerical(df, categorical_cols, numerical_cols )

def create_info_conduct_eda(data, positive_cols=[], drop_cols=[], type_conv={},
                  feature_info={}, modify_dict={}, ddict_str="", cat_orders={}, to_plot = False):
  data = create_info(data, positive_cols = positive_cols,
                        type_conv = type_conv,
                        feature_info = feature_info,
                  modify_dict = modify_dict,
                            drop_cols = drop_cols, ddict_str = ddict_str, cat_orders=cat_orders)

  data, categorical_cols, numerical_cols, palette_dict_cat, colors_dict_num = get_numerical_categorical_info(data)

  if to_plot:
    missing_data_cols = [c.split(" (missing)")[0] for c in data.columns if '(missing)' in c]
    for c in missing_data_cols:
      data[f'{c} (missing)'] = data[f'{c} (missing)'].fillna('Unknown')
      data[f'{c} (missing)'] = pd.Categorical(data[f'{c} (missing)'], categories=['Unknown']+ cat_orders[c], ordered=True)
    plot_all_univariate(data, categorical_cols, numerical_cols, palette_dict_cat, colors_dict_num)
    categorical_cols = [c for c in categorical_cols if c not in missing_data_cols]
    numerical_cols = [c for c in numerical_cols if c not in missing_data_cols]
    data_drop_extra = data.drop(missing_data_cols, axis = 1)
    plot_all_bivariate(data_drop_extra, categorical_cols, numerical_cols, palette_dict_cat)

  return data
