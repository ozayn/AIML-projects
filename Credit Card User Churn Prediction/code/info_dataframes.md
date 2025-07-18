There are 10,127 (~10K) _rows_ and 24 _columns_ in the dataset.

The __memory usage__ is approximately  1898.9 KB.

There are _3,380 null values_ in the dataset. 14.999506270366346% of the missing data is in Education_Level (missing).

There are __no duplicated__ rows in the data.

Anamolous Values

No feature engineering is done.

Column(s) CLIENTNUM is/are dropped.

'Attrition_Flag' is converted to category type.
'Card_Category' is converted to category type.
'Contacts_Count_12_mon' is converted to category type.
'Dependent_count' is converted to category type.
'Education_Level' is converted to category type.
'Gender' is converted to category type.
'Income_Category' is converted to category type.
'Marital_Status' is converted to category type.
'Months_Inactive_12_mon' is converted to category type.
'Total_Relationship_Count' is converted to category type.


| Column                    | Data Type   | Description                                                                                                                                                |   # unique |
|:--------------------------|:------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------|-----------:|
| CLIENTNUM                 | int64       | Client number. Unique identifier for the customer holding the account                                                                                      |      10127 |
| Attrition_Flag            | object      | Internal event (customer activity) variable - if the account is closed then "Attrited Customer" else "Existing Customer"                                   |          2 |
| Customer_Age              | int64       | Age in Years                                                                                                                                               |         45 |
| Gender                    | object      | Gender of the account holder                                                                                                                               |          2 |
| Dependent_count           | int64       | Number of dependents                                                                                                                                       |          6 |
| Education_Level           | object      | Educational Qualification of the account holder - Graduate, High School, Unknown, Uneducated, College(refers to college student), Post-Graduate, Doctorate |          6 |
| Marital_Status            | object      | Marital Status of the account holder                                                                                                                       |          3 |
| Income_Category           | object      | Annual Income Category of the account holder                                                                                                               |          5 |
| Card_Category             | object      | Type of Card                                                                                                                                               |          4 |
| Months_on_book            | int64       | Period of relationship with the bank (in months)                                                                                                           |         44 |
| Total_Relationship_Count  | int64       | Total no. of products held by the customer                                                                                                                 |          6 |
| Months_Inactive_12_mon    | int64       | No. of months inactive in the last 12 months                                                                                                               |          7 |
| Contacts_Count_12_mon     | int64       | No. of Contacts in the last 12 months                                                                                                                      |          7 |
| Credit_Limit              | float64     | Credit Limit on the Credit Card                                                                                                                            |       6205 |
| Total_Revolving_Bal       | int64       | Total Revolving Balance on the Credit Card                                                                                                                 |       1974 |
| Avg_Open_To_Buy           | float64     | Open to Buy Credit Line (Average of last 12 months)                                                                                                        |       6813 |
| Total_Amt_Chng_Q4_Q1      | float64     | Change in Transaction Amount (Q4 over Q1)                                                                                                                  |       1158 |
| Total_Trans_Amt           | int64       | Total Transaction Amount (Last 12 months)                                                                                                                  |       5033 |
| Total_Trans_Ct            | int64       | Total Transaction Count (Last 12 months)                                                                                                                   |        126 |
| Total_Ct_Chng_Q4_Q1       | float64     | Change in Transaction Count (Q4 over Q1)                                                                                                                   |        830 |
| Avg_Utilization_Ratio     | float64     | Average Card Utilization Ratio                                                                                                                             |        964 |
| Education_Level (missing) | object      | nan                                                                                                                                                        |          6 |
| Marital_Status (missing)  | object      | nan                                                                                                                                                        |          3 |
| Income_Category (missing) | object      | nan                                                                                                                                                        |          5 |

|                 |           |
|:----------------|:----------|
| Memory Usage    | 1898.9 KB |
|                 |           |
| #               |           |
| Rows            | 10127     |
| Columns         | 24        |
| Null Values     | 3380      |
| Duplicated Rows | 0         |
|                 |           |
| int64           | 10        |
| object          | 9         |
| float64         | 5         |

|                           |   # Null |
|:--------------------------|---------:|
| Education_Level (missing) |     1519 |
| Income_Category (missing) |     1112 |
| Marital_Status (missing)  |      749 |

| Column                    | Data Type   | Description                                                                                                                                                |   # unique |
|:--------------------------|:------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------|-----------:|
| Attrition_Flag            | category    | Internal event (customer activity) variable - if the account is closed then "Attrited Customer" else "Existing Customer"                                   |          2 |
| Customer_Age              | int64       | Age in Years                                                                                                                                               |         45 |
| Gender                    | category    | Gender of the account holder                                                                                                                               |          2 |
| Dependent_count           | category    | Number of dependents                                                                                                                                       |          6 |
| Education_Level           | category    | Educational Qualification of the account holder - Graduate, High School, Unknown, Uneducated, College(refers to college student), Post-Graduate, Doctorate |          6 |
| Marital_Status            | category    | Marital Status of the account holder                                                                                                                       |          3 |
| Income_Category           | category    | Annual Income Category of the account holder                                                                                                               |          5 |
| Card_Category             | category    | Type of Card                                                                                                                                               |          4 |
| Months_on_book            | int64       | Period of relationship with the bank (in months)                                                                                                           |         44 |
| Total_Relationship_Count  | category    | Total no. of products held by the customer                                                                                                                 |          6 |
| Months_Inactive_12_mon    | category    | No. of months inactive in the last 12 months                                                                                                               |          7 |
| Contacts_Count_12_mon     | category    | No. of Contacts in the last 12 months                                                                                                                      |          7 |
| Credit_Limit              | float64     | Credit Limit on the Credit Card                                                                                                                            |       6205 |
| Total_Revolving_Bal       | int64       | Total Revolving Balance on the Credit Card                                                                                                                 |       1974 |
| Avg_Open_To_Buy           | float64     | Open to Buy Credit Line (Average of last 12 months)                                                                                                        |       6813 |
| Total_Amt_Chng_Q4_Q1      | float64     | Change in Transaction Amount (Q4 over Q1)                                                                                                                  |       1158 |
| Total_Trans_Amt           | int64       | Total Transaction Amount (Last 12 months)                                                                                                                  |       5033 |
| Total_Trans_Ct            | int64       | Total Transaction Count (Last 12 months)                                                                                                                   |        126 |
| Total_Ct_Chng_Q4_Q1       | float64     | Change in Transaction Count (Q4 over Q1)                                                                                                                   |        830 |
| Avg_Utilization_Ratio     | float64     | Average Card Utilization Ratio                                                                                                                             |        964 |
| Education_Level (missing) | object      | nan                                                                                                                                                        |          6 |
| Marital_Status (missing)  | object      | nan                                                                                                                                                        |          3 |
| Income_Category (missing) | object      | nan                                                                                                                                                        |          5 |

|                 |           |
|:----------------|:----------|
| Memory Usage    | 1129.7 KB |
|                 |           |
| #               |           |
| Rows            | 10127     |
| Columns         | 23        |
| Null Values     | 3380      |
| Duplicated Rows | 0         |
|                 |           |
| category        | 10        |
| int64           | 5         |
| float64         | 5         |
| object          | 3         |

|                           |   # Null |
|:--------------------------|---------:|
| Education_Level (missing) |     1519 |
| Income_Category (missing) |     1112 |
| Marital_Status (missing)  |      749 |

| Object/Categorical Column   |   unique | top            |   freq |
|:----------------------------|---------:|:---------------|-------:|
| Attrition_Flag              |        2 | Existing       |   8500 |
| Gender                      |        2 | F              |   5358 |
| Dependent_count             |        6 | 3              |   2732 |
| Education_Level             |        6 | Graduate       |   4647 |
| Marital_Status              |        3 | Married        |   5436 |
| Income_Category             |        5 | Less than $40K |   4673 |
| Card_Category               |        4 | Blue           |   9436 |
| Total_Relationship_Count    |        6 | 3              |   2305 |
| Months_Inactive_12_mon      |        7 | 3              |   3846 |
| Contacts_Count_12_mon       |        7 | 3              |   3380 |
| Education_Level (missing)   |        6 | Graduate       |   3128 |
| Marital_Status (missing)    |        3 | Married        |   4687 |
| Income_Category (missing)   |        5 | Less than $40K |   3561 |

| Numerical Column      |   mean |    std |    min |      25% |      50% |       75% |       max |      IQR |   # Outliers (Upper) |   # Outliers (Lower) |   # Outliers |   Outliers % |
|:----------------------|-------:|-------:|-------:|---------:|---------:|----------:|----------:|---------:|---------------------:|---------------------:|-------------:|-------------:|
| Customer_Age          |   46.3 |    8   |   26   |   41     |   46     |    52     |    73     |   11     |                    2 |                    0 |            2 |          0   |
| Months_on_book        |   35.9 |    8   |   13   |   31     |   36     |    40     |    56     |    9     |                  198 |                  188 |          386 |          3.8 |
| Credit_Limit          | 8632   | 9088.8 | 1438.3 | 2555     | 4549     | 11067.5   | 34516     | 8512.5   |                  984 |                    0 |          984 |          9.7 |
| Total_Revolving_Bal   | 1162.8 |  815   |    0   |  359     | 1276     |  1784     |  2517     | 1425     |                    0 |                    0 |            0 |          0   |
| Avg_Open_To_Buy       | 7469.1 | 9090.7 |    3   | 1324.5   | 3474     |  9859     | 34516     | 8534.5   |                  963 |                    0 |          963 |          9.5 |
| Total_Amt_Chng_Q4_Q1  |    0.8 |    0.2 |    0   |    0.631 |    0.736 |     0.859 |     3.397 |    0.228 |                  350 |                   48 |          398 |          3.9 |
| Total_Trans_Amt       | 4404.1 | 3397.1 |  510   | 2155.5   | 3899     |  4741     | 18484     | 2585.5   |                  896 |                    0 |          896 |          8.8 |
| Total_Trans_Ct        |   64.9 |   23.5 |   10   |   45     |   67     |    81     |   139     |   36     |                    2 |                    0 |            2 |          0   |
| Total_Ct_Chng_Q4_Q1   |    0.7 |    0.2 |    0   |    0.582 |    0.702 |     0.818 |     3.714 |    0.236 |                  300 |                   96 |          396 |          3.9 |
| Avg_Utilization_Ratio |    0.3 |    0.3 |    0   |    0.023 |    0.176 |     0.503 |     0.999 |    0.48  |                    0 |                    0 |            0 |          0   |

