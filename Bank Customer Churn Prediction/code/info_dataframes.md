There are 10,000 (~10K) _rows_ and 13 _columns_ in the dataset.

The __memory usage__ is approximately  1015.8 KB.

There are __no missing__ values in the data.

There are __no duplicated__ rows in the data.

Anamolous Values

No feature engineering is done.

Column(s) Surname, CustomerId is/are dropped.

'Exited' is converted to category type.
'Gender' is converted to category type.
'Geography' is converted to category type.
'HasCrCard' is converted to category type.
'IsActiveMember' is converted to category type.
'NumOfProducts' is converted to category type.


| Column              | Data Type   | Description                                                                                         |   # unique |
|:--------------------|:------------|:----------------------------------------------------------------------------------------------------|-----------:|
| CustomerId          | int64       | Unique ID which is assigned to each customer                                                        |      10000 |
| Surname             | object      | Last name of the customer                                                                           |       2932 |
| CreditScore         | int64       | It defines the credit history of the customer                                                       |        460 |
| Geography           | object      | Customer’s location                                                                                 |          3 |
| Gender              | object      | It defines the gender of the customer                                                               |          2 |
| Age                 | int64       | Age of the customer                                                                                 |         70 |
| Tenure              | int64       | Number of years for which the customer has been with the bank                                       |         11 |
| Balance (K)         | float64     | Account balance                                                                                     |       6382 |
| NumOfProducts       | int64       | Refers to the number of products that a customer has purchased through the bank                     |          4 |
| HasCrCard           | int64       | It is a categorical variable which decides whether the customer has credit card or not              |          2 |
| IsActiveMember      | int64       | Is is a categorical variable which decides whether the customer is active member of the bank or not |          2 |
| EstimatedSalary (K) | float64     | Estimated salary                                                                                    |       9999 |
| Exited              | int64       | Whether or not the customer left the bank within six month, 0 = No, 1 = Yes                         |          2 |

|                 |           |
|:----------------|:----------|
| Memory Usage    | 1015.8 KB |
|                 |           |
| #               |           |
| Rows            | 10000     |
| Columns         | 13        |
| Null Values     | 0         |
| Duplicated Rows | 0         |
|                 |           |
| int64           | 8         |
| object          | 3         |
| float64         | 2         |



| Column              | Data Type   | Description                                                                                         |   # unique |
|:--------------------|:------------|:----------------------------------------------------------------------------------------------------|-----------:|
| CreditScore         | int64       | It defines the credit history of the customer                                                       |        460 |
| Geography           | category    | Customer’s location                                                                                 |          3 |
| Gender              | category    | It defines the gender of the customer                                                               |          2 |
| Age                 | int64       | Age of the customer                                                                                 |         70 |
| Tenure              | int64       | Number of years for which the customer has been with the bank                                       |         11 |
| Balance (K)         | float64     | Account balance                                                                                     |       6382 |
| NumOfProducts       | category    | Refers to the number of products that a customer has purchased through the bank                     |          4 |
| HasCrCard           | category    | It is a categorical variable which decides whether the customer has credit card or not              |          2 |
| IsActiveMember      | category    | Is is a categorical variable which decides whether the customer is active member of the bank or not |          2 |
| EstimatedSalary (K) | float64     | Estimated salary                                                                                    |       9999 |
| Exited              | category    | Whether or not the customer left the bank within six month, 0 = No, 1 = Yes                         |          2 |

|                 |          |
|:----------------|:---------|
| Memory Usage    | 450.2 KB |
|                 |          |
| #               |          |
| Rows            | 10000    |
| Columns         | 11       |
| Null Values     | 0        |
| Duplicated Rows | 0        |
|                 |          |
| category        | 6        |
| int64           | 3        |
| float64         | 2        |



| Object/Categorical Column   |   unique | top    |   freq |
|:----------------------------|---------:|:-------|-------:|
| Geography                   |        3 | France |   5014 |
| Gender                      |        2 | Male   |   5457 |
| NumOfProducts               |        4 | 1      |   5084 |
| HasCrCard                   |        2 | 1      |   7055 |
| IsActiveMember              |        2 | 1      |   5151 |
| Exited                      |        2 | 0      |   7963 |

| Numerical Column    |   mean |   std |       min |      25% |      50% |     75% |     max |      IQR |   # Outliers (Upper) |   # Outliers (Lower) |   # Outliers |   Outliers % |
|:--------------------|-------:|------:|----------:|---------:|---------:|--------:|--------:|---------:|---------------------:|---------------------:|-------------:|-------------:|
| CreditScore         |  650.5 |  96.7 | 350       | 584      | 652      | 718     | 850     | 134      |                    0 |                   16 |           16 |          0.2 |
| Age                 |   38.9 |  10.5 |  18       |  32      |  37      |  44     |  92     |  12      |                  411 |                    0 |          411 |          4.1 |
| Tenure              |    5   |   2.9 |   0       |   3      |   5      |   7     |  10     |   4      |                    0 |                    0 |            0 |          0   |
| Balance (K)         |   76.5 |  62.4 |   0       |   0      |  97.1985 | 127.644 | 250.898 | 127.644  |                    0 |                    0 |            0 |          0   |
| EstimatedSalary (K) |  100.1 |  57.5 |   0.01158 |  51.0021 | 100.194  | 149.388 | 199.992 |  98.3861 |                    0 |                    0 |            0 |          0   |

