There are 5,000 (~5K) _rows_ and 15 _columns_ in the dataset.

The __memory usage__ is approximately  586.1 KB.

There are _3,462 null values_ in the dataset. 69.24% of the missing data is in Mortgage > 0.

There are __no duplicated__ rows in the data.

Anamolous Values

There are 52 negative values in Experience column.

52 values in Experience column are converted to positive values.

The ZIPCode (SCF) column is created from the ZIPCode column.
ZIPCode (SCF): (sectional center facility) is the first two digits of the ZIPCode.

Column(s) ID, ZIPCode is/are dropped.

'CD_Account' is converted to category type.
'CreditCard' is converted to category type.
'Education' is converted to category type.
'Family' is converted to category type.
'Online' is converted to category type.
'Personal_Loan' is converted to category type.
'Securities_Account' is converted to category type.
'ZIPCode' is converted to str type.


| Column             | Data Type   | Description                                                                                            |   # unique |
|:-------------------|:------------|:-------------------------------------------------------------------------------------------------------|-----------:|
| ID                 | int64       | Customer ID                                                                                            |       5000 |
| Age                | int64       | Customer’s age in completed years                                                                      |         45 |
| Experience         | int64       | Number of years of professional experience                                                             |         47 |
| Income             | int64       | Annual income of the customer (in thousand dollars)                                                    |        162 |
| ZIPCode            | int64       | Home address ZIP code                                                                                  |        467 |
| Family             | int64       | The family size of the customer                                                                        |          4 |
| CCAvg              | float64     | Average spending on credit cards per month (in thousand dollars)                                       |        108 |
| Education          | int64       | Education level (1- Undergrad; 2- Graduate; 3- Advanced/Professional)                                  |          3 |
| Mortgage           | int64       | Value of house mortgage if any (in thousand dollars)                                                   |        347 |
| Personal_Loan      | int64       | Did this customer accept the personal loan offered in the last campaign? (0- No, 1-Yes)                |          2 |
| Securities_Account | int64       | Does the customer have securities account with the bank? (0- No, 1- Yes)                               |          2 |
| CD_Account         | int64       | Does the customer have a certificate of deposit (CD) account with the bank? (0- No, 1- Yes)            |          2 |
| Online             | int64       | Do customers use internet banking facilities? (0- No, 1- Yes)                                          |          2 |
| CreditCard         | int64       | Does the customer use a credit card issued by any other Bank (excluding AllLife Bank)? (0- No, 1- Yes) |          2 |
| Mortgage > 0       | float64     | nan                                                                                                    |        346 |

|                 |          |
|:----------------|:---------|
| Memory Usage    | 586.1 KB |
|                 |          |
| #               |          |
| Rows            | 5000     |
| Columns         | 15       |
| Null Values     | 3462     |
| Duplicated Rows | 0        |
|                 |          |
| int64           | 13       |
| float64         | 2        |

|              |   # Null |
|:-------------|---------:|
| Mortgage > 0 |     3462 |

| Column             | Data Type   | Description                                                                                            |   # unique |
|:-------------------|:------------|:-------------------------------------------------------------------------------------------------------|-----------:|
| Age                | int64       | Customer’s age in completed years                                                                      |         45 |
| Experience         | int64       | Number of years of professional experience                                                             |         44 |
| Income             | int64       | Annual income of the customer (in thousand dollars)                                                    |        162 |
| Family             | category    | The family size of the customer                                                                        |          4 |
| CCAvg              | float64     | Average spending on credit cards per month (in thousand dollars)                                       |        108 |
| Education          | category    | Education level (1- Undergrad; 2- Graduate; 3- Advanced/Professional)                                  |          3 |
| Mortgage           | int64       | Value of house mortgage if any (in thousand dollars)                                                   |        347 |
| Personal_Loan      | category    | Did this customer accept the personal loan offered in the last campaign? (0- No, 1-Yes)                |          2 |
| Securities_Account | category    | Does the customer have securities account with the bank? (0- No, 1- Yes)                               |          2 |
| CD_Account         | category    | Does the customer have a certificate of deposit (CD) account with the bank? (0- No, 1- Yes)            |          2 |
| Online             | category    | Do customers use internet banking facilities? (0- No, 1- Yes)                                          |          2 |
| CreditCard         | category    | Does the customer use a credit card issued by any other Bank (excluding AllLife Bank)? (0- No, 1- Yes) |          2 |
| Mortgage > 0       | float64     | nan                                                                                                    |        346 |
| ZIPCode (SCF)      | category    | (sectional center facility) is the rightmost two digits of the ZIPCode                                 |          7 |

|                 |          |
|:----------------|:---------|
| Memory Usage    | 274.8 KB |
|                 |          |
| #               |          |
| Rows            | 5000     |
| Columns         | 14       |
| Null Values     | 3462     |
| Duplicated Rows | 1        |
|                 |          |
| category        | 8        |
| int64           | 4        |
| float64         | 2        |

|              |   # Null |
|:-------------|---------:|
| Mortgage > 0 |     3462 |

| Object/Categorical Column   |   unique |   top |   freq |
|:----------------------------|---------:|------:|-------:|
| Family                      |        4 |     1 |   1472 |
| Education                   |        3 |     1 |   2096 |
| Personal_Loan               |        2 |     0 |   4520 |
| Securities_Account          |        2 |     0 |   4478 |
| CD_Account                  |        2 |     0 |   4698 |
| Online                      |        2 |     1 |   2984 |
| CreditCard                  |        2 |     0 |   3530 |
| ZIPCode (SCF)               |        7 |    94 |   1472 |

| Numerical Column   |   mean |   std |   min |   25% |   50% |   75% |   max |   IQR |   # Outliers (Upper) |   # Outliers (Lower) |   # Outliers |   Outliers % |
|:-------------------|-------:|------:|------:|------:|------:|------:|------:|------:|---------------------:|---------------------:|-------------:|-------------:|
| Age                |   45.3 |  11.5 |    23 |  35   |  45   |  55   |    67 |  20   |                    0 |                    0 |            0 |          0   |
| Experience         |   20.1 |  11.4 |     0 |  10   |  20   |  30   |    43 |  20   |                    0 |                    0 |            0 |          0   |
| Income             |   73.8 |  46   |     8 |  39   |  64   |  98   |   224 |  59   |                   96 |                    0 |           96 |          1.9 |
| CCAvg              |    1.9 |   1.7 |     0 |   0.7 |   1.5 |   2.5 |    10 |   1.8 |                  340 |                    0 |          340 |          6.8 |
| Mortgage           |   56.5 | 101.7 |     0 |   0   |   0   | 101   |   635 | 101   |                  291 |                    0 |          291 |          5.8 |
| Mortgage > 0       |  183.7 | 101.4 |    75 | 109   | 153   | 227   |   635 | 118   |                   70 |                    0 |           70 |          1.4 |

