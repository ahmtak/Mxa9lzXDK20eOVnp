# Mxa9lzXDK20eOVnp

The accuracy is 93.5%

Most important feature is = duration
Second most important feature is = balance

If a customer's duration is bigger than 451 , this customer will highly subscribe to a term deposit.
If a customer's balance is bigger than 1418 , this customer will highly subscribe to a term deposit.

The company must focus these customers more than others.



***************************************************************************************************************
import pandas as pd

data_set = pd.read_csv('term-deposit-marketing-2020.csv')

data_set.head()

	age 	job 	marital 	education 	default 	balance 	housing 	loan 	contact 	day 	month 	duration 	campaign 	y
0 	58 	management 	married 	tertiary 	no 	2143 	yes 	no 	unknown 	5 	may 	261 	1 	no
1 	44 	technician 	single 	secondary 	no 	29 	yes 	no 	unknown 	5 	may 	151 	1 	no
2 	33 	entrepreneur 	married 	secondary 	no 	2 	yes 	yes 	unknown 	5 	may 	76 	1 	no
3 	47 	blue-collar 	married 	unknown 	no 	1506 	yes 	no 	unknown 	5 	may 	92 	1 	no
4 	33 	unknown 	single 	unknown 	no 	1 	no 	no 	unknown 	5 	may 	198 	1 	no

columns = data_set.columns.to_list()

columns

for column in columns:

    print(column , data_set[column].unique())

age [58 44 33 47 35 28 42 43 41 29 53 57 51 45 60 56 32 25 40 39 52 46 36 49
 59 37 50 54 55 48 24 38 31 30 27 34 23 26 61 22 21 20 66 62 83 75 67 70
 65 68 64 69 72 71 19 76 85 63 90 82 73 74 78 80 94 79 77 86 95 81]
job ['management' 'technician' 'entrepreneur' 'blue-collar' 'unknown'
 'retired' 'admin' 'services' 'self-employed' 'unemployed' 'housemaid'
 'student']
marital ['married' 'single' 'divorced']
education ['tertiary' 'secondary' 'unknown' 'primary']
default ['no' 'yes']
balance [  2143     29      2 ...   7222   3402 102127]
housing ['yes' 'no']
loan ['no' 'yes']
contact ['unknown' 'cellular' 'telephone']
day [ 5  6  7  8  9 12 13 14 15 16 19 20 21 23 26 27 28 29 30  2  3  4 11 17
 18 24 25  1 10 22 31]
month ['may' 'jun' 'jul' 'aug' 'oct' 'nov' 'dec' 'jan' 'feb' 'mar' 'apr']
duration [ 261  151   76 ... 1880 1460 2219]
campaign [ 1  2  3  5  4  6  7  8  9 10 11 12 13 19 14 24 16 32 18 22 15 17 25 21
 43 51 63 41 26 28 55 50 38 23 20 29 31 37 30 46 27 58 33 35 34 36 39 44]
y ['no' 'yes']

logic_dictionary = {'no' : 0 , 'yes' : 1}

data_set.y = data_set.y.map(logic_dictionary)

data_set.head()

	age 	job 	marital 	education 	default 	balance 	housing 	loan 	contact 	day 	month 	duration 	campaign 	y
0 	58 	management 	married 	tertiary 	no 	2143 	yes 	no 	unknown 	5 	may 	261 	1 	0
1 	44 	technician 	single 	secondary 	no 	29 	yes 	no 	unknown 	5 	may 	151 	1 	0
2 	33 	entrepreneur 	married 	secondary 	no 	2 	yes 	yes 	unknown 	5 	may 	76 	1 	0
3 	47 	blue-collar 	married 	unknown 	no 	1506 	yes 	no 	unknown 	5 	may 	92 	1 	0
4 	33 	unknown 	single 	unknown 	no 	1 	no 	no 	unknown 	5 	may 	198 	1 	0

logic_dictionary = {'no' : 0 , 'yes' : 1}

data_set.default = data_set.default.map(logic_dictionary)

data_set.housing = data_set.housing.map(logic_dictionary)

data_set.loan = data_set.loan.map(logic_dictionary)

print(data_set.dtypes)

data_set.head()

age           int64
job          object
marital      object
education    object
default       int64
balance       int64
housing       int64
loan          int64
contact      object
day           int64
month        object
duration      int64
campaign      int64
y             int64
dtype: object

	age 	job 	marital 	education 	default 	balance 	housing 	loan 	contact 	day 	month 	duration 	campaign 	y
0 	58 	management 	married 	tertiary 	0 	2143 	1 	0 	unknown 	5 	may 	261 	1 	0
1 	44 	technician 	single 	secondary 	0 	29 	1 	0 	unknown 	5 	may 	151 	1 	0
2 	33 	entrepreneur 	married 	secondary 	0 	2 	1 	1 	unknown 	5 	may 	76 	1 	0
3 	47 	blue-collar 	married 	unknown 	0 	1506 	1 	0 	unknown 	5 	may 	92 	1 	0
4 	33 	unknown 	single 	unknown 	0 	1 	0 	0 	unknown 	5 	may 	198 	1 	0

data_set.groupby('y').mean()

	age 	default 	balance 	housing 	loan 	day 	duration 	campaign
y 								
0 	40.599208 	0.020483 	1249.752533 	0.608209 	0.176531 	16.032072 	221.408042 	2.918688
1 	39.844959 	0.016920 	1588.495856 	0.505525 	0.131215 	15.827003 	682.958564 	2.414365

month_dictionary = {'may':5, 'jun':6, 'jul':7, 'aug':8, 'oct':9 ,'nov':11, 'dec':12, 'jan':1, 'feb':2, 'mar':3, 'apr':4}

data_set.month = data_set.month.map(month_dictionary)

data_set.head()

	age 	job 	marital 	education 	default 	balance 	housing 	loan 	contact 	day 	month 	duration 	campaign 	y
0 	58 	management 	married 	tertiary 	0 	2143 	1 	0 	unknown 	5 	5 	261 	1 	0
1 	44 	technician 	single 	secondary 	0 	29 	1 	0 	unknown 	5 	5 	151 	1 	0
2 	33 	entrepreneur 	married 	secondary 	0 	2 	1 	1 	unknown 	5 	5 	76 	1 	0
3 	47 	blue-collar 	married 	unknown 	0 	1506 	1 	0 	unknown 	5 	5 	92 	1 	0
4 	33 	unknown 	single 	unknown 	0 	1 	0 	0 	unknown 	5 	5 	198 	1 	0

education_dictionary = {'tertiary':3, 'secondary':2, 'unknown':0, 'primary':1}

data_set.education = data_set.education.map(education_dictionary)

​

marital_dictionary = {'married':1, 'single':0, 'divorced':2}

data_set.marital = data_set.marital.map(marital_dictionary)

​

contact_dictionary = {'unknown':0, 'cellular':1, 'telephone':2}

data_set.contact = data_set.contact.map(contact_dictionary)

​

job_dictionary = {'management':10, 'technician':9, 'entrepreneur':7, 'blue-collar':8, 'unknown':0,

 'retired':4, 'admin':3, 'services':11, 'self-employed':6, 'unemployed':1, 'housemaid':2,

 'student':5 }

data_set.job = data_set.job.map(job_dictionary)

​

data_set.head()

​

	age 	job 	marital 	education 	default 	balance 	housing 	loan 	contact 	day 	month 	duration 	campaign 	y
0 	58 	10 	1 	3 	0 	2143 	1 	0 	0 	5 	5 	261 	1 	0
1 	44 	9 	0 	2 	0 	29 	1 	0 	0 	5 	5 	151 	1 	0
2 	33 	7 	1 	2 	0 	2 	1 	1 	0 	5 	5 	76 	1 	0
3 	47 	8 	1 	0 	0 	1506 	1 	0 	0 	5 	5 	92 	1 	0
4 	33 	0 	0 	0 	0 	1 	0 	0 	0 	5 	5 	198 	1 	0

data_set.groupby('age').mean()

	job 	marital 	education 	default 	balance 	housing 	loan 	contact 	day 	month 	duration 	campaign 	y
age 													
19 	5.000000 	0.000000 	0.916667 	0.000000 	961.666667 	0.000000 	0.000000 	1.250000 	11.166667 	3.666667 	161.916667 	5.000000 	0.166667
20 	5.727273 	0.090909 	1.727273 	0.000000 	1057.681818 	0.590909 	0.090909 	0.727273 	16.181818 	4.636364 	201.045455 	2.363636 	0.181818
21 	6.205128 	0.128205 	1.717949 	0.051282 	729.820513 	0.589744 	0.153846 	0.666667 	16.205128 	4.589744 	186.923077 	2.076923 	0.179487
22 	6.873239 	0.126761 	2.000000 	0.000000 	777.464789 	0.802817 	0.084507 	0.619718 	14.676056 	4.859155 	254.478873 	2.225352 	0.126761
23 	7.382550 	0.174497 	1.919463 	0.026846 	883.744966 	0.845638 	0.127517 	0.543624 	15.147651 	5.087248 	295.288591 	2.429530 	0.114094
... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	...
85 	4.000000 	1.333333 	1.000000 	0.000000 	6608.333333 	0.000000 	0.000000 	1.333333 	20.000000 	2.666667 	312.000000 	2.000000 	1.000000
86 	4.000000 	1.000000 	1.000000 	0.000000 	0.000000 	0.000000 	0.000000 	1.000000 	15.000000 	4.000000 	141.000000 	12.000000 	0.000000
90 	4.000000 	2.000000 	2.000000 	0.000000 	1.000000 	0.000000 	0.000000 	1.000000 	13.000000 	2.000000 	152.000000 	3.000000 	1.000000
94 	4.000000 	2.000000 	2.000000 	0.000000 	1234.000000 	0.000000 	0.000000 	1.000000 	3.000000 	3.000000 	212.000000 	1.000000 	0.000000
95 	4.000000 	2.000000 	1.000000 	0.000000 	2282.000000 	0.000000 	0.000000 	2.000000 	21.000000 	4.000000 	207.000000 	17.000000 	1.000000

70 rows × 13 columns

data_set.groupby('y').mean()

	age 	job 	marital 	education 	default 	balance 	housing 	loan 	contact 	day 	month 	duration 	campaign
y 													
0 	40.599208 	7.63066 	0.851013 	2.037462 	0.020483 	1249.752533 	0.608209 	0.176531 	0.727442 	16.032072 	6.037112 	221.408042 	2.918688
1 	39.844959 	7.48895 	0.780387 	2.167127 	0.016920 	1588.495856 	0.505525 	0.131215 	0.885704 	15.827003 	5.632251 	682.958564 	2.414365

boxplot = data_set.boxplot(column=['age'])

def get_age_group(age):

    if(31 > age):

        return 0

    elif(39 > age):

        return 1

    elif(49 > age):

        return 2

    elif(70 > age):

        return 3

    else:

        return 4

for i in range(len(data_set.age)):

    data_set.age[i] = get_age_group(data_set.age[i])

​

boxplot_balance = data_set.boxplot(column=['balance'], return_type=None)

print(boxplot_balance)

AxesSubplot(0.125,0.125;0.775x0.755)

'''

print(data_set.mean())

for i in range(len(data_set.age)):

    data_set.age[i] = get_age_group(data_set.age[i])

print(data_set.mean())

'''

def get_balance_group(balance):

    if(50>balance):

        return 0

    elif(250 > balance):

        return 1

    elif(500 > balance):

        return 2

    elif(1000 > balance):

        return 3

    else:

        return 4

​

​

for i in range(len(data_set.balance)):

    data_set.balance[i] = get_balance_group(data_set.balance[i])

 

boxplot_duration = data_set.boxplot(column=['duration'])

def get_duration_group(duration):

    if(50>duration):

        return 0

    elif(200 > duration):

        return 1

    elif(300 > duration):

        return 2

    elif(500 > duration):

        return 3

    else:

        return 4

​

​

for i in range(len(data_set.duration)):

    data_set.duration[i] = get_duration_group(data_set.duration[i])

 

​

data_set.head()

	age 	job 	marital 	education 	default 	balance 	housing 	loan 	contact 	day 	month 	duration 	campaign 	y
0 	3 	10 	1 	3 	0 	2143 	1 	0 	0 	5 	5 	261 	1 	0
1 	2 	9 	0 	2 	0 	29 	1 	0 	0 	5 	5 	151 	1 	0
2 	1 	7 	1 	2 	0 	2 	1 	1 	0 	5 	5 	76 	1 	0
3 	2 	8 	1 	0 	0 	1506 	1 	0 	0 	5 	5 	92 	1 	0
4 	1 	0 	0 	0 	0 	1 	0 	0 	0 	5 	5 	198 	1 	0

hr_vars=data_set.columns.to_list()[:]

​

yy=['y']

XX=[i for i in hr_vars if i not in yy]

​

​

from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

​

#from sklearn.svm import SVC

​

rfc_model = RandomForestClassifier()

​

lr_model = LogisticRegression()

​

rfe = RFE(rfc_model, 8)

rfe = rfe.fit(data_set[XX], data_set[yy])

print(rfe.support_)

print(rfe.ranking_)

​

cols = []

for i in range(len(rfe.support_)):

    if(rfe.support_[i]):

        cols.append(XX[i])

print(cols)

​

cols_false = []

for i in range(len(rfe.support_)):

    if(rfe.support_[i]!= 1):

        cols_false.append(XX[i])

print(cols_false)


[ True  True False  True False  True False False False  True  True  True
  True]
[1 1 2 1 6 1 3 5 4 1 1 1 1]
['age', 'job', 'education', 'balance', 'day', 'month', 'duration', 'campaign']
['marital', 'default', 'housing', 'loan', 'contact']

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import ShuffleSplit



log_reg = LogisticRegression()



cv_shuffle = ShuffleSplit(n_splits=5, test_size=0.2)



x_all = data_set.iloc[:,:-1]

y = data_set.iloc[:,-1]



scores = cross_val_score(log_reg, x_all, y, cv=cv_shuffle)



print(scores.mean())

scores



from sklearn.svm import SVC

svc = SVC()





x_all = data_set.iloc[:,:-1]

y = data_set.iloc[:,-1]



cv_shuffle = ShuffleSplit(n_splits=5, test_size=0.2)



scores = cross_val_score(svc, x_all, y, cv=cv_shuffle)

print(scores.mean())

scores

x_cols = data_set[cols].values

y = data_set.y





from sklearn.model_selection import cross_val_score

from sklearn.model_selection import ShuffleSplit



log_reg = LogisticRegression()



cv_shuffle = ShuffleSplit(n_splits=5, test_size=0.2)



scores = cross_val_score(log_reg, x_cols, y, cv=cv_shuffle)

print(scores.mean())

scores

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import ShuffleSplit

from sklearn.ensemble import RandomForestClassifier



rfc_reg = RandomForestClassifier()



cv_shuffle = ShuffleSplit(n_splits=5, test_size=0.2)

​

x_cols = data_set[cols].values

x_all = data_set.iloc[:,:-1]

y = data_set.iloc[:,-1]



scores = cross_val_score(rfc_reg, x_all, y, cv=cv_shuffle)



print(scores.mean())

scores

0.93665

array([0.937125, 0.93925 , 0.93625 , 0.93475 , 0.935875])



rfc_clf.feature_importances_

array([0.04212896, 0.06201822, 0.02557882, 0.0328286 , 0.00326429,
       0.13823357, 0.01962799, 0.01216962, 0.01985525, 0.11581987,
       0.07540354, 0.40284345, 0.05022782])

most_important_feature_index = -1

most_important_feature_value = -1

second_most_imp_feature_index = -1

second_most_imp_feature_value = -1

​

for i in range(len(rfc_clf.feature_importances_)):

    if(rfc_clf.feature_importances_[i] > most_important_feature_value):

        second_most_imp_feature_value = most_important_feature_value

        second_most_imp_feature_index = most_important_feature_index

        

        most_important_feature_value = rfc_clf.feature_importances_[i]

        most_important_feature_index = i

        

print("Most important feature is =" , data_set.columns[most_important_feature_index])

print("Second most important feature is =" , data_set.columns[second_most_imp_feature_index])

Most important feature is = duration
Second most important feature is = balance

data_set.groupby(["y"]).mean()

	age 	job 	marital 	education 	default 	balance 	housing 	loan 	contact 	day 	month 	duration 	campaign
y 													
0 	1.615082 	7.63066 	0.851013 	2.037462 	0.020483 	1249.752533 	0.608209 	0.176531 	0.727442 	16.032072 	6.037112 	221.408042 	2.918688
1 	1.494130 	7.48895 	0.780387 	2.167127 	0.016920 	1588.495856 	0.505525 	0.131215 	0.885704 	15.827003 	5.632251 	682.958564 	2.414365

(221+682)/2

451.5

(1249+1588)/2

1418.5

The accuracy is 93.5%
Most important feature is = duration
Second most important feature is = balance
If a customer's duration is bigger than 451 , this customer will highly subscribe to a term deposit.
If a customer's balance is bigger than 1418 , this customer will highly subscribe to a term deposit.

The company must focus these customers more than others.

​


