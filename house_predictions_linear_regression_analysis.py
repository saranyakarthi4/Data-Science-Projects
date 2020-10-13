import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
import statsmodel.api as sm
%matplotlib inline

df_train = pd.read_csv('/Users/skarthi/Documents/Python_Scripts/Data/house_pred_train.csv')
df_test = pd.read_csv('/Users/skarthi/Documents/Python_Scripts/Data/house_pred_test.csv')
print (df_train.head(2))

x_train = ['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
       'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
       'SaleCondition']
est =sm.OLS(df1['SalePrice'],df1[x_train])
est2 =est.fit()
est2.summary()

# to get no of rows and columns
print (df_train.shape)
print ('Number of rows in dataset',len(df_train))
#######################
# Data Analysis
# 1. Missing values, 2. All the numerical values 
# 3. Distribution of numerical values 4. categorical variables
# 5. cardiablilty of categorical variables
# 6. outliers  7. relationship b/w dependent and independent variable
#######################

# to get the column list
print (df_train.columns)

df_train_withnull =[]
# 1. find the columns that has atleast 1 null value
for col in df_train.columns:
    if df_train[col].isnull().sum() > 0: 
    # which means if it has atleast 1 null value
        df_train_withnull.append(col)
print (df_train_withnull)  # shows the columns that has atleast 1 null value

# finding the % of missing value in those columns
for col in df_train_withnull:
    print ('{} having {} % of missing values '.format(col,np.round(df_train[col].isnull().mean(),4)))

# 2. Finding Numerical values
numerical_columns = [col for col in df_train.columns if df_train[col].dtypes != 'O' ]
print ('printing all numerical columns from the dataset', df_train[numerical_columns])

# temporal variables examples year columns
year_columns = [col for col in df_train.columns if 'Yr' in col or 'Year' in col]
print (df_train[year_columns])

# avg sale price by year sold
df_train.groupby('YrSold')['SalePrice'].median().plot()
plt.ylabel('Sale Price') 
plt.title('Sale PRice by year sold')

# plotting sale price versus difference between yrsold and other year columns
for col in df_train[year_columns].columns:
    if col != 'YrSold':
        df_train['year_diff'] = df_train['YrSold'] - df_train[col]
        plt.scatter(df_train['year_diff'],df_train['SalePrice'])
        plt.xlabel(col)
        plt.ylabel('Sale Price') 
        plt.show()

# finding numerical variables are of 2 types
# discrete and continuous varibale variables 
discrete_columns = [col for col in numerical_columns if col not in [year_columns]+['Id']+['YrSold'] and len(df_train[col].unique()) < 25 ]
print (discrete_columns,len(discrete_columns),year_columns)

for col in discrete_columns:
    #copy_ds = df_train.copy()
    df_train.groupby(col)['SalePrice'].median().plot.bar()
    plt.ylabel('Sale Price') 
    plt.xlabel(col)
    plt.show()

# continuous variable
continuous_columns = [col for col in numerical_columns if col not in discrete_columns+year_columns+['Id']]
print (continuous_columns,len(continuous_columns))

# 3. Distribution of Conti variables creating histograms using continuous variables
for col in continuous_columns:
    df_train[col].hist(bins=25)
    plt.ylabel('Count') 
    plt.xlabel(col)
    plt.show()


##### Now, we need to transformed the skewed data into normal distributed data to
# apply regression algorithm
# we will do doing Logarthimic transformation
for col in continuous_columns:
    data = df_train.copy()
    if 0 in data[col].unique():
        pass
    else:
        data[col]=np.log(data[col])
        data['SalePrice'] = np.log(data['SalePrice'])
        plt.scatter(data[col], data['SalePrice'])
        plt.xlabel(col)
        plt.ylabel('Sale Price')
        plt.show()

# 4. Categorical variables
categorical_columns = [col for col in df_train.columns if df_train[col].dtypes == 'O' ]
print ('printing all numerical columns from the dataset', df_train[categorical_columns])

#  5. Cardiablity of Categorical variables is finding the unique values of eCH COLUMN 
for col in categorical_columns:
    print ('for column {} , the number of values are {}'.format(col,len(df_train[col].unique())))

################6.    OUTLIERS
# we can use box plot to identify
for col in continuous_columns:
    data = df_train.copy()
    if 0 in data[col].unique():
        pass
    else:
        data[col]=np.log(data[col])
        data.boxplot(column = col)
        plt.title(col)
        plt.show()

# 7. relationship b/w categorical variable and and dependent variable(saleprice)

for col in categorical_columns:
    #copy_ds = df_train.copy()
    df_train.groupby(col)['SalePrice'].median().plot.bar()
    plt.ylabel('Sale Price') 
    plt.xlabel(col)
    plt.show()



###############    Feature Engineering      ###############
# 1. Handling Missing Values
# 2. Temporal variables
# 3. Categorical variables : remove rare labels
# 4. standarize the values of the variables to the same range

# 1. A) Missing value in categorical value
for col in categorical_columns:
    if df_train[col].isnull().sum() > 0:
        np.round(df_train[col].isnull().mean(),4)
        print ('missing null vales in {} is {}'.format(col,np.round(df_train[col].isnull().mean(),4)))

# filling the missing values with 'unknown' label
for col in categorical_columns:
    if df_train[col].isnull().sum() >0 :
        df_train[col] = df_train[col].fillna('Unknown')
print ('After filling the missing values in categorical variables ')
print (' there are no null values in the category',df_train[categorical_columns].isnull().sum())

# 1. B) Missing value in Numerical Columns
# finding NULL values in numrical columns
for col in numerical_columns:
    if df_train[col].isnull().sum() > 0:
        print ("Null values in column {} is {}".format(col,np.round(df_train[col].isnull().mean(),4)))

# fill the missing values in numerical columns
for col in numerical_columns:
    # create a new column with suffix 'with_null' in the dataset to determine where we repalced null valeus in the original columns
    # np.where will search the column for NULL values and if the value is null then 1 else 0
    df_train[col+'_with_null'] = np.where(df_train[col].isnull,1,0)
    df_train[col].fillna(df_train[col].median(),inplace=True)
print ('checking all numerical columns for null values',df_train[numerical_columns].isnull().sum() )      


# 2 Handing Temporal variables - Date time variables
for col in year_columns:
    if col !='YrSold':
        df_train[col+'_yrs'] = df_train['YrSold'] - df_train[col]
print (df_train.head())

# For the numerical columns, if the diftribution is skwed then we need to 
# transform into logarithmic distribution
# these numerical columns are having the hist values not starting from 0, 
# we can ignore the columns if it has values 0 becasue np.log(0) throws error
columns =['LotFrontage','LotArea','1stFlrSF','GrLivArea','SalePrice']
for col in columns:
    df_train[col] = np.log(df_train[col])
print (df_train.head())

# 3. handling Rare Categorical variables
# if any categorical columns has any specific value/category rarely occurs in the data (< 1% of dataset rows), 
# we are gonna group them into value called 'rare_category'
# reason of doing this if we have multiple categorical column having  many 'rare_category' value ,
# we can ignore them for prediction as a good practice
# for ex: for 'MSZoning' column , the value of 'C (all) ' in tha dataset is just 10 rows, so we c
# are converting into 'rare_category' , simillary for all columns
print (df_train['MSZoning'].value_counts())


for col in categorical_columns:
    temp = (df_train.groupby(col)['SalePrice'].count())/len(df_train) # gives the % of the column value in dataset
    temp_df = temp[temp > 0.01].index  # get only the values of the columns which is >1% in the dataset
    df_train[col]= np.where(df_train[col].isin(temp_df) , df_train[col] , 'Rare_category')
    # if the value in the column is < 1% then rename it as 'Rare_category' else use the column value itself


###############    Encoding
for col in categorical_columns:    
    # for each columns, based on avg_saleprice of each column value and sort it asc , '.index' would return all column values
    label_ordered = (df_train.groupby([col])['SalePrice'].mean().sort_values().index) 
    # for MSZoning column, this line would return {'Rare_category': 0, 'RM': 1, 'RH': 2, 'RL': 3, 'FV': 4}
    label_ordered = ({k:i for i,k in enumerate(label_ordered,0)}) 
    # if the dictionary key matches with column value then replace with dictionary value , 
    # this converts all catgorical values into numerics
    df_train[col] = df_train[col].map(label_ordered)

###############   Feature Scaling

# All the categorical an dnumerical values needs to numbers before doing Feature Scaling
# for this, we dont need to include key attributes like 'Id' and dependent variable like' SalePrice'
# 2 types of scaling, StandardScaler and MinMaxScaler
# StandardScaler works for column shaving -ve values and uses Standard Normal Distribution
# MinMaxScaler uses 0 to 1 values, we need to try both the scaler methods and see which gives better results

feature_scale = [col for col in df_train.columns if col not in ['Id','SalePrice']]

scaler = MinMaxScaler()
scaler.fit(df_train[feature_scale])
# This would transform all the categorical and numerical values to Array of values between 0 and 1
scaler.transform(df_train[feature_scale])
scaler_df = pd.DataFrame(scaler.transform(df_train[feature_scale]),columns=feature_scale)
data = pd.concat([df_train[['Id', 'SalePrice']],
                  pd.DataFrame(scaler.transform(df_train[feature_scale]), columns=feature_scale)],
                    axis=1)

# write this to csv data.to_csv('xyz.csv',index=False)


######## Apply Lasso Regression Model
y_train = df_train[['SalePrice']]
x_train = df_train.drop(['Id','SalePrice'],axis = 1)









