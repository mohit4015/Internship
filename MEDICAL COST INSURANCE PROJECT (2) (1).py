#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing requied library
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Read dataset 
df = pd.read_csv("medical_cost_insurance.csv")


# In[3]:


#Dataset
df


# In[4]:


# Head of dataset
df.head()


# In[5]:


#   Name of all columns
df.columns


# In[6]:


#  Information of dataset
df.info()


# In[7]:


df['sex'].unique()


# In[8]:


df['smoker'].unique()


# In[9]:


df['region'].unique()


# In[10]:


df.shape


# In[11]:


df.describe()


# In[12]:


print("Find most important features relative to target")
corr = df.corr()
corr.sort_values(["charges"], ascending = False, inplace = True)
corr
print(corr['charges'])


# In[13]:


import warnings
warnings.filterwarnings("ignore")


# In[14]:


#Data Visualization
sns.distplot(df['charges'])


# In[15]:


df['charges'].skew()


# In[16]:


#  The above distribution tells us the charge is not normally distributed, 
#  It has high skewness. We will deal with it later. Also most of charges are below 50k


# In[17]:


#SEX
sns.countplot(df['sex'])


# In[18]:


#   Count of male and female is more or less equal


# In[19]:


g=sns.FacetGrid(df,col='sex')
g=g.map(sns.distplot,"charges")


# In[21]:


sns.catplot(y = 'charges',x = 'sex',data = df , kind = "swarm")


# In[22]:


# It does not look there is any difference in the charging based on sex of an individual. 
# We will explore further with other variable. 
# Till now insurance does not depend if you are male or female.


# In[23]:


#SMOKER
sns.catplot(y = 'charges',x = 'sex',col = 'smoker',data = df , kind = "swarm")


# In[24]:


sns.catplot(y = 'charges', x = 'sex', col = 'smoker', data = df , kind = "box")


# In[25]:


sns.catplot(y = 'charges',x = 'sex',col = 'smoker' , data = df , kind = "violin")


# In[26]:


df[(df['smoker']=='yes')&(df['sex']=='male')]['charges'].mean()


# In[27]:


df[(df['smoker']=='yes')&(df['sex']=='female')]['charges'].mean()


# In[28]:


#    From the diagram it is clear that smoking has high correlation with charges, if you smoke , 
#    then insurance paid will be high irrespective of gender. Further in the smoking categroy the mean of 
#    male smoker is higher than female smoker, 
#    need to compare with other variable like age and bmi to know why are they charged more than female.


# In[29]:


#Age
sns.distplot(df['age'])


# In[30]:


sns.scatterplot(x='age', y='charges',data=df)


# In[31]:


df['age'].describe()


# In[32]:


#  From the data it is visible that the minimum age is 18 and the maximum age is 64. 
#  So if we convert this numerical data into categorical data we can get better insight of the data.

#  SO lets convert the age into: 18yrs-26 yrs as young , 
#  27yrs-40yrs as adult,41yrs to 55yrs as senior adult and 56yrs to 64yrs as Elder


# In[35]:


df['age_cat']=np.NAN
lst = [df]

for col in lst:
    col.loc[(col['age'] >= 18) & (col['age'] <= 26), 'age_cat'] = 'Young'
    col.loc[(col['age'] > 26) & (col['age'] <= 40), 'age_cat'] = 'Adult'
    col.loc[(col['age'] > 40) & (col['age'] <= 55), 'age_cat'] = 'Senior Adult'
    col.loc[col['age'] > 55, 'age_cat'] = 'Elder'


# In[36]:


lst


# In[37]:


df['age_cat'].value_counts()


# In[38]:


sns.catplot(x='age_cat', y='charges',data=df,kind='bar')


# In[39]:


sns.catplot(x='age_cat', y='charges',data=df,kind='box')


# In[40]:


sns.catplot(x='age_cat', y='charges',data=df,kind='violin')


# In[41]:


sns.catplot(x='age_cat', y='charges',data=df,kind='swarm')


# In[42]:


#    From the above we can now easily see the corelation(ordinal) between age and charges,
#    as the age increase insurance charge increase. 
#    But each group has some high values need to understand the reason, using other features.


# In[43]:


sns.catplot(x='age_cat', y='charges',data=df,kind='box',col='smoker')


# In[44]:


sns.catplot(x='age_cat', y='charges',data=df,kind='box',col='smoker',hue='sex')


# In[45]:


sns.catplot(x='age_cat', y='charges',data=df,kind='violin',col='smoker',hue='sex')


# In[46]:


df[(df['age_cat']=='Adult')&(df['smoker']=='yes')&(df['sex']=='male')]['charges'].describe()


# In[47]:


df[(df['age_cat']=='Adult')&(df['smoker']=='yes')&(df['sex']=='female')]['charges'].describe()


# In[48]:


#   So from this we can confirm that non smoker of any age has low insurance charges in comparison to smokers.

#  In non smoker category, Males has slightly less insurance cost in comparison to female. 2)in smoker category, mean cost of male are higher than female.


# In[49]:


df.head()


# In[50]:


#BMI
sns.scatterplot(x='bmi',y='charges',data=df)


# In[51]:


sns.distplot(df['bmi'])


# In[52]:


df['bmi'].describe()


# In[53]:


#    If we see a BMI chart, there is 5 classes, 15 to 18 is underweight , 
#    19 to 24 is healthy ,25 to 29 is overweight ,30 to 39 is obese and greater than 40 is extremly obese.
#    So we will divide our data into that categories.


# In[54]:


df.loc[(df['bmi']>= 15)&(df['bmi']<19), 'bmi_cat'] = 'underweight'
df.loc[(df['bmi']>= 19)&(df['bmi']<25), 'bmi_cat'] = 'healthy'
df.loc[(df['bmi']>= 25)&(df['bmi']<30), 'bmi_cat'] = 'overweight'
df.loc[(df['bmi']>= 30)&(df['bmi']<40), 'bmi_cat'] = 'obese'
df.loc[(df['bmi']>= 40), 'bmi_cat'] = 'ext_obese'


# In[55]:


df['bmi_cat'].value_counts()


# In[56]:


df['bmi_cat'].unique()


# In[57]:


sns.catplot(x='bmi_cat',y='charges',kind='bar',data=df)


# In[58]:


sns.catplot(x='bmi_cat',y='charges',kind='box',data=df)


# In[59]:


sns.catplot(x='bmi_cat',y='charges',kind='swarm',data=df)


# In[60]:


sns.catplot(x='bmi_cat',y='charges',col='smoker',kind='box',data=df)


# In[61]:


sns.catplot(x='bmi_cat',y='charges',col='age_cat',hue='smoker',kind='box',data=df)


# In[62]:


#   In all age category one thing is common, 
#   if you are obese or more and you smoke the expense will be very high. 
#   The same is applicabe for overweight and healthy but the ratio of increase is lesser than obese. 
#   Further with increasein age, expense increases as we have seen earlier.


# In[63]:


#CHILDREN
sns.countplot(df['children'])


# In[64]:


sns.stripplot(x="children", y="charges", data=df, size = 5, jitter = True)


# In[65]:


df['children'].value_counts()


# In[66]:


df['children']=df['children'].map({0:'0',1:'1',2:'2',3:'3+',4:'3+',5:'3+'})


# In[67]:


df['children'].value_counts()


# In[68]:


sns.catplot(x='children',y='charges',data=df,kind='bar')


# In[69]:


#  if people tend to quit smoking or dont smoke if they have children


# In[70]:


sns.countplot(x='children',hue='smoker',data=df)


# In[71]:


df['smoker'].value_counts()


# In[72]:


#    No children present or not people do smoke


# In[73]:


df.head()


# In[74]:


#   Lets see if you have more children then do you get good time to take care of yourselve, 
#   eat well, do workout etc..or you dont get time and time and get overweight.


# In[75]:


sns.catplot(x='children',y='bmi',data=df,kind='swarm')


# In[76]:


sns.catplot(x='children',y='bmi',hue='smoker',data=df,kind='box')


# In[77]:


sns.catplot(x='children',y='charges',data=df,kind='bar')


# In[78]:


#   It looks like having children dont affect your BMI. 
#   However with increase in no of children medical expense increases. 
#   but if you see above distribution we can safely say having 0 or 1 child expnse is more or less equal, 
#   similarly having 2 or more child expense will be similar. 
#    So lets disribute into two part. 'less' if no of children is less than equal to 1 and 'more; if it is greater than equal to 2.


# In[79]:


df['child_cat']=np.NAN
df.loc[(df['children']=='0')|(df['children']=='1'), 'child_cat'] = 'less'
df.loc[(df['children']=='2')|(df['children']=='3+'), 'child_cat'] = 'more'


# In[80]:


df.head()


# In[81]:


#REGION
df['region'].value_counts()


# In[82]:


#REGION
df['region'].value_counts()


# In[83]:


#    distribution of smoker based on region


# In[84]:


sns.countplot(x='region',hue='smoker',data=df)


# In[85]:


#    Southeast has the maximum no of smoker followed by northeast , 
#    thus the charges will also be high for southeast followed by northeast.


# In[86]:


sns.catplot(x='region',y='charges',hue='smoker',data=df,kind='box')


# In[87]:


sns.catplot(x='region',y='charges',hue='smoker',data=df,kind='swarm')


# In[88]:


#    southeast had maximum no of smoker followed by northeast, charges are higher for southeast.


# In[89]:


sns.catplot(x='region',y='charges',col='smoker',data=df,kind='bar')


# In[90]:


sns.catplot(x='region',y='charges',col='bmi_cat',hue='smoker',data=df,kind='bar')


# In[91]:


sns.catplot(x='region',y='charges',data=df,kind='bar')


# In[92]:


# We cannot find any trend in charges based on region, when we are consider all the factors. 
#  thus we will one hot code this feature.


# In[93]:


df1=df.copy()


# In[94]:


#   FEATURE ENGINEERING
df1.head()


# In[95]:


df1=df1.drop(['age','bmi','children'],axis=1)


# In[96]:


df1['age_cat'].unique()


# In[97]:


df1['bmi_cat'].unique()


# In[98]:


df1.info()


# In[99]:


df1['age_cat']=df1['age_cat'].map({'Young':0, 'Adult':1, 'Senior Adult':2, 'Elder':3})
df1['bmi_cat']=df1['bmi_cat'].map({'underweight':0, 'healthy':1, 'overweight':2, 'obese':3,'ext_obese':4})
df1['child_cat']=df1['child_cat'].map({'less':0, 'more':1})


# In[101]:


#data_hot=data1[['sex','smoker','region','age_cat','bmi_cat','child_cat']]


# In[102]:


#data_hot1 = pd.get_dummies(data_hot)


# In[103]:


#data_final=pd.concat([data_hot1,data1['charges']],axis=1)


# In[104]:


df1.head()


# In[105]:


df1['smoker']=df1['smoker'].map({'no':0, 'yes':1})


# In[106]:


df1.head()


# In[107]:


df_hot = pd.get_dummies(df1['region'])
df_hot1 = pd.get_dummies(df1['sex'])


# In[108]:


df1


# In[109]:


df1=df1.drop(['region','sex'],axis=1)


# In[110]:


df_final=pd.concat([df1,df_hot,df_hot1],axis=1)


# In[111]:


df_final.head()


# In[112]:


#   Now the final part, we will work on charges. Let check the skewness.


# In[113]:


#Skewness of dependent variable
from scipy import stats
from scipy.stats import norm, skew #for some statistics

sns.distplot(df['charges'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(df_final['charges'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('charge distribution')

fig = plt.figure()
res = stats.probplot(df['charges'], plot=plt)
plt.show()


# In[114]:


df_final.head()


# In[115]:


df_final.shape


# In[116]:


#Modelling
Df_out=df_final['charges']
input_df=df_final.drop(['charges'],axis=1)


# In[117]:


from sklearn.model_selection import train_test_split


# In[118]:


from sklearn.preprocessing import PolynomialFeatures
quad = PolynomialFeatures (degree = 2)
x_quad = quad.fit_transform(input_df)


# In[119]:


r2_Scores = []
models1 = ['Linear Regression' , 'GradientBoosting Regression' ,'DecisionTreeRegressor','SVR','RandomForestRegressor','KNeighbours Regression']


# In[120]:


#df_y = result['Class']
# split the data into test and train by maintaining same distribution of output varaible 'y_true' [stratify=y_true]
X_train, X_test, y_train, y_test = train_test_split(input_df,Df_out,test_size=0.10)
# split the train data into train and cross validation by maintaining same distribution of output varaible 'y_train' [stratify=y_train]
#X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train,test_size=0.10)


# In[121]:


print('Number of data points in train data:', X_train.shape[0])
print('Number of data points in test data:', X_test.shape[0])
#print('Number of data points in cross validation data:', X_cv.shape[0])


# In[122]:


df_final.shape


# In[123]:


#Standarization
# X_train.describe()
y_train= y_train.values.reshape(-1,1)
y_test= y_test.values.reshape(-1,1)
#y_cv= y_cv.values.reshape(-1,1)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
y_train = sc_X.fit_transform(y_train)
y_test = sc_y.fit_transform(y_test)
#X_cv = sc_X.fit_transform(X_cv)
#y_cv = sc_y.fit_transform(y_cv)


# In[124]:


#Linear regression
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
print(lm)
# print the intercept
print(lm.intercept_)


# In[125]:


print(lm.coef_)


# In[126]:


lm_pred = lm.predict(X_test)
lm_pred= lm_pred.reshape(-1,1)
print("accuracy: "+ str(lm.score(X_test,y_test)*100) + "%")


# In[127]:


plt.figure(figsize=(15,8))
plt.scatter(y_test,lm_pred)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show()


# In[128]:


from sklearn import metrics
from sklearn import ensemble
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error, r2_score


# In[129]:


r2=r2_score(y_test, lm_pred)
print('MAE:', metrics.mean_absolute_error(y_test, lm_pred))
print('MSE:', metrics.mean_squared_error(y_test, lm_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, lm_pred)))
print('r2 score:',r2)
R2_Scores.append(r2)


# In[130]:


#   Gradient Boosting

params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)

clf.fit(X_train, y_train)


# In[131]:


clf_pred=clf.predict(X_test)
clf_pred= clf_pred.reshape(-1,1)


# In[132]:


r2=r2_score(y_test, clf_pred)
print('MAE:', metrics.mean_absolute_error(y_test, clf_pred))
print('MSE:', metrics.mean_squared_error(y_test, clf_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, clf_pred)))
print('r2 score:',r2)
R2_Scores.append(r2)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




