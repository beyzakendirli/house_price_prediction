#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Udacity House Price Prediction Project


# In[32]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


filename = "C:/Users/admKendirliB/data.csv"
print(filename)
df = pd.read_csv(filename)


# In[406]:


#let's get familiar with data


# In[33]:


df.head()


# In[29]:


df.shape


# In[30]:


df.columns


# In[34]:


df.info() 
#there are no null values. this is why no need to dropping null values or imputing new generated ones.
#there are just a few categorical features, we have mostly numerical features, which is good.


# In[411]:


df.count()


# In[9]:


df.describe()


# In[413]:


df.max()


# In[414]:


df.min()


# In[35]:


print(df.groupby(['bedrooms'])['bedrooms'].count().sort_values(ascending=False)) 
#houses mostly have bedrooms between 2 to 5.


# In[36]:


plt.hist(df['bedrooms']);


# In[39]:


print(df.groupby(['bathrooms'])['bathrooms'].count().sort_values(ascending=False))
#houses mostly have bathrooms between 1 to 4.


# In[40]:


plt.hist(df['bathrooms']);


# In[42]:


print(df.groupby(['floors'])['floors'].count().sort_values(ascending=False)) 
#houses mostly have bathrooms between 1 to 2.


# In[43]:


plt.hist(df['floors']);


# In[44]:


print(df.groupby(['condition'])['condition'].count().sort_values(ascending=False)) 
#houses mostly have been considered in a good condition.


# In[45]:


plt.hist(df['condition']);


# In[46]:


print(df.groupby(['view'])['view'].count().sort_values(ascending=False)) 
#we can say that big proportion of houses has not a great view.


# In[47]:


plt.hist(df['view']);


# In[48]:


print(df.groupby(['waterfront'])['waterfront'].count().sort_values(ascending=False)) 


# In[421]:


print(df.groupby(['country'])['country'].count().sort_values(ascending=False)) 


# In[422]:


print(df.groupby(['city'])['city'].count().sort_values(ascending=False)) 


# In[423]:


#data visualization for better understanding & correlation analysis


# In[424]:


df.hist();


# In[425]:


sns.heatmap(df.corr(), annot= True, fmt='.2f');
#bedrooms - price correlation equals to 0.20 (bedrooms can be used as a feature in prediction)
#bathrooms - price correlation equals to 0.33 (bathrooms can be used as a feature in prediction)
#floors - price correlation equals to 0.15 (floors can be a misleading feature in prediction)
#condition - price correlation equals to 0.03 (condition can be a misleading feature in prediction)
#view - price correlation equals to 0.23 (view can be used as a feature in prediction)
#sqft_living - price correlation equals to 0.43 (sqft_living can be used as a feature in prediction)


# In[426]:


#let's answer some questions to gain insights


# In[427]:


# Q1: what percentage of the houses have been renovated?


# In[428]:


def renovated_or_not(x):
    
    #create a renovated flag to show if the house has been renovated or not
    df['renovated_flag'] = df['yr_renovated'].apply(lambda x: False if x == 0 else True)
    
    #count the number of houses which have been renovated
    renovated = df.groupby(['renovated_flag'])['renovated_flag'].count()[True]
    all = df.count()['renovated_flag']
    
    print("Percentage of houses which have been renovated is ", '{:.2f}%'.format(renovated/all))


# In[429]:


renovated_or_not(df)


# In[430]:


# Q2: Does renovation has a big impact on the condition of a house?


# In[431]:


def renovation_impact(x):
    avg_yes_renovation = df.groupby(['renovated_flag'])['condition'].mean()[True]
    avg_no_renovation = df.groupby(['renovated_flag'])['condition'].mean()[False]
    avg_all = df['condition'].mean()
    
     
    if (avg_yes_renovation > avg_no_renovation) & (avg_yes_renovation > avg_all):
        print("Renovation has a big impact on the condition of a house.\n Average condition point of a house which has been renovated is", '{:.2f}'.format(avg_yes_renovation) , "\n Average condition point of a house which has not been renovated is", '{:.2f}'.format(avg_no_renovation) ,  "\n Average condition point of all houses no matter if it has been renovated or not is", '{:.2f}'.format(avg_all))
    else:
        print("Renovation has not a big impact on the condition of a house.\n Average condition point of a house which has been renovated is", '{:.2f}'.format(avg_yes_renovation) , "\n Average condition point of a house which has not been renovated is", '{:.2f}'.format(avg_no_renovation) ,  "\n Average condition point of all houses no matter if it has been renovated or not is", '{:.2f}'.format(avg_all))


# In[432]:


renovation_impact(df)


# In[433]:


# Q3: How many households are using more than half of their land as living space and what is the percentage?


# In[434]:


def living_space(x):
    df['surface'] = df['sqft_living'] / df['sqft_lot']
    df['surface'] = df['surface'].apply(lambda x: 'large living space' if x > 0.50 else 'narrow living space')

    large = df.groupby(['surface'])['surface'].count()['large living space']
    all = df['surface'].count()
    
    print("Number of households who are using more than half of their land as living space is ", '{:.2f}'.format(large) , "and the percentage is ", '{:.2f}%'.format(large/all))


# In[435]:


living_space(df)


# In[436]:


#let's perform the model for prediction


# In[96]:


def find_optimal_lm_mod(X, y, cutoffs, test_size = .45, random_state=42, plot=True):
    
    r2_scores_test, r2_scores_train, num_feats, results = [], [], [], dict() 
    
    for cutoff in cutoffs:
        
        
        #reduce X matris
        reduce_X = X.iloc[:, np.where((X.sum() > int(cutoff)) == True)[0]]
        num_feats.append(reduce_X.shape[1])
        
        #split the data into train and test
        X_train, X_test, y_train, y_test = train_test_split(reduce_X, y, test_size = .45, random_state=42)
        
        #global variable definition is crucial if you'll use the same variable out side of this function
        global lm_model 


        #fit the model and obtain pred response
        lm_model = LinearRegression(normalize=True)
        lm_model.fit(X_train, y_train)
        y_test_preds = lm_model.predict(X_test)
        y_train_preds = lm_model.predict(X_train)
        
        #append the r2 value from the test set
        r2_scores_test.append(r2_score(y_test, y_test_preds))
        r2_scores_train.append(r2_score(y_train, y_train_preds))
        results[str(cutoff)] = r2_score(y_test, y_test_preds)
        
    if plot:
        plt.plot(num_feats, r2_scores_test, label= 'Test', alpha=.5)
        plt.plot(num_feats, r2_scores_train, label= 'Train', alpha=.5)
        plt.xlabel('Number of features')
        plt.ylabel('Rsquared')
        plt.title('Rsquared by Number of features')
        plt.legend(loc=1)
        plt.show()
        
        
    best_cutoff = max(results, key=results.get)
    
    #reduce X matris
    reduce_X = X.iloc[:, np.where((X.sum() > int(best_cutoff)) == True)[0]]
    
    #split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.45, random_state = 42)
    
    #fit the model
    lm_model = LinearRegression(normalize=True)
    lm_model.fit(X_train, y_train)
    
    return r2_scores_test, r2_scores_train, lm_model, X_train, X_test, y_train, y_test


# In[109]:


#cutoffs here pertains to the number of missing values allowed in the used columns
#therefore, lower values for the cutoff provides more predictors in the model
X = df[['bedrooms','bathrooms','view','sqft_living']]
y = df['price']

cutoffs = [5000, 3500, 2500, 1000, 100, 50, 30, 25]

r2_scores_test, r2_scores_train, lm_model, X_train, X_test, y_train, y_test = find_optimal_lm_mod(X, y, cutoffs)


# In[110]:


#this function helps us to see which features are more efficient for model to predict the price
def coef_weights(coefficients, X_train):
    
    coefs_df = pd.DataFrame()
    coefs_df['est_int'] = X_train.columns
    coefs_df['coefs'] = lm_model.coef_
    coefs_df['abc_coefs'] = np.abs(lm_model.coef_)
    coefs_df = coefs_df.sort_values('abc_coefs', ascending=False)
    return coefs_df


# In[111]:


coefs_df = coef_weights(lm_model.coef_, X_train)


# In[112]:


coefs_df.head()


# In[113]:


y_test_preds = lm_model.predict(X_test)


# In[114]:


#r square is a statistical measure that represents the goodness of fit. the best possible score is 1.0
"The r square score for the model using only quantitative variables was {} on {} values".format(r2_score(y_test,y_test_preds), len(y_test))


# In[115]:


y_train_preds = lm_model.predict(X_train)


# In[116]:


#r square is a statistical measure that represents the goodness of fit. the best possible score is 1.0
"The r square score for the model using only quantitative variables was {} on {} values".format(r2_score(y_train,y_train_preds), len(y_train))


# In[ ]:




