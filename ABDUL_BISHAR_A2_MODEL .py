#!/usr/bin/env python
# coding: utf-8

# In[12]:


# Student Name : Abdul Bishar
# Cohort       : 5


################################
####### Import Packages ########
###############################


import pandas as pd                                                   # data science essentials
import numpy as np                                                    # Numeric Python
import matplotlib.pyplot as plt                                       # data viz
import seaborn as sns                                                 # Enhance data viz   


import statsmodels.formula.api as smf                                 # stats model for regression
from sklearn.model_selection import train_test_split,cross_val_score  # Train Tests Split data 
from sklearn.neighbors import KNeighborsClassifier                    # KNN for Classification
from sklearn.preprocessing import StandardScaler                      # Standard scaler
from sklearn.linear_model import LogisticRegression                   # Logistic Regression
from sklearn.metrics import roc_auc_score                             # AUC curve
from sklearn.ensemble import GradientBoostingClassifier               # Gradient Boosting Regressor
from sklearn.model_selection import GridSearchCV                      # GridSearch
from sklearn.metrics import confusion_matrix                          # Confusion Matrix
from sklearn.metrics import make_scorer                               # customizable scorer
 
# libraries for classification trees
from sklearn.tree import DecisionTreeClassifier                       # classification trees
from sklearn.ensemble import GradientBoostingClassifier               # Gradient Boosting Classifier
from sklearn.ensemble import RandomForestClassifier                   # Random Forest for classification
from sklearn.tree import export_graphviz                              # exports graphics
from sklearn.externals.six import StringIO                            # saves objects in memory
from IPython.display import Image                                     # displays on frontend
from sklearn.tree import export_graphviz                              # exports graphics
import pydotplus                                                      # interprets dot objects 


###############################
######### lOAD DATA  ##########
###############################

#Reading in dataset and printing first 5 rows
file = 'Apprentice_Chef_Dataset.xlsx'

original = pd.read_excel(file)

#copy of original dataset
chef = original.copy()

################################
########### Missing Values######
###############################
#47 missing values identified for FAMILY_NAME 
fill = 'Unknown'
chef['FAMILY_NAME'] = chef['FAMILY_NAME'].fillna(fill)  


################################
######### Emails ##############
###############################


# Dummie variables from the email domain.
# Converting dataset to a dataFrame in order to use .iterrows()
chef_email       = pd.DataFrame(chef['EMAIL'])

placeholder_lst  = []

for index, col in chef_email.iterrows():
    split_email  = chef_email.loc[index, 'EMAIL'].split(sep = '@')
    
    placeholder_lst.append(split_email)
    
email_df         = pd.DataFrame(placeholder_lst)
email_df.columns = ['name', 'domain']


# Domain groups
personal_email     = ['@gmail.com', '@yahoo.com','@protonmail.com']
professional_email = ['@mmm.com', '@amex.com','@apple.com',
                      '@boeing.com','@caterpillar.com',
                      '@chevron.com','@cisco.com','@cocacola.com',
                      '@disney.com','@dupont.com','@exxon.com',
                      '@ge.org','@goldmansacs.com','@homedepot.com',
                      '@ibm.com','@intel.com','@jnj.com',
                      '@jpmorgan.com','@mcdonalds.com','@merck.com',
                      '@microsoft.com','@nike.com','@pfizer.com',
                      '@pg.com','@travelers.com','@unitedtech.com',
                      '@unitedhealth.com','@verizon.com','@visa.com',
                      '@walmart.com']
junk_email          = ['@me.com', '@aol.com', '@hotmail.com', '@live.com',
                       '@msn.com','@passport.com']


# For loop categorising the different email domains
placeholder_lst = []

for domain in email_df['domain']:
    
    if '@' + domain in personal_email:
        placeholder_lst.append('personal')
    elif '@' + domain in professional_email:
        placeholder_lst.append('professional')
    else:
        placeholder_lst.append('junk')
        
# make the columns into a series to append it to original dataset        
email_df['DOMAIN_GROUP'] = pd.Series(placeholder_lst)

# Concatenate the email domains as a new column in the chef DataFrame 
chef['DOMAIN'] = email_df['DOMAIN_GROUP']

# Get dummies from the domain variable and drop the original column
one_hot_DOMAIN = pd.get_dummies(chef['DOMAIN'])

# Remove the old and add the 3 new columns
chef           = chef.join([one_hot_DOMAIN])


################################
######### Number of names ######
###############################
# Adding variable, counting the number of names in NAME column 

def text_split_feature(col, df, sep=' ', new_col_name=None):
    """
Splits values in a string Series (as part of a DataFrame) and sums the number
of resulting items. Automatically appends summed column to original DataFrame.

PARAMETERS
----------
col          : column to split
df           : DataFrame where column is located
sep          : string sequence to split by, default ' '
new_col_name : name of new column after summing split, default
               'number_of_names'
"""
    
    chef[new_col_name] = 0
    
    
    for index, val in chef.iterrows():
        chef.loc[index, new_col_name] = len(chef.loc[index, col].split(sep = ' '))
        
text_split_feature(col = 'NAME', df = chef, new_col_name = 'NUMBER_NAMES' )   



################################
######### Nobility  ###########
###############################

# Creating variable that determines whether individual is NOBLE

#empty list to be appended at the end
placeholder_lst = []

#for loop to loop over names and match noble titles

for row,pattern in chef.iterrows():
    if ' of ' in chef.loc[row,'NAME'] or     'lord' in chef.loc[row,'NAME'] or     'Lord' in chef.loc[row,'NAME'] or     ' mo ' in chef.loc[row,'NAME'] or     ' zo ' in chef.loc[row,'NAME'] or     ' Mo ' in chef.loc[row,'NAME'] or     'Knight' in chef.loc[row, 'NAME'] or     'knight'in chef.loc[row, 'NAME']:
        placeholder_lst.append(1)
    else:
        placeholder_lst.append(0)
        

#creating column by appending empty list
chef['NOBLE'] = pd.Series(placeholder_lst)



#################################
## Willingness try new product ## 
#################################



#Creating a column for % of unique meals purchased
chef['PERCENT_UNIQUE_MEALS']= round(chef['UNIQUE_MEALS_PURCH']/ chef['TOTAL_MEALS_ORDERED']*100, 2)

placeholder_lst = []


for row,col in chef.iterrows():
    if chef.loc[row,'PERCENT_UNIQUE_MEALS'] >= 20:
        placeholder_lst.append(1)
    else:
        placeholder_lst.append(0)

# Adding the new variable to the original dataset
chef['WILLIGNESS_NEW_PRODUCTS'] = pd.Series(placeholder_lst)

#chef['WILLIGNESS_NEW_PRODUCTS'].value_counts()



#################################
####### Outlier thresholds ######
#################################


# Outliers thresholds determined based on the histograms and scatterplots
revenue_hi                    = 6200
total_meals_hi                = 230
unique_meals_hi               = 10
contact_w_customer_service_hi = 12
avg_time_per_site_hi          = 250
cancel_before_noon_hi         = 7
late_deliveries_hi            = 15
early_delivery_lo             = 0
total_photos_viewed_lo        = 0
follow_recommendations_pct_hi = 40
pc_log_hi                     = 7
pc_log_lo                     = 3
mobile_log_hi                 = 3
mobile_log_lo                 = 0
weekly_plan_hi                = 15
avg_prep_video_hi             = 300
total_photos_viewed_lo        = 0
percent_unique_meals_hi       = 30
avg_meal_price_hi             = 120
follow_recommendations_pct_lo = 10


# REVENUE
chef['out_REVENUE']  = 0
condition_hi = chef.loc[0:,'out_REVENUE'][chef['REVENUE'] 
                                                          > revenue_hi]

chef['out_REVENUE'].replace(to_replace = condition_hi,
                                   value      = 1,
                                   inplace    = True)
# TOTAL_MEALS_ORDERED
chef['out_TOTAL_MEALS_ORDERED']  = 0
condition_hi = chef.loc[0:,'out_TOTAL_MEALS_ORDERED'][chef['TOTAL_MEALS_ORDERED'] 
                                                             > total_meals_hi]

chef['out_TOTAL_MEALS_ORDERED'].replace(to_replace = condition_hi,
                                               value      = 1,
                                               inplace    = True)

# UNIQUE_MEALS_PURCH
chef['out_UNIQUE_MEALS_PURCH']  = 0
condition_hi = chef.loc[0:,'out_UNIQUE_MEALS_PURCH'][chef['UNIQUE_MEALS_PURCH'] 
                                                            > unique_meals_hi]

chef['out_UNIQUE_MEALS_PURCH'].replace(to_replace = condition_hi,
                                              value      = 1,
                                              inplace    = True)

# CONTACTS_W_CUSTOMER_SERVICE
chef['out_CONTACTS_W_CUSTOMER_SERVICE']  = 0
condition_hi = chef.loc[0:,'out_CONTACTS_W_CUSTOMER_SERVICE'][chef['CONTACTS_W_CUSTOMER_SERVICE'] 
                                                                     > contact_w_customer_service_hi]

chef['out_CONTACTS_W_CUSTOMER_SERVICE'].replace(to_replace = condition_hi,
                                                       value      = 1,
                                                       inplace    = True)

# AVG_TIME_PER_SITE_VISIT
chef['out_AVG_TIME_PER_SITE_VISIT']  = 0
condition_hi = chef.loc[0:,'out_AVG_TIME_PER_SITE_VISIT'][chef['AVG_TIME_PER_SITE_VISIT'] 
                                                                 > avg_time_per_site_hi]

chef['out_AVG_TIME_PER_SITE_VISIT'].replace(to_replace = condition_hi,
                                                   value      = 1,
                                                   inplace    = True)

# CANCELLATIONS_BEFORE_NOON
chef['out_CANCELLATIONS_BEFORE_NOON']  = 0
condition_hi = chef.loc[0:,'out_CANCELLATIONS_BEFORE_NOON'][chef['CANCELLATIONS_BEFORE_NOON'] 
                                                                   > cancel_before_noon_hi]

chef['out_CANCELLATIONS_BEFORE_NOON'].replace(to_replace = condition_hi,
                                                     value      = 1,
                                                     inplace    = True)


# LATE_DELIVERIES
chef['out_LATE_DELIVERIES']  = 0
condition_hi = chef.loc[0:,'out_LATE_DELIVERIES'][chef['LATE_DELIVERIES'] 
                                                         > late_deliveries_hi]

chef['out_LATE_DELIVERIES'].replace(to_replace = condition_hi,
                                           value      = 1,
                                           inplace    = True)


# EARLY_DELIVERIES
chef['out_EARLY_DELIVERIES']  = 0
condition_lo = chef.loc[0:,'out_EARLY_DELIVERIES'][chef['EARLY_DELIVERIES'] 
                                                         < early_delivery_lo]

chef['out_EARLY_DELIVERIES'].replace(to_replace = condition_lo,
                                           value      = 1,
                                           inplace    = True)

# TOTAL_PHOTOS_VIEWED
chef['out_TOTAL_PHOTOS_VIEWED']  = 0
condition_lo = chef.loc[0:,'out_TOTAL_PHOTOS_VIEWED'][chef['TOTAL_PHOTOS_VIEWED'] 
                                                         < total_photos_viewed_lo]

chef['out_TOTAL_PHOTOS_VIEWED'].replace(to_replace = condition_lo,
                                           value      = 1,
                                           inplace    = True)


# FOLLOW_REC_PCT
chef['out_FOLLOWED_RECOMMENDATIONS_PCT']  = 0
condition_hi = chef.loc[0:,'out_FOLLOWED_RECOMMENDATIONS_PCT'][chef['FOLLOWED_RECOMMENDATIONS_PCT'] 
                                                                     > follow_recommendations_pct_hi  ]
condition_lo = chef.loc[0:,'out_FOLLOWED_RECOMMENDATIONS_PCT'][chef['FOLLOWED_RECOMMENDATIONS_PCT'] 
                                                < follow_recommendations_pct_lo ]


chef['out_FOLLOWED_RECOMMENDATIONS_PCT'].replace(to_replace = condition_hi,
                                                       value      = 1,
                                                       inplace    = True)

chef['out_FOLLOWED_RECOMMENDATIONS_PCT'].replace(to_replace = condition_lo,
                                  value      = 1,
                                  inplace    = True)


# PC_LOGINS
chef['out_PC_LOGINS'] = 0
condition_hi = chef.loc[0:,'out_PC_LOGINS'][chef['PC_LOGINS'] 
                                            > pc_log_hi ]
condition_lo = chef.loc[0:,'out_PC_LOGINS'][chef['PC_LOGINS'] 
                                                < pc_log_lo ]

chef['out_PC_LOGINS'].replace(to_replace = condition_hi,
                              value      = 1,
                              inplace    = True)
chef['out_PC_LOGINS'].replace(to_replace = condition_lo,
                             value      = 1,
                             inplace    = True)


# MOBILE_LOGINS
chef['out_MOBILE_LOGINS'] = 0
condition_hi = chef.loc[0:,'out_MOBILE_LOGINS'][chef['MOBILE_LOGINS'] 
                                                > mobile_log_hi ]
condition_lo = chef.loc[0:,'out_MOBILE_LOGINS'][chef['MOBILE_LOGINS'] 
                                                < mobile_log_lo ]

chef['out_MOBILE_LOGINS'].replace(to_replace = condition_hi,
                                  value      = 1,
                                  inplace    = True)
chef['out_MOBILE_LOGINS'].replace(to_replace = condition_lo,
                                  value      = 1,
                                  inplace    = True)

# WEEKLY_PLAN
chef['out_WEEKLY_PLAN'] = 0
condition_hi = chef.loc[0:,'out_WEEKLY_PLAN'][chef['WEEKLY_PLAN'] 
                                              > weekly_plan_hi]

chef['out_WEEKLY_PLAN'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)


# AVG_PREP_VID_TIME
chef['out_AVG_PREP_VID_TIME'] = 0
condition_hi = chef.loc[0:,'out_AVG_PREP_VID_TIME'][chef['AVG_PREP_VID_TIME'] 
                                                    > avg_prep_video_hi]


chef['out_AVG_PREP_VID_TIME'].replace(to_replace = condition_hi,
                                      value      = 1,
                                      inplace    = True)



# TOTAL_PHOTOS_VIEWED
chef['out_TOTAL_PHOTOS_VIEWED']  = 0
condition_lo = chef.loc[0:,'out_TOTAL_PHOTOS_VIEWED'][chef['TOTAL_PHOTOS_VIEWED'] 
                                                         < total_photos_viewed_lo]

chef['out_TOTAL_PHOTOS_VIEWED'].replace(to_replace = condition_lo,
                                           value      = 1,
                                           inplace    = True)


# PERCENT_UNIQUE_MEALS
chef['out_PERCENT_UNIQUE_MEALS'] = 0
condition_hi = chef.loc[0:,'out_PERCENT_UNIQUE_MEALS'][chef['PERCENT_UNIQUE_MEALS'] 
                                                    > percent_unique_meals_hi]


chef['out_PERCENT_UNIQUE_MEALS'].replace(to_replace = condition_hi,
                                      value      = 1,
                                      inplace    = True)








#################################
###### Variable dictionary ######
#################################

# Variable dictionary for signingicant variables and the full dataset

variable_dict = {
    'logit_sig' : ['MOBILE_NUMBER',
                   'CANCELLATIONS_BEFORE_NOON' ,
                   'TASTES_AND_PREFERENCES' ,
                   'PC_LOGINS',
                   'MOBILE_LOGINS' ,
                   'junk',
                   'professional',
                   'NUMBER_NAMES',
                   'NOBLE',
                   'REFRIGERATED_LOCKER' ,
                   'WILLIGNESS_NEW_PRODUCTS' ,
                   'FOLLOWED_RECOMMENDATIONS_PCT'],
}





#################################
###### Train/TEST/SPLIT #########
#################################

# Seperating data and target variables
chef_data   =  chef.loc[ : , variable_dict['logit_sig']]
chef_target =  chef.loc[ : , 'CROSS_SELL_SUCCESS']


# preparing training and testing sets (Training = 75% , Testing = 25%)
X_train, X_test, y_train, y_test = train_test_split(
            chef_data,
            chef_target,
            random_state = 222,
            test_size    = 0.25,
            stratify     = chef['FOLLOWED_RECOMMENDATIONS_PCT'])




# INSTANTIATING a classification 
g_boost = GradientBoostingClassifier(loss = 'deviance',
                                     criterion = 'mae',
                                     learning_rate =  0.1,
                                     n_estimators = 200,
                                     max_depth = 1,
                                     max_features = 3,
                                     random_state  = 222)




#################################
###### Gradient Boost model #####
#################################

g_boost = GradientBoostingClassifier(loss = 'deviance',
                                     criterion = 'mae',
                                     learning_rate =  0.1,
                                     n_estimators = 95,
                                     max_features = 3,
                                     random_state  = 222)

# FITTING the training data
g_boost_fit = g_boost.fit(X_train, y_train)

# PREDICTING on test data
g_boost_pred = g_boost_fit.predict(X_test)



#################################
###### Final score ##############
#################################



# SCORING the model
print('Training ACCURACY:', g_boost_fit.score(X_train, y_train).round(4))
print('Testing  ACCURACY:', g_boost_fit.score(X_test, y_test).round(4))
print('AUC Score        :', roc_auc_score(y_true  = y_test,
                                          y_score = g_boost_pred).round(4))

