
# code using this link
# https://github.com/tkannab/Udacity-DAND-T1-P4-Analyze-AB-Test-Results/blob/master/Analyze_ab_test_results_notebook.ipynb
# borrow H0/H1 for the following the link
# https://github.com/SThornewillvE/Udacity-Project---AB-testing/blob/master/Analyze_ab_test_results.ipynb
# this H0/H1 conforms to code calculation

import pandas as pd
import numpy as np
import random
from tqdm import *
import matplotlib 
matplotlib.use('TkAgg')

#We are setting the seed to assure that we get the same answers on quizzes as we set up
random.seed(42)


###########################################
# part 1: probablity evaluation
###########################################


###########################################
#Quiz 1: Understanding the Dataset
#DESCRIPTION
#VALUE
#The number of rows in the dataset.
#294478
#The number of unique users in the dataset.
#290584
#The proportion of users converted.
#12%
#The number of times the new_page and treatment don't line up.
#3893
#Do any of the rows have missing values?
#No
###########################################


# a. Read in the dataset and take a look at the top few rows here:
# read dataset
# import data
df = pd.read_csv('ab_data.csv')

# show top rows
# ? conversion 這邊的case 不知道什麼case 才會是1 要查查
df.head()


# b. Use the below cell to find the number of rows in the dataset.
# we use shape function to see number of rows [first element]
df.shape[0]



# c.Number of unique users in the dataset.
#use unique() function
df.user_id.nunique()


# d. The proportion of users converted.
df.converted.mean()


# e. The number of times the new_page and treatment don't line up.
# Looking for rows where treatment/control doesn't line up with old/new pages respectively
pd.crosstab(df.group, df.landing_page, margins=True)


# f. Do any rows have missing values?
# Check for missing values?
df.isnull().values.any()

######################
#Quiz 2: Messy Data
#In this case, how should we handle the rows where the landing_page and group columns don't align?
#Remove these rows
######################

# df2 是去除不aligned 的rows 的結果


# a. Now use the answer to the quiz to create a new dataset that meets the specifications from the quiz. Store your new dataframe in df2.
# Copy dataframe
# define a new dataframe df2 from df where we excluded the 3893 records where new_page and trement didn't line up

df2 = df[((df['group']=='treatment') & (df['landing_page']=='new_page')) | ((df['group']=='control') & (df['landing_page']=='old_page'))]
In [9]:
# Double Check all of the correct rows were removed - this should be 0
df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]



###########################################
#Quiz 3: Updated DataFrame
#QUIZ QUESTION
#Match each description to the correct corresponding value.
#DESCRIPTION
#VALUE
#The number of unique ids in df2.
#290584
#The user_id for the non-unique id in df2.
#773192
#The landing_page for the non-unique id.
#new_page
#The group for the non-unique id.
#treatment
#The value of converted column for the non-unique id.
#0
###########################################

## df2 是去除不aligned 的rows  + duplication 的結果


# No. of unique user_ids after cleaning our dataset.

df2['user_id'].nunique()

# display the repeated user_id

df2[df2['user_id'].duplicated()]['user_id']


# Display the information of the repeated user_id

df2[df2['user_id'].duplicated()]


# No. of rows before removing the duplicate 

df2.shape


# Drop one of the rows that belongs to the repeated user_id

df2 = df2.drop_duplicates(subset='user_id');
# No. of rows after removing the duplicate 

df2.shape



###########################################
#Quiz 4: Probability
#QUIZ QUESTION
#Use your solutions to Question 4. in the notebook to match each description to its corresponding value.
#DESCRIPTION
#VALUE
#Probability of converting regardless of page.
#0.1196
#Given an individual received the control page, the probability of converting.
#0.1204
#Given that an individual received the treatment, the probability of converting.
#0.1188
#The probability of receiving the new page.
#0.5001


# e: Evidence that one page leads to more conversions?

# Based on the output above, 
# it looks like that the control group has a slightly higher conversion rate (0.1204) 
# than the treatment group (0.1195), however, 
# these results don't provide a solid evidence if one page leads to more 
# conversions as we still don't know the significance of these results and the factors 
#that might have contributed to the results above, 
# such as change resistence or test time duration. 
# In order to provide a meaningful information to support the decision whether to 
# implement the new page or keep the old page, we need to define our test hypothesis and 
# calculate p-value for the new and old pages.


###########################################


# The probability of an individual converting regardless of the page they receive
df2['converted'].mean()


# Probability of control group converting
df2.query('group =="control"').converted.mean()



# Probability of treatment group converting
df2.query('group =="treatment"').converted.mean()


# Probability an individual recieved new page
len(df2[df2['landing_page'] == 'new_page'])/len(df2)





###########################################
# part 2: A/B test

# Notice that because of the time stamp associated with each event, 
# you could technically run a hypothesis test continuously as each observation was observed.

# A/B test 時間上要多長  才能說這個test 是重要的?
# However, then the hard question is do you stop as soon as one page is considered 
# significantly better than another or does it need to happen consistently for a certain amount of time? 
# How long do you run to render a decision that neither page is better than another?
###########################################



###########################################
#Quiz 5: Hypothesis Testing
#QUIZ QUESTION
#Use your solutions to Part II Question 2 in the notebook to assist in this quiz.
#DESCRIPTION
#SOLUTION
#p_new under the null.
#0.1196
#p_old under the null.
#0.1196
#n_new
#145310
#n_old
#145274
#p_new - p_old under the null.
#0





###########################################


#For now, consider you need to make the decision just based on "all the data provided".
# 指的是 dfs (after  part 1 remove missing data, remove duplicated)
#If you want to assume that the old page is better unless the new page proves to be definitely
#better at a Type I error rate of 5% (type I error（α)), 
#what should your null and alternative hypotheses be? 
# You can state your hypothesis in terms of words or in terms of $p_{old}$ and $p_{new}$, 
# $p_{old}$ and $p_{new}$ are the converted rates for the old and new pages.

# ？這邊的自由度沒有像chi square 還要考慮


#  p 指的是conversion rate
# 要證明 new page 的conversion rate 比較好

#Hypothesis
#$H_0:  p_{new} - p_{old} \leq 0$
#$H_1: p_{new} - p_{old} &gt; 0$



# ? 這個assumption 是幹嘛的
# 就是假設  母體的 p_{new}$ and $p_{old}$  = converted rate in ab_data.csv regardless of the page. 在樣本中
#Assume under the null hypothesis, 
#$p_{new}$ and $p_{old}$ both have "true" success rates equal to the converted success rate regardless of page - 
# that is $p_{new}$ and $p_{old}$ are equal. 
#Furthermore, assume they are equal to the converted rate in ab_data.csv regardless of the page.


# ? 不知道這個說明是要幹嘛的
#Use a sample size for each page equal to the ones in ab_data.csv. 
#Perform the sampling distribution for the difference in converted between the two pages over 10,000 iterations of calculating an estimate from the null. 
#Use the cells below to provide the necessary parts of this simulation. If this doesn't make complete sense right now, don't worry - you are going to work through the problems below to complete this problem. You can use Quiz 5 in the classroom to make sure you are on the right track.


# under the null => unde the null hypothesis

# assume 這兩個是相等的在前面
# 然後根據真正的模擬然後做調整
# ??所以這邊是一樣的

# What is the convert rate for $p_{new}$ under the null?
# As per the instruction above, p_old = p_new = converted rate in ab_data.csv regardless of the page
p_new = df2.converted.mean()
p_new

# What is the convert rate for $p_{old}$ under the null? 
# As per the instruction above, p_old = p_new = converted rate in ab_data.csv regardless of the page
p_old = df2.converted.mean()
p_old

# treatment group samples
# What is $n_{new}$?
# Create a dataframe with all new page records from df2
newPage_df = df2.query('landing_page == "new_page"')
n_new = newPage_df.shape[0]
n_new

# control group samples
# What is $n_{old}$?  
# Create a dataframe with all old page records from df2
oldPage_df = df2.query('landing_page == "old_page"')
n_old = oldPage_df.shape[0]
n_old



#Simulate $n_{new}$ transactions with a convert rate of $p_{new}$ under the null. Store these $n_{new}$ 1's and 0's in new_page_converted.
new_page_converted = np.random.binomial(n_new,p_new)
new_page_converted


#Simulate $n_{old}$ transactions with a convert rate of $p_{old}$ under the null. Store these $n_{old}$ 1's and 0's in old_page_converted.
old_page_converted = np.random.binomial(n_old,p_old)
old_page_converted


# Find $p_{new}$ - $p_{old}$ for your simulated values 
p_diff = (new_page_converted/n_new) - (old_page_converted/n_old)
p_diff



# 根據上面p_diff 的過程  計算10000 次
# 但是不用truncate size 而是用mean 來幫助
# ? 不是完全懂
# Simulate 10,000 $p_{new}$ - $p_{old}$ values using this same process similarly to the one you calculated in parts a. through 
# Store all 10,000 values in p_diffs.
p_diffs = []

for _ in range(10000):
    new_converted_simulation = np.random.binomial(n_new,p_new)/n_new
    old_converted_simulation = np.random.binomial(n_old,p_old)/n_old
    diff = new_converted_simulation - old_converted_simulation 
    p_diffs.append(diff)



plt.hist(p_diffs);

# Plot a histogram of the p_diffs. Does this plot look like what you expected? 
# Use the matching problem in the classroom to assure you fully understand what was computed here.
plt.hist(p_diffs);


# j. What proportion of the p_diffs are greater than the actual difference observed in ab_data.csv?
# (是跟df 比  不是df2 比較)
# meaning 
# Calculate the actucl difference observed in ab_data
org_old_mean = df.query('group =="control"').converted.mean()
org_new_mean = df.query('group =="treatment"').converted.mean()
org_diff = org_new_mean - org_old_mean

# Convert p_diffs to array

p_diffs = np.array(p_diffs)

# Calculate the propotion of the p_diffs are greater than the actual difference observed in ab_data.csv
(p_diffs > org_diff).mean()

plt.hist(p_diffs);
plt.axvline(org_diff,c='r',linewidth = 2);





# k. In words, explain what you just computed in part j.. 
#We are computing p-values here.
#Answer: The value above represents the p-value of 
# observing the statistic given the Null is true. 
# As the p-value is large enough, 
# we would fail to reject the Null hypothesis and keep the old page.

