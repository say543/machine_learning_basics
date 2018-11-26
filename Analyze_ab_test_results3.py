
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
df_length = len(df)         
print(df_length)



# c.Number of unique users in the dataset.
#use unique() function
len(df.user_id.unique())


# d. The proportion of users converted.
df.converted.sum()/df_length


# e. The number of times the new_page and treatment don't line up.
# Looking for rows where treatment/control doesn't line up with old/new pages respectively
df_t_not_n = df[(df['group'] == 'treatment') & (df['landing_page'] == 'old_page')]
df_not_t_n = df[(df['group'] == 'control') & (df['landing_page'] == 'new_page')]

# Add lengths
mismatch= len(df_t_not_n) + len(df_not_t_n)

# Create one dataframe from it
mismatch_df = pd.concat([df_t_not_n, df_not_t_n])

mismatch


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
df2 = df

# Remove incriminating rows
mismatch_index = mismatch_df.index
df2 = df2.drop(mismatch_index)


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


# Find unique users
print("Unique users:", len(df2.user_id.unique()))

# Check for not unique users
print("Non-unique users:", len(df2)-len(df2.user_id.unique()))

# Find duplicated user
df2[df2.duplicated('user_id')]

# Find duplicates under user ids
df2[df2['user_id']==773192]

# Drop duplicated user
df2.drop(labels=1899, axis=0, inplace=True)

# Check the drop worked
df2[df2['user_id']==773192]

# inspect number of entries in df2 after deleting duplicate record
df2.info()



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


# According to the probabilities, the control group (the group with the old page) 
# converted at a higher rate than the teatment (the group with the new page). 
# However, the magnitude of this change is very small with a difference of roughly 0.2%.

# ? 這句話  我覺得不太make sense  不太懂
# 是因為0.5 不夠大  所以不行
# Given the data in Question 4 so far, 
# the probability that an individual recieved a new page is roughly 0.5, 
# this means that it is not possible for there to be a difference in conversion based on being given more opportunities to do so. For instance, if the probability of recieving a new page was higher relative to the old page then it would be observed that the rate of conversion would naturally increase.


###########################################


# Probability of user converting
print("Probability of user converting:", df2.converted.mean())


# Probability of control group converting
print("Probability of control group converting:", 
      df2[df2['group']=='control']['converted'].mean())



# Probability of treatment group converting
print("Probability of treatment group converting:", 
      df2[df2['group']=='treatment']['converted'].mean())


# Probability an individual recieved new page
print("Probability an individual recieved new page:", 
      df2['landing_page'].value_counts()[0]/len(df2))





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


# If our sample conformed to the null hypothesis 
# then we'd expect the proportion greater than the actual difference to be 0.5. 
# (from quiz 4 the probability that an individual recieved a new page is roughly 0.5, )

#here
# almost 90% of the population in our simulated sample lies above the 
# real difference which does not only suggest that the new page does not do significantly better 
# than the old page, it might even be worse!




###########################################


#For now, consider you need to make the decision just based on "all the data provided".
# 指的是 dfs (after  part 1 remove missing data, remove duplicated)
#If you want to assume that the old page is better unless the new page proves to be definitely
#better at a Type I error rate of 5% (type I error（α)), 
#what should your null and alternative hypotheses be? 
# You can state your hypothesis in terms of words or in terms of $p_{old}$ and $p_{new}$, 
# $p_{old}$ and $p_{new}$ are the converted rates for the old and new pages.


#Hypothesis
#$H_0:  p_{new} - p_{old} \leq 0$
# $H_1: p_{new} - p_{old} &gt; 0$


# ? 不知道這個assumption 是幹嘛的
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

# Calculate probability of conversion for new page
p_new = df2[df2['landing_page']=='new_page']['converted'].mean()
print("Probability of conversion for new page (p_new):", p_new)


# Calculate probability of conversion for old page
p_old = df2[df2['landing_page']=='old_page']['converted'].mean()

print("Probability of conversion for old page (p_old):", p_old)




# Calc. differences in probability of conversion for new and old page (not under H_0)
p_diff = p_new-p_old

print("Difference in probability of conversion for new and old page (not under H_0):", p_diff)

# Take the mean of these two probabilities
# ? 不知道這邊為什麼可以求mean 就是under null hypothesis
p_mean = np.mean([p_new, p_old])

print("Probability of conversion udner null hypothesis (p_mean):", p_mean)


# Calculate n_new and n_old
n_new, n_old = df2['landing_page'].value_counts()

print("new:", n_new, "\nold:", n_old)



# Simulate conversion rates under null hypothesis
new_page_converted = np.random.choice([1, 0], size=n_new, p=[p_mean, (1-p_mean)])

new_page_converted.mean()


# Simulate conversion rates under null hypothesis
old_page_converted = np.random.choice([1, 0], size=n_old, p=[p_mean, (1-p_mean)])

old_page_converted.mean()

# Calculate difference in p under the null hypothesis
new_page_converted.mean()-old_page_converted.mean()


# 根據上面p_diff 的過程  計算10000 次
p_diffs = []

# Re-run simulation 10,000 times
# trange creates an estimate for how long this program will take to run
for i in trange(10000):
    new_page_converted = np.random.choice([1, 0], size=n_new, p=[p_mean, (1-p_mean)])
    old_page_converted = np.random.choice([1, 0], size=n_old, p=[p_mean, (1-p_mean)])
    p_diff = new_page_converted.mean()-old_page_converted.mean()
    p_diffs.append(p_diff)


# Plot histogram
# ?這個說明原因不太懂
# The simulated data creates a normal distribution (no skew) as expected due to how the data was generated. 
#The mean of this normal distribution is 0, which which is what the data should look like under the null hypothesis.
plt.hist(p_diffs, bins=25)
plt.title('Simulated Difference of New Page and Old Page Converted Under the Null')
plt.xlabel('Page difference')
plt.ylabel('Frequency')
plt.axvline(x=(p_new-p_old), color='r', linestyle='dashed', linewidth=1, label="Real difference")
plt.axvline(x=(np.array(p_diffs).mean()), color='g', linestyle='dashed', linewidth=1, label="Simulated difference")
plt.legend()
plt.show()


# j. What proportion of the p_diffs are greater than the actual difference observed in ab_data.csv?
# meaning 


p_diff = p_new - p_old

# Find proportion of p_diffs greater than the actual difference
greater_than_diff = [i for i in p_diffs if i > p_diff]


# Calculate values
# The value calculated above is the p-value.

#the above number makes sense because the probability for a conversion of a new page is lower than both the mean and old page conversion rates.
print("Actual difference:" , p_diff)

p_greater_than_diff = len(greater_than_diff)/len(p_diffs)

print('Proportion greater than actual difference:', p_greater_than_diff)

print('As a percentage: {}%'.format(p_greater_than_diff*100))




# k. In words, explain what you just computed in part j.. 

