
# https://github.com/nirupamaprv/Analyze-AB-test-Results/blob/master/Analyze%20A:B%20Test%20Results-Quiz-Answers.txt
# hypothesis is less related to p-value

import pandas as pd
import numpy as np
import random
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

# read dataset
df = pd.read_csv('ab_data.csv')

# inspect dataset
# ? conversion 這邊的case 不知道什麼case 才會是1 要查查
df.head()


# we use shape function to see number of rows [first element]
row_num = df.shape[0]
print("Number of rows is: {}".format(row_num))



#use unique() function
user_total = df.nunique()['user_id']
print("Number of unique users is : {}".format(user_total))


# met1
# 這個只是恰巧  因為convertion 的value  只有1
# we can find proportion of users converted by taking mean since values are 1 and 0
#print("Converted users proportion is {}%".format((df['converted'].mean())*100))

# met2
# alternate method to find number of converted users 
sum(df['converted'].values)/row_num



# rows where treatment group user lands incorrectly on old_page 
# columa name = value 
# treatment group 希望在新的landing page
mismatch_grp1 = df.query("group == 'treatment' and landing_page == 'old_page'")
print("Times treatment group user lands incorrectly on old_page is {}".format(len(mismatch_grp1)))

# rows where control group user incorrectly lands on new_page
# control group 希望在舊的landing page
mismatch_grp2 = df.query("group == 'control' and landing_page == 'new_page'")
print("Times control group user incorrectly lands on new_page is {}".format(len(mismatch_grp2)))

#  number of times the new_page and treatment don't line up is sum of above two values
print("Times new_page and treatment don't line up is {}".format(len(mismatch_grp1) + len(mismatch_grp2)))


# we check number of values in each rows using info function
# entry values denote if any column has missing values
df.info()

######################
#Quiz 2: Messy Data
#In this case, how should we handle the rows where the landing_page and group columns don't align?
#Remove these rows.
######################

# Delete Rows
# drop rows for mismatched treatment groups
df.drop(df.query("group == 'treatment' and landing_page == 'old_page'").index, inplace=True)
# drop rows for mismatched control groups
df.drop(df.query("group == 'control' and landing_page == 'new_page'").index, inplace=True)


# display after delete
df.info()


# save new clean dataset which contains no duplicates or records with missing or mismatched values
# we will use this dataset in next sections
# ? 沒看到有去重的function 啊  我覺得還沒有去重
df.to_csv('ab_edited.csv', index=False)


# read newly created dataset into another dataframe
df2 = pd.read_csv('ab_edited.csv')


# Double Check all of the uncorrect rows were removed - this should be 0
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
# inspect df2
df2.info()

# unique user ids count is
len(df2['user_id'].unique())

# check if duplicates in user_id
# we know that one user id is repeated due to difference between #userids and #unique ids
sum(df2['user_id'].duplicated())

# inspect duplicate userid
df2[df2.duplicated(['user_id'], keep=False)]['user_id']


# delete duplicate record 
# we choose one with timestamp as "2017-01-09 05:37:58.781806"
# 實際行為要看怎麼remove
time_dup = "2017-01-09 05:37:58.781806"
df2 = df2[df2.timestamp != time_dup]


# inspect number of entries in df2 after deleting duplicate record
df2.info()


# compare df2 with df (original without preprocessing)
# as seen above, 290584 entries now as entry with index 1876 is deleted
# we can confirm by checking unique values of user ids
len(df['user_id'].unique())


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


# Evidence that one page leads to more conversions?

# Given that an individual was in the treatment group, the probability they converted is 0.118807
# Given that an individual was in the control group, the probability they converted is 0.120386
# We find that old page does better, but by a very tiny margin.
# Change aversion, test span durations and other potentially influencing factors are not accounted for. So, we cannot state with certainty that one page leads to more conversions. This is even more important due to almost similar perforamnce of both pages.


###########################################

# since values are 1 and 0, we can calculate mean to get probability of an individual converting 
# ? individual converting 是什麼意思 這個probability 不太懂
df['converted'].mean()



# for this we group by column 'group'
# then we compute the statistics using describe function
# as conversions are assigned boolean values, we can use mean to find probability of conversion

df_grp = df.groupby('group')
df_grp.describe()



# number of individuals who got new page is same as those in treatment group
new_user = len(df.query("group == 'treatment'"))

# calculate total number of users
users=df.shape[0]

# thus, probability that an individual received the new page is new_user/users
new_user_p = new_user/users
print(new_user_p)



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


#Hypothesis
#$H_{0}$ : $p_{old}$ >=  $p_{new}$
#$H_{1}$ : $p_{old}$ <  $p_{new}$


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
# What is the convert rate for $p_{new}$ under the null?
p_new = df2['converted'].mean()
print(p_new)

# What is the convert rate for $p_{old}$ under the null? 
p_old = df2['converted'].mean()
print(p_old)

# treatment group samples
# What is $n_{new}$?
n_new = len(df2.query("group == 'treatment'"))
print(n_new)

# control group samples
# What is $n_{old}$?  
n_old = len(df2.query("group == 'control'"))
print(n_old)



#Simulate $n_{new}$ transactions with a convert rate of $p_{new}$ under the null. Store these $n_{new}$ 1's and 0's in new_page_converted.
new_page_converted = np.random.choice([1, 0], size=n_new, p=[p_new, (1-p_new)])
# print(len(new_page_converted)) #code to check values


#Simulate $n_{old}$ transactions with a convert rate of $p_{old}$ under the null. Store these $n_{old}$ 1's and 0's in old_page_converted.
old_page_converted = np.random.choice([1, 0], size=n_old, p=[p_old, (1-p_old)])
# print(len(old_page_converted))  #code to check values


# Find $p_{new}$ - $p_{old}$ for your simulated values 
# from part new_page_converted(145310) and old_page_converted(145274)
# since new_page_converted and old_page_converted have different sizes, we cannot directly compute p_diff
# since, differernce is only 36 values of thousands, we truncate the excess in new_page_converted
new_page_converted = new_page_converted[:145274]


p_diff = (new_page_converted/n_new) - (old_page_converted/n_old)
# print(p_diff) #code to check values



# 根據上面p_diff 的過程  計算10000 次
# 但是不用truncate size 而是用mean 來幫助
# ? 不是完全懂
# Simulate 10,000 $p_{new}$ - $p_{old}$ values using this same process similarly to the one you calculated in parts a. through 
# Store all 10,000 values in p_diffs.
p_diffs = []

for _ in range(10000):
    new_page_converted = np.random.choice([1, 0], size=n_new, p=[p_new, (1-p_new)]).mean()
    old_page_converted = np.random.choice([1, 0], size=n_old, p=[p_old, (1-p_old)]).mean()
    diff = new_page_converted - old_page_converted 
    p_diffs.append(diff)


# Plot a histogram of the p_diffs. Does this plot look like what you expected? 
# Use the matching problem in the classroom to assure you fully understand what was computed here.
plt.hist(p_diffs)
plt.xlabel('p_diffs')
plt.ylabel('Frequency')
plt.title('Plot of 10K simulated p_diffs');


# j. What proportion of the p_diffs are greater than the actual difference observed in ab_data.csv?
# (是跟df 比  不是df2 比較)
# meaning 


# compute difference from original dataset ab_data.csv
act_diff = df[df['group'] == 'treatment']['converted'].mean() -  df[df['group'] == 'control']['converted'].mean()
act_diff



# list to array
p_diffs = np.array(p_diffs)
p_diffs

# proportion of p_diffs greater than the actual difference observed in ab_data.csv is computed as:
(act_diff < p_diffs).mean()




# k. In words, explain what you just computed in part j.. 
#We are computing p-values here.
#As explained in the videos and quizzes, this is the probability of observing our statistic (or one more extreme in favor of the alternative) if the null hypothesis is true.
#The more extreme in favor of the alternative portion of this statement determines the shading associated with your p-value.
#Here, we find that there is no conversion advantage with new pages. We conclude that null hypothesis is true as old and new pages perform almost similarly. Old pages, as the numbers show, performed slightly better.
