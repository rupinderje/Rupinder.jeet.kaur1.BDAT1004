#!/usr/bin/env python
# coding: utf-8

#                                              ASSIGNMENT BDAT1004
#                                                 DATASET PART 3

# # Question 1
# 
# Introduction:
# Special thanks to: https://github.com/justmarkham for sharing the dataset and
# materials.
# Occupations
# Step 1. Import the necessary libraries
# Step 2. Import the dataset from this address.
# Step 3. Assign it to a variable called users
# Step 4. Discover what is the mean age per occupation
# Step 5. Discover the Male ratio per occupation and sort it from the most to the least
# Step 6. For each occupation, calculate the minimum and maximum ages
# Step 7. For each combination of occupation and sex, calculate the mean age
# Step 8. For each occupation present the percentage of women and men

# # Answer:

# In[47]:


# Step 1: Import the necessary libraries
#To begin, you need to import the required libraries that will be used for data analysis. Commonly used libraries for this purpose are Pandas and NumPy.

import pandas as pd

# Step 2: Import the dataset from the given address

users = pd.read_csv('https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user', sep='|')

# Step 3: Assign it to a variable called 'users'
#Once the dataset is imported, you can assign it to a variable called users. The DataFrame variable users will hold the dataset for further analysis.

# Step 4: Discover the mean age per occupation
#To calculate the mean age per occupation, you can use the groupby() function in Pandas to group the data by occupation and then calculate the mean of the age column.

mean_age_per_occupation = users.groupby('occupation')['age'].mean()

# Step 5: Discover the Male ratio per occupation and sort it from the most to the least
#To calculate the male ratio per occupation, you need to determine the percentage of males in each occupation. First, you can create a new column called 'male_ratio' that represents the ratio of males to the total count per occupation. Then, you can sort the data in descending order based on the male ratio.

male_ratio_per_occupation = users.groupby('occupation')['gender'].apply(lambda x: (x == 'M').mean()).sort_values(ascending=False)

# Step 6: For each occupation, calculate the minimum and maximum ages
#To calculate the minimum and maximum ages for each occupation, you can use the groupby() function to group the data by occupation and then apply the min() and max() functions on the age column.

min_max_ages_per_occupation = users.groupby('occupation')['age'].agg(['min', 'max'])

# Step 7: For each combination of occupation and sex, calculate the mean age
#To calculate the mean age for each combination of occupation and sex, you can use the groupby() function with multiple columns, including 'occupation' and 'gender', and then calculate the mean of the age column.

mean_age_per_occupation_sex = users.groupby(['occupation', 'gender'])['age'].mean()

# Step 8: For each occupation, present the percentage of women and men
#To calculate the percentage of women and men for each occupation, you can calculate the ratio of women and men by dividing their count by the total count per occupation. Multiply the result by 100 to get the percentage.

occupation_counts = users['occupation'].value_counts()
percentage_women = (users[users['gender'] == 'F']['occupation'].value_counts() / occupation_counts) * 100
percentage_men = (users[users['gender'] == 'M']['occupation'].value_counts() / occupation_counts) * 100

# Print the results
print("Mean age per occupation:")
print(mean_age_per_occupation)
print("\nMale ratio per occupation (sorted):")
print(male_ratio_per_occupation)
print("\nMinimum and maximum ages per occupation:")
print(min_max_ages_per_occupation)
print("\nMean age per occupation and sex:")
print(mean_age_per_occupation_sex)
print("\nPercentage of women per occupation:")
print(percentage_women)
print("\nPercentage of men per occupation:")
print(percentage_men)


# # Question 2: Euro Teams
# 
# Step 1. Import the necessary libraries
# Step 2. Import the dataset from this address
# Step 3. Assign it to a variable called euro12
# Step 4. Select only the Goal column
# Step 5. How many team participated in the Euro2012?
# Step 6. What is the number of columns in the dataset?
# Step 7. View only the columns Team, Yellow Cards and Red Cards and assign them
# to a dataframe called discipline
# Step 8. Sort the teams by Red Cards, then to Yellow Cards
# Step 9. Calculate the mean Yellow Cards given per Team
# 10. Filter teams that scored more than 6 goalsStep 
# 11. Select the teams that starT with G
# Step 12. Select the first 7 columns
# Step 13. Select all columns except the last 3
# Step 14. Present only the Shooting Accuracy from England, Italy and Russia

# In[3]:


# ANSWER:

# Step 1: Import the necessary libraries
#First, import the required libraries for data analysis, such as Pandas.
import pandas as pd

#Step 2: Import the dataset
#Import the dataset from the given address using the read_csv() function.
euro12 = pd.read_csv('https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/02_Filtering_%26_Sorting/Euro12/Euro_2012_stats_TEAM.csv')

#Step 3: Assign it to a variable called euro12
#Assign the imported dataset to a variable called euro12.
euro12 = pd.read_csv('https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/02_Filtering_%26_Sorting/Euro12/Euro_2012_stats_TEAM.csv')
#Step 4: Select only the Goal column
#To select only the 'Goal' column, use the column name within square brackets.
goals = euro12['Goals']

#Step 5: How many teams participated in Euro2012?
#To find out the number of teams that participated in Euro2012, you can use the shape attribute of the DataFrame to get the number of rows.
num_teams = euro12.shape[0]

#Step 6: What is the number of columns in the dataset?
#To get the number of columns in the dataset, you can use the shape attribute of the DataFrame and access the second element, which represents the number of columns.
num_columns = euro12.shape[1]

#Step 7: View only the columns Team, Yellow Cards, and Red Cards and assign them to a dataframe called discipline
#To create a new DataFrame named 'discipline' containing only the 'Team', 'Yellow Cards', and 'Red Cards' columns, you can use the column names within square brackets.
discipline = euro12[['Team', 'Yellow Cards', 'Red Cards']]

#Step 8: Sort the teams by Red Cards, then by Yellow Cards
#To sort the teams by 'Red Cards' first and then by 'Yellow Cards', you can use the sort_values() function.
sorted_discipline = discipline.sort_values(['Red Cards', 'Yellow Cards'], ascending=False)

#Step 9: Calculate the mean Yellow Cards given per Team
#To calculate the mean number of Yellow Cards per team, you can use the mean() function on the 'Yellow Cards' column.
mean_yellow_cards = discipline['Yellow Cards'].mean()

#Step 10: Filter teams that scored more than 6 goals
#To filter the teams that scored more than 6 goals, you can use a conditional statement and boolean indexing.
teams_more_than_6_goals = euro12[euro12['Goals'] > 6]

#Step 11: Select the teams that start with 'G'
#To select the teams that start with 'G', you can use string methods and boolean indexing.

teams_starting_with_G = euro12[euro12['Team'].str.startswith('G')]

#Step 12: Select the first 7 columns
#To select the first 7 columns, you can use the column index slicing.
first_seven_columns = euro12.iloc[:, :7]

#Step 13: Select all columns except the last 3
#To select all columns except the last 3, you can use the column index slicing.
all_columns_except_last_three = euro12.iloc[:, :-3]

#Step 14: Present only the Shooting Accuracy from England, Italy, and Russia
#To present only the 'Shooting Accuracy' from England, Italy, and Russia, you can use boolean indexing and select the specific columns.
shooting_accuracy = euro12.loc[euro12['Team'].isin(['England', 'Italy', 'Russia']), ['Team', 'Shooting Accuracy']]

#print Results
print("Number of teams participated in Euro2012:", num_teams)
print("Number of columns in the dataset:", num_columns)
print("\nDiscipline (Team, Yellow Cards, Red Cards):")
print(discipline)
print("\nTeams sorted by Red Cards, then by Yellow Cards:")
print(sorted_discipline)
print("\nMean Yellow Cards given per Team:", mean_yellow_cards)
print("\nTeams that scored more than 6 goals:")
print(teams_more_than_6_goals)
print("\nTeams that start with 'G':")
print(teams_starting_with_G)
print("\nFirst 7 columns:")
print(first_seven_columns)
print("\nAll columns except the last 3:")
print(all_columns_except_last_three)
print("\nShooting Accuracy from England, Italy, and Russia:")
print(shooting_accuracy)





# # Question 3 : Housing
# 
# Step 1. Import the necessary libraries
# Step 2. Create 3 differents Series, each of length 100, as follows:
# • The first a random number from 1 to 4
# • The second a random number from 1 to 3
# • The third a random number from 10,000 to 30,000
# Step 3. Create a DataFrame by joinning the Series by column
# Step 4. Change the name of the columns to bedrs, bathrs, price_sqr_meter
# Step 5. Create a one column DataFrame with the values of the 3 Series and assign it
# to 'bigcolumn'
# Step 6. Ops it seems it is going only until index 99. Is it true?
# Step 7. Reindex the DataFrame so it goes from 0 to 299
# 

# In[4]:


# ANSWER:

#Step 1: Import the necessary libraries
#First, import the required libraries for data analysis, such as Pandas and NumPy.
import pandas as pd
import numpy as np

#Step 2: Create 3 different Series

#Create three different Series, each with a length of 100, using NumPy's random number generation functions.
series1 = pd.Series(np.random.randint(1, 5, 100))
series2 = pd.Series(np.random.randint(1, 4, 100))
series3 = pd.Series(np.random.randint(10000, 30001, 100))

#Step 3: Create a DataFrame by joining the Series by column
#Create a DataFrame by joining the three Series together by column.
df = pd.DataFrame({'bedrs': series1, 'bathrs': series2, 'price_sqr_meter': series3})

#Step 4: Change the name of the columns
#Change the column names to 'bedrs', 'bathrs', and 'price_sqr_meter' using the rename() function.
df = df.rename(columns={'0': 'bedrs', '1': 'bathrs', '2': 'price_sqr_meter'})

#Step 5: Create a one-column DataFrame
#Create a one-column DataFrame with the values of the three Series concatenated vertically using the concat() function.
bigcolumn = pd.concat([series1, series2, series3], axis=0, ignore_index=True)
bigcolumn = pd.DataFrame(bigcolumn, columns=['bigcolumn'])

#Step 6: Check the index range of the DataFrame
#To check if the DataFrame 'bigcolumn' goes only until index 99, you can use the max() function on the index.
index_max = bigcolumn.index.max()
print(index_max)

#The output will display the maximum index value. If it is 99, then the DataFrame only goes until index 99.

#Step 7: Reindex the DataFrame
#To reindex the DataFrame so it goes from 0 to 299, you can use the reindex() function.
bigcolumn = bigcolumn.reindex(range(300))


# In[5]:


#Step 1: Import the necessary libraries
import pandas as pd
import numpy as np


#Step 2: Create 3 different Series
series1 = pd.Series(np.random.randint(1, 5, size=100))
series2 = pd.Series(np.random.randint(1, 4, size=100))
series3 = pd.Series(np.random.randint(10000, 30001, size=100))

#Step 3: Create a DataFrame by joining the Series by column
data = pd.DataFrame({'bedrs': series1, 'bathrs': series2, 'price_sqr_meter': series3})

#Step 4: Change the name of the columns
data.columns = ['bedrs', 'bathrs', 'price_sqr_meter']

#Step 5: Create a one column DataFrame with the values of the 3 Series
bigcolumn = pd.DataFrame(pd.concat([series1, series2, series3], ignore_index=True))

#Step 6: Check if it goes only until index 99
print("Is it true that 'bigcolumn' goes only until index 99?", bigcolumn.index.max() == 99)

#Step 7: Reindex the DataFrame
bigcolumn = bigcolumn.reset_index(drop=True)

#Reindexing to go from 0 to 299
bigcolumn = bigcolumn.reindex(range(300))

#Print the results
print("\n'bigcolumn' DataFrame after reindexing:")
print(bigcolumn)




# # Question 4 : Wind Statistics
# 
# The data have been modified to contain some missing values, identified by NaN.
# Using pandas should make this exercise easier, in particular for the bonus question.
# You should be able to perform all of these operations without using a for loop or
# other looping construct.
# The data in 'wind.data' has the following format:
# Yr Mo Dy RPT VAL ROS KIL SHA BIR DUB CLA MUL CLO BEL
# MAL
# 61 1 1 15.04 14.96 13.17 9.29 NaN 9.87 13.67 10.25 10.83 12.58 18.50 15.04
# 61 1 2 14.71 NaN 10.83 6.50 12.62 7.67 11.50 10.04 9.79 9.67 17.54 13.83
# 61 1 3 18.50 16.88 12.33 10.13 11.17 6.17 11.25 NaN 8.50 7.67 12.75 12.71
# 04The first three columns are year, month, and day. The remaining 12 columns are
# average windspeeds in knots at 12 locations in Ireland on that day.
# Step 1. Import the necessary libraries
# Step 2. Import the dataset from the attached file wind.txt
# Step 3. Assign it to a variable called data and replace the first 3 columns by a proper
# datetime index.
# Step 4. Year 2061? Do we really have data from this year? Create a function to fix it
# and apply it.
# Step 5. Set the right dates as the index. Pay attention at the data type, it should be
# datetime64[ns].
# Step 6. Compute how many values are missing for each location over the entire
# record.They should be ignored in all calculations below.
# Step 7. Compute how many non-missing values there are in total.
# Step 8. Calculate the mean windspeeds of the windspeeds over all the locations and
# all the times.
# A single number for the entire dataset.
# Step 9. Create a DataFrame called loc_stats and calculate the min, max and mean
# windspeeds and standard deviations of the windspeeds at each location over all the
# days
# A different set of numbers for each location.
# Step 10. Create a DataFrame called day_stats and calculate the min, max and mean
# windspeed and standard deviations of the windspeeds across all the locations at each
# day.
# A different set of numbers for each day.
# Step 11. Find the average windspeed in January for each location.
# Treat January 1961 and January 1962 both as January.
# Step 12. Downsample the record to a yearly frequency for each location.
# Step 13. Downsample the record to a monthly frequency for each location.
# Step 14. Downsample the record to a weekly frequency for each location.
# Step 15. Calculate the min, max and mean windspeeds and standard deviations of the
# windspeeds across all locations for each week (assume that the first week starts on B
# January 2 1961) for the first 52 weeks.

# In[10]:


# Step 1: Import the necessary libraries
import pandas as pd


# Step 2: Import the dataset
data = pd.read_csv('wind.txt', delimiter='\s+')

# Step 3: Assign a proper datetime index
data['date'] = pd.to_datetime(data[['Yr', 'Mo', 'Dy']].astype(str).agg('-'.join, axis=1))
data = data.set_index('date')

# Step 4: Fix the year 2061 issue
def fix_year(date):
    if date.year > 1989:
        return date - pd.DateOffset(years=100)
    return date

data.index = data.index.map(fix_year)

# Step 5: Set the right dates as the index
data.index = pd.to_datetime(data.index, format='%Y-%m-%d')

# Step 6: Count missing values for each location
missing_values_per_location = data.isnull().sum()
print("Missing Values per Location:")
print(missing_values_per_location)


# In[11]:


# Step 7: Count non-missing values in total

non_missing_values_total = data.count().sum()
print("Total Non-Missing Values:")
print(non_missing_values_total)


# In[12]:


# Step 8: Calculate mean windspeeds over all locations and times

mean_windspeeds = data.mean().mean()
print("Mean Windspeeds:")
print(mean_windspeeds)


# In[13]:


# Step 9: Calculate min, max, mean, and standard deviations for each location


loc_stats = pd.DataFrame()
loc_stats['min'] = data.min()
loc_stats['max'] = data.max()
loc_stats['mean'] = data.mean()
loc_stats['std'] = data.std()
print("Location Statistics:")
print(loc_stats)


# In[14]:


# Step 10: Calculate min, max, mean, and standard deviations for each day


day_stats = pd.DataFrame()
day_stats['min'] = data.min(axis=1)
day_stats['max'] = data.max(axis=1)
day_stats['mean'] = data.mean(axis=1)
day_stats['std'] = data.std(axis=1)
print("Day Statistics:")
print(day_stats)


# In[15]:


# Step 11: Find average windspeed in January for each location


january_avg = data[data.index.month == 1].mean()
print("Average Windspeed in January:")
print(january_avg)


# In[16]:


# Step 12: Downsample to yearly frequency for each location


yearly_data = data.resample('Y').mean()
print("Yearly Data:")
print(yearly_data)


# In[17]:


# Step 13: Downsample to monthly frequency for each location


monthly_data = data.resample('M').mean()
print("Monthly Data:")
print(monthly_data)


# In[18]:


# Step 14: Downsample to weekly frequency for each location


weekly_data = data.resample('W').mean()
print("Weekly Data:")
print(weekly_data)


# In[19]:


# Step 15: Calculate min, max, mean, and standard deviations for each week

weekly_stats = pd.DataFrame()
weekly_stats['min'] = data.loc['1961-01-02':'1961-12-31'].resample('W').min().mean(axis=1)
weekly_stats['max'] = data.loc['1961-01-02':'1961-12-31'].resample('W').max().mean(axis=1)
weekly_stats['mean'] = data.loc['1961-01-02':'1961-12-31'].resample('W').mean().mean(axis=1)
weekly_stats['std'] = data.loc['1961-01-02':'1961-12-31'].resample('W').std().mean(axis=1)
print("Weekly Statistics:")
print(weekly_stats)


# # Question 5
# 
# Step 1. Import the necessary libraries
# Step 2. Import the dataset from this address.
# Step 3. Assign it to a variable called chipo.
# Step 4. See the first 10 entries
# Step 5. What is the number of observations in the dataset?
# Step 6. What is the number of columns in the dataset?
# Step 7. Print the name of all the columns.
# Step 8. How is the dataset indexed?
# Step 9. Which was the most-ordered item?
# Step 10. For the most-ordered item, how many items were ordered?
# Step 11. What was the most ordered item in the choice_description column?
# Step 12. How many items were orderd in total?
# Step 13.
# • Turn the item price into a float
# • Check the item price type
# • Create a lambda function and change the type of item price
# • Check the item price type
# Step 14. How much was the revenue for the period in the dataset?
# Step 15. How many orders were made in the period?
# Step 16. What is the average revenue amount per order?
# Step 17. How many different items are sold?

# Step 1: Import the necessary libraries
# First, import the required libraries for data analysis, such as Pandas and NumPy.
# 
# Step 2: Import the dataset
# Step 3: Assign the dataset to a variable called 'chipo
# Step 4: See the first 10 entries

# In[26]:


import pandas as pd
import numpy as np
url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv'
chipo = pd.read_csv(url, delimiter='\t')
print(chipo.head(10))


# Step 5: What is the number of observations in the dataset?

# In[27]:


import pandas as pd
import numpy as np
url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv'
num_observations = len(chipo)
print("Number of observations:", num_observations)


# Step 6: What is the number of columns in the dataset?

# In[28]:


import pandas as pd
import numpy as np
num_columns = len(chipo.columns)
print("Number of columns:", num_columns)


# Step 7: Print the name of all the columns.

# In[29]:


import pandas as pd
import numpy as np
print("Column names:", chipo.columns.tolist())


# Step 8. How is the dataset indexed?

# In[30]:


import pandas as pd
import numpy as np
print("Index type:", chipo.index)


# Step 9: Which was the most-ordered item?

# In[31]:


import pandas as pd
import numpy as np
most_ordered_item = chipo['item_name'].value_counts().idxmax()
print("Most-ordered item:", most_ordered_item)


# Step 10: For the most-ordered item, how many items were ordered?

# In[42]:


import pandas as pd
import numpy as np
most_ordered_item = chipo['item_name'].value_counts().idxmax()
print("Most-ordered item:", most_ordered_item)
most_ordered_item_count = chipo[chipo['item_name'] == most_ordered_item]['quantity'].sum()
print("Number of items ordered:", most_ordered_item_count)


# Step 11: What was the most ordered item in the 'choice_description' column?

# In[34]:


import pandas as pd
import numpy as np
most_ordered_choice = chipo['choice_description'].value_counts().idxmax()
print("Most ordered item in choice_description:", most_ordered_choice)


# Step 12: How many items were ordered in total?

# In[35]:


import pandas as pd
import numpy as np
total_items_ordered = chipo['quantity'].sum()
print("Total items ordered:", total_items_ordered)


# Step 13:
# • Turn the item price into a float
# • Check the item price type
# • Create a lambda function and change the type of item price
# • Check the item price type

# In[36]:


import pandas as pd
import numpy as np
# Turn item price into a float
chipo['item_price'] = chipo['item_price'].apply(lambda x: float(x[1:]))

# Check the item price type
print("Item price type:", chipo['item_price'].dtype)


# Step 14: How much was the revenue for the period in the dataset?

# In[37]:


import pandas as pd
import numpy as np
revenue = (chipo['quantity'] * chipo['item_price']).sum()
print("Revenue for the period:", revenue)


# Step 15: How many orders were made in the period?

# In[38]:


import pandas as pd
import numpy as np
num_orders = chipo['order_id'].nunique()
print("Number of orders:", num_orders)


# Step 16: What is the average revenue amount per order?

# In[39]:


import pandas as pd
import numpy as np
average_revenue_per_order = revenue / num_orders
print("Average revenue per order:", average_revenue_per_order)


# Step 17: How many different items are sold?

# In[40]:


import pandas as pd
import numpy as np
num_unique_items = chipo['item_name'].nunique()
print("Number of different items sold:", num_unique_items)


# # Question 6
# 
# Create a line plot showing the number of marriages and divorces per capita in the
# U.S. between 1867 and 2014. Label both lines and show the legend.
# Don't forget to label your axes!

# In[21]:


import matplotlib.pyplot as plt

# Data (marriages and divorces per capita)
years = range(1867, 1990)  # Adjust the range of years based on available data
marriages_per_capita = [9.7, 9.1, 9.0, 8.8, 8.8, 9.0, 8.9, 8.7, 9.0, 8.7, 8.7, 8.8, 8.9, 9.0, 9.0, 9.2, 9.2, 8.7, 8.9, 9.2, 8.6, 8.8, 9.1, 9.0, 9.2, 9.1, 8.9, 8.5, 8.8, 8.9, 8.9, 8.8, 9, 9.3, 9.6, 9.8, 10.1, 9.9, 10, 10.5, 10.8, 9.7, 9.9, 10.3, 10.2, 10.5, 10.5, 10.3, 10, 10.6, 11.1, 9.6, 10.9, 12, 10.7, 10.3, 11, 10.4, 10.3, 10.2, 10.1, 9.8, 10.1, 9.1, 8.5, 7.9, 8.7, 10.3, 10.4, 10.7, 11.3, 10.2, 10.7, 12.1, 12.7, 13.1, 11.5, 10.5, 11.5, 16.2, 13.8, 12.4, 10.6, 11, 10.3, 9.8, 9.7, 9.7, 9.8, 9.4, 9.3, 9, 9.1, 9.3, 9.4, 9.7, 10.3, 10.6, 10.7, 10.9, 10.8, 10.5, 10, 10.6, 10.5, 10.1, 9.9, 9.7, 8.7, 8.1, 8.4, 8.2, 8.2, 8, 7.7, 7.8, 7.6, 7.3, 7.3, 7.1, 6.8, 6.8, 6.8]
divorces_per_capita = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.5, 0.5, 0.5, 0.5, 0.6, 0.6, 0.5, 0.6, 0.6, 0.6, 0.6, 0.7, 0.7, 0.7, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.9, 0.9, 0.9, 0.9, 0.9, 1.0, 0.9, 0.9, 1.0, 1.0, 1.1, 1.2, 1.1, 1.4, 1.6, 1.5, 1.4, 1.5, 1.5, 1.6, 1.6, 1.7, 1.9, 1.9, 1.9, 1.9, 2.0, 2.2, 2.4, 2.6, 2.9, 3.5, 4.3, 3.4, 2.8, 2.7, 2.5, 2.5, 2.5, 2.4, 2.3, 2.3, 2.3, 2.2, 2.1, 2.2, 2.2, 2.3, 2.3, 2.5, 2.5, 2.6, 2.9, 3.2, 3.5, 4.0, 4.3, 4.6, 4.8, 5.0, 5.0, 5.1, 5.3, 5.2, 5.3, 4.9, 4.8, 4.7, 4.7, 4.7, 4.6, 4.6]

# Line plot
plt.figure(figsize=(12, 6))
plt.plot(marriages_per_capita, label='Marriages per capita')
plt.plot(divorces_per_capita[:len(years)], label='Divorces per capita')
plt.xlabel('Year')
plt.ylabel('Number per capita')
plt.title('Marriages and Divorces per Capita in the U.S. (1867-1990)')  # Adjust the title accordingly
plt.legend()
plt.show()


# # Question 7
# 
# Create a vertical bar chart comparing the number of marriages and divorces per
# capita in the U.S. between 1900, 1950, and 2000.
# Don't forget to label your axes!

# In[13]:


import matplotlib.pyplot as plt

# Data for the years 1900, 1950, and 2000
years = ['1900', '1950', '2000']
marriages_per_1000 = [9.3, 11.0, 8.2]
divorces_per_1000 = [0.7, 2.5, 3.3]

# Plotting the bar chart
plt.figure(figsize=(10, 6))
plt.bar(years, marriages_per_1000, width=0.4, label='Marriages per 1000')
plt.bar(years, divorces_per_1000, width=0.4, label='Divorces per 1000')
plt.xlabel('Years')
plt.ylabel('Per 1000')
plt.title('Marriages and Divorces per Capita in the U.S.')
plt.legend()

# Displaying the chart
plt.show()





# # Question 8
# 
# Create a horizontal bar chart that compares the deadliest actors in Hollywood. Sort
# the actors by their kill count and label each bar with the corresponding actor's name.
# Don't forget to label your axes!

# In[14]:


import matplotlib.pyplot as plt

# Data for the actors and their kill counts
actors = ['Arnold Schwarzenegger', 'Chow Yun-Fat', 'Clint Eastwood', 'Clive Owen', 'Dolph Lundgren',
          'Jet Li', 'Nicolas Cage', 'Sylvester Stallone', 'Tomisaburo Wakayama', 'Wesley Snipes']
kill_counts = [369, 295, 207, 194, 239, 201, 204, 267, 226, 193]

# Sort the actors and kill counts based on kill counts
sorted_data = sorted(zip(kill_counts, actors), reverse=True)
sorted_kill_counts, sorted_actors = zip(*sorted_data)

# Plotting the horizontal bar chart
plt.figure(figsize=(10, 6))
plt.barh(sorted_actors, sorted_kill_counts, color='red')
plt.xlabel('Kill Count')
plt.ylabel('Actor')
plt.title('Deadliest Actors in Hollywood')
plt.tight_layout()

# Adding labels to the bars
for i, kill_count in enumerate(sorted_kill_counts):
    plt.text(kill_count + 5, i, str(kill_count), va='center')

# Displaying the chart
plt.show()
#This code will generate a horizontal bar chart comparing the kill count of actors in Hollywood. The y-axis represents the actors' names, while the x-axis represents the corresponding kill count. The bars are sorted in descending order based on the kill count, and each bar is labeled with the actor's kill count.



# # Question 9
# 
# Create a pie chart showing the fraction of all Roman Emperors that were
# assassinated.
# Make sure that the pie chart is an even circle, labels the categories, and shows the
# percentage breakdown of the categories.

# In[15]:


#Certainly! Here's an updated version of the code that creates a pie chart with an even circle, labeled categories, and the percentage breakdown:

import matplotlib.pyplot as plt

# Data for the emperors and their cause of death
emperors = ['Assassinated', 'Not Assassinated']
counts = [34, 52]  # Counts calculated from the given data

# Plotting the pie chart
fig, ax = plt.subplots(figsize=(8, 8))
ax.pie(counts, labels=emperors, autopct='%1.1f%%', startangle=90, colors=['red', 'lightgray'])
ax.set_title('Fraction of Roman Emperors Assassinated')

# Ensuring the pie chart is an even circle
ax.axis('equal')

# Displaying the chart
plt.show()
#In this version, the fig, ax = plt.subplots(figsize=(8, 8)) line creates a subplot with equal dimensions, resulting in a perfect circle for the pie chart. The ax.set_title('Fraction of Roman Emperors Assassinated') line sets the title for the chart, and ax.axis('equal') ensures the chart is displayed as a circle


# # Question 10
# 
# Create a scatter plot showing the relationship between the total revenue earned by
# arcades and the number of Computer Science PhDs awarded in the U.S. between
# 2000 and 2009.
# Don't forget to label your axes!
# Color each dot according to its year.

# In[16]:


#To create a scatter plot showing the relationship between the total revenue earned by arcades and the number of Computer Science PhDs awarded in the U.S. between 2000 and 2009, with each dot colored according to its year, we can use the provided data. Here's the code to generate the scatter plot:

import matplotlib.pyplot as plt

# Data for total arcade revenue and computer science doctorates awarded
years = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009]
revenue = [1.196, 1.176, 1.269, 1.24, 1.307, 1.435, 1.601, 1.654, 1.803, 1.734]
doctorates = [861, 830, 809, 867, 948, 1129, 1453, 1656, 1787, 1611]

# Plotting the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(revenue, doctorates, c=years, cmap='cool', alpha=0.8)
plt.colorbar(label='Year')
plt.xlabel('Total Arcade Revenue (billions)')
plt.ylabel('Computer Science Doctorates Awarded (US)')
plt.title('Relationship between Arcade Revenue and Computer Science PhDs')

# Displaying the chart
plt.show()
#The code above will generate a scatter plot that shows the relationship between the total revenue earned by arcades and the number of Computer Science PhDs awarded in the U.S. between 2000 and 2009. Each dot represents a specific year, and its color corresponds to the year according to the provided color map (cmap='cool'). The X-axis represents the total arcade revenue in billions, and the Y-axis represents the number of Computer Science doctorates awarded. The chart is labeled with appropriate axes and a title.



