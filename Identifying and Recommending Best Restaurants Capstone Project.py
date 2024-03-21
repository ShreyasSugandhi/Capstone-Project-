#!/usr/bin/env python
# coding: utf-8

# # Importing libraries

# In[94]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # Loading the dataset

# In[95]:


df = pd.read_excel('data.xlsx')
df.head()


# In[96]:


#Loading 2nd dataset 
op = pd.read_excel("Country-Code.xlsx")
op.head ()


# # LEFT JOIN

# In[97]:


#merging two datasets in one  
df_rest = pd.merge(df,op,on='Country Code',how='left')
df_rest.head()


# In[98]:


df_rest.columns = df_rest.columns.str.replace(' ','_')
df_rest.columns


# In[99]:


#Check for null values
df_rest.isnull().sum()


# # As Restaurant_name has null / missing value and check for what restraunt it belongs to

# In[100]:


df_rest[df_rest['Restaurant_Name'].isnull()]


# # Since the restaurant name is missing, we dropped the record and reset the index.

# In[101]:


df_rest.dropna(axis=0,subset=['Restaurant_Name'],inplace=True)
df_rest.reset_index(drop=True,inplace=True)


# In[102]:


df_rest[df_rest['Cuisines'].isnull()]


# In[103]:


#Since there were only 9 records without cuisines, we have replace the null values with Others.
df_rest['Cuisines'].fillna('Others',inplace=True)


# In[104]:


df_rest.isnull().sum()
df_rest.info()


# 
# # Explore the geographical distribution of the restaurants. Finding out the cities with maximum / minimum number of restaurants

# In[105]:


cntry_dist = df_rest.groupby(['Country_Code','Country']).agg( Count = ('Restaurant_ID','count'))
cntry_dist.sort_values(by='Count',ascending=False)


# In[106]:


# Create a countplot with country on the x-axis and count on the y-axis
sns.countplot(x="Country", data= df_rest)
plt.xticks(size=5)
plt.xticks(rotation=20)
plt.xlabel('Country', size=20)

# Show the plot
plt.show()


# #We observe that India has then highest number of restaurants with 8651 restaurants and USA is number 2 with 434 restaurants

# In[107]:


cntry_dist.plot(kind='barh')


# # Finding out cities having maximum and minimum number of restraunts.

# In[108]:


city_dist = df_rest.groupby(['Country','City']).agg(Count = ('Restaurant_ID','count'))


# In[109]:


city_dist.sort_values(by='Count',ascending=False)


# #We observe that New Delhi has maximum number of restraunts in the world followed by Gurgaon and Noida
# #We also observe that multiple cities in multiple countries has only 1 restruant.

# In[110]:


min_cnt_rest = city_dist[city_dist['Count']==1]
min_cnt_rest.info()
min_cnt_rest


#  #Here we got to know that there are 46 cities in 7 different countries with 1 restaurants

# In[111]:


restaurant_counts = pd.Series()
restaurant_counts['India'] = len(df_rest[df_rest.Country == 'India'])
restaurant_counts['Others'] = len(df_rest[df_rest.Country != 'India'])
restaurant_counts.plot.pie(radius = 2,autopct = '%1.1f%%' , textprops = {'size':15 }, explode
= [0.1,0.1], shadow = True, cmap ='Set2')
plt.xticks(size = 12, rotation = 10)
plt.ylabel('')
plt.show()


# #From above Pi chart we can say that The country India alone has the 90.6% of total restaurants count and other 14 countries together holds only 9.4%
# Analysing data from India should give us a pretty accurate representation of the entire data.

# #Restaurant franchise is a thriving venture. So, it becomes very important to explore the franchise with most national presence.

# In[112]:


plt.figure(figsize = (15,10))
vc=df_rest.Restaurant_Name.value_counts()[:10]
g = sns.barplot(y = vc.index, x = vc.values, palette = 'Set2')
g.set_yticklabels(g.get_yticklabels(),fontsize = 13)
for i in range(10):
 value = vc[i]
 g.text(x = value - 2,y = i +0.125 , s = value, color='black', ha="center",fontsize = 15)
g.set_xlabel('Count', fontsize = 15)
g.set_title('Restaurant Presence', fontsize = 30, color = 'darkred')
plt.show()


# #cafe Coffe day has most national presesence

# In[113]:


df_rest1 = df_rest.copy()
df_rest1.columns


# In[114]:


dummy = ['Has_Table_booking','Has_Online_delivery']
df_rest1 = pd.get_dummies(df_rest1,columns=dummy,drop_first=True)
df_rest1.head()
# 0 indicates 'NO'
# 1 indicates 'YES'


# In[115]:


df_rest1.columns


# #Ratio between restaurants allowing table booking and those which don't

# In[116]:


table_booking = df_rest1[df_rest1['Has_Table_booking_Yes']==1]['Restaurant_ID'].count()
table_nbooking =df_rest1[df_rest1['Has_Table_booking_Yes']==0]['Restaurant_ID'].count()
print('Ratio between restaurants that allow table booking vs. those that do not allow table booking: ',
      round((table_booking/table_nbooking),2))


# #Pie chart to show percentage of restaurants which allow table booking and those which don't

# In[117]:


axes = plt.subplots(figsize = (7,20))
labels ='No Table Booking', 'Table Booking' 
df_rest1.Has_Table_booking_Yes.value_counts().plot.pie(ax = axes[1],labels=labels, autopct = '%0.1f%%', radius = 1.25,wedgeprops = {'width' : 0.75}, cmap = 'Set2_r',
textprops = {'size' : 18} )
axes[1].set_title('Table Booking Vs No Table Booking\n', fontsize = 16)
axes[1].set_ylabel('')


plt.show()


# #Find out the percentage of restaurants providing online delivery

# In[118]:


axes = plt.subplots(figsize = (20,7))
labels = 'No Online Delivery','Online Delivery'

df_rest1.Has_Online_delivery_Yes.value_counts().plot.pie(ax = axes[1],labels=labels, autopct = '%0.1f%%', radius = 1.25,wedgeprops = {'width' : 0.75}, cmap = 'Set2_r',
textprops = {'size' : 18} )
axes[1].set_title('Online Delivery\n', fontsize = 16)
axes[1].set_ylabel('')
plt.show()


# #Difference in number of votes for restaurants that deliver and dont deliver

# In[119]:


dc= df_rest.pivot_table(index = ['Has_Online_delivery'],values = 'Votes', aggfunc = 'sum')
dc


# In[120]:


dc['Perc'] = (dc.Votes / dc.Votes.sum() *100).round(2)

sns.barplot(x = dc.index, y = dc.Votes,)
plt.xticks( rotation = 0, fontsize = 14)
plt.xlabel('')
for i in range(len(dc)):
 plt.annotate(str(dc.Perc.iloc[i]) + '%',xy = (i-0.15, int(dc.Votes.iloc[i]/
2)), fontsize = 12 )
plt.ylabel('No. of Votes',fontsize = 20)


# #From observing the above table we can say that the difference between number of votes for restaurents that dont deliver and  deliver

# # What are the top 10 cuisines served across cities?

# In[121]:


l = []
for i in df_rest.Cuisines.str.split(','):
    l.extend(i)
s = pd.Series([i.strip() for i in l])
plt.figure(figsize = (15,5))
sns.barplot(x = s.value_counts()[:10].index, y = s.value_counts()[:10] )
for i in range(10):
    plt.annotate(s.value_counts()[i], xy = (i-0.15,s.value_counts()[i]+50),fontsize = 14)
plt.ylim(0, round(s.value_counts()[0]+300))
plt.show()


# # What is the maximum and minimum no. of cuisines that a restaurant serves? & what is the relationship between No. of cuisines served and Ratings?

# In[122]:


df_rest['no_cuisines']=df_rest.Cuisines.str.split(',').apply(len)
df_rest['no_cuisines']


# In[ ]:





# In[123]:


plt.figure(figsize = (15,5))
vc = df_rest.no_cuisines.value_counts()
sns.countplot(x='no_cuisines', data=df_rest, order = vc.index)
for i in range(len(vc)):
    plt.annotate(vc.iloc[i], xy = (i-0.07,vc.iloc[i]+10), fontsize = 12)
plt.show()


# the maximum no of cuisines served by a single restaurant is 8
# most of the restaurant are serving atleast 2 or 1 cuisine

# # Explore how ratings are distributed overall.

# In[124]:


plt.figure( figsize = (15, 4))
sns.countplot(x='Aggregate_rating', data = df_rest[df_rest.Aggregate_rating !=0] ,palette = 'magma')
plt.tick_params('x', rotation = 70)
plt.title('Y')
plt.show()


# # Rating Vs Delevery Options(Has_Online_delivery, Yes, No)

# In[125]:


plt.figure(figsize=(20,6))
sns.countplot(data=df_rest[df_rest.Aggregate_rating !=0],x='Aggregate_rating',hue='Has_Online_delivery',palette='viridis')
plt.show()


# #From observing the above chart we can say  that the delevery options can be a factor to decide the rating of restaurent.

# # No of Cuisines vs Rating

# In[126]:


df_rest['Rating_cat'] = df_rest['Aggregate_rating'].round(0).astype(int)
fusion_rate = df_rest.loc[df_rest.Aggregate_rating >0,['no_cuisines', 'Rating_cat','Aggregate_rating']].copy()
fusion_rate


# In[127]:


sns.regplot(x='no_cuisines',y='Aggregate_rating',data=fusion_rate)


# In[128]:


fusion_rate[['no_cuisines', 'Aggregate_rating']].corr()


# In[129]:


sns.barplot(x='no_cuisines',y='Aggregate_rating',data=fusion_rate)


# #From the above graphs we can observe that the number of cuisines and Aggregate_rating has positive corelation. We also observe that higher the number of cuisines higher the rating. 

# # Overall Cost Destribution

# In[130]:


plt.figure(figsize = (15,5))
sns.distplot(df_rest[df_rest.Average_Cost_for_two != 0].Average_Cost_for_two)
plt.show()


# # Discuss the cost(Average_Cost_for_two) vs rating

# In[131]:


df_rest['Average_Cost_for_two_cat']= pd.cut(df_rest[df_rest.Average_Cost_for_two != 0].Average_Cost_for_two,
bins = [0, 200, 500, 1000, 3000, 5000,10000],
labels = ['<=200', '<=500', '<=1000', '<=3000', '<=5000', '<=10000'])


# In[132]:


f = plt.figure(figsize = (20,10))
ax = plt.subplot2grid((2,5), (0,0),colspan = 2)
sns.countplot(x=df_rest['Average_Cost_for_two_cat'], ax = ax, palette = sns.color_palette('magma', 7))
ax.set_title('Average Cost for 2')
ax.set_xlabel('')
ax.tick_params('x', rotation = 70)
ax = plt.subplot2grid((2,5), (0,2), colspan = 3)
sns.boxplot(x = 'Average_Cost_for_two_cat', y = 'Aggregate_rating', data =df_rest, ax = ax, palette = sns.color_palette('magma', 7))
plt.show()


# #From Observing the above graph we can say that as the average cost for 2 increases the aggregate rating also increases

# # Price Range Vs Rating

# In[133]:


count = df_rest['Price_range'].value_counts().reset_index()
count.columns = ['Price_range', 'Count']


# In[134]:


f = plt.figure(figsize = (20,10))
ax = plt.subplot2grid((2,5), (1,0),colspan = 2)
sns.barplot(x = 'Price_range', y = 'Count', data = count, ax=ax, palette = sns.color_palette('magma', 5))
ax.set_title('Price Range')
ax.set_xlabel('')
ax = plt.subplot2grid((2,5), (1,2), colspan = 3)
sns.boxplot(x='Price_range', y ='Aggregate_rating', data = df_rest, ax = ax,palette = sns.color_palette('magma', 5))
plt.subplots_adjust(wspace = 0.3, hspace = 0.4,)
plt.suptitle('Price Range & Rating Distribution', size = 30)
plt.show()


# # Aggregate Rating vs Votes

# In[135]:


sns.scatterplot(data=df_rest,x='Aggregate_rating',y='Votes', palette ='Set2')


# In[136]:


df_rest[['Votes','Aggregate_rating']].corr()


# # We see that there is no single variable that affects the rating strongly, however table booking, online delivery, avg price for two and price range, number of votes do play a part in affecting the rating of a restaurant.
