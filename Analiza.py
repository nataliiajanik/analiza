#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# In[2]:


df = pd.read_csv('Wellbeing_and_lifestyle_data_Kaggle.csv')


# Zbiór danych

# In[3]:


df.head()


# In[8]:


pd.set_option('display.max_columns', None)
df.head()


# Etykiety danych

# In[34]:


df['GENDER']


# In[5]:


dfetykiety = pd.read_excel('etykiety.xlsx')


# In[35]:


dfetykiety.head()


# In[7]:


dfetykiety.to_numpy()


# In[8]:


df.shape


# Jakiego typu zmienne występują w zbiorze?

# In[9]:


df.dtypes


# Wartości puste, typy danych

# In[10]:


df.info()


# Suma wystąpień wartości pustych

# In[11]:


df.isnull().sum()


# Podsumowanie zmiennych:
# - count: zliczenie wartości w kolumnie
# - mean: średnia
# - std: odchylenie standardowe
# - min: wartość minimalna
# - 25%: I kwartyl
# - 50%: II kwartyl
# - 75%: III kwartyl
# - max: wartość maksymalna

# In[12]:


df.describe()


# In[4]:


df['AGE']=df['AGE'].replace('Less than 20', '20 or less')


# In[5]:


dfcopy = df.copy()


# In[6]:


df['MONTH'] = pd.DatetimeIndex(df['Timestamp']).month
df['MONTH']


# In[7]:


df['GENDER'] = df['GENDER'].map( {'Female': 0, 'Male': 1} ).astype(int)
df['AGE'] = df['AGE'].map( {'20 or less': 0, '21 to 35': 1, '36 to 50': 2, '51 or more': 3} ).astype(int)


# In[17]:


plt.figure(figsize=(16,16))
plt.title('Współczynniki korelacji liniowej Pearsona', y=1.05, size=15)
sns.heatmap(df.corr(),linewidths=0.1,vmax=1.0, square=True, cmap='coolwarm_r', linecolor='white', annot=True)


# In[18]:


def descriptive(df):
    desc=df.describe().round(1).drop({'count'}, axis=0)
    i=-0.1
    j=0
    Row = int(round(len(desc.columns.tolist())/2+0.1)) 
    f,ax = plt.subplots(Row,2, figsize=(28,18))
    for name in desc.columns.tolist():
        desc[name].plot(title = name, kind='bar', figsize=(14,34), ax=ax[round(i), j], fontsize=16, color='green')
        ax[round(i), j].tick_params(axis='x', rotation=0)
        for k, v in enumerate(desc[name].tolist()):
            ax[round(i), j].text(k -0.15 , v +0.05, str(v), color='black', size = 12) #v to x, k to y
        i +=0.5 
        if j==0: j=1 
        else: j=0
    f.tight_layout()
descriptive(df)


# In[19]:


df1 = dfcopy.pivot_table(values='WORK_LIFE_BALANCE_SCORE', index=['AGE'], columns=['GENDER'])
df1.head()


# In[20]:


result = df[df['BMI_RANGE']==1]
result


# In[21]:


wlb = result['WORK_LIFE_BALANCE_SCORE']
wlb


# In[24]:


f,ax = plt.subplots(7,3,figsize=(16,42))
f.suptitle('WHAT DRIVES OUR GOOD/BAD WORK LIFE BALANCE SCORE?', fontsize=20)

ax[0,0].set_title('AVERAGE WORK LIFE BALANCE BY GENDER AND AGE')
ax[0,1].set_title('AVERAGE WORK LIFE BALANCE & DAILY SHOUTING')
ax[0,2].set_title('AVERAGE WORK LIFE BALANCE & LOST VACATION')
ax[1,0].set_title('AVERAGE WORK LIFE BALANCE & ACHIEVEMENT')
ax[1,1].set_title('AVERAGE WORK LIFE BALANCE & TODO COMPLETED')
ax[1,2].set_title('AVERAGE WORK LIFE BALANCE & SUPPORTING OTHERS')
ax[2,0].set_title('AVERAGE WORK LIFE BALANCE & PLACES VISITED')
ax[2,1].set_title('AVERAGE WORK LIFE BALANCE & TIME FOR PASSION')
ax[2,2].set_title('AVERAGE WORK LIFE BALANCE & PERSONAL AWARDS')
ax[3,0].set_title('AVERAGE WORK LIFE BALANCE & FLOW')
ax[3,1].set_title('AVERAGE WORK LIFE BALANCE & LIVE VISION')
ax[3,2].set_title('AVERAGE WORK LIFE BALANCE & DONATION')
ax[4,0].set_title('AVERAGE WORK LIFE BALANCE & FRUITS AND VEGGIES')
ax[4,1].set_title('AVERAGE WORK LIFE BALANCE & WEEKLY MEDITATION')
ax[4,2].set_title('AVERAGE WORK LIFE BALANCE & SOCIAL NETWORK')
ax[5,0].set_title('AVERAGE WORK LIFE BALANCE & DAILY STEPS \n STEPS IN THOUSAND')
ax[5,1].set_title('AVERAGE WORK LIFE BALANCE & SLEEP HOURS')
ax[5,2].set_title('AVERAGE WORK LIFE BALANCE & MONTH')
ax[6,0].set_title('AVERAGE WORK LIFE BALANCE & DAILY STRESS')

ax[0,0].set_ylim([655, 685])
df1.plot(kind='bar', color=('darksalmon', 'cornflowerblue'), alpha=0.7, ax = ax[0,0])
ax[0,0].tick_params(axis='x', rotation=0)

sns.pointplot(x = 'DAILY_SHOUTING', y = 'WORK_LIFE_BALANCE_SCORE', data = df, ax = ax[0,1])
sns.pointplot(x = 'LOST_VACATION', y = 'WORK_LIFE_BALANCE_SCORE', data = df, ax = ax[0,2])
sns.pointplot(x = 'ACHIEVEMENT',  y = 'WORK_LIFE_BALANCE_SCORE', data=df, ax = ax[1,0])
sns.pointplot(x= 'TODO_COMPLETED',y='WORK_LIFE_BALANCE_SCORE', data=df, ax = ax[1,1])
sns.pointplot(x = 'SUPPORTING_OTHERS',  y = 'WORK_LIFE_BALANCE_SCORE', data=df, ax = ax[1,2])
sns.pointplot(x = 'PLACES_VISITED',  y = 'WORK_LIFE_BALANCE_SCORE', data=df, ax = ax[2,0])
sns.pointplot(x= 'TIME_FOR_PASSION',y='WORK_LIFE_BALANCE_SCORE', data=df, ax = ax[2,1])
sns.pointplot(x = 'PERSONAL_AWARDS',  y = 'WORK_LIFE_BALANCE_SCORE', data=df, ax = ax[2,2])
sns.pointplot(x = 'FLOW',  y = 'WORK_LIFE_BALANCE_SCORE', data=df, ax = ax[3,0])
sns.pointplot(x= 'LIVE_VISION',y='WORK_LIFE_BALANCE_SCORE', data=df, ax = ax[3,1])
sns.pointplot(x = 'DONATION',  y = 'WORK_LIFE_BALANCE_SCORE', data=df, ax = ax[3,2])
sns.pointplot(x = 'FRUITS_VEGGIES',  y = 'WORK_LIFE_BALANCE_SCORE', data=df, ax = ax[4,0])
sns.pointplot(x= 'WEEKLY_MEDITATION',y='WORK_LIFE_BALANCE_SCORE', data=df, ax = ax[4,1])
sns.pointplot(x = 'SOCIAL_NETWORK',  y = 'WORK_LIFE_BALANCE_SCORE', data=df, ax = ax[4,2])
sns.pointplot(x = 'DAILY_STEPS',  y = 'WORK_LIFE_BALANCE_SCORE', data=df, ax = ax[5,0])
sns.pointplot(x= 'SLEEP_HOURS',y='WORK_LIFE_BALANCE_SCORE', data=df, ax = ax[5,1])
sns.pointplot(x = 'MONTH',  y = 'WORK_LIFE_BALANCE_SCORE', data=df, ax = ax[5,2])
sns.pointplot(x = 'DAILY_STRESS',  y = 'WORK_LIFE_BALANCE_SCORE', data=df, ax = ax[6,0])

plt.show()


# In[25]:


df2 = dfcopy.pivot_table(values='ACHIEVEMENT', index=['AGE'], columns=['GENDER'], )
df2.head()


# In[26]:


f,ax = plt.subplots(3,3,figsize=(20,18))
f.suptitle('WHAT DRIVES OUR ACHIEVEMENTS?', fontsize=20)

ax[0,0].set_title('AVERAGE ACHIEVEMENT BY GENDER AND AGE')
ax[0,1].set_title('AVERAGE ACHIEVEMENT BY TIME FOR PASSION')
ax[0,2].set_title('AVERAGE ACHIEVEMENT BY SUPPORTING OTHERS')
ax[1,0].set_title('AVERAGE ACHIEVEMENT BY TODO COMPLETED')
ax[1,1].set_title('AVERAGE ACHIEVEMENT BY LIVE VISION')
ax[1,2].set_title('AVERAGE ACHIEVEMENT BY PERSONAL AWARDS')
ax[2,0].set_title('AVERAGE ACHIEVEMENT BY DAILY STRESS')
ax[2,1].set_title('AVERAGE ACHIEVEMENT BY TIME FOR PASSION')
ax[2,2].set_title('AVERAGE ACHIEVEMENT BY FLOW')

ax[0,0].set_ylim([3.5, 4.5])
df2.plot(kind='bar', color=('darksalmon', 'cornflowerblue'), alpha=0.7, ax = ax[0,0])
ax[0,0].tick_params(axis='x', rotation=0)

sns.pointplot(x= 'TIME_FOR_PASSION',y='ACHIEVEMENT', data=df, ax = ax[0,1])
sns.pointplot(x = 'SUPPORTING_OTHERS', y = 'ACHIEVEMENT', data = df, ax = ax[0,2])
sns.pointplot(x = 'TODO_COMPLETED',  y = 'ACHIEVEMENT',    data=df, ax = ax[1,0])
sns.pointplot(x = 'LIVE_VISION',  y = 'ACHIEVEMENT',    data=df, ax = ax[1,1])
sns.pointplot(x = 'PERSONAL_AWARDS',  y = 'ACHIEVEMENT',    data=df, ax = ax[1,2])
sns.pointplot(x = 'DAILY_STRESS',  y = 'ACHIEVEMENT',    data=df, ax = ax[2,0])
sns.pointplot(x = 'TIME_FOR_PASSION',  y = 'ACHIEVEMENT',    data=df, ax = ax[2,1])
sns.pointplot(x = 'FLOW',  y = 'ACHIEVEMENT',    data=df, ax = ax[2,2])

plt.show()


# In[27]:


f,ax = plt.subplots(2,3,figsize=(20,14))
f.suptitle('WHAT DRIVES US TO BE PRODUCTIVE?', fontsize=20)

ax[0,0].set_title('TODO COMPLETED BY GENDER AND AGE')
ax[0,1].set_title('TODO COMPLETED BY TIME FOR PASSION')
ax[0,2].set_title('TODO COMPLETED BY LIVE VISION')
ax[1,0].set_title('TODO COMPLETED BY FLOW')
ax[1,1].set_title('TODO COMPLETED BY ACHIEVEMENT')
ax[1,2].set_title('TODO COMPLETED BY SUPPORTING OTHERS')

ax[0,0].set_ylim([650, 685])
df1.plot(kind='bar', color=('darksalmon', 'cornflowerblue'), alpha=0.7, ax = ax[0,0])
ax[0,0].tick_params(axis='x', rotation=0)

sns.pointplot(x= 'TIME_FOR_PASSION',y='TODO_COMPLETED', data=df, ax = ax[0,1])
sns.pointplot(x = 'LIVE_VISION', y = 'TODO_COMPLETED', data = df, ax = ax[0,2])
sns.pointplot(x = 'FLOW',  y = 'TODO_COMPLETED',    data=df, ax = ax[1,0])
sns.pointplot(x = 'ACHIEVEMENT',  y = 'TODO_COMPLETED',    data=df, ax = ax[1,1])
sns.pointplot(x = 'SUPPORTING_OTHERS',  y = 'TODO_COMPLETED',    data=df, ax = ax[1,2])

plt.show()

