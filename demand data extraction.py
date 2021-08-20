#!/usr/bin/env python
# coding: utf-8

# # Read Excel and Import Pandas

# In[2]:


#import all python essential tool
import pandas as pd
df = pd.read_excel (r'raw_fes_2020.xlsx')


# ### Set Up the FES Outlook - And Choose the DSO Liscence Location 

# In[3]:


#choose the scenario - ST = System Transform Scenario (Base Case)
ST = df[df['scenario'] == 'ST']
#ID of the location, see FES Location Excel File
GSP = ST[ST['GSP'] == 'BRWA1']

 #we are extracting the column for average demand peak for each year of each demand type 
C = GSP[GSP['type'] == 'C']['avg_demand_peak']
D = GSP[GSP['type'] == 'D']['avg_demand_peak']
E = GSP[GSP['type'] == 'E']['avg_demand_peak']
H = GSP[GSP['type'] == 'H']['avg_demand_peak']
I = GSP[GSP['type'] == 'I']['avg_demand_peak']
R = GSP[GSP['type'] == 'R']['avg_demand_peak']
Z = GSP[GSP['type'] == 'Z']['avg_demand_peak']


# In[7]:


C 


# ### Create a Data Frame for Putting All Relevant Data into An Excel File

# In[4]:


new_df = pd.DataFrame()
for element in range(len(C)):
    CD = C.iloc[element]
    DD = D.iloc[element]
    ED = E.iloc[element]
    HD = H.iloc[element]
    ID = I.iloc[element]
    RD = R.iloc[element]
    ZD = Z.iloc[element]
    
    row = {'Commercial' : CD, 'District_Heating' : DD, 'Electric_Vehicle' : ED, 'Heating' : HD,           'Industrial' : ID, 'Residential' : RD, 'Electrolyzer' : ZD}
    new_df = new_df.append(row, ignore_index = True)
    
new_df['Total'] = new_df['Commercial'] + new_df['District_Heating'] + new_df['Electric_Vehicle'] + new_df['Electrolyzer'] + new_df['Heating'] + new_df['Industrial'] + new_df['Residential']

#name the excel file to be exported
#new_df.to_csv('node_29_BRWA1_demand.csv')


# In[5]:


new_df


# In[ ]:




