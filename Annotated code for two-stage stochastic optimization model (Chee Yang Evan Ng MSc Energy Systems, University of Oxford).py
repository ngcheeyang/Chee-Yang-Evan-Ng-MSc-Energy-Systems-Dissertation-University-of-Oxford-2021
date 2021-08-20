#!/usr/bin/env python
# coding: utf-8

# # Two-Stage Stochastic Optimization Model - MSc Energy Systems 
# # Student: Evan Ng
# # Supervisor: Dr Iacopo Savelli

# ### Import Libraries  

# In[1]:


#import all python essential tool
import pandas as pd

#Load PYOMO libraries 
import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.opt import SolverFactory, TerminationCondition

# Create an object representing the solver, in this case CPLEX
solver = SolverFactory("cplex")


# ### Create Sets Involved for Indexing  

# In[2]:


#Create a pyomo concrete model:
model = ConcreteModel (name="(two_stage_stochastic_optimization)")

#Create a set for the nodes
number_of_node = 29
model.node_set = Set(initialize=range(1, number_of_node+1))

#Create a set for the time
number_of_time = 24
model.time_set = Set(initialize=range(1, number_of_time+1))

#Create a set for the years
#the analysis consider 5 years average of each block of year of operation to reduce computational time
number_of_year = 5
model.year_set = Set(initialize=range(1, number_of_year+1))

#Create a set for the years
number_of_scenario = 100
model.scenario_set = Set(initialize=range(1, number_of_scenario+1))

#Create a set for existing transmission line 
#import the existing line data from excel
Existing_transmission_line = pd.read_excel (r'updated_existing_transmission_line.xlsx')
from_node = Existing_transmission_line['from_bus']
to_node = Existing_transmission_line['to_bus']
maximum_flow = Existing_transmission_line['rating_MVA'] #maximum flow in MW, we assume that maximum negative flow is - maximum flow
reactance = Existing_transmission_line['reactance_pu']

existing_lines = {}
for line in range(len(Existing_transmission_line + 1)):
        existing_lines[(from_node[line] , to_node[line])] = [float(reactance[line]) , float(maximum_flow[line])]         
model.existing_line_set = Set(initialize=existing_lines.keys(),dimen=2)

#Create a set for candidate transmission lines
#import the candidate line data from excel
Candidate_transmission_line = pd.read_excel (r'updated_candidate_transmission_line.xlsx')
from_node_c = Candidate_transmission_line['from_bus']
to_node_c = Candidate_transmission_line['to_bus']
maximum_flow_c = Candidate_transmission_line['rating_MVA'] #maximum flow in MW, we assume that maximum negative flow is - maximum flow
reactance_c = Candidate_transmission_line['reactance_pu']
line_cost = Candidate_transmission_line['CAPEX_GBP']

candidate_lines = {}
for linec in range(len(Candidate_transmission_line)):
        candidate_lines[(from_node[linec] , to_node[linec])] = [float(reactance_c[linec]) , float(maximum_flow_c[linec]),float(line_cost[linec])]         
model.candidate_line_set = Set(initialize=candidate_lines.keys(),dimen=2)


# ### Create the First Stage Variables Involved (Generation)

# In[3]:


#Creating the first stage variables
#Read the excel file to obtain the maximum installed capacity of each technology across all nodes

Capacity_constraint_for_generator = pd.read_excel (r'Generation_constraints.xlsx')
lower = Capacity_constraint_for_generator['lower'] #this is basically all 0
upper_RS = Capacity_constraint_for_generator['max_RS_MW']
upper_US = Capacity_constraint_for_generator['max_US_MW']
upper_ON = Capacity_constraint_for_generator['max_ON_MW']
upper_OF = Capacity_constraint_for_generator['max_OF_MW']
upper_NP = Capacity_constraint_for_generator['max_NP_MW']
upper_HP = Capacity_constraint_for_generator['max_NP_MW']

#This set the upper bound to fufil equation (7),(8),(9)
#setting the maximum value of installed capacity at each node
def RS_domain(model,i):
    return  (lower[i-1],upper_RS[i-1])
def US_domain(model,i):
    return  (lower[i-1],upper_US[i-1])
def ON_domain(model,i):
    return  (lower[i-1],upper_ON[i-1])
def OF_domain(model,i):
    return  (lower[i-1],upper_OF[i-1])
def NP_domain(model,i):
    return  (lower[i-1],upper_NP[i-1])
def HP_domain(model,i):
    return  (lower[i-1],upper_HP[i-1])

#Create the variables for installed capacity of each technology in first stage
#note this, this first stage variable is only being indexed over all nodes - not other sets
model.RS_capacity_var = pyo.Var(model.node_set,bounds=RS_domain,initialize = 0)
model.US_capacity_var = pyo.Var(model.node_set,bounds=US_domain,initialize = 0)
model.ON_capacity_var = pyo.Var(model.node_set,bounds=ON_domain,initialize = 0)
model.OF_capacity_var = pyo.Var(model.node_set,bounds=OF_domain,initialize = 0)
model.NP_capacity_var = pyo.Var(model.node_set,bounds=NP_domain,initialize = 0)
model.HP_capacity_var = pyo.Var(model.node_set,bounds=NP_domain,initialize = 0)
model.NG_capacity_var = pyo.Var(model.node_set,within=pyo.NonNegativeReals,initialize = 0)
model.BE_capacity_var = pyo.Var(model.node_set,within=pyo.NonNegativeReals,initialize = 0)
model.CP_capacity_var = pyo.Var(model.node_set,within=pyo.NonNegativeReals,initialize = 0)


# ### Create the First Stage Variables Involved (Transmission Lines - Binary) 

# In[4]:


model.new_transmission_line_var = pyo.Var(model.candidate_line_set, within=pyo.Binary, initialize = 0)


# ### Create the First Stage Variables Involved (Interconnectors - Binary) 

# In[5]:


model.new_interconnector_var = pyo.Var(model.node_set,within=pyo.Binary, initialize = 0)


# ### Create the Second Stage Variables Involved (Energy Generation, Load Shedding, and Energy Import/Export) 

# In[6]:


#Creating the variable of energy generation, index across nodes, time, year, and scenario
model.RS_gen_var = pyo.Var(model.scenario_set,model.year_set,model.time_set,model.node_set, domain=NonNegativeReals)
model.US_gen_var = pyo.Var(model.scenario_set,model.year_set,model.time_set,model.node_set, domain=NonNegativeReals)
model.ON_gen_var = pyo.Var(model.scenario_set,model.year_set,model.time_set,model.node_set, domain=NonNegativeReals)
model.OF_gen_var = pyo.Var(model.scenario_set,model.year_set,model.time_set,model.node_set, domain=NonNegativeReals)
model.NP_gen_var = pyo.Var(model.scenario_set,model.year_set,model.time_set,model.node_set, domain=NonNegativeReals)
model.HP_gen_var = pyo.Var(model.scenario_set,model.year_set,model.time_set,model.node_set, domain=NonNegativeReals)
model.NG_gen_var = pyo.Var(model.scenario_set,model.year_set,model.time_set,model.node_set, domain=NonNegativeReals)
model.BE_gen_var = pyo.Var(model.scenario_set,model.year_set,model.time_set,model.node_set, domain=NonNegativeReals)
model.CP_gen_var = pyo.Var(model.scenario_set,model.year_set,model.time_set,model.node_set, domain=NonNegativeReals)

#The variables for interconnector import and export is defined, which will be treated as generation and demand, respectively
model.existing_inter_import_var = pyo.Var(model.scenario_set,model.year_set,model.time_set,model.node_set, domain=NonNegativeReals)
model.existing_inter_export_var = pyo.Var(model.scenario_set,model.year_set,model.time_set,model.node_set, domain=NonNegativeReals)
model.candidate_inter_import_var = pyo.Var(model.scenario_set,model.year_set,model.time_set,model.node_set, domain=NonNegativeReals)
model.candidate_inter_export_var = pyo.Var(model.scenario_set,model.year_set,model.time_set,model.node_set, domain=NonNegativeReals)


# ### Create A Dictionary Parameter for Scenario Data - Capacity Factor

# In[7]:


#Input all relevant data from excel files
RS_data = pd.read_excel(r'RS_capacity_factor_first_8_scenario.xlsx')
US_data = pd.read_excel(r'US_capacity_factor_first_8_scenario.xlsx')
ON_data = pd.read_excel(r'ON_capacity_factor_first_8_scenario.xlsx')
OF_data = pd.read_excel(r'OF_capacity_factor_first_8_scenario.xlsx')

#In the excel files, scenarios k = (1-25) are winter scenario, scenario k = (26-50) are summer, k =(51-100) are spring/fall 

    
#RS_data[RS_data['k'] == 1]
df_RS = pd.DataFrame(RS_data) 
df_US = pd.DataFrame(US_data) 
df_ON = pd.DataFrame(ON_data) 
df_OF = pd.DataFrame(OF_data) 

#Stochastic scenario - RS capacity factor
dict_RS_CF = {}  #{(k,t,n): for k in model.scenario_set for t in model.time_set for n in model.node_set}
for k in range(1,number_of_scenario+1):
    for t in range(1,number_of_time+1):
        for n in range(1,number_of_node+1):
            data_label = 'n'+str(n)+'_t'+str(t)
            dict_RS_CF[(k,t,n)] = (float(df_RS.loc[df_RS['k'] == k][data_label].iloc[0]))

#Stochastic scenario - US capacity factor
dict_US_CF = {}  #{(k,t,n): for k in model.scenario_set for t in model.time_set for n in model.node_set}
for k in range(1,number_of_scenario+1):
    for t in range(1,number_of_time+1):
        for n in range(1,number_of_node+1):
            data_label = 'n'+str(n)+'_t'+str(t)
            dict_US_CF[(k,t,n)] = (float(df_US.loc[df_US['k'] == k][data_label].iloc[0]))

dict_ON_CF = {}  #{(k,t,n): for k in model.scenario_set for t in model.time_set for n in model.node_set}
for k in range(1,number_of_scenario+1):
    for t in range(1,number_of_time+1):
        for n in range(1,number_of_node+1):
            data_label = 'n'+str(n)+'_t'+str(t)
            dict_ON_CF[(k,t,n)] = (float(df_ON.loc[df_ON['k'] == k][data_label].iloc[0]))  
            
dict_OF_CF = {}  #{(k,t,n): for k in model.scenario_set for t in model.time_set for n in model.node_set}
for k in range(1,number_of_scenario+1):
    for t in range(1,number_of_time+1):
        for n in range(1,number_of_node+1):
            data_label = 'n'+str(n)+'_t'+str(t)
            dict_OF_CF[(k,t,n)] = (float(df_OF.loc[df_OF['k'] == k][data_label].iloc[0]))


# ### Create A Dictionary Parameter for Scenario Data - Demand Load Profile

# In[8]:


#demand dictionary for year group 1 - 5

#Input all relevant data from excel files
demand_data_y1 = pd.read_excel(r'load_profile_y1_first_8_scenario.xlsx')
demand_data_y2 = pd.read_excel(r'load_profile_y2_first_8_scenario.xlsx')
demand_data_y3 = pd.read_excel(r'load_profile_y3_first_8_scenario.xlsx')
demand_data_y4 = pd.read_excel(r'load_profile_y4_first_8_scenario.xlsx')
demand_data_y5 = pd.read_excel(r'load_profile_y5_first_8_scenario.xlsx')

#In the excel files, scenarios k = (1-25) are winter scenario, scenario k = (26-50) are summer, k =(51-100) are spring/fall 
    
#RS_data[RS_data['k'] == 1]
df_demand_y1 = pd.DataFrame(demand_data_y1) 
df_demand_y2 = pd.DataFrame(demand_data_y2) 
df_demand_y3 = pd.DataFrame(demand_data_y3) 
df_demand_y4 = pd.DataFrame(demand_data_y4) 
df_demand_y5 = pd.DataFrame(demand_data_y5) 


dict_demand = {}  #{(k,t,n): for k in model.scenario_set for t in model.time_set for n in model.node_set}
for k in range(1,number_of_scenario+1):
    for t in range(1,number_of_time+1):
        for n in range(1,number_of_node+1):
            data_label = 'n'+str(n)+'_t'+str(t)
            dict_demand[(k,1,t,n)] = (float(df_demand_y1.loc[df_demand_y1['k'] == k][data_label].iloc[0]))

for k in range(1,number_of_scenario+1):
    for t in range(1,number_of_time+1):
        for n in range(1,number_of_node+1):
            data_label = 'n'+str(n)+'_t'+str(t)
            dict_demand[(k,2,t,n)] = (float(df_demand_y2.loc[df_demand_y2['k'] == k][data_label].iloc[0]))

for k in range(1,number_of_scenario+1):
    for t in range(1,number_of_time+1):
        for n in range(1,number_of_node+1):
            data_label = 'n'+str(n)+'_t'+str(t)
            dict_demand[(k,3,t,n)] = (float(df_demand_y3.loc[df_demand_y3['k'] == k][data_label].iloc[0]))
            
for k in range(1,number_of_scenario+1):
    for t in range(1,number_of_time+1):
        for n in range(1,number_of_node+1):
            data_label = 'n'+str(n)+'_t'+str(t)
            dict_demand[(k,4,t,n)] = (float(df_demand_y4.loc[df_demand_y4['k'] == k][data_label].iloc[0]))
            
for k in range(1,number_of_scenario+1):
    for t in range(1,number_of_time+1):
        for n in range(1,number_of_node+1):
            data_label = 'n'+str(n)+'_t'+str(t)
            dict_demand[(k,5,t,n)] = (float(df_demand_y5.loc[df_demand_y5['k'] == k][data_label].iloc[0]))


# ### Implement the Generation Constrain for Each Technology 

# In[9]:


#Generation constraint for renewable energy generation - equation (3.10)
def RS_gen_rule(self,k,y,t,n): 
    return model.RS_gen_var[k,y,t,n] == model.RS_capacity_var[n] * dict_RS_CF[k,t,n]
model.RS_generation_con = pyo.Constraint(model.scenario_set,model.year_set,model.time_set,model.node_set, rule=RS_gen_rule)

def US_gen_rule(self,k,y,t,n): 
    return model.US_gen_var[k,y,t,n] == model.US_capacity_var[n] * dict_US_CF[k,t,n]
model.US_generation_con = pyo.Constraint(model.scenario_set,model.year_set,model.time_set,model.node_set, rule=US_gen_rule)

def ON_gen_rule(self,k,y,t,n): 
    return model.ON_gen_var[k,y,t,n] == model.ON_capacity_var[n] * dict_ON_CF[k,t,n]
model.ON_generation_con = pyo.Constraint(model.scenario_set,model.year_set,model.time_set,model.node_set, rule=ON_gen_rule)

def OF_gen_rule(self,k,y,t,n): 
    return model.OF_gen_var[k,y,t,n] == model.OF_capacity_var[n] * dict_OF_CF[k,t,n]
model.OF_generation_con = pyo.Constraint(model.scenario_set,model.year_set,model.time_set,model.node_set, rule=OF_gen_rule)


#Generation constraint for baseload energy generation - equation (3.11)
#nuclear is define to have an operating load factor between 23% and 100% of installed capacity - see report
def NP_gen_rule_min (self,k,y,t,n):
    return model.NP_gen_var[k,y,t,n] >= 0.23 * model.NP_capacity_var[n]
model.NP_generation_min_con = pyo.Constraint(model.scenario_set,model.year_set,model.time_set,model.node_set, rule=NP_gen_rule_min)

def NP_gen_rule_max (self,k,y,t,n):
    return model.NP_gen_var[k,y,t,n] <= model.NP_capacity_var[n]
model.NP_generation_max_con = pyo.Constraint(model.scenario_set,model.year_set,model.time_set,model.node_set, rule=NP_gen_rule_max)

def HP_gen_rule_min (self,k,y,t,n):
    return model.NP_gen_var[k,y,t,n] >= 0.05 * model.HP_capacity_var[n]
model.HP_generation_min_con = pyo.Constraint(model.scenario_set,model.year_set,model.time_set,model.node_set, rule=HP_gen_rule_min)

def HP_gen_rule_max (self,k,y,t,n):
    return model.NP_gen_var[k,y,t,n] <= 0.362 * model.HP_capacity_var[n]
model.NP_generation_max_con = pyo.Constraint(model.scenario_set,model.year_set,model.time_set,model.node_set, rule=HP_gen_rule_max)

#Generation constraint for thermal energy generation - equation (3.12)
#PS: the minimum for thermal generation is 0 - and this is notdefined as a constrait because the variable defination
#has already set this variable in the domain of non-negative real number

def NG_gen_rule_max (self,k,y,t,n):
    return model.NG_gen_var[k,y,t,n] <= model.NG_capacity_var[n]
model.NG_generation_max_con = pyo.Constraint(model.scenario_set,model.year_set,model.time_set,model.node_set, rule=NG_gen_rule_max)

def BE_gen_rule_max (self,k,y,t,n):
    return model.NG_gen_var[k,y,t,n] <= model.NG_capacity_var[n]
model.NG_generation_max_con = pyo.Constraint(model.scenario_set,model.year_set,model.time_set,model.node_set, rule=BE_gen_rule_max) 

def CP_gen_rule_max (self,k,y,t,n):
    return model.NG_gen_var[k,y,t,n] <= model.NG_capacity_var[n]
model.NG_generation_max_con = pyo.Constraint(model.scenario_set,model.year_set,model.time_set,model.node_set, rule=CP_gen_rule_max) 


# ### Implement the Power Import and Export Constraint via Existing and Candidate Interconnector 

# In[10]:


#Import the technical and economical data for existing and candidate interconnector - create a dictionary
##Approach taken - to model interconnector as generator/ demand: make every node has "interconnector" 
#but most has a rating (generation and demand) of 0 MW - aka non-existence 

#Import all relevant interconnector data for both existing and candidate interconnectors 
#Load the file for existing interconnectors
Existing_interconnector = pd.read_excel (r'all_node_existing_interconnector.xlsx')
EI_node = Existing_interconnector['node']
EI_name = Existing_interconnector['name']
EI_rating = Existing_interconnector['capacity_MW']

existing_interconnectors = {}
for inter in range(len(Existing_interconnector)):
    existing_interconnectors[EI_node[inter]] = [EI_name[inter],float(EI_rating[inter])]
#model.existing_interconnector_set = Set(model.node_set,initialize=existing_interconnectors.keys())

#Load the file for candidate interconnectors
Candidate_interconnector = pd.read_excel (r'all_node_candidate_interconnector.xlsx')
CI_node = Candidate_interconnector['node']
CI_name = Candidate_interconnector['name']
CI_rating = Candidate_interconnector['capacity_MW']
CI_capex = Candidate_interconnector['CAPEX']

candidate_interconnectors = {}
for interc in range(len(Existing_interconnector)):
    candidate_interconnectors[CI_node[interc]] = [CI_name[interc],float(CI_rating[interc]),float(CI_capex[interc])]    
#model.candidate_interconnector_set = Set(model.node_set,initialize=candidate_interconnectors.keys())
#^^ do we still need this? Maybe for constraint?

#Now we can set the energy trading constraints - Equation (16) - (19)
def existing_inter_import_rule (self,k,y,t,n):
    return model.existing_inter_import_var[k,y,t,n] <= existing_interconnectors[n][1]
model.existing_inter_import_con = pyo.Constraint(model.scenario_set,model.year_set,model.time_set,model.node_set,rule=existing_inter_import_rule)
    
def existing_inter_export_rule (self,k,y,t,n):
    return model.existing_inter_export_var[k,y,t,n] <= existing_interconnectors[n][1]
model.existing_inter_export_con = pyo.Constraint(model.scenario_set,model.year_set,model.time_set,model.node_set,rule=existing_inter_export_rule) 

#candidate interconnectors' will have zero capacity, if the binary variable to build it is zero
#meaning that the interconnector will not be constructed 
def candidate_inter_import_rule (self,k,y,t,n):
    return model.candidate_inter_import_var[k,y,t,n] <= candidate_interconnectors[n][1] * model.new_interconnector_var[n] 
model.candidate_inter_import_con = pyo.Constraint(model.scenario_set,model.year_set,model.time_set,model.node_set,rule=candidate_inter_import_rule)

def candidate_inter_export_rule (self,k,y,t,n):
    return model.candidate_inter_export_var[k,y,t,n] <= candidate_interconnectors[n][1] * model.new_interconnector_var[n] 
model.candidate_inter_export_con = pyo.Constraint(model.scenario_set,model.year_set,model.time_set,model.node_set,rule=candidate_inter_export_rule)


# ### Power Flow Constraint: Create the Incident Matrix for Existing and Candidate Transmission Line

# In[11]:


# incidence matrix for existing line: 
#these incident matrix is needed for power balance
LE = {}
for (i, j) in model.existing_line_set:
    for n in model.node_set:
        if n == i:
            LE[i,j,n] = 1
        elif n == j:
            LE[i,j,n] = -1
        else:
            LE[i,j,n] = 0
 
            
# incidence matrix for candidate line
LC = {}
for (i, j) in model.candidate_line_set:
    for n in model.node_set:
        if n == i:
            LC[i,j,n] = 1
        elif n == j:
            LC[i,j,n] = -1
        else:
            LC[i,j,n] = 0

#**PS make sure power balance consider both LE and LC - so power flowring through candidate line is also considered
#How is this incident matrix use? Havent use it yet


# ### Set the Power Flow Constraint (Flow Capacity) for both Exisiting and Candidate Line 

# In[12]:


#Set an indexted variable for power flow through existing line
#Note: The domain for power flow is set as Reals instead of non-negative real because we are 
#modelling this power flow as two-way flow, in which negative flow represent a flow in the opposite direction
model.power_flow_through_existing_line = pyo.Var(model.scenario_set,model.year_set,model.time_set,model.existing_line_set, domain=Reals)
model.power_flow_through_candidate_line = pyo.Var(model.scenario_set,model.year_set,model.time_set,model.candidate_line_set, domain=Reals)

#Setting the capacity constraint for power flow - equation (14)
def min_flow_rule_for_existing_line (self,k,y,t,i,j):
    return model.power_flow_through_existing_line[k,y,t,i,j] >= -existing_lines[(i,j)][1]
model.min_flow_existing_line_con = pyo.Constraint(model.scenario_set,model.year_set,model.time_set,model.existing_line_set,rule=min_flow_rule_for_existing_line)
    
def max_flow_rule_for_existing_line (self,k,y,t,i,j):
    return model.power_flow_through_existing_line[k,y,t,i,j] <= existing_lines[(i,j)][1]
model.max_flow_existing_line_con = pyo.Constraint(model.scenario_set,model.year_set,model.time_set,model.existing_line_set,rule=max_flow_rule_for_existing_line)
    
#For candidate line, the binary variable of constructing the candidate transmission line is also included   
def min_flow_rule_for_candidate_line (self,k,y,t,i,j):
    return model.power_flow_through_candidate_line[k,y,t,i,j] >= -candidate_lines[(i,j)][1] * model.new_transmission_line_var[(i,j)]
model.min_flow_candidate_line_con = pyo.Constraint(model.scenario_set,model.year_set,model.time_set,model.candidate_line_set,rule=min_flow_rule_for_candidate_line)
    
def max_flow_rule_for_candidate_line (self,k,y,t,i,j):
    return model.power_flow_through_candidate_line[k,y,t,i,j] <= candidate_lines[(i,j)][1] * model.new_transmission_line_var[(i,j)]
model.max_flow_candidate_line_con = pyo.Constraint(model.scenario_set,model.year_set,model.time_set,model.candidate_line_set,rule=max_flow_rule_for_candidate_line)


# ### Set the Power Flow Constraint (Phase-Angle) for both Exisiting and Candidate Line  

# In[13]:


#Create variable of phase angle
model.theta_existing_line = pyo.Var(model.scenario_set,model.year_set,model.time_set,model.node_set, domain=Reals)
model.theta_candidate_line = pyo.Var(model.scenario_set,model.year_set,model.time_set,model.node_set, domain=Reals)

#Create the constrait that leads to a DC load flow meshed network model for existing line
def DC_loadflow_rule(self,k, y, t, i, j):
    return model.power_flow_through_existing_line[k,y,t,(i,j)] == (model.theta_existing_line[k,y,t,i] - model.theta_existing_line[k,y,t,j])/(existing_lines[(i,j)][0]) 
model.DC_load_flow_existing_line_con = pyo.Constraint(model.scenario_set,model.year_set,model.time_set,model.existing_line_set,rule=DC_loadflow_rule)     

#Create the constrait that leads to a DC load flow meshed network model for candidate line - introduce binary variable
def DC_loadflow_rule_c(self,k, y, t, i, j):
    return model.power_flow_through_candidate_line[k,y,t,i,j] == (model.theta_candidate_line[k,y,t,i] - model.theta_candidate_line[k,y,t,j])/(candidate_lines[(i,j)][0])
model.DC_load_flow_candidate_line_con = pyo.Constraint(model.scenario_set,model.year_set,model.time_set,model.candidate_line_set,rule=DC_loadflow_rule_c)     


# ### Set the Power Balance Constraint 

# In[14]:


#Power balance constraint 

def power_balance_rule(self,k,y,t,n):
    total_local_demand = dict_demand[k,y,t,n] + model.existing_inter_export_var[k,y,t,n] + model.candidate_inter_export_var[k,y,t,n]
    total_local_supply = model.RS_gen_var[k,y,t,n] + model.US_gen_var[k,y,t,n] + model.ON_gen_var[k,y,t,n] + model.OF_gen_var[k,y,t,n] + model.NP_gen_var[k,y,t,n] + model.NG_gen_var[k,y,t,n] + model.existing_inter_import_var[k,y,t,n] + model.candidate_inter_import_var[k,y,t,n]
    flows_e = sum(LE[i,j,n]*model.power_flow_through_existing_line[k,y,t,i,j] for (i,j) in model.existing_line_set)
    flows_c = sum(LC[i,j,n]*model.power_flow_through_candidate_line[k,y,t,i,j] for (i,j) in model.candidate_line_set)
    return total_local_demand - total_local_supply + flows_e + flows_c == 0
    
model.power_balance = pyo.Constraint(model.scenario_set,model.year_set,model.time_set,model.node_set, rule = power_balance_rule)   


# ### Insert All Capital Cost Parameter to Model 

# In[15]:


#Capital cost parameters - these are inserted as a scalar 

#setting a CAPEX parameter for cadidate transmission lines to be built 
def line_capex_rule(self, i,j):
    return candidate_lines[(i,j)][2]
model.new_transmission_line_capex_param = pyo.Param(model.candidate_line_set, initialize = line_capex_rule)

generation_capex = pd.read_excel (r'CAPEX_of_generation.xlsx')
technology = generation_capex['Technology']
technology_capex = generation_capex['CAPEX_GBP/MW']

#Set a scalar parameter for the unit cost of CAPEX required to construct each MW of different generation technology
model.unit_cost_RS = pyo.Param(initialize = int(technology_capex[0]))
model.unit_cost_US = pyo.Param(initialize = int(technology_capex[1]))
model.unit_cost_ON = pyo.Param(initialize = int(technology_capex[2]))
model.unit_cost_OF = pyo.Param(initialize = int(technology_capex[3]))
model.unit_cost_NP = pyo.Param(initialize = int(technology_capex[4]))
model.unit_cost_HP = pyo.Param(initialize = int(technology_capex[5]))
model.unit_cost_NG = pyo.Param(initialize = int(technology_capex[6]))
model.unit_cost_BE = pyo.Param(initialize = int(technology_capex[7]))
model.unit_cost_CP = pyo.Param(initialize = int(technology_capex[8]))


def interconnector_capex_rule(self, interconnector):
    return candidate_interconnectors[interconnector][2]
model.new_interconnector_capex_param = pyo.Param(model.node_set, initialize = interconnector_capex_rule)


# ### Create Dictionary for Annual Costs - Energy Generation Cost

# In[16]:


#OPEX of Each Generation Technology 
#opex includes operation and maintenence cost needed to geneerate unit of energy [£/MWh]
generation_opex = pd.read_excel (r'gen_annual_cost.xlsx')
#opex_RS[y] for y in mode.set_year
opex_RS = generation_opex['RS_opex']
opex_US = generation_opex['US_opex']
opex_ON = generation_opex['ON_opex']
opex_OF = generation_opex['OF_opex']
opex_NP = generation_opex['NP_opex']
opex_HP = generation_opex['HP_opex']
opex_BE = generation_opex['BE_opex']
opex_CP = generation_opex['CP_opex']


# ### Create Dictionary for Annual Costs - Energy Trading Cost

# In[17]:


#annual cost for energy trading
trading_price = pd.read_excel (r'trading_annual_cost.xlsx')

#read time series data for energy import 
North_Connect_import = trading_price['North_Connect_import_3']
Moyle_import = trading_price['Moyle_import_5']
North_Sea_import = trading_price['North_Sea_import_7']
East_West_import = trading_price['East_West_import_12']
Viking_Link_import = trading_price['Viking_Link_import_19']
BritNed_import = trading_price['BritNed_import_20']
Nemo_Link_import = trading_price['Nemo_Link_import_26']
IFA_1_import = trading_price['IFA_1_import_27']
IFA_2_import = trading_price['IFA_2_import_28']

Ice_Link_import = trading_price['Ice_Link_import_1']
Green_Link_import = trading_price['Green_Link_import_8']
Neu_Connect_import = trading_price['Neu_Connect_import_26']
FAB_Link_import = trading_price['FAB_Link_import_29']

#read time series data for energy export 
North_Connect_export = trading_price['North_Connect_export_3']
Moyle_export = trading_price['Moyle_export_5']
North_Sea_export = trading_price['North_Sea_export_7']
East_West_export = trading_price['East_West_export_12']
Viking_Link_export = trading_price['Viking_Link_export_19']
BritNed_export = trading_price['BritNed_export_20']
Nemo_Link_export = trading_price['Nemo_Link_export_26']
IFA_1_export = trading_price['IFA_1_export_27']
IFA_2_export = trading_price['IFA_2_export_28']

Ice_Link_export = trading_price['Ice_Link_export_1']
Green_Link_export = trading_price['Green_Link_export_8']
Neu_Connect_export = trading_price['Neu_Connect_export_26']
FAB_Link_export = trading_price['FAB_Link_export_29']


# ### Total Emissions Constraint and Objective

# In[18]:


#Equation (3): Cost of carbon abatement 
#annual cost for carbon abatement

#Reserve for emisions constraints
#For ease of altering the emissions related parameter to perform sensitivity analysis
#All emission related parameter will be entered manually

#NG_emission_factor = 0.240 #tonCO2e/MWh
#
#For simplicity, we assume UK carbon abatement potential with NB as total annual emissions
NB_limit = 3404169 #tonCO2e/year - this is for the base case of using UK's current LULUCF credit
#total_annual_emission = 10000000 #tonCO2e/year

model.NG_F = pyo.Param(initialize = NG_emission_factor)
model.BE_F = pyo.Param(initialize = BE_emission_factor)
model.CP_F = pyo.Param(initialize = CP_emission_factor)
model.NB_max = pyo.Param(initialize = NB_limit)
model.total_CO2_max = pyo.Param(initialize = total_annual_emission)

carbon_cost = pd.read_excel (r'carbon_abatement_annual_cost.xlsx')
NB_cost = carbon_cost['NB_cost']
CCS_cost = carbon_cost['CCS_cost']

#Set up the rule to define the variable of annual emissions
model.annual_emission_var = pyo.Var(model.scenario_set, model.year_set,domain=NonNegativeReals)

def annual_CO2_rule (self,k,y):
    emissions_NG = sum(model.NG_gen_var[k,y,t,n] for t in model.time_set for n in model.node_set)
    emissions_BE = sum(model.BE_gen_var[k,y,t,n] for t in model.time_set for n in model.node_set)
    emissions_CP = sum(model.CP_gen_var[k,y,t,n] for t in model.time_set for n in model.node_set)
    return model.annual_emission_var[k,y] == 365*(emissions_NG * model.NG_F + emissions_BE * model.BE_F + emissions_CP * model.CP_F)
model.annual_emission_con = pyo.Constraint(model.scenario_set, model.year_set,rule=annual_CO2_rule)

#Total emissions constraint - equation (6)
#This constraint ensure that the annual emission that can be abated using negative emission technology
#in each scenario (also each year) is limited by the NB_limit - hence, the remaining emissions needed to be abated will done with CCS 
#def total_emission_constraint_rule(self,k,y):
#    return model.annual_emission_var[k,y] <= total_annual_emission 
#model.total_emission_con = pyo.Constraint(model.scenario_set, model.year_set, rule = total_emission_constraint_rule)


# ### Set The Objective Function for The First Stage 

# In[19]:


#def objective_rule(n):
#    return model.unit_cost_RS * model.RS_capacity_var[n+1] + model.unit_cost_US * model.US_capacity_var[n+1] 
#model.objective_function = pyo.Objective(model.node_set,rule = objective_rule, sense=minimize)

#model.objective = pyo.Objective(expr=modelunit_cost_RS * model.RS_capacity_var[n+1])
#Equation (1) the cost of stage 1 in objective function 
total_gen_capex = sum(model.unit_cost_RS * model.RS_capacity_var[n] + model.unit_cost_US * model.US_capacity_var[n] 
                      + model.unit_cost_ON * model.ON_capacity_var[n] + model.unit_cost_OF * model.OF_capacity_var[n]
                      + model.unit_cost_NP * model.NP_capacity_var[n] + model.unit_cost_NG * model.NG_capacity_var[n] + model.unit_cost_HP * model.HP_capacity_var[n] + + model.unit_cost_BE * model.BE_capacity_var[n] + + model.unit_cost_CP * model.CP_capacity_var[n] for n in model.node_set)
total_line_capex = sum(model.new_transmission_line_var[(i,j)] * model.new_transmission_line_capex_param[(i,j)] for (i,j) in model.candidate_line_set)
total_interconnector_capex = sum(model.new_interconnector_var[inter] * model.new_interconnector_capex_param[inter] for inter in model.new_interconnector_var)

total_stage_1_cost = total_gen_capex + total_line_capex + total_interconnector_capex


#Equation (2) the total cost of energy production and load shedding across all scenario, year, location, and time
RS_total_gen_cost = sum(sum(sum(sum(model.RS_gen_var[k,y,t,n] * opex_RS[y-1] for n in model.node_set) for t in model.time_set) for y in model.year_set) for k in model.scenario_set)      
US_total_gen_cost = sum(sum(sum(sum(model.US_gen_var[k,y,t,n] * opex_US[y-1] for n in model.node_set) for t in model.time_set) for y in model.year_set) for k in model.scenario_set)      
ON_total_gen_cost = sum(sum(sum(sum(model.ON_gen_var[k,y,t,n] * opex_ON[y-1] for n in model.node_set) for t in model.time_set) for y in model.year_set) for k in model.scenario_set)      
OF_total_gen_cost = sum(sum(sum(sum(model.OF_gen_var[k,y,t,n] * opex_OF[y-1] for n in model.node_set) for t in model.time_set) for y in model.year_set) for k in model.scenario_set)      
NP_total_gen_cost = sum(sum(sum(sum(model.NP_gen_var[k,y,t,n] * opex_NP[y-1] for n in model.node_set) for t in model.time_set) for y in model.year_set) for k in model.scenario_set)      
HP_total_gen_cost = sum(sum(sum(sum(model.HP_gen_var[k,y,t,n] * opex_HP[y-1] for n in model.node_set) for t in model.time_set) for y in model.year_set) for k in model.scenario_set)      
NG_total_gen_cost = sum(sum(sum(sum(model.NG_gen_var[k,y,t,n] * opex_NG[y-1] for n in model.node_set) for t in model.time_set) for y in model.year_set) for k in model.scenario_set)      
BE_total_gen_cost = sum(sum(sum(sum(model.BE_gen_var[k,y,t,n] * opex_BE[y-1] for n in model.node_set) for t in model.time_set) for y in model.year_set) for k in model.scenario_set)      
CP_total_gen_cost = sum(sum(sum(sum(model.CP_gen_var[k,y,t,n] * opex_CP[y-1] for n in model.node_set) for t in model.time_set) for y in model.year_set) for k in model.scenario_set)      
total_LS_cost = sum(sum(sum(sum(model.LS_var[k,y,t,n] * cost_LS[y-1] for n in model.node_set) for t in model.time_set) for y in model.year_set) for k in model.scenario_set)      

total_gen_cost = RS_total_gen_cost + US_total_gen_cost + ON_total_gen_cost + OF_total_gen_cost + NP_total_gen_cost + NG_total_gen_cost + HP_total_gen_cost + BE_total_gen_cost + CP_total_gen_cost 


#Equation (3) - Carbon Abatement Cost
#Given that multiple trial has shown that the total annual emissions is higher than what the natural based solution can fufil
#The binary variable of using CCS, WkCS in equation (3) is assume to always equal to 1
#This approach is taken as there is currently no a clearer idea on how to set the indexed binary variable
#test assume Xbinary = 0 from your result: - assume NB solution is used for all carbon abatement
#total_abatement_cost= sum((model.annual_emission_var[k,y] * NB_cost[y-1])/(number_of_scenario) for k in model.scenario_set for y in model.year_set)
#Xbinary = 1:
total_abatement_cost = sum(((CCS_cost[y-1]*(model.annual_emission_var[k,y]-NB_limit) + NB_limit*NB_cost[y-1])/(number_of_scenario)) for k in model.scenario_set for y in model.year_set)

#Equation (4) - Energy Import Cost and Export Revenue
#for energy import:
total_North_Connect_import_cost = sum(sum(sum(model.existing_inter_import_var[k,y,t,3] * North_Connect_import[y-1] for t in model.time_set) for y in model.year_set) for k in model.scenario_set)
total_Moyle_import_cost = sum(sum(sum(model.existing_inter_import_var[k,y,t,5] * Moyle_import[y-1] for t in model.time_set) for y in model.year_set) for k in model.scenario_set)
total_North_Sea_import_cost = sum(sum(sum(model.existing_inter_import_var[k,y,t,7] * North_Sea_import[y-1] for t in model.time_set) for y in model.year_set) for k in model.scenario_set)
total_East_West_import_cost = sum(sum(sum(model.existing_inter_import_var[k,y,t,12] * East_West_import[y-1] for t in model.time_set) for y in model.year_set) for k in model.scenario_set)
total_Viking_Link_import_cost = sum(sum(sum(model.existing_inter_import_var[k,y,t,19] * Viking_Link_import[y-1] for t in model.time_set) for y in model.year_set) for k in model.scenario_set)
total_BritNed_import_cost = sum(sum(sum(model.existing_inter_import_var[k,y,t,20] * BritNed_import[y-1] for t in model.time_set) for y in model.year_set) for k in model.scenario_set)
total_Nemo_Link_import_cost = sum(sum(sum(model.existing_inter_import_var[k,y,t,26] * Nemo_Link_import[y-1] for t in model.time_set) for y in model.year_set) for k in model.scenario_set)
total_IFA_1_import_cost = sum(sum(sum(model.existing_inter_import_var[k,y,t,27] * IFA_1_import[y-1] for t in model.time_set) for y in model.year_set) for k in model.scenario_set)
total_IFA_2_import_cost = sum(sum(sum(model.existing_inter_import_var[k,y,t,28] * IFA_2_import[y-1] for t in model.time_set) for y in model.year_set) for k in model.scenario_set)

total_Ice_Link_import_cost = sum(sum(sum(model.candidate_inter_import_var[k,y,t,1] * Ice_Link_import[y-1] for t in model.time_set) for y in model.year_set) for k in model.scenario_set)
total_Green_Link_import_cost = sum(sum(sum(model.candidate_inter_import_var[k,y,t,8] * Green_Link_import[y-1] for t in model.time_set) for y in model.year_set) for k in model.scenario_set)
total_Neu_Connect_import_cost = sum(sum(sum(model.candidate_inter_import_var[k,y,t,26] * Neu_Connect_import[y-1] for t in model.time_set) for y in model.year_set) for k in model.scenario_set)
total_FAB_Link_import_cost = sum(sum(sum(model.candidate_inter_import_var[k,y,t,29] * FAB_Link_import[y-1] for t in model.time_set) for y in model.year_set) for k in model.scenario_set)

total_import = total_North_Connect_import_cost + total_Moyle_import_cost + total_North_Sea_import_cost + total_East_West_import_cost + total_Viking_Link_import_cost + total_BritNed_import_cost + total_Nemo_Link_import_cost + total_IFA_1_import_cost + total_IFA_2_import_cost + total_Ice_Link_import_cost + total_Green_Link_import_cost + total_Neu_Connect_import_cost + total_FAB_Link_import_cost 

#for energy export:
total_North_Connect_export_cost = sum(sum(sum(model.existing_inter_export_var[k,y,t,3] * North_Connect_export[y-1] for t in model.time_set) for y in model.year_set) for k in model.scenario_set)
total_Moyle_export_cost = sum(sum(sum(model.existing_inter_export_var[k,y,t,5] * Moyle_export[y-1] for t in model.time_set) for y in model.year_set) for k in model.scenario_set)
total_North_Sea_export_cost = sum(sum(sum(model.existing_inter_export_var[k,y,t,7] * North_Sea_export[y-1] for t in model.time_set) for y in model.year_set) for k in model.scenario_set)
total_East_West_export_cost = sum(sum(sum(model.existing_inter_export_var[k,y,t,12] * East_West_export[y-1] for t in model.time_set) for y in model.year_set) for k in model.scenario_set)
total_Viking_Link_export_cost = sum(sum(sum(model.existing_inter_export_var[k,y,t,19] * Viking_Link_export[y-1] for t in model.time_set) for y in model.year_set) for k in model.scenario_set)
total_BritNed_export_cost = sum(sum(sum(model.existing_inter_export_var[k,y,t,20] * BritNed_export[y-1] for t in model.time_set) for y in model.year_set) for k in model.scenario_set)
total_Nemo_Link_export_cost = sum(sum(sum(model.existing_inter_export_var[k,y,t,26] * Nemo_Link_export[y-1] for t in model.time_set) for y in model.year_set) for k in model.scenario_set)
total_IFA_1_export_cost = sum(sum(sum(model.existing_inter_export_var[k,y,t,27] * IFA_1_export[y-1] for t in model.time_set) for y in model.year_set) for k in model.scenario_set)
total_IFA_2_export_cost = sum(sum(sum(model.existing_inter_export_var[k,y,t,28] * IFA_2_export[y-1] for t in model.time_set) for y in model.year_set) for k in model.scenario_set)

total_Ice_Link_export_cost = sum(sum(sum(model.candidate_inter_export_var[k,y,t,1] * Ice_Link_export[y-1] for t in model.time_set) for y in model.year_set) for k in model.scenario_set)
total_Green_Link_export_cost = sum(sum(sum(model.candidate_inter_export_var[k,y,t,8] * Green_Link_export[y-1] for t in model.time_set) for y in model.year_set) for k in model.scenario_set)
total_Neu_Connect_export_cost = sum(sum(sum(model.candidate_inter_export_var[k,y,t,26] * Neu_Connect_export[y-1] for t in model.time_set) for y in model.year_set) for k in model.scenario_set)
total_FAB_Link_export_cost = sum(sum(sum(model.candidate_inter_export_var[k,y,t,29] * FAB_Link_export[y-1] for t in model.time_set) for y in model.year_set) for k in model.scenario_set)

total_export = total_North_Connect_export_cost + total_Moyle_export_cost + total_North_Sea_export_cost + total_East_West_export_cost + total_Viking_Link_export_cost + total_BritNed_export_cost + total_Nemo_Link_export_cost + total_IFA_1_export_cost + total_IFA_2_export_cost + total_Ice_Link_export_cost + total_Green_Link_export_cost + total_Neu_Connect_export_cost + total_FAB_Link_export_cost 


#Objective Function
model.objective_function = pyo.Objective(expr = total_stage_1_cost + 5*(365*(total_gen_cost)/(number_of_scenario) + total_abatement_cost + 365*(total_import - total_export)/(number_of_scenario)), sense = minimize)
#need to have objective function for second stage too 


# ### Solve the Optimization Problem 

# In[20]:


# solve the optimization problem
#results = solver.solve(model, tee=True)
#set tolerance to 1% (should have a new line in solver's print)
solver.options["mip tolerances mipgap"] = 0.015 #1.5%
results = solver.solve(model, tee=True)


#print(results.solver.status)
#print(results.solver.termination_condition)
#print(results.solver.termination_message)
#print(results.solver.time)


# In[21]:


# solve the optimization problem
results = solver.solve(model, tee=True)
results = solver.solve(model, tee=True)

print(results.solver.status)
print(results.solver.termination_condition)
print(results.solver.termination_message)
print(results.solver.time)


# In[22]:


print(results.solver.status)
print(results.solver.termination_condition)
print(results.solver.termination_message)
print(results.solver.time) # = 11.hours 35 min


# In[23]:


#can we reduce tolerant to like 1%?
print('This section shows the result output: [BASE CASE]')
print('')
print(str('The total Net Present Cost for 25 years is: ') + str((int(value(model.objective_function)/1000000))) + str(' million GBP-2025'))
print('')
print('')
print('GENERATION CAPACITY ACROSS LOCATION')
print('')
print('ROOFTOP SOLAR: ')
for n in model.node_set:
    print ('node ' + str(n) + 'RS capacity [MW]: ' + str(int(value(model.RS_capacity_var[n]))))
print('')   
print('UTILITY-SCALE SOLAR: ')
for n in model.node_set:
    print ('node ' + str(n) + 'US capacity [MW]: ' + str(int(value(model.US_capacity_var[n]))))
print('')    
print('ONSHORE WIND: ')
for n in model.node_set:
    print ('node ' + str(n) + 'ON capacity [MW]: ' + str(int(value(model.ON_capacity_var[n]))))
print('')  
print('OFFSHORE WIND: ')
for n in model.node_set:
    print ('node ' + str(n) + 'OF capacity [MW]: ' + str(int(value(model.OF_capacity_var[n]))))
print('')    
print('NUCLEAR POWER: ')
for n in model.node_set:
    print ('node ' + str(n) + 'NP capacity [MW]: ' + str(int(value(model.NP_capacity_var[n]))))
print('')       
print('HYDRO POWER: ')
for n in model.node_set:
    print ('node ' + str(n) + 'HP capacity [MW]: ' + str(int(value(model.HP_capacity_var[n]))))
print('')     
print('NATURAL GAS: ')
for n in model.node_set:
    print ('node ' + str(n) + 'NG capacity [MW]: ' + str(int(value(model.NG_capacity_var[n]))))
print('')
print('BIOELECTRIC: ')
for n in model.node_set:
    print ('node ' + str(n) + 'BE capacity [MW]: ' + str(int(value(model.BE_capacity_var[n]))))
print('')     
print('COAL PEAKING PLANT: ')
for n in model.node_set:
    print ('node ' + str(n) + 'CP capacity [MW]: ' + str(int(value(model.CP_capacity_var[n]))))
print('')     
print('AVERAGE ANNUAL EMISSIONS: ')
for y in model.year_set: 
    average_annual_emission = sum((value(model.annual_emission_var[k,y]))/(number_of_scenario) for k in model.scenario_set)
    print('year_step ' + str(y) + ' average emissions: ' + str(int(average_annual_emission)) + ' tonCO2e')

    
    
#**
#def average_annual_emission_rule(self,y):
#    return sum((model.annual_emission_var[_,y])/(number_of_scenario)  for y in model.year_set)
#model.average_annual_emission = pyo.Var(model.year_set,rule = average_annual_emission_rule)

#for y in model.year_set: 
#    if y ==1:
#        for k in model.scenario_set:
#            if k ==1:
#                for t in model.time_set:
#                    if t == 2:
#                        for n in model.node_set:
#                            print(model.LS_var[y,k,t,n].value)



#
#for lc in model.candidate_line_set:
#    print(value(model.new_transmission_line_var[lc]))
#
#for y in model.year_set: 
#    if y ==1:
#        for k in model.scenario_set:
#            if k ==1:
#                for t in model.time_set:
#                    if t == 2:
#                        for n in model.node_set:
#                            print(model.candidate_inter_import_var[y,k,t,n].value)

#    
#model.new_transmission_line_var
#model.existing_inter_import_var

#model.existing_inter_import_var
#model.power_flow_through_existing_line[k,y,t,(i,j)]

#for k in model.scenario_set:
#    if k == 1:
#        for y in model.year_set:
#            if y == 1:
#                for t in model.time_set:
#                    if t == 1:
#                        for (i,j) in model.existing_line_set:
#                            print(i,j,model.power_flow_through_existing_line[k,y,t,(i,j)].value)
                
#power flow has issue - however is flowing from 1 - 2 is the same as however much is flowing from 2 to 1 - so net flow is zero
#thats why so much natural gas is needed and no new line is build - so maybe we need to check power balance? or theta
#problem: we have 2 line from node 1 to node 2; but the model/ dictionary only read one of them


# In[32]:


#for s in model.node_set:
#    print(value(model.new_interconnector_var[s]))
print('TOTAL CAPEX: ' + str(int(value(total_stage_1_cost)/1000000))+ ' million GBP-2025') 
print('Generation CAPEX: ' + str(int(value(total_gen_capex)/1000000) ) +' million GBP-2025') 
print('Transmission Line CAPEX: '+ str(int(value(total_line_capex)/1000000))+' million GBP-2025')
print('Interconnector CAPEX: ' + str(int(value(total_interconnector_capex)/1000000))+' million GBP-2025')
value(total_abatement_cost)

#emission with binary = 1:  3360593
#emission limit for NB sol: 3404169
#so, binary should be 0? for all cases - cuz even with NB case, current NB will be good enough to offset emissions economically 

#NB_limit = 3404169 #tonCO2e/year - this is for the base case of using UK's current LULUCF credit
#total_annual_emission = 10000000 #tonCO2e/year

#86549


# In[25]:


#NExt step: makes dictionaries for parameter - demand, capacity factor
#average_annual_emission = sum((value(model.annual_emission_var[k,y]))/(number_of_scenario) for k in model.scenario_set)
#demand: location: use current load profiel; predicted growrh each year using that location for fes (for winter, summer - total growth across all sectors)
##The total Net Present Cost for 25 years is: 86549 million GBP-2025


# In[26]:


for (i,j) in model.candidate_line_set:
    print(value(model.new_transmission_line_var[(i,j)]))


# In[27]:


print('TOTAL CAPEX: ' + str(int(value(total_stage_1_cost)/1000000))+ ' million GBP-2025') 
print('Generation CAPEX: ' + str(int(value(total_gen_capex)/1000000) ) +' million GBP-2025') 
print('Transmission Line CAPEX: '+ str(int(value(total_line_capex)/1000000))+' million GBP-2025')
print('Interconnector CAPEX: ' + str(int(value(total_interconnector_capex)/1000000))+' million GBP-2025')
print('Total carbon abatement cost: ' + str(int(value(5*total_abatement_cost)/1000000)) +' million GBP-2025')


# In[28]:


print(' ')
print('Operation cost over 26 years in NPV')
print('RS Generation Cost:' + str(int(value(365*5*RS_total_gen_cost/(number_of_scenario))/1000000)) + ' million GBP-2025')
print('US Generation Cost:' + str(int(value(365*5*US_total_gen_cost/(number_of_scenario))/1000000)) + ' million GBP-2025')
print('ON Generation Cost:' + str(int(value(365*5*ON_total_gen_cost/(number_of_scenario))/1000000)) + ' million GBP-2025')
print('OF Generation Cost:' + str(int(value(365*5*OF_total_gen_cost/(number_of_scenario))/1000000)) + ' million GBP-2025')
print('NP Generation Cost:' + str(int(value(365*5*NP_total_gen_cost/(number_of_scenario))/1000000)) + ' million GBP-2025')
print('NG Generation Cost:' + str(int(value(365*5*NG_total_gen_cost/(number_of_scenario))/1000000)) + ' million GBP-2025')


# In[29]:


for (i,j) in model.candidate_line_set:
    print('from node' + str(i) + ' to node'+ str(j) + ' binary: ' + str(value(model.new_transmission_line_var[(i,j)])))


# In[30]:


print('Interconnector to be built: ')
for n in model.node_set:
    print ('node ' + str(n) + ' binary interconnector: ' + str(int(value(model.new_interconnector_var[n]))))


# In[31]:


print('Total import cost: ' + str(int(value((total_import)/(number_of_scenario))/1000000)) + ' million GBP 2025')
print('Total export revenue: ' + str(int(value((total_export)/(number_of_scenario))/1000000)) + ' million GBP 2025')

