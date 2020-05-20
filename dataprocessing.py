import numpy as np
import pandas as pd
import datetime as dt

def cal_dwelltime(depart, arrive, schedule, schedule_type):
    """ Calculate dwell time given load record """
    if (schedule_type == 'Open') or (depart < schedule) or (arrive > schedule):
        return depart - arrive
    else:
        return depart - schedule

def apply_cal_dwelltime(df):
    """ Apply Cal_DwellTime function to each row of DataFrame """
    df['DwellTime'] = df.apply(lambda row: cal_dwelltime(row['DepartDateTime'], row['ArriveDateTime'], row['ScheduleOpenTime'], row['ScheduleType']), axis=1).astype('timedelta64[m]')/60
    return df

def readFile(filenames):
    """ Read and process data from multiple files """
    data_list = []
    for file in filenames:
        data = pd.read_csv(file,encoding='mac_roman',parse_dates=['LoadDate','ScheduleOpenTime','ScheduleCloseTime','ArriveDateTime','DepartDateTime'])
        data = cleanData_preload(data)
        data_list.append(data)
    truck = pd.concat(data_list)
    return apply_cal_dwelltime(truck)

def cleanData_preload(data):
    # Clean ArriveDateTime
    data = data[data.LoadDate - data.ArriveDateTime <= dt.timedelta(days = 15)]
    # Clean Total Pallets between 0-80
    data = data[(data.TotalPallets > 0) & (data.TotalPallets <= 80)]
    # Clean Total Pallets between 0-50,000
    data = data[(data.TotalWeight > 0) & (data.TotalWeight <= 5e4)]
    # Exclude Trailer Dropped
    data = data[data.TrailerDropped == False]
    # Replace Bouncecount
    data.BounceCount.replace(np.nan,0,inplace=True)
    return data

def cleanData_postload(data):
    # Clean Dwell Time between 0-6 hours
    data = data[(data.DwellTime > 0) & (data.DwellTime <= 6)]
    # Replace Bouncecount
    data.BounceCount.replace(np.nan,0,inplace=True)
    # Clean Geographical data (ClusterName)
    data = data[(data.ClusterName != 'HI Region') | (data.ClusterName != 'AK Region')]
    return data

def add_calvars(data):
    """ Generate calculated variables """
    data['DayOfWeek'] = data.ArriveDateTime.dt.dayofweek
    data['DayOfWeek'] = data['DayOfWeek'].map(lambda x: {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}.get(x))
    data['HourOfDay'] = data.ArriveDateTime.dt.hour
    data['PeakHour'] = data.HourOfDay.apply(lambda x: 1 if (x>=6)&(x<16) else 0)
    return data

def add_aggvars(data):
    """ Generate Aggregated Variables """
    # Add Facility Traffic (Count of records for each hour of day)
    facility_traffic = data.groupby(['FacilityID','DayOfWeek','HourOfDay']).agg({'DwellTime':'count'}).reset_index()
    facility_traffic.columns = ['FacilityID','DayOfWeek','HourOfDay','facility_traffic']
    data = pd.merge(data, facility_traffic, on=['FacilityID','DayOfWeek','HourOfDay'])

    ## Aggregated by CarrierID
    # Add Driver Experience (Count of records for each CarrierID)
    # Add Driver Complexity (Count unique customerID for each CarrierID)
    by_carrierid = data.groupby('CarrierID').agg({'DwellTime':'count','CustomerID':'nunique'}).reset_index()
    by_carrierid.columns = ['CarrierID','driver_exp','driver_complexity']
    data = pd.merge(data, by_carrierid, on='CarrierID')

    ## Aggregated by FacilityID
    # Add Facility Complexity (Count of unique CarrierID for each FacilityID)
    # Add Average/Median/Min/Max/Std Dwell Time
    by_facilityid = data.groupby('FacilityID').agg({'CarrierID':'nunique','DwellTime':['mean','median','min','max','std']}).reset_index()
    by_facilityid.columns = ['FacilityID','facility_complexity','avg_dt','median_dt','min_dt','max_dt','std_dt']
    data = pd.merge(data, by_facilityid, on ='FacilityID')

    return data

#### MAIN ####
### Use datasets from 2017-2019
truck = readFile(["./raw_data_2017.csv","./raw_data_2018.csv","./raw_data_2019.csv"])
truck = cleanData_postload(truck)

# Add calculated variables
truck = add_calvars(truck)

# Add aggregated variables
truck = add_aggvars(truck)

### OPTION 1 One-Hot Encoder ###
# Finalize Variables
X = truck[['DwellTime','Miles','MilesToNextStop','ClusterId','ArriveTimeUpdateType','BounceCount','TotalPallets','TotalWeight','Hot','DnBIndustry','ScheduleType','EquipmentType','EquipmentLength','LoadStopType','LoadStopSequence','WorkType','OnTime','DayOfWeek','HourOfDay','PeakHour','facility_traffic','driver_exp','driver_complexity','facility_complexity','avg_dt','median_dt','min_dt','max_dt','std_dt']]
X.dropna(inplace=True)
X.isna().sum() # count NA
X = pd.get_dummies(X, prefix='', prefix_sep='')
y = X.pop('DwellTime')

# Export data
X.to_csv("./indpt_vars.csv",index=False)
y.to_csv("./dwelltime.csv",index=False)

### OPTION 2 Numerical Encoder ###
truck.drop('DayOfWeek',axis=1, inplace=True)
truck['DayOfWeek'] = truck.ArriveDateTime.dt.dayofweek
X = truck[['Miles','MilesToNextStop','ClusterId','ArriveTimeUpdateType','BounceCount','TotalPallets','TotalWeight','Hot','DnBIndustry','ScheduleType','EquipmentType','EquipmentLength','LoadStopType','LoadStopSequence','WorkType','OnTime','DayOfWeek','HourOfDay','PeakHour','facility_traffic','driver_exp','driver_complexity','facility_complexity','avg_dt','median_dt','min_dt','max_dt','std_dt']]
X.dropna(inplace=True)
X.isna().sum()
# Convert categorical variables with one-hot encoding
X['LoadStopType'] = X.LoadStopType.map(lambda x: 1 if x == 'Pick Up' else 0)
X['Hot'] = X.Hot.map(lambda x: 1 if x is True else 0)
X['ArriveTimeUpdateType'] = X.ArriveTimeUpdateType.map(lambda x: 1 if x == 'Automated' else 0)
X['EquipmentType'] = X.EquipmentType.map(lambda x: 1 if x == 'R' else 0)
# Convert categorical variables with numerical encoding
X['ClusterId'], unique_cluster = pd.factorize(X['ClusterId'])
X['DnBIndustry'], unique_industry = pd.factorize(X['DnBIndustry'])
X['ScheduleType'], unique_schedule = pd.factorize(X['ScheduleType'])
X['WorkType'], unique_work = pd.factorize(X['WorkType'])

# Export data
X.to_csv("./indpt_vars_num.csv",index=False)
