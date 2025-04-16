import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#take the UKC data and the weather data and load them to dataframe. Collate all the time-series data after cleaning. Then reorganise the UKC logs by date. Then collate the weather data with the UKC data.
rain_file='%PLACEHOLDER%'
weather_file = '%PLACEHOLDER%'
#First clean the weather data.

def rainfall_clean(file): #for rainfall, just want to keep precipitation amount. Need to remove some rows which have cumulative information as well. Then concatenate all the data together.
    csv_raw = pd.read_csv(file) #read file
    csv_pre = csv_raw[csv_raw.iloc[:,3] == 1] #get rid of total rainfall over 12h
    csv_clean = csv_pre.iloc[:,[0,8]] #just keep time and precip amount
    return csv_clean

def weather_clean(file):
    csv_raw = pd.read_csv(file)  #read file
    csv_pre = csv_raw[csv_raw.iloc[:,3] == 'AWSHRLY'] #delete rows that contain daily info data
    csv_clean = csv_pre.iloc[:, [0, 35, 36, 38]]  # just keep time, air temp, dew point, rel humid
    return csv_clean

def merge_data(rain, weather): #merge the rainfall and weather data
    rain_cleaned = rainfall_clean(rain)
    weather_cleaned = weather_clean(weather)
    if rain_cleaned.shape[0] == weather_cleaned.shape[0]: #check if two frames are the same shape
        merged_data = weather_cleaned.join(rain_cleaned.set_index('ob_end_time'), on='ob_time').reset_index(drop=True) #join two frames together by adding the humidity column to the weather frame. Need to have a failsafe for if the two dataframes have different number of rows
    else:
        print('Rain and weather data have different number of observations: rain_cleaned: %d, weather_cleaned: %d' %(rain_cleaned.shape[0],weather_cleaned.shape[0])) #here need to come up with a routine to handle missing data in one of the data
        weather_cleaned=weather_cleaned.reset_index(drop=True) #reset indices
        rain_cleaned1=rain_cleaned.reset_index(drop=True)
        rain_cleaned2= rain_cleaned1.rename(columns={rain_cleaned.columns[0]: 'ob_time'}) #change column heading to be the same as that of weather data: 'ob_time'
        merged_data = weather_cleaned.merge(rain_cleaned2, on='ob_time') #this merges the two data sets by adding the humidity data to the weather data. If the time is the same the humidity will be read out from rain_cleaned. If the time in weather_cleaned is not present in rain_cleaned then humidity is NaN
        #above: have how='left' or no??
        print('Merged data has %d rows'%(merged_data.shape[0]))
    return merged_data



def make_timeseries(): #concatenate the timeseries in order
    indices = list(range(2008, 2024))
    weather_list = []
    for i in indices:
        rain_i = '%PLACEHOLDER%' % (i)
        weather_i = '%PLACEHOLDER%' % (i)
        merged_i = merge_data(rain_i, weather_i)
        weather_list.append(merged_i) #adds ith entry to a list of dataframe for each year
    weather_frame = pd.concat(weather_list, ignore_index=True) #concatenates the list into one big dataframe
    weather_frame['ob_time']=pd.to_datetime(weather_frame['ob_time']) #change dates to datetime format
    weather_frame = weather_frame.set_index('ob_time') #set date as index
    print(weather_frame.shape)
    weather_frame = weather_frame.resample('h').interpolate(limit=2)
    print(weather_frame.shape)
    weather_frame['hour'] = weather_frame.index.hour
    weather_frame['date'] = weather_frame.index.date
    weather_frame = weather_frame.reset_index()
    #print(weather_frame.head())
    # daily_weather=[]
    # for col in ['air_temperature', 'dewpoint', 'rltv_hum', 'prcp_amt']:
    #     pivoted=weather_frame[[col,'hour']].pivot_table()
    weather_melted = weather_frame.melt(id_vars=['date', 'hour'], value_vars=['air_temperature', 'dewpoint', 'rltv_hum', 'prcp_amt']) # Melt to long format (one value per row)
    weather_reshaped = weather_melted.pivot(index='date', columns=['variable','hour'], values='value') # Pivot so each variable + hour becomes a column
    weather_reshaped.columns = [f'{var}_hour_{hr}' for var, hr in weather_reshaped.columns] # Flatten column MultiIndex
    weather_final=weather_reshaped.reset_index()
    #print(weather_final.head())
    # weather_final.to_csv('weather_data.csv', index=False) #
    return weather_final



def UKC_clean_data():
    plantation='ukc_logs_stanage-plantation-101.csv'
    north='ukc_logs_stanage_north-99.csv'
    popular='ukc_logs_stanage_popular-104.csv'
    all_crags_files=[plantation,north,popular]
    all_crags_list=[]
    for crag in all_crags_files:
        df = pd.read_csv(crag)
        all_crags_list.append(df)
    all_crags_df=pd.concat(all_crags_list)
    all_crags_df['Date'] = pd.to_datetime(all_crags_df['Date'],format='%d %b, %Y', errors='coerce')
    crags_clean=all_crags_df.dropna()
    crags_clean = crags_clean.sort_values(by='Date',ascending=True)
    crags_clean=crags_clean[crags_clean['Style'].str.contains('dnf')==False] #remove dnf from rows
    crags_clean= crags_clean[crags_clean['Notes'].str.contains('wet')==False]
    crags_clean = crags_clean[crags_clean['Notes'].str.contains('damp') == False]
    filtered_crags = crags_clean.loc[crags_clean['Date']< '2023-12-29']
    filtered_crags = filtered_crags.loc[filtered_crags['Date'] > '2008-09-21']
    filtered_crags.insert(2,'Ticks',1)
    filtered_crags=filtered_crags[['Date','Ticks']]
    pivot_crags= filtered_crags.pivot_table(index='Date',aggfunc='sum')
    pivot_crags = pivot_crags.resample('d').asfreq()
    pivot_crags = pivot_crags.fillna(0)
    pivot_crags = pivot_crags.reset_index()
    return pivot_crags







if __name__ == '__main__':
    #merged_frame=merge_data(rain, weather)
    #print(merged_frame.head())
    #hopp=make_timeseries()
    #hopp.to_csv('weather_data.csv', index=False)
    #ukc_data=UKC_clean_data()
    # ukc_data.to_csv('ukc_logs_stanage_barebone.csv')
    # hopp['Ticks']=ukc_data['Ticks']
    # hopp.to_csv('data_final_clean.csv',index=False)
    # data= pd.read_csv('data_final_clean.csv')
    # data2 = data.loc[data['date'] > '2023-01-01']
    # data2=data2[['date','Ticks']]
    # print(data2.shape)
    # plot = data2.plot(x='date',y='Ticks')
    # # plt.show()
    # data3 = data2.loc[data2['Ticks']<5]
    # print(data3)