import io
from imghdr import tests

import requests
from bs4 import BeautifulSoup
import json
import time
import re
import pandas as pd
import numpy as np
import warnings

headers = {"User-Agent": "%PLACEHOLDER%"}  # Some sites block requests without a user-agent
page = '%PLACEHOLDER%' #specify which crag(s) - make sure to add a slash at the end
crag_name='%PLACEHOLDER%'
login_url = 'https://www.ukclimbing.com/user/'
username= '%PLACEHOLDER%'
password= '%PLACEHOLDER%' 



def get_climb_urls(page): #outputs a tuple of lists consisting of the URLs of climbs for a specific crag, as well as their corresponding names, grades, type and number of stars.

    url_base = 'https://www.ukclimbing.com/logbook/crags/' #base url for ukc logbook crags
    url= url_base + page #load our specific crag page
    response = requests.get(url,headers=headers) #html request
    soup = BeautifulSoup(response.text, 'html.parser') #parse the html using soup
    #print(soup)
    soup_json=soup.select("script") #select only scripts from the soup object
    string_soup=str(soup_json) #turn into string
    list_climbs=re.findall(r'"slug":"(.*?)"', string_soup) #parse above object to select only the climb urls
    attributes_soup = re.findall(r'table_data\s*=\s*(\[\{.*?\}\])\s*,',string_soup, re.DOTALL)
    list_names_unicode=re.findall(r'"name":"(.*?)"', attributes_soup[0], re.DOTALL) #makes list of all climb names
    list_names = [] #initialise list of names
    for name in list_names_unicode:
        name2 = name.replace("\/", "/") #avoids a deprecation warning for climbs that have a / in their name
        list_names.append(name2.encode().decode('unicode-escape')) #convert unicode format to proper string with special characters
    list_grade_type=re.findall(r'"gradetype"\s*:\s*(\d+)', attributes_soup[0], re.DOTALL) #list of all types of climb ie boulder (4) trad (2) lead
    list_stars=re.findall(r'"stars"\s*:\s*(\d+)', attributes_soup[0],re.DOTALL) #makes list of number of stars
    climb_urls=[url + climb for climb in list_climbs] #concatenate into new url for each climb at the crag
    return climb_urls,list_names,list_grade_type,list_stars

def get_tick_info(url,name,grade_type,stars): #outputs a pandas dataframe for each climb, with the logs information including user, note, date, style as well as the name grade and type of climb
    with (requests.session() as s): #create session
        payload={
            '%PLACEHOLDER%' } #prepare payload to feed into request
        cookies={'%PLACEHOLDER%'} #prepare cookies
        climbing_login = s.post(login_url, data=payload, headers=headers, cookies=cookies) #log in to UKC with the correct cookies
        print(f"Status code: {climbing_login.status_code}") #if returns 200 then it worked
        climbing_session = s.get(url) #get html from climb page
        climb_soup = BeautifulSoup(climbing_session.text, 'html.parser') #parse the climb page
        climb_soup.find('h1',{'class': 'sr-only'}).decompose()
        climb_info_header = climb_soup.find('h1')
        if grade_type=='4':#here change the 0 to index in for loop
            route_grade = climb_info_header.find('em').text.strip() #climb is a boulder
        elif grade_type=='2':
            route_grade = climb_info_header.find('small').text.strip()  # climb is a route
        else:
            route_grade = 0
            print('There was an error recovering the grade of this climb')
        public_logbooks = climb_soup.find('div', {'id': "public_logbooks"}) #extract the part of the page containing the public logbooks
        if public_logbooks is not None:
            table = public_logbooks.find('table', class_= 'table table-sm mb-0') #extract the tick list table
            ticks_frame = pd.read_html(io.StringIO(str(table)))[0] #read the html table into a pandas dataframe
            ticks_frame=ticks_frame.rename(columns={ticks_frame.columns[3] : 'Notes'}) #rename the last column
            ticks_frame_preclean=ticks_frame[ticks_frame.User != ticks_frame.Notes] #remove rows that have the same User and Notes entry
            ticks_frame_clean= ticks_frame_preclean.dropna(subset= ['User']) #remove rows that have NaN user entry
            ticks_frame_ext = ticks_frame_clean.copy() #avoids a deprecation error
            ticks_frame_ext.loc[:,'Name']=name #adds a column of just the name of the route, same with below lines
            ticks_frame_ext.loc[:,'Grade'] = route_grade
            ticks_frame_ext.loc[:,'Stars'] = stars
            ticks_frame_ext.loc[:,'Type'] = grade_type
        if public_logbooks is None: #adding a failsafe for when there are no logs on a climb
            ticks_frame_ext = pd.DataFrame(0,index=[0],columns=['User','Date','Style','Notes']) #fills user, date, style and note with a 0
            ticks_frame_ext.loc[:, 'Grade'] = route_grade
            ticks_frame_ext.loc[:, 'Stars'] = stars
            ticks_frame_ext.loc[:, 'Type'] = grade_type
        #pd.set_option("display.max_columns", None)
        #print(ticks_frame_clean['User'])
        #print(ticks_frame_clean['Date'])
        #print(ticks_frame_clean['Style'])
        #print(ticks_frame_clean['Notes'])
        #print(ticks_frame_ext['Name'])
        #print(ticks_frame_ext['Grade'])
        #print(ticks_frame_ext['Stars'])
        #print(ticks_frame_ext['Type'])
        #print(ticks_frame_clean)
        return ticks_frame_ext

def scrape (page): #actual scraper
    climbs_frame= get_climb_urls(page) #load in the climb urls
    climb_urls= climbs_frame[0] #list of climb urls, and corresponding below
    list_names= climbs_frame[1]
    list_grade_type= climbs_frame[2]
    list_stars= climbs_frame[3]
    big_tick_frame=[] #initialise list of ticks
    for i in range(len(climb_urls)): #iterate over all climbs at the crag
        climb_url = climb_urls[i] #individual climb url etc
        climb_name = list_names[i]
        climb_type = list_grade_type[i]
        climb_stars = list_stars[i]
        tick_info = get_tick_info(climb_url,climb_name,climb_type,climb_stars) #run tick info ie generate dataframe for individual climb
        big_tick_frame.append(tick_info) #add climb tick frame to the list of frames
        print('Climb %d out of %d : %s' %(i+1,len(climb_urls),climb_name)) #progress info
        time.sleep(1) #wait one second between each climb
    all_logs_frame = pd.concat(big_tick_frame, ignore_index=True) #concatenate all the ticks for all climbs into one big dataframe
    return all_logs_frame



if __name__ == '__main__':
    all_logs=scrape(page)
    all_logs.to_csv('ukc_logs_'+crag_name+'.csv',index=False)

