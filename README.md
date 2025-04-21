# Is Stanage dry? 
## A Machine Learning approach to predicting outdoor climbing weather conditions.

Stanage edge is the most popular outdoor climbing destination in the UK. Due to the fragile nature of the rock type - gritstone - at Stanage, it is crucial to avoid climbing there when the rock is not fully dry. 

This code aims to predict whether Stanage is dry, and therefore whether it is possible to climb there, given the weather in the previous days.

By combining user-logged data from ukclimbing.com which indicates when people have climbed at Stanage, with historical weather data from the MET office, the code trains a neural-network to classify a set of weather data into a simple yes or no answer to the question: is Stanage dry?

Simply put, users of the UKC website log a climb after they have done it, with the option to add a note describing their ascent. I obtained the historical data from all logs from UKC at stanage since the start of the website, then reorganised this as a time-series data, showing the number of logs at Stanage every day. Stanage is so popular that it is reasonable to assume days with 0 logs correspond to days the weather was too bad to climb there.

In this version of the code the labels for the classifier are as follow: Dry if number_of_logs > 0, Wet if number_of_logs = 0. For any given day on which one wishes to know wether Stanage will be dry, the network will take as input the hourly weather data from the last N days. The input data is then a vector of length N * 4 *24, where the 4 refers to 4 features and 24 is the number of hours in a day. For each feature, being hourly temperature, dew point, relative humidity and rainfall, there is a list of 24h hourly measurements. The input vector is then of the form: {day1_temp_h_0, day1_temp_h1, ... , dayN_rain_h22, dayN_rain_h23} where day1 is the day for which you want the classifier to work, and dayN is (N-1) days prior.

The code includes a scraping script - UKCscraper.py - to obtain the data from UKC, with some information redacted.

DataCleaner.py is a purpose-built pipeline taking the downloaded weather data and collating it with the UKC logs data.

learn.py implements a neural network with 1 input layer, 1 hidden layer and 1 output layer. 

Future versions of this code will include different architectures to maximise accuracy, as well as a automatic data collection pipeline to get real time weather data, allowing for direct prediction of the conditions at Stanage.


