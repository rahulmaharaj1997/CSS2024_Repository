#Date and Time Code was First Written : 31 January  2024 @ 10:43 PM 
#Date and Time Code was Last Written :   4 February 2024 @  11:11 PM 
#Author: Rahul Jaichund Maharaj

import pandas as pd 

#Extraction
df = pd.read_csv("movie_dataset.csv") 


#Transformation 
#Dropping rank and description columns as we don't need them
df.drop(["Rank"], inplace=True, axis=1) 
df.drop(["Description"], inplace=True, axis =1)

#There are empty cells in the revenue and metascore column
# Therefore I calculate the average of Revenue and Metascore 

x = df["Revenue (Millions)"].mean()  #Mean of All Revenue 
y = df["Metascore"].mean()  #Mean of All Metascore

#Fill all those empty cells with the mean values x and y 

df["Revenue (Millions)"].fillna(x, inplace=True) 
df["Metascore"].fillna(y, inplace=True)   

#To display all rows and columns .
pd.set_option("display.max_rows",None) 
pd.set_option("display.max_columns",None)

#print(df)
#print(df.describe()) 

#Filtering files that have zero revenue. 
print( df[df["Revenue (Millions)"] == 0])  

# Row 231 has a zero revenue 
# A Kind of Murder made zero revenue.
# Google says 91.149 Million US Dollars so I am using it 
# Locate the cell and input the value 
df.loc[231,"Revenue (Millions)"] = 91.149 

#Next I want to filter all the ratings that exceed 10 
print(df[df["Rating"]>10])

#Next I want to filter all the metascores that exceed 100
print(df[df["Metascore"] > 100]) 


print(df)
print(df.describe())


#Loading File 

df.to_csv("movie_dataset_cleaned.csv", index= False)


#Quiz Questions 
#Question 1 
#What is the highest rated movie in the data set 

#Find the maximum rating 
print("The highest movie rating is", df["Rating"].max()) 

#The highest movie rating is 9.0
#Filter out the movie with the highest rating 

print(df[df["Rating"] == 9.0]) 
#The Dark Knight is the highest rated

#What is the average revenue of all movies in the dataset 

print("The average revenue is", df["Revenue (Millions)"].mean() , "million US Dollars")  


#which is between 70-100 million us dollars 

#What is the average revenue in dollar for movies between 2015-2017 in dataset
x2 = df[df["Year"] < 2017].count()
print("x2=",x2)
x1 = df[df["Year"] < 2015].count()
print("x1=",x1)

number_2015_2017 = x2 -x1 


sum_2 = df[df["Year"] < 2017].sum()
sum_1 = df[df["Year"] < 2015].sum()



















#How many movies are there in 2016
v = df[df["Year"]==2016].count()
print("There are ", v , "movies in 2016") 
#297 movies in 2016 

#How many movies were directed by Christopher Nolan? 
p = df[df["Director"] == "Christopher Nolan"].count()
print("There are", p, "movies directed by Christopher Nolan") 

#How many movies in the dataset have at least a median of 8.0? 
d = df[df["Rating"]>= 8.0].count() 
print("There are", d , "movies that have a rating of at least 8.0") 

#What is the median rating of movies directed by Christopher Nolan ? 
#Use the describe function()
#50% of the data is the median
z = df[df["Director"] == "Christopher Nolan"].describe() 
print(z)

#Find the Year with the Highest Average Rating : 
# 2007 
year_2007 = df[df["Year"] == 2007].describe() 
print("2007 stats",year_2007)

# 2016 
year_2016 = df[df["Year"] == 2016].describe() 
print("2016 stats",year_2016) 

# 2006 
year_2006 = df[df["Year"] == 2006].describe() 
print("2006 stats",year_2006)

# 2008
year_2008 = df[df["Year"] == 2008].describe() 
print("2008 stats",year_2008)

#2007 average rating 7.133
#2016                6.44
#2006                7.125
#2008                6.78

# What is the percentage increase in number of movies made between 2006 and 2016?

x = df[df["Year"] == 2006].count()
y = df[df["Year"] == 2016].count()

perc_inc = ((y - x)/x)*100

print("% inc =", perc_inc)

#Find the most common actor in all the movies?

# Note, the "Actors" column has
#multiple actors names. 
#You must find a way to search for the most common actor in all the movies.

# Chris Pratt
# Mark Wahlberg 
# Matthew McConaughey 
# Bradley Cooper 

print("Chris Pratt", df[df["Actors"]=="Chris Pratt"].count()) 
print("Mark Wahlberg", df[df["Actors"]=="Mark Wahlberg"].count()) 
print("Matthew McConaughey", df[df["Actors"]=="Matthew McCoughey"].count()) 
print("Bradley Cooper", df[df["Actors"]=="Bradley Cooper"].count()) 


#Do a correlation of the numerical features. 
#What advice can you give to make better movies. 

#Plotting Ratings Against Votes 

import  numpy as np 
import matplotlib.pyplot as plt 

x = np.array(df["Rating"]) 
y = np.array(df["Votes"])   

#xfit = np.arange(0,10, 0.01)  
#y_xfit = np.exp(xfit) + 0.05
#curve_fit = np.polyfit(xfit,y_xfit,2) 
#yfit = np.polyval(curve_fit,xfit)

#plt.bar(x,y) 
#plt.plot(xfit,yfit) 
#plt.xlabel("Rating") 
#plt.ylabel("Votes")
#plt.title("Bar Plot of Votes vs Ratings to Determine Effect of Rating on Votes")
#plt.show() 
 
##########################################################################################
#R-Squared Correlation Btw Ratings and Votes 
sum_x = np.sum(x) 
sum_y = np.sum(y) 
n = len(x)
sum_xy = np.sum(x*y)  

sum_x_squared = np.sum(x**2) 
sum_y_squared = np.sum(y**2) 

numerator = n*sum_xy -sum_x*sum_y 
denominator = np.sqrt(n*sum_x_squared - sum_x**2)*np.sqrt(n*sum_y_squared - sum_y**2) 

R_squaredcorr = (numerator/denominator)**2 

print("R Squared Correlation of Ratings and Votes  =",R_squaredcorr)

####################################################################################################
# Plot of Revenue and Metascore 
import numpy as np 
import matplotlib.pyplot as plt

x = np.array(df["Metascore"]) 
y = np.array(df["Revenue (Millions)"]) 

plt.scatter(x,y)
plt.xlabel("Metascore") 
plt.ylabel("Revenue (Millions)")

################################################################################################
#R Squared Correlation Between Revenue and Metascore 
n = len(x)

sum_x = np.sum(x) 
sum_y = np.sum(y) 

sum_xy = np.sum(x*y)  

sum_x_squared = np.sum(x**2) 
sum_y_squared = np.sum(y**2) 

numerator = n*sum_xy -sum_x*sum_y 
denominator = np.sqrt(n*sum_x_squared - sum_x**2)*np.sqrt(n*sum_y_squared - sum_y**2) 

R_squaredcorr = (numerator/denominator)**2 

print("R Squared Correlation Between Metascore and Revenue=",R_squaredcorr) 
###########################################################################################
#R Squared Correlation Between Votes and Revenue 

x = np.array(df["Votes"]) 
y = np.array(df["Revenue (Millions)"])

n = len(x)

sum_x = np.sum(x) 
sum_y = np.sum(y) 

sum_xy = np.sum(x*y)  

sum_x_squared = np.sum(x**2) 
sum_y_squared = np.sum(y**2) 

numerator = n*sum_xy -sum_x*sum_y 
denominator = np.sqrt(n*sum_x_squared - sum_x**2)*np.sqrt(n*sum_y_squared - sum_y**2) 

R_squaredcorr = (numerator/denominator)**2 

print("R Squared Correlation Between Votes and Revenue =",R_squaredcorr)
########################################################################################
#R Squared Correlation Between Year and Revenue 
x = np.array(df["Year"]) 
y = np.array(df["Revenue (Millions)"]) 

n = len(x)

sum_x = np.sum(x) 
sum_y = np.sum(y) 

sum_xy = np.sum(x*y)  

sum_x_squared = np.sum(x**2) 
sum_y_squared = np.sum(y**2) 

numerator = n*sum_xy -sum_x*sum_y 
denominator = np.sqrt(n*sum_x_squared - sum_x**2)*np.sqrt(n*sum_y_squared - sum_y**2) 

R_squaredcorr = (numerator/denominator)**2 

print("R Squared Correlation Between Year and Revenue =",R_squaredcorr)
####################################################################################################
#R Squared Correlation Between Ratings and Revenue 

x = np.array(df["Rating"]) 
y = np.array(df["Revenue (Millions)"]) 

n = len(x)

sum_x = np.sum(x) 
sum_y = np.sum(y) 

sum_xy = np.sum(x*y)  

sum_x_squared = np.sum(x**2) 
sum_y_squared = np.sum(y**2) 

numerator = n*sum_xy -sum_x*sum_y 
denominator = np.sqrt(n*sum_x_squared - sum_x**2)*np.sqrt(n*sum_y_squared - sum_y**2) 

R_squaredcorr = (numerator/denominator)**2 

print("R Squared Correlation Between Ratings and Revenue =", R_squaredcorr)
########################################################################################
#R Squared Between Revenue and Metascore
x = np.array(df["Revenue (Millions)"]) 
y = np.array(df["Metascore"]) 

n = len(x)

sum_x = np.sum(x) 
sum_y = np.sum(y) 

sum_xy = np.sum(x*y)  

sum_x_squared = np.sum(x**2) 
sum_y_squared = np.sum(y**2) 

numerator = n*sum_xy -sum_x*sum_y 
denominator = np.sqrt(n*sum_x_squared - sum_x**2)*np.sqrt(n*sum_y_squared - sum_y**2) 

R_squaredcorr = (numerator/denominator)**2 

print("R Squared Correlation Between Revenue and Metascore =", R_squaredcorr)
#########################################################################################'
x = np.array(df["Runtime (Minutes)"]) 
y = np.array(df["Votes"]) 

n = len(x)

sum_x = np.sum(x) 
sum_y = np.sum(y) 

sum_xy = np.sum(x*y)  

sum_x_squared = np.sum(x**2) 
sum_y_squared = np.sum(y**2) 

numerator = n*sum_xy -sum_x*sum_y 
denominator = np.sqrt(n*sum_x_squared - sum_x**2)*np.sqrt(n*sum_y_squared - sum_y**2) 

R_squaredcorr = (numerator/denominator)**2 

print("R Squared Correlation Between Runtime and Votes =", R_squaredcorr)

#Number of Genres Question 11 of Quiz
a1 = df[df["Genre"] == "Horror"].count()
a2 = df[df["Genre"] =="Comedy"].count()  
a3 = df[df["Genre"] == "Action"].count() 
a4 = 




print(df["Genre"])






