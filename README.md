# Day-Trading-Stock-AI


Abstract

This project aims to explore a way to utilize neural network and deep learning algorithms to try to predict
tomorrow’s close price of a stock, and (if possible) give recommendation to which company to invest in by all profits
and losses for tomorrow. The program was built using Keras with the backend being Tensorflow - a free popular
modern solution for creating neural networks and machine learning. The programming language used in the project
is Python 3. The dataset used is called “Huge Stock Market Dataset” from “Kaggle” [1] which contains all historical
data of daily prices and volumes of all U.S. stocks and ETFs. In this project, the data of ETFs have not been used.
There are 7195 CSV files included in the dataset where each file indicates the historical data of a company in U.S.
trading on the NYSE, NASDAQ, and NYSE MKT. The data was last updated on 11/10/2017 and is presented in CSV
format as follows: Date, Open, High, Low, Close, Volume, OpenInt. Note that prices have been adjusted for dividends
and splits.

The program is trained on the dataset and creates and saves an exclusive model for every company. The created model
is a deep neural network with 4 layers. The prediction is done by feeding in the close price of the past 60 days and the
output will be the predicted price for the 61st day.
The accuracy of each model differs due to different data length of each file (for each company). It is presented as a
root-mean-squared error value for each stock.





Problems and Aim

Problem: Financial analysts who invest in stock market usually are not aware of the stock market behavior.
They are facing the problem of stock trading as they do not know which buying which stock will gain them most
profit. They have to analyze all news on financial newspapers, magazines, or even social media like Twitter, and
be up to date with current events to extract useful information to aid them in investing efficiently in the stock
market. [6] Is there an easier way of doing so? Can we have an AI system that handles the analysis of these data
and only give us the future prices?

Goal: To predict future stock price of a company with high accuracy and low error using only statistical data
of previous prices. The trained models are aimed to have lowest root-mean-squared error values possible (close
to zero) preferably less than 8.
* Possible additional goal: To predict tomorrow’s price of all stocks in the data base and recommend the stocks
with most day-profit to the user to invest in
