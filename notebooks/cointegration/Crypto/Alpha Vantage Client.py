#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: warrenfulton
"""
'''
This is a client program to authenticate and create calls and requests to the Alpha Vantage financial API
'''

from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.foreignexchange import ForeignExchange
from alpha_vantage.techindicators import TechIndicators
import requests
import pandas as pd
import time


class AVClient:
    #initialize client with api key
    def __init__(self):
        try:
            with open("api_key.txt", "r") as f:
                self.key = f.read()
        except IOError:
            print("IOError: Could Not Find API Key")

        self.url = "https://www.alphavantage.co/query?"
            
    
    #request function to return stock time series
    #function and symbol are required
    #default interval is 1min for intraday data
    #default outputsize and data type are 'compact' and 'pandas' respectively
    #returns pandas dataframe of price information
    def getTimeSeries(self, function, symbols, interval='1min', outputsize="compact", datatype="pandas"):
        #check if datatype is valid and create instance of time series
        if datatype != 'pandas' and datatype != 'json' and datatype != 'csv':
            print("Invalid Datatype: Vaible options are 'pandas', 'json', 'csv'")
            return -1
        else:
            ts = TimeSeries(key=self.key, output_format=datatype)
            
        #check if outputsize is a valid input
        if outputsize != 'compact' and outputsize != 'full':
            print("Invalide Output Size: Viable options are 'compact' and 'full'")
            return -1
        
        #determine what time series the user wants and build request
        if function == 'intraday':
            valid_intervals = ['1min', '5min', '15min', '30min', '60min']
            if interval not in valid_intervals:
                print("Invalid Interval: Viable options are '1min', '5min', '15min', '30min', '60min'")
                return -1
            else:
                data, meta_data = ts.get_intraday(symbol=symbols, interval=interval, outputsize=outputsize)
        elif function == 'daily':
            data, meta_data = ts.get_daily(symbol=symbols, outputsize=outputsize)
        elif function == 'weekly':
            data, meta_data = ts.get_weekly(symbol=symbols)
        elif function == 'monthly':
            data, meta_data = ts.get_monthly(symbol=symbols)
        else:
            print("Invalid Function: Viable options are 'intraday', 'daily', 'weekly', and 'monthly'" )
            return -1
        
        return data
        
        
    #ruequest function to return fundamental data of a company
    #required parameters are function and symbol
    #returns a dictionary of fundamental data
    def getFundamentals(self, function, symbol):
        #check if function is valid
        valid_functions = ['OVERVIEW', 'INCOME_STATEMENT', 'BALANCE_SHEET', 'CASH_FLOW']
        if function not in valid_functions:
            print("Invalid Function: Viable options are 'OVERVIEW', 'INCOME_STATEMENT', 'BALANCE_SHEET', 'CASH_FLOW'")
            return -1
        
        #build url and data
        params = {'function': function,
                  'symbol': symbol,
                  'apikey': self.key}
        
        response = requests.get(url=self.url, params=params)
        
        data = response.json()

        return data
 

    #Function to grab technical indicators for any equity or currency pair
    #parameters are 'function' (string): which technical indicator you want (e.g. 'SMA', 'EMA', 'VWAP', etc.)
    #'symbol' (string): the ticker of the equity or currency pair
    #'interval' (string): daily, weekly, or monthly data
    #'time_period' (int): the time period for the technical indicator (e.g. 10, 50, 100, OR 200 SMA)
    #'series_type' (string): open, high, low, or close data
    #returns a dictionary of the values over the given interval
    def getTechIndicators(self, function, symbol, interval, time_period, series_type):
        
        params = {'function': function,
        		'symbol': symbol,
        		'interval': interval,
        		'time_period': time_period,
        		'series_type': series_type,
        		'apikey': self.key}

        response = requests.get(url=self.url, params=params)
        
        data = response.json()
        
        return data

    #works with any cryptocurrency and/or real currency
    #paramters are 'from_currency' (string), 'to_currency' (string) (e.g. from 'BTC' to 'XRP')
    #returens a dictionary of the given current exchange rate 
    def getCurrencyExchangeRates(self, from_currency, to_currency):

    	params = {'function': 'CURRENCY_EXCHANGE_RATE',
    			  'from_currency': from_currency,
    			  'to_currency': to_currency,
    			  'apikey': self.key}

    	response = requests.get(url=self.url, params=params)

    	data = response.json()

    	return data

    #function to grab time sereies data for forex currency
    #parameters are 'function' (string): intraday, daily, weekly, or monthly data
    #'from_symbol' (string): ticker of first currency
    #'to_symbol' (string): ticker of second currency
    #returns a dictionary time series data for the given exchange rate
    def getForexTimeSeries(self, function, from_symbol, to_symbol):

    	if function == 'daily':
    		function = 'FX_DAILY'
    	elif function == 'weekly':
    		function = 'FX_WEEKLY'
    	elif function == 'monthly':
    		function = 'FX_MONTHLY'
    	else:
    		print('Invalid Function: valid functions are intraday, daily, weekly and monthly.')

    	params = {'function': function,
    			  'from_symbol': from_symbol,
    			  'to_symbol': to_symbol,
    			  'outputsize': 'full',
    			  'apikey': self.key}

    	response = requests.get(url=self.url, params=params)

    	data = response.json()

    	return data

    #works the same as getForexTimeSeries but with crypto tickers
    #parameter 'market' (string) specifies which market you want to grab the current price data from (e.g. CNY)
    #returns a dictionary of the time series data
    def getCryptoTimeSeries(self, function, symbol, market):

    	if function == 'daily':
    		function = 'DIGITAL_CURRENCY_DAILY'
    	elif function == 'weekly':
    		function = 'DIGITAL_CURRENCY_WEEKLY'
    	elif function == 'monthly':
    		function = 'DIGITAL_CURRENCY_MONTHLY'
    	else:
    		print("Invalid Function: Valid functions are 'daily', 'weekly', and 'monthly'")

    	params = {'function': function,
    			  'symbol': symbol,
    			  'market': market,
    			  'apikey': self.key}

    	response = requests.get(url=self.url, params=params)

    	data = response.json()

    	return data


client = AVClient()

data = client.getCryptoTimeSeries('daily', 'BTC', 'USD')

df = pd.DataFrame.from_dict(data)

df.to_csv('Bitcoin.csv')



