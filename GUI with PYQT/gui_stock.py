import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
sns.set_style('darkgrid')
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')
import os
import plotly.graph_objects as go
import joblib
import itertools
from sklearn.metrics import roc_auc_score, roc_curve, explained_variance_score, r2_score
from sklearn.model_selection import cross_val_score
from statsmodels.tsa.seasonal import seasonal_decompose as season
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
from sklearn.metrics import classification_report, f1_score
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import learning_curve
from mlxtend.plotting import plot_decision_regions
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRFRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.neural_network import MLPRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from pmdarima.utils import decomposed_plot
from pmdarima.arima import decompose
from pmdarima import auto_arima
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.mixture import GaussianMixture
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
from matplotlib.colors import ListedColormap


class DemoGUI_Yahoo(QDialog):
    def __init__(self):
        QDialog.__init__(self)
        loadUi("gui_yahoo.ui", self)
        self.setWindowTitle("GUI Demo of Forecasting Yahoo Stock Price with Machine Learning")
        self.initial_state(False)
        self.pbLoad.clicked.connect(self.import_dataset)
        self.cbData.currentIndexChanged.connect(self.choose_plot)
        self.cbForecasting.currentIndexChanged.connect(self.choose_forecasting)
        self.pbTrainML.clicked.connect(self.train_model_ML)
        self.cbClassifier.currentIndexChanged.connect(self.choose_ML_model)
    

    def initial_state(self, state):
        self.pbTrainML.setEnabled(state)
        self.cbData.setEnabled(state)
        self.cbForecasting.setEnabled(state)
        self.cbClassifier.setEnabled(state)
        self.rbRobust.setEnabled(state)
        self.rbMinMax.setEnabled(state)
        self.rbStandard.setEnabled(state)


    # Takes a df and writes it to a qtable provided. df headers become qtable headers
    @staticmethod
    def write_df_to_qtable(df, table):
        headers = list(df)
        table.setRowCount(df.shape[0])
        table.setColumnCount(df.shape[1])
        table.setHorizontalHeaderLabels(headers)
        # getting data from df is computationally costly so convert it to array first
        df_array = df.values
        for row in range(df.shape[0]):
            for col in range(df.shape[1]):
                table.setItem(row,col, QTableWidgetItem(str(df_array[row, col])))


    def populate_table(self, data, table):
        #Populate two tables
        self.write_df_to_qtable(data, table)

        table.setAlternatingRowColors(True)
        table.setStyleSheet("alternate-background-color: #ffb07c;background-color:#e6daa6;");

    
    def compute_year_month_wise(self, df):
        cols = list(df.columns)
        cols.remove("Month")
        cols.remove("Day")
        cols.remove("Week")
        cols.remove("Year")
        cols.remove("Quarter")

        #Resample the data year-wise by mean
        year_data_mean = df[cols].resample('y').mean()

        #Resample the data year-wise by ewm
        year_data_ewm = year_data_mean.ewm(span=5).mean()

        #Resample the data month-wise by mean
        monthly_data_mean = df[cols]. resample('m').mean()

        #Resample the data month-wise by EWM
        monthly_data_ewm = monthly_data_mean.ewm(span=5).mean()

        return year_data_mean, year_data_ewm, monthly_data_mean, monthly_data_ewm
    

    def create_new_dfs(self, df):
        #Extracts day, month, week, quarter and year
        df['Date'] = pd.to_datetime(df['Date'])
        df['Day'] = df['Date'].dt.weekday
        df['Month'] = df['Date'].dt.month
        df['Year'] = df['Date'].dt.year
        df['Week'] = df['Date'].dt.isocalendar().week
        df['Quarter'] = df['Date'].dt.quarter

        #Sets Date column as index
        df = df.set_index('Date')

        #Creates a dummy dataframe for visualisation
        df_dummy = df.copy()

        #Computes year-wise and month-wise data
        self.year_data_mean, self.year_data_ewm, self.monthly_data_mean, self.monthly_data_ewm = self.compute_year_month_wise(df_dummy)

        #Converts days, months and quarters from numerics to meaningful string
        days = {0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 4:'Friday'}
        df_dummy['Day'] = df_dummy['Day'].map(days)
        months = {1:'January', 2:'February', 3:'March', 4:'April', 5:'May', 6:'June', 7:'July', 8:'August', 9:'September', 10:'October', 11:'November', 12:'December'}
        df_dummy['Month'] = df_dummy['Month'].map(months)
        quarters = {1:'Jan-March', 2:'April-June', 3:'July-Sept', 4:'Oct-Dec'}
        df_dummy['Quarter'] = df_dummy['Quarter'].map(quarters)
        return df, df_dummy
    

    def compute_daily_returns(self, df):
        '''Compute and return the daily return values.'''
        # TODO: Your code here
        # Note: Returned DataFrame must have the same number of rows
        daily_return = (df / df.shift(1)) - 1
        daily_return[0] = 0
        return daily_return
    

    def calculate_SMA(self, df, periods=15):
        SMA = df.rolling(window=periods, min_periods=periods, center=False).mean()
        return SMA
    

    def calculate_MACD(self, df, nslow=26, nfast=12):
        emaslow = df.ewm(span=nslow, min_periods=nslow, adjust=True, ignore_na=False).mean()
        emafast = df.ewm(span=nfast, min_periods=nfast, adjust=True, ignore_na=False).mean()
        dif = emafast - emaslow
        MACD = dif.ewm(span=9, min_periods=9, adjust=True, ignore_na=False).mean()
        return dif, MACD
    

    def calculate_RSI(self, df, periods=14):
        #wilder's RSI
        delta = df.diff()
        up, down = delta.copy(), delta.copy()

        up[up < 0] = 0
        down[down > 0] = 0

        rUp = up.ewm(com=periods, adjust=False).mean()
        rDown = down.ewm(com=periods, adjust=False).mean().abs()

        rsi = 100 - 100 / (1 + rUp / rDown)
        return rsi
    

    def calculate_BB(self, df, periods=15):
        STD = df.rolling(window=periods, min_periods=periods, center=False).std()
        SMA = self.calculate_SMA(df)
        upper_band = SMA + (2 * STD)
        lower_band = SMA - (2 * STD)
        return upper_band, lower_band
    

    def calculate_stdev(self, df, periods=5):
        STDEV = df.rolling(periods).std()
        return STDEV
    

    def compute_technical_indicators(self, df):
        stock_close = df['Adj Close']
        daily_returns = self.compute_daily_returns(stock_close)
        SMA_CLOSE = self.calculate_SMA(stock_close)
        upper_band, lower_band = self.calculate_BB(stock_close)
        DIF, MACD = self.calculate_MACD(stock_close)
        RSI = self.calculate_RSI(stock_close)
        STDEV = self.calculate_stdev(stock_close)
        Open_Close = df.Open - df['Adj Close']
        High_Low = df.High - df.Low

        df['daily_returns'] = daily_returns
        df['SMA'] = SMA_CLOSE
        df['Upper_band'] = upper_band
        df['Lower_band'] = lower_band
        df['DIF'] = DIF
        df['MACD'] = MACD
        df['RSI'] = RSI
        df['STDEV'] = STDEV
        df['Open_Close'] = Open_Close
        df['High_Low'] = High_Low

        #Checks null values because of technical indicators 
        print(df.isnull().sum().to_string())
        print('Total number of null values: ', df.isnull().sum().sum())

        #Fills each null value in every column with mean value
        cols = list(df.columns)
        for n in cols:
            df[n].fillna(df[n].mean(), inplace=True)

        #Checks again null values
        print(df.isnull().sum().to_string())
        print('Total number of null values: ', df.isnull().sum().sum())

    
    def populate_cbForecasting(self):
        self.cbForecasting.addItems(["Linear Regression", "Random Forest Regression"])
        self.cbForecasting.addItems(["Decision Tree Regression", "KNN Regression"])
        self.cbForecasting.addItems(["Adaboost Regression", "Gradient Boosting Regression"])
        self.cbForecasting.addItems(["XGB Regression", "LGBM Regression"])
        self.cbForecasting.addItems(["Catboost Regression", "SVR Regression"])
        self.cbForecasting.addItems(["MLP Regression", "Lasso Regression", "Ridge Regression"])
    

    def populate_cbData(self):
        self.cbData.addItems(['Case Distribution'])
        self.cbData.addItems(['High', 'Low', 'Open', 'Close'])
        self.cbData.addItems(['Adj Close', 'Volume', 'Technical Indicators'])
        self.cbData.addItems(['Year-Wise', 'Month-Wise'])


    def read_dataset(self, dir):
        #Reads dataset
        df = pd.read_csv(dir)

        #Creates new Dataframes
        self.df, self.df_dummy=self.create_new_dfs(df)

        #Splits data for regression
        self.X_final, self.y_final, self.X_train, self.y_train, self.X_test, self.y_test, self.X_val, self.y_val=self.split_data_regression(self.df)

        #Computes technical indicators
        self.compute_technical_indicators(self.df)


    def import_dataset(self):
        curr_path = os.getcwd()
        dataset_dir = curr_path + "/VOO-1.csv"

        self.read_dataset(dataset_dir)
        print("Dataframe has been read...")

        #Populates cbData and cbForecasting
        self.populate_cbData()
        self.populate_cbForecasting()
        
        #Populates tables with data
        self.populate_table(self.df, self.twData1)
        self.label1.setText('Data for Forecasting')

        self.populate_table(self.df_dummy, self.twData2)
        self.label2.setText('Data for Visualisation')

        #Turns off pbLoad
        self.pbLoad.setEnabled(False)

        #Turns on cbForecasting and cbData
        self.cbForecasting.setEnabled(True)
        self.cbData.setEnabled(True)

        #Turns on pbTrainML widget
        self.pbTrainML.setEnabled(True)


    #Defines function to plot case distribution of a categorical feature bar plot
    def plot_barchart(self, df, var, widget):
        ax = df[var].value_counts().plot(kind="barh",ax = widget.canvas.axis1)
        for i, j in enumerate(df[var].value_counts().values):
            ax.text(.7, i, j, weight = 'bold', fontsize=10)

        widget.canvas.axis1.set_title("Case distribution " + " of " + var + " variable", fontsize=14)
        widget.canvas.figure.tight_layout()
        widget.canvas.draw()


    #Defines function to plot case distribution of a categorical feature in pie chart
    def plot_piechart(self, df, var, widget):
        label_list = list(df[var].value_counts().index)
        df[var].value_counts().plot.pie(ax = widget.canvas.axis1, autopct="%1.1f%%", colors=sns.color_palette("prism", 7), startangle=60, labels=label_list, wedgeprops={"linewidth":2, "edgecolor":"k"}, shadow=True, textprops={'fontsize':10})
        widget.canvas.axis1.set_title("Case distribution "+ " of "+ var + " variable", fontsize=14)
        widget.canvas.figure.tight_layout()
        widget.canvas.draw()


    def color_month(self, month):
        if month == 1:
            return "January", "blue"
        elif month == 2:
            return "February", "green"
        elif month == 3:
            return "March", "orange"
        elif month == 4:
            return "April", "yellow"
        elif month == 5:
            return "May", "red"
        elif month == 6:
            return "June", "violet"
        elif month == 7:
            return "July", "purple"
        elif month == 8:
            return "August", "black"
        elif month == 9:
            return "September", "brown"
        elif month == 10:
            return "October", "darkblue"
        elif month == 11:
            return "November", "grey"
        else:
            return "December", "pink"
        

    def line_plot_month(self, month, data, ax):
        label, color = self.color_month(month)
        mdata = data[data.index.month == month]
        sns.lineplot(data=mdata, ax=ax, label=label, color=color, marker='o', linewidth=3)


    def sns_plot_month(self, data, feat, ax):
        for i in range(1,13):
            self.line_plot_month(i,data[feat],ax)


    def choose_plot(self):
        strCB = self.cbData.currentText()

        if strCB == "Case Distribution":
            self.widgetPlot1.canvas.figure.clf()
            self.widgetPlot1.canvas.axis1 = self.widgetPlot1.canvas.figure.add_subplot(121, facecolor='#a9f971')
            self.plot_barchart(self.df_dummy, "Year", self.widgetPlot1)

            self.widgetPlot1.canvas.axis1 = self.widgetPlot1.canvas.figure.add_subplot(122, facecolor='#a9f971')
            self.plot_piechart(self.df_dummy, "Year", self.widgetPlot1)

            self.widgetPlot2.canvas.figure.clf()
            self.widgetPlot2.canvas.axis1 = self.widgetPlot2.canvas.figure.add_subplot(121, facecolor='#a9f971')
            self.plot_barchart(self.df_dummy, "Month", self.widgetPlot2)

            self.widgetPlot2.canvas.axis1 = self.widgetPlot2.canvas.figure.add_subplot(122, facecolor ='#a9f971')
            self.plot_piechart(self.df_dummy, "Month", self.widgetPlot2)

            self.widgetPlot3.canvas.figure.clf()
            self.widgetPlot3.canvas.axis1 = self.widgetPlot3.canvas.figure.add_subplot(221, facecolor = '#a9f971')
            self.plot_barchart(self.df_dummy, "Day", self.widgetPlot3)

            self.widgetPlot3.canvas.axis1 = self.widgetPlot3.canvas.figure.add_subplot(222, facecolor = '#a9f971')
            self.plot_piechart(self.df_dummy, "Day", self.widgetPlot3)

            self.widgetPlot3.canvas.axis1 = self.widgetPlot3.canvas.figure.add_subplot(223, facecolor = '#a9f971')
            self.plot_barchart(self.df_dummy, "Quarter", self.widgetPlot3)

            self.widgetPlot3.canvas.axis1 = self.widgetPlot3.canvas.figure.add_subplot(224, facecolor = '#a9f971')
            self.plot_piechart(self.df_dummy, "Quarter", self.widgetPlot3)

        elif strCB == "High":
            self.plot_distribution(strCB, "Adj Close", "Volume")

        elif strCB == "Low":
            self.plot_distribution(strCB, "Adj Close", "Volume")

        elif strCB == "Open":
            self.plot_distribution(strCB, "Adj Close", "Volume")

        elif strCB == "Close":
            self.plot_distribution(strCB, "Open", "Volume")

        elif strCB == "Volume":
            self.widgetPlot1.canvas.figure.clf()
            self.widgetPlot1.canvas.axis1 = self.widgetPlot1.canvas.figure.add_subplot(121, facecolor='#a9f971')
            self.plot_group_barchart(self.df_dummy.groupby('Year')['Volume'].sum(), "Volume", "The distribution of Volume by Year", self.widgetPlot1)

            self.widgetPlot1.canvas.axis1 = self.widgetPlot1.canvas.figure.add_subplot(122, facecolor = '#a9f971')
            self.plot_group_piechart(self.df_dummy.groupby("Year")['Volume'].sum(), "The distribution of Volume by Year", self.widgetPlot1)

            self.widgetPlot2.canvas.figure.clf()
            self.widgetPlot2.canvas.axis1 = self.widgetPlot2.canvas.figure.add_subplot(121, facecolor = '#a9f971')
            self.plot_group_barchart(self.df_dummy.groupby('Quarter')['Volume'].sum(), "Volume", "The distribution of Volume by Quarter", self.widgetPlot2)

            self.widgetPlot2.canvas.axis1 = self.widgetPlot2.canvas.figure.add_subplot(122, facecolor = '#a9f971')
            self.plot_group_piechart(self.df_dummy.groupby("Quarter")["Volume"].sum(), "The distribution of Volume by Quarter", self.widgetPlot2)

            self.widgetPlot3.canvas.figure.clf()
            self.widgetPlot3.canvas.axis1 = self.widgetPlot3.canvas.figure.add_subplot(221, facecolor = '#a9f971')
            self.plot_group_barchart(self.df_dummy.groupby("Day")["Volume"].sum(), "Volume", "The distribution of Volume by Days of Week", self.widgetPlot3)

            self.widgetPlot3.canvas.axis1 = self.widgetPlot3.canvas.figure.add_subplot(222, facecolor = '#a9f971')
            self.plot_group_piechart(self.df_dummy.groupby("Day")["Volume"].sum(), "The distribution of Volume by Days of Week", self.widgetPlot3)

            self.widgetPlot3.canvas.axis1 = self.widgetPlot3.canvas.figure.add_subplot(223, facecolor = '#a9f971')
            self.plot_group_barchart(self.df_dummy.groupby("Month")["Volume"].sum(), "Volume", "The distribution of Volume by Month", self.widgetPlot3)

            self.widgetPlot3.canvas.axis1 = self.widgetPlot3.canvas.figure.add_subplot(224, facecolor='#a9f971')
            self.plot_group_piechart(self.df_dummy.groupby("Month")["Volume"].sum(), "The distribution of Volume by Month", self.widgetPlot3)

        elif strCB == 'Year-Wise':
            self.widgetPlot3.canvas.figure.clf()
            self.widgetPlot3.canvas.axis1 = self.widgetPlot3.canvas.figure.add_subplot(221, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis2 = self.widgetPlot3.canvas.figure.add_subplot(222, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis3 = self.widgetPlot3.canvas.figure.add_subplot(223, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis4 = self.widgetPlot3.canvas.figure.add_subplot(224, facecolor='#a9f971')
            norm_data = (self.year_data_mean - self.year_data_mean.min()) / (self.year_data_mean.max() - self.year_data_mean.min())
            self.plot_norm_wise_data(norm_data, self.widgetPlot3.canvas.axis1, self.widgetPlot3.canvas.axis2, self.widgetPlot3.canvas.axis3, self.widgetPlot3.canvas.axis4, "normalised year-wise data")
            self.widgetPlot3.canvas.figure.tight_layout()
            self.widgetPlot3.canvas.draw()

        #plotting month-wise distribution
        elif strCB == 'Month-Wise':
            self.widgetPlot3.canvas.figure.clf()
            self.widgetPlot3.canvas.axis1 = self.widgetPlot3.canvas.figure.add_subplot(221, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis2 = self.widgetPlot3.canvas.figure.add_subplot(222, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis3 = self.widgetPlot3.canvas.figure.add_subplot(223, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis4 = self.widgetPlot3.canvas.figure.add_subplot(224, facecolor='#a9f971')
            norm_data = (self.monthly_data_mean - self.monthly_data_mean.min()) / (self.monthly_data_mean.max() - self.monthly_data_mean.min())
            self.plot_norm_wise_data(norm_data, self.widgetPlot3.canvas.axis1, self.widgetPlot3.canvas.axis2, self.widgetPlot3.canvas.axis3, self.widgetPlot3.canvas.axis4, "normalised month-wise data")
            self.widgetPlot3.canvas.figure.tight_layout()
            self.widgetPlot3.canvas.draw()


        elif strCB == "Technical Indicators":
            self.widgetPlot3.canvas.figure.clf()
            self.widgetPlot3.canvas.axis1 = self.widgetPlot3.canvas.figure.add_subplot(311, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis2 = self.widgetPlot3.canvas.figure.add_subplot(312, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis3 = self.widgetPlot3.canvas.figure.add_subplot(313, facecolor='#a9f971')

            self.widgetPlot1.canvas.figure.clf()
            self.widgetPlot1.canvas.axis4 = self.widgetPlot1.canvas.figure.add_subplot(111, facecolor='#a9f971')

            self.widgetPlot2.canvas.figure.clf()
            self.widgetPlot2.canvas.axis5 = self.widgetPlot2.canvas.figure.add_subplot(111, facecolor='#a9f971')

            self.plot_technical_indicators(self.df, self.widgetPlot3.canvas.axis1, self.widgetPlot3.canvas.axis2, self.widgetPlot3.canvas.axis3, self.widgetPlot1.canvas.axis4, self.widgetPlot2.canvas.axis5)
            self.widgetPlot3.canvas.figure.tight_layout()
            self.widgetPlot2.canvas.figure.tight_layout()
            self.widgetPlot1.canvas.figure.tight_layout()
            self.widgetPlot3.canvas.draw()
            self.widgetPlot2.canvas.draw()
            self.widgetPlot1.canvas.draw()


    def plot_distribution(self, strCB, feat1, feat2):
        self.widgetPlot1.canvas.figure.clf()
        self.widgetPlot1.canvas.axis1 = self.widgetPlot1.canvas.figure.add_subplot(111, facecolor='#a9f971')
        sns.lineplot(data=self.df_dummy[strCB], color='red', linewidth=3, ax=self.widgetPlot1.canvas.axis1)
        self.widgetPlot1.canvas.axis1.set_title(strCB + " feature all year", fontsize=20)
        self.widgetPlot1.canvas.figure.tight_layout()
        self.widgetPlot1.canvas.draw()

        self.widgetPlot2.canvas.figure.clf()
        self.widgetPlot2.canvas.axis1 = self.widgetPlot2.canvas.figure.add_subplot(111, facecolor='#a9f971')
        sns.scatterplot(data=self.df_dummy, x=strCB, y=feat1, hue='Year', palette='deep', ax=self.widgetPlot2.canvas.axis1)
        self.widgetPlot2.canvas.axis1.set_title("Scatter distribution of " + strCB + " vs " + feat1 + " vs Year", fontsize=20)
        self.widgetPlot2.canvas.figure.tight_layout()
        self.widgetPlot2.canvas.draw()

        self.widgetPlot3.canvas.figure.clf()
        self.widgetPlot3.canvas.axis1 = self.widgetPlot3.canvas.figure.add_subplot(221, facecolor="#a9f971")
        sns.scatterplot(data=self.df_dummy, x=strCB, y=feat2, hue="Day", palette="deep", ax=self.widgetPlot3.canvas.axis1)
        self.widgetPlot3.canvas.axis1.set_title("Scatter distribution of " + strCB + " vs " + feat2 + " vs Day", fontsize=20)
        self.widgetPlot3.canvas.figure.tight_layout()
        self.widgetPlot3.canvas.draw()

        self.widgetPlot3.canvas.axis1 = self.widgetPlot3.canvas.figure.add_subplot(222, facecolor="#a9f971")
        self.year_data_mean[strCB].plot(linewidth=5, ax=self.widgetPlot3.canvas.axis1)
        self.year_data_ewm[strCB].plot(linewidth=5, ax=self.widgetPlot3.canvas.axis1)
        self.widgetPlot3.canvas.axis1.set_title("Year-Wise Data: Mean and EWM", fontsize=14)
        self.widgetPlot3.canvas.axis1.set_ylabel(strCB, fontsize=12)
        self.widgetPlot3.canvas.axis1.legend(["Mean", "EWM"], fontsize=20)
        self.widgetPlot3.canvas.figure.tight_layout()
        self.widgetPlot3.canvas.draw()

        self.widgetPlot3.canvas.axis1 = self.widgetPlot3.canvas.figure.add_subplot(223, facecolor='#a9f971')
        self.monthly_data_mean[strCB].plot(linewidth=5, ax=self.widgetPlot3.canvas.axis1)
        self.monthly_data_ewm[strCB].plot(linewidth=5, ax=self.widgetPlot3.canvas.axis1)
        self.widgetPlot3.canvas.axis1.set_title(strCB + ": Month-Wise Data: Mean and EWM", fontsize=14)
        self.widgetPlot3.canvas.axis1.set_ylabel(strCB, fontsize=12)
        self.widgetPlot3.canvas.axis1.legend(["Mean", "EWM"], fontsize=20)
        self.widgetPlot3.canvas.figure.tight_layout()
        self.widgetPlot3.canvas.draw()

        self.widgetPlot3.canvas.axis1 = self.widgetPlot3.canvas.figure.add_subplot(224, facecolor='#a9f971')
        self.sns_plot_month(self.monthly_data_mean, strCB, ax=self.widgetPlot3.canvas.axis1)
        self.widgetPlot3.canvas.axis1.set_title(strCB + ": Month-Wise Data for Every Month", fontsize=14)
        self.widgetPlot3.canvas.axis1.set_ylabel(strCB, fontsize=12)
        self.widgetPlot3.canvas.figure.tight_layout()
        self.widgetPlot3.canvas.draw()


    #Plotting grouped Distribution
    #Defines function to plot grouped distribution of a categorical feature bar plot

    def plot_group_barchart(self, df, var, title, widget):
        ax = df.plot(kind='barh', ax=widget.canvas.axis1)
        for i, j in enumerate(df.values):
            ax.text(.7, i, j, weight='bold', fontsize=10)

        widget.canvas.axis1.set_title(title, fontsize=14)
        widget.canvas.axis1.set_xlabel(var)
        widget.canvas.figure.tight_layout()
        widget.canvas.draw()

    #Defines function to plot case grouped distribution of a categorical feature in pie chart

    def plot_group_piechart(self, df, title, widget):
        label_list = list(df.index)
        df.plot.pie(ax = widget.canvas.axis1, autopct="%1.1f%%", colors=sns.color_palette('prism', 7), startangle=60, labels=label_list, wedgeprops={"linewidth":2, "edgecolor":"k"}, shadow=True, textprops={'fontsize': 10})
        widget.canvas.axis1.set_title(title, fontsize=14)
        widget.canvas.figure.tight_layout()
        widget.canvas.draw()


    #Plotting Year-Wise Distribution 
    #Plots boxplot, violinplot, stripplot and heatmap of normalised year-wise data
    def plot_norm_wise_data(self, norm_data, ax1, ax2, ax3, ax4, title):
        g=sns.boxplot(data=norm_data, ax=ax1)
        g.xaxis.get_label().set_fontsize(10)
        g.set_title("The box plot of " + title, fontsize=15)

        g=sns.violinplot(data=norm_data, ax=ax2)
        g.xaxis.get_label().set_fontsize(10)
        g.set_title("The violin plot of " + title, fontsize=15)

        g=sns.stripplot(data=norm_data, jitter=True, s=18, alpha=0.3, ax=ax3)
        g.xaxis.get_label().set_fontsize(10)
        g.set_title("The strip plot of " + title, fontsize=15)

        g=sns.lineplot(data=norm_data, marker='s', ax=ax4, linewidth=5)
        g.xaxis.get_label().set_fontsize(30)
        g.set_title("The line plot of " + title, fontsize=15)
        g.set_xlabel("YEAR")


    #Plotting Technical Indicators
    #Plots MACD, SMA, RSI, upper and lower bands, standard deviation and daily returns of Adj Close column
    def plot_technical_indicators(self, df, ax1, ax2, ax3, ax4, ax5):
        stock_close = df['Adj Close']
        SMA_CLOSE = df['SMA']
        stock_close[:365].plot(title='GLD Moving Average', label='GLD', linewidth=3, ax=ax1)
        SMA_CLOSE[:365].plot(label='SMA', linewidth=3, ax=ax1)

        upper_band = df['Upper_band']
        lower_band = df['Lower_band']
        upper_band[:365].plot(title="Upper-Lower Band", label='upper band', linewidth=3, ax=ax1)
        lower_band[:365].plot(label='lower band', linewidth=3, ax=ax1)

        DIF = df['DIF']
        MACD = df['MACD']
        DIF[:365].plot(title='DIF and MACD', label='DIF', linewidth=3, ax=ax2)
        MACD[:365].plot(label='MACD', linewidth=3, ax=ax2)

        RSI = df['RSI']
        RSI[:365].plot(title='RSI', label='RSI', linewidth=3, ax=ax3)

        STDEV = df['STDEV']
        STDEV[:365].plot(title='STDEV', label='STDEV', linewidth=3, ax=ax4)

        Daily_Return = df['daily_returns']
        Daily_Return[:365].plot(title='Daily Returns', label='Daily Returns', linewidth=3, ax=ax5)

        ax1.set_ylabel('Price')
        ax2.set_ylabel('Price')
        ax3.set_ylabel('Price')
        ax4.set_ylabel('Price')
        ax5.set_ylabel('Price')


    #Preparing Data For Forecasting
    def split_data_regression(self, X):
        #Sets target column
        y_final = pd.DataFrame(X["Adj Close"])
        X = X.drop(["Adj Close"], axis=1)

        #Normalises data
        scaler = MinMaxScaler()
        X_minmax_data = scaler.fit_transform(X)
        X_final = pd.DataFrame(columns=X.columns, data=X_minmax_data, index=X.index)
        print('Shape of features : ', X_final.shape)
        print('Shape of target : ', y_final.shape)

        #Shifts target array to predict the n + 1 samples
        n=90
        y_final = y_final.shift(-1)
        y_val = y_final[-n:-1]
        y_final = y_final[:-n]

        #Takes last n rows of data to be validation set
        X_val = X_final[-n:-1]
        X_final = X_final[:-n]

        print("\n -----After process----- \n")
        print('Shape of features : ', X_final.shape)
        print('Shape of target : ', y_final.shape)
        print(y_final.tail().to_string())

        y_final = y_final.astype('float64')

        #Splits data intp training and test data at 90% and 10% respectively 

        self.split_idx=round(0.9*len(X))
        print('split_idx=', self.split_idx)
        X_train = X_final[:self.split_idx]
        y_train = y_final[:self.split_idx]
        X_test = X_final[self.split_idx:]
        y_test = y_final[self.split_idx:]

        return X_final, y_final, X_train, y_train, X_test, y_test, X_val, y_val


    def perform_regression(self, model, X, y, xtrain, ytrain, xtest, ytest, xval, yval, label, feat, ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9):
        model.fit(xtrain, ytrain)
        predictions_test = model.predict(xtest)
        predictions_train = model.predict(xtrain)
        prediction_val = model.predict(xval)

        str_label = 'RMSE using ' + label
        print(str_label + f': {np.sqrt(mean_squared_error(ytest, predictions_test))}')
        print("mean square error: ", mean_squared_error(ytest, predictions_test))
        print("variance or r-squared: ", explained_variance_score(ytest, predictions_test))
        print('PREDICTED: Avg. ' + feat + f': {predictions_test.mean()}')
        print('PREDICTED: Median ' + feat + f': {np.median(predictions_test)}')

        #Evaluation of regression an all dataset
        all_pred = model.predict(X)
        print('mean square error (whole dataset): ', mean_squared_error(y, all_pred))
        print('variance or r-squared (whole dataset): ', explained_variance_score(y, all_pred))

        #Visualises the training set result in scatter plot
        ax0.scatter(x=ytrain, y=predictions_train, color='blue')
        ax0.set_title('The scatter of actual versus predicted Concrete compressive strength (Training set): ' + label, fontweight='bold', fontsize=10)
        ax0.set_xlabel('Actual Train Set', fontsize=10)
        ax0.set_ylabel('Predicted Train Set', fontsize=10)
        ax0.plot([ytrain.min(), ytrain.max()], [ytrain.min(), ytrain.max()], 'r--', linewidth=3)

        #visualises the test set results in a scatter plot
        ax1.scatter(x=ytest, y=predictions_test, color='red')
        ax1.set_title('The scatter of actual vs predicted Concrete compressive strength (Test set):' + label, fontweight='bold', fontsize=10)
        ax1.plot([ytest.min(), ytest.max()], [ytest.min(), ytest.max()], 'b--', linewidth=3)
        ax1.set_xlabel('Actual Test Set', fontsize=10)
        ax1.set_ylabel('Predicted Test Set', fontsize=10)

        #Visualises the validation set results in a scatter plot
        ax2.scatter(x=yval, y=prediction_val, color='red')
        ax2.set_title('The scatter of actual vs predicted Concrete compressive strength (Validation set): ' + label, fontweight='bold', fontsize=10)
        ax2.plot([yval.min(), yval.max()], [yval.min(), yval.max()], 'b--', linewidth=3)
        ax2.set_xlabel('Actual Validation Set', fontsize=10)
        ax2.set_ylabel('Predicted Validation Set', fontsize=10)

        #Visualises the density of errorr of training and testing 
        sns.distplot(np.array(ytrain - predictions_train.reshape(len(ytrain), 1)), ax=ax3, color='red', kde_kws=dict(linewidth=3))
        ax3.set_xlabel('Error', fontsize=10)
        sns.distplot(np.array(ytest - predictions_test.reshape(len(ytest), 1)), ax=ax3, color='blue', kde_kws=dict(linewidth=3))
        sns.distplot(np.array(yval - prediction_val.reshape(len(yval), 1)), ax=ax3, color='green', kde_kws=dict(linewidth=3))
        ax3.set_title('The density of training, testing and validation errors: ' + label, fontsize=10, fontweight='bold')
        ax3.set_xlabel('Error', fontsize=10)
        ax3.legend(["Training Error", "Testing Error", "Validation Error"], prop={'size': 8})

        #Histogram distribution of regression on train data
        sns.histplot(predictions_train, ax=ax4, kde=True, bins=50, color='red', line_kws={'lw': 3});
        sns.histplot(ytrain, ax=ax4, kde=True, bins=50, color='blue', line_kws={'lw': 3});
        ax4.set_title("Histogram Distribution of " + label + " on " + feat + "feature on train data", fontsize=10, fontweight='bold');
        ax4.set_xlabel(feat, fontsize=10)
        ax4.set_ylabel('Count', fontsize=10)
        ax4.legend(['Prediction', 'Actual'], prop={'size': 8})

        for p in ax4.patches:
            ax4.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), weight='bold', fontsize=10, textcoords='offset points')
        
        #Histogram distribution of regression on test data 
        sns.histplot(predictions_test, ax=ax5, kde=True, bins=50, color='red', line_kws={'lw': 3});
        sns.histplot(ytest, ax=ax5, kde=True, bins=50, color='blue', line_kws={'lw': 3});
        ax5.set_title("Histogram Distribution of " + label + " on " + feat + "feature on test data", fontsize=10, fontweight='bold');
        ax5.set_xlabel(feat, fontsize=10)
        ax5.set_ylabel("Count", fontsize=10)
        ax5.legend(["Prediction", "Actual"])

        for p in ax5.patches:
            ax5.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), weight='bold', fontsize=10, textcoords='offset points')
        
        ax6.plot(X.index[:self.split_idx], ytrain, color='blue', linewidth=3, linestyle='-', label='Actual')
        ax6.plot(X.index[:self.split_idx], predictions_train, color='red', linewidth=3, linestyle='-', label='Predicted')
        ax6.set_title('Actual and Predicted Training Set: ' + label, fontsize=10)
        ax6.set_xlabel('Date', fontsize=10)
        ax6.set_ylabel(feat, fontsize=10)
        ax6.legend(prop={'size': 10})

        ax7.plot(X.index[self.split_idx:], ytest, color='blue', linewidth=3, linestyle='-', label='Actual')
        ax7.plot(X.index[self.split_idx:], predictions_test, color='red', linewidth=3, linestyle='-', label='Predicted')
        ax7.set_title('Actual and predicted Test Set: ' + label, fontsize=10)
        ax7.set_xlabel('Date', fontsize=10)
        ax7.set_ylabel(feat, fontsize=10)
        ax7.legend(prop={'size': 10})

        ax8.plot(yval.index, yval, color='blue', linewidth=3, linestyle='-', label='Actual')
        ax8.plot(yval.index, prediction_val, color='red', linewidth=3, linestyle='-', label='Predicted')
        ax8.set_title('Actual and Predicted Validation Set (90 days forecasting): ' + label, fontsize=10)
        ax8.set_xlabel('Date', fontsize=10)
        ax8.set_ylabel(feat, fontsize=10)
        ax8.legend(prop={'size': 10})

        ax9.plot(X.index, y, color='blue', linewidth=3, linestyle='-', label='Actual')
        ax9.plot(X.index, all_pred, color='red', linewidth=3, linestyle='-', label='Predicted')
        ax9.set_title('Actual and Predicted Whole Dataset: ' + label, fontsize=10)
        ax9.set_xlabel('Date', fontsize=10)
        ax9.set_ylabel(feat, fontsize=10)
        ax9.legend(prop={'size': 10})


    #Linear Regression
    def choose_forecasting(self):
        strCB = self.cbForecasting.currentText()

        if strCB == "Linear Regression":
            self.widgetPlot1.canvas.figure.clf()
            self.widgetPlot1.canvas.axis1 = self.widgetPlot1.canvas.figure.add_subplot(111, facecolor = '#a9f971')
            self.widgetPlot2.canvas.figure.clf()
            self.widgetPlot2.canvas.axis2 = self.widgetPlot2.canvas.figure.add_subplot(111, facecolor = '#a9f971')

            self.widgetPlot3.canvas.figure.clf()
            self.widgetPlot3.canvas.axis3 = self.widgetPlot3.canvas.figure.add_subplot(421, facecolor = '#a9f971')
            self.widgetPlot3.canvas.axis4 = self.widgetPlot3.canvas.figure.add_subplot(422, facecolor = '#a9f971')
            self.widgetPlot3.canvas.axis5 = self.widgetPlot3.canvas.figure.add_subplot(423, facecolor = '#a9f971')
            self.widgetPlot3.canvas.axis6 = self.widgetPlot3.canvas.figure.add_subplot(424, facecolor = '#a9f971')
            self.widgetPlot3.canvas.axis7 = self.widgetPlot3.canvas.figure.add_subplot(425, facecolor = '#a9f971')
            self.widgetPlot3.canvas.axis8 = self.widgetPlot3.canvas.figure.add_subplot(426, facecolor = '#a9f971')
            self.widgetPlot3.canvas.axis9 = self.widgetPlot3.canvas.figure.add_subplot(427, facecolor = '#a9f971')
            self.widgetPlot3.canvas.axis10 = self.widgetPlot3.canvas.figure.add_subplot(428, facecolor = '#a9f971')

            lin_reg = LinearRegression()
            self.perform_regression(lin_reg, self.X_final, self.y_final, self.X_train, self.y_train, self.X_test, self.y_test, self.X_val, self.y_val, "Linear Regression", "Adj Close", self.widgetPlot1.canvas.axis1, self.widgetPlot2.canvas.axis2, self.widgetPlot3.canvas.axis3, self.widgetPlot3.canvas.axis4, self.widgetPlot3.canvas.axis5, self.widgetPlot3.canvas.axis6, self.widgetPlot3.canvas.axis7, self.widgetPlot3.canvas.axis8, self.widgetPlot3.canvas.axis9, self.widgetPlot3.canvas.axis10)
            self.widgetPlot3.canvas.figure.tight_layout()
            self.widgetPlot2.canvas.figure.tight_layout()
            self.widgetPlot1.canvas.figure.tight_layout()
            self.widgetPlot3.canvas.draw()
            self.widgetPlot2.canvas.draw()
            self.widgetPlot1.canvas.draw()
            joblib.dump(lin_reg, 'LRmodel.joblib')
        
        #Random Forest Regression
        elif strCB == "Random Forest Regression":
            self.widgetPlot1.canvas.figure.clf()
            self.widgetPlot1.canvas.axis1 = self.widgetPlot1.canvas.figure.add_subplot(111, facecolor='#a9f971')

            self.widgetPlot2.canvas.figure.clf()
            self.widgetPlot2.canvas.axis2 = self.widgetPlot2.canvas.figure.add_subplot(111, facecolor='#a9f971')

            self.widgetPlot3.canvas.figure.clf()
            self.widgetPlot3.canvas.axis3 = self.widgetPlot3.canvas.figure.add_subplot(421, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis4 = self.widgetPlot3.canvas.figure.add_subplot(422, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis5 = self.widgetPlot3.canvas.figure.add_subplot(423, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis6 = self.widgetPlot3.canvas.figure.add_subplot(424, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis7 = self.widgetPlot3.canvas.figure.add_subplot(425, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis8 = self.widgetPlot3.canvas.figure.add_subplot(426, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis9 = self.widgetPlot3.canvas.figure.add_subplot(427, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis10 = self.widgetPlot3.canvas.figure.add_subplot(428, facecolor='#a9f971')

            rf_reg = RandomForestRegressor()
            self.perform_regression(rf_reg, self.X_final, self.y_final, self.X_train, self.y_train, self.X_test, self.y_test, self.X_val, self.y_val, "RF Regression", "Adj Close", self.widgetPlot1.canvas.axis1, self.widgetPlot2.canvas.axis2, self.widgetPlot3.canvas.axis3, self.widgetPlot3.canvas.axis4, self.widgetPlot3.canvas.axis5, self.widgetPlot3.canvas.axis6, self.widgetPlot3.canvas.axis7, self.widgetPlot3.canvas.axis8, self.widgetPlot3.canvas.axis9, self.widgetPlot3.canvas.axis10)
            self.widgetPlot3.canvas.figure.tight_layout()
            self.widgetPlot2.canvas.figure.tight_layout()
            self.widgetPlot1.canvas.figure.tight_layout()
            self.widgetPlot3.canvas.draw()
            self.widgetPlot2.canvas.draw()
            self.widgetPlot1.canvas.draw()

        #Decision Tree Regression
        elif strCB == "Decision Tree Regression":
            self.widgetPlot1.canvas.figure.clf()
            self.widgetPlot1.canvas.axis1 = self.widgetPlot1.canvas.figure.add_subplot(111, facecolor='#a9f971')

            self.widgetPlot2.canvas.figure.clf()
            self.widgetPlot2.canvas.axis2 = self.widgetPlot2.canvas.figure.add_subplot(111, facecolor='#a9f971')

            self.widgetPlot3.canvas.figure.clf()
            self.widgetPlot3.canvas.axis3 = self.widgetPlot3.canvas.figure.add_subplot(421, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis4 = self.widgetPlot3.canvas.figure.add_subplot(422, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis5 = self.widgetPlot3.canvas.figure.add_subplot(423, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis6 = self.widgetPlot3.canvas.figure.add_subplot(424, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis7 = self.widgetPlot3.canvas.figure.add_subplot(425, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis8 = self.widgetPlot3.canvas.figure.add_subplot(426, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis9 = self.widgetPlot3.canvas.figure.add_subplot(427, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis10 = self.widgetPlot3.canvas.figure.add_subplot(428, facecolor='#a9f971')

            dt_reg = DecisionTreeRegressor(random_state=100)
            self.perform_regression(dt_reg, self.X_final, self.y_final, self.X_train, self.y_train, self.X_test, self.y_test, self.X_val, self.y_val, "DT Regression", "Adj Close", self.widgetPlot1.canvas.axis1, self.widgetPlot2.canvas.axis2, self.widgetPlot3.canvas.axis3, self.widgetPlot3.canvas.axis4, self.widgetPlot3.canvas.axis5, self.widgetPlot3.canvas.axis6, self.widgetPlot3.canvas.axis7, self.widgetPlot3.canvas.axis8, self.widgetPlot3.canvas.axis9, self.widgetPlot3.canvas.axis10)
            self.widgetPlot3.canvas.figure.tight_layout()
            self.widgetPlot2.canvas.figure.tight_layout()
            self.widgetPlot1.canvas.figure.tight_layout()
            self.widgetPlot3.canvas.draw()
            self.widgetPlot2.canvas.draw()
            self.widgetPlot1.canvas.draw()
            joblib.dump(dt_reg, 'DTmodel.joblib')

        #KNN Regression
        elif strCB == "KNN Regression":
            self.widgetPlot1.canvas.figure.clf()
            self.widgetPlot1.canvas.axis1 = self.widgetPlot1.canvas.figure.add_subplot(111, facecolor='#a9f971')

            self.widgetPlot2.canvas.figure.clf()
            self.widgetPlot2.canvas.axis2 = self.widgetPlot2.canvas.figure.add_subplot(111, facecolor='#a9f971')

            self.widgetPlot3.canvas.figure.clf()
            self.widgetPlot3.canvas.axis3 = self.widgetPlot3.canvas.figure.add_subplot(421, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis4 = self.widgetPlot3.canvas.figure.add_subplot(422, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis5 = self.widgetPlot3.canvas.figure.add_subplot(423, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis6 = self.widgetPlot3.canvas.figure.add_subplot(424, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis7 = self.widgetPlot3.canvas.figure.add_subplot(425, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis8 = self.widgetPlot3.canvas.figure.add_subplot(426, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis9 = self.widgetPlot3.canvas.figure.add_subplot(427, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis10 = self.widgetPlot3.canvas.figure.add_subplot(428, facecolor='#a9f971')

            knn_reg = KNeighborsRegressor(n_neighbors=7)
            self.perform_regression(knn_reg, self.X_final, self.y_final, self.X_train, self.y_train, self.X_test, self.y_test, self.X_val, self.y_val, "KNN Regression", "Adj Close", self.widgetPlot1.canvas.axis1, self.widgetPlot2.canvas.axis2, self.widgetPlot3.canvas.axis3, self.widgetPlot3.canvas.axis4, self.widgetPlot3.canvas.axis5, self.widgetPlot3.canvas.axis6, self.widgetPlot3.canvas.axis7, self.widgetPlot3.canvas.axis8, self.widgetPlot3.canvas.axis9, self.widgetPlot3.canvas.axis10)
            self.widgetPlot3.canvas.figure.tight_layout()
            self.widgetPlot2.canvas.figure.tight_layout()
            self.widgetPlot1.canvas.figure.tight_layout()
            self.widgetPlot3.canvas.draw()
            self.widgetPlot2.canvas.draw()
            self.widgetPlot1.canvas.draw()

        #Adaboost Regression
        elif strCB == "Adaboost Regression":
            self.widgetPlot1.canvas.figure.clf()
            self.widgetPlot1.canvas.axis1 = self.widgetPlot1.canvas.figure.add_subplot(111, facecolor='#a9f971')

            self.widgetPlot2.canvas.figure.clf()
            self.widgetPlot2.canvas.axis2 = self.widgetPlot2.canvas.figure.add_subplot(111, facecolor='#a9f971')

            self.widgetPlot3.canvas.figure.clf()
            self.widgetPlot3.canvas.axis3 = self.widgetPlot3.canvas.figure.add_subplot(421, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis4 = self.widgetPlot3.canvas.figure.add_subplot(422, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis5 = self.widgetPlot3.canvas.figure.add_subplot(423, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis6 = self.widgetPlot3.canvas.figure.add_subplot(424, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis7 = self.widgetPlot3.canvas.figure.add_subplot(425, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis8 = self.widgetPlot3.canvas.figure.add_subplot(426, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis9 = self.widgetPlot3.canvas.figure.add_subplot(427, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis10 = self.widgetPlot3.canvas.figure.add_subplot(428, facecolor='#a9f971')

            ada_reg = AdaBoostRegressor(random_state=100, n_estimators=200)
            self.perform_regression(ada_reg, self.X_final, self.y_final, self.X_train, self.y_train, self.X_test, self.y_test, self.X_val, self.y_val, "Adaboost Regression", "Adj Close", self.widgetPlot1.canvas.axis1, self.widgetPlot2.canvas.axis2, self.widgetPlot3.canvas.axis3, self.widgetPlot3.canvas.axis4, self.widgetPlot3.canvas.axis5, self.widgetPlot3.canvas.axis6, self.widgetPlot3.canvas.axis7, self.widgetPlot3.canvas.axis8, self.widgetPlot3.canvas.axis9, self.widgetPlot3.canvas.axis10)
            self.widgetPlot3.canvas.figure.tight_layout()
            self.widgetPlot2.canvas.figure.tight_layout()
            self.widgetPlot1.canvas.figure.tight_layout()
            self.widgetPlot3.canvas.draw()
            self.widgetPlot2.canvas.draw()
            self.widgetPlot1.canvas.draw()


        #Gradient Boosting Regression
        elif strCB == "Gradient Boosting Regression":
            self.widgetPlot1.canvas.figure.clf()
            self.widgetPlot1.canvas.axis1 = self.widgetPlot1.canvas.figure.add_subplot(111, facecolor='#a9f971')

            self.widgetPlot2.canvas.figure.clf()
            self.widgetPlot2.canvas.axis2 = self.widgetPlot2.canvas.figure.add_subplot(111, facecolor='#a9f971')

            self.widgetPlot3.canvas.figure.clf()
            self.widgetPlot3.canvas.axis3 = self.widgetPlot3.canvas.figure.add_subplot(421, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis4 = self.widgetPlot3.canvas.figure.add_subplot(422, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis5 = self.widgetPlot3.canvas.figure.add_subplot(423, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis6 = self.widgetPlot3.canvas.figure.add_subplot(424, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis7 = self.widgetPlot3.canvas.figure.add_subplot(425, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis8 = self.widgetPlot3.canvas.figure.add_subplot(426, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis9 = self.widgetPlot3.canvas.figure.add_subplot(427, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis10 = self.widgetPlot3.canvas.figure.add_subplot(428, facecolor='#a9f971')

            gb_reg = GradientBoostingRegressor(random_state=100)
            self.perform_regression(gb_reg, self.X_final, self.y_final, self.X_train, self.y_train, self.X_test, self.y_test, self.X_val, self.y_val, "GB Regression", "Adj Close", self.widgetPlot1.canvas.axis1, self.widgetPlot2.canvas.axis2, self.widgetPlot3.canvas.axis3, self.widgetPlot3.canvas.axis4, self.widgetPlot3.canvas.axis5, self.widgetPlot3.canvas.axis6, self.widgetPlot3.canvas.axis7, self.widgetPlot3.canvas.axis8, self.widgetPlot3.canvas.axis9, self.widgetPlot3.canvas.axis10)
            self.widgetPlot3.canvas.figure.tight_layout()
            self.widgetPlot2.canvas.figure.tight_layout()
            self.widgetPlot1.canvas.figure.tight_layout()
            self.widgetPlot3.canvas.draw()
            self.widgetPlot2.canvas.draw()
            self.widgetPlot1.canvas.draw()


        #Extreme Gradient Boosting Regression
        elif strCB == "XGB Regression":
            self.widgetPlot1.canvas.figure.clf()
            self.widgetPlot1.canvas.axis1 = self.widgetPlot1.canvas.figure.add_subplot(111, facecolor='#a9f971')

            self.widgetPlot2.canvas.figure.clf()
            self.widgetPlot2.canvas.axis2 = self.widgetPlot2.canvas.figure.add_subplot(111, facecolor='#a9f971')

            self.widgetPlot3.canvas.figure.clf()
            self.widgetPlot3.canvas.axis3 = self.widgetPlot3.canvas.figure.add_subplot(421, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis4 = self.widgetPlot3.canvas.figure.add_subplot(422, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis5 = self.widgetPlot3.canvas.figure.add_subplot(423, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis6 = self.widgetPlot3.canvas.figure.add_subplot(424, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis7 = self.widgetPlot3.canvas.figure.add_subplot(425, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis8 = self.widgetPlot3.canvas.figure.add_subplot(426, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis9 = self.widgetPlot3.canvas.figure.add_subplot(427, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis10 = self.widgetPlot3.canvas.figure.add_subplot(428, facecolor='#a9f971')

            xgb_reg = XGBRFRegressor(random_state=100)
            self.perform_regression(xgb_reg, self.X_final, self.y_final, self.X_train, self.y_train, self.X_test, self.y_test, self.X_val, self.y_val, "XGB Regression", "Adj Close", self.widgetPlot1.canvas.axis1, self.widgetPlot2.canvas.axis2, self.widgetPlot3.canvas.axis3, self.widgetPlot3.canvas.axis4, self.widgetPlot3.canvas.axis5, self.widgetPlot3.canvas.axis6, self.widgetPlot3.canvas.axis7, self.widgetPlot3.canvas.axis8, self.widgetPlot3.canvas.axis9, self.widgetPlot3.canvas.axis10)
            self.widgetPlot3.canvas.figure.tight_layout()
            self.widgetPlot2.canvas.figure.tight_layout()
            self.widgetPlot1.canvas.figure.tight_layout()
            self.widgetPlot3.canvas.draw()
            self.widgetPlot2.canvas.draw()
            self.widgetPlot1.canvas.draw()


        #Light Gradient Boosting Regression
        elif strCB == "LGBM Regression":
            self.widgetPlot1.canvas.figure.clf()
            self.widgetPlot1.canvas.axis1 = self.widgetPlot1.canvas.figure.add_subplot(111, facecolor='#a9f971')

            self.widgetPlot2.canvas.figure.clf()
            self.widgetPlot2.canvas.axis2 = self.widgetPlot2.canvas.figure.add_subplot(111, facecolor='#a9f971')

            self.widgetPlot3.canvas.figure.clf()
            self.widgetPlot3.canvas.axis3 = self.widgetPlot3.canvas.figure.add_subplot(421, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis4 = self.widgetPlot3.canvas.figure.add_subplot(422, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis5 = self.widgetPlot3.canvas.figure.add_subplot(423, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis6 = self.widgetPlot3.canvas.figure.add_subplot(424, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis7 = self.widgetPlot3.canvas.figure.add_subplot(425, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis8 = self.widgetPlot3.canvas.figure.add_subplot(426, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis9 = self.widgetPlot3.canvas.figure.add_subplot(427, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis10 = self.widgetPlot3.canvas.figure.add_subplot(428, facecolor='#a9f971')

            lgbm_reg = LGBMRegressor(random_state=100)
            self.perform_regression(lgbm_reg, self.X_final, self.y_final, self.X_train, self.y_train, self.X_test, self.y_test, self.X_val, self.y_val, "LGBM Regression", "Adj Close", self.widgetPlot1.canvas.axis1, self.widgetPlot2.canvas.axis2, self.widgetPlot3.canvas.axis3, self.widgetPlot3.canvas.axis4, self.widgetPlot3.canvas.axis5, self.widgetPlot3.canvas.axis6, self.widgetPlot3.canvas.axis7, self.widgetPlot3.canvas.axis8, self.widgetPlot3.canvas.axis9, self.widgetPlot3.canvas.axis10)
            self.widgetPlot3.canvas.figure.tight_layout()
            self.widgetPlot2.canvas.figure.tight_layout()
            self.widgetPlot1.canvas.figure.tight_layout()
            self.widgetPlot3.canvas.draw()
            self.widgetPlot2.canvas.draw()
            self.widgetPlot1.canvas.draw()


        #Catboost Regression
        elif strCB == "Catboost Regression":
            self.widgetPlot1.canvas.figure.clf()
            self.widgetPlot1.canvas.axis1 = self.widgetPlot1.canvas.figure.add_subplot(111, facecolor='#a9f971')

            self.widgetPlot2.canvas.figure.clf()
            self.widgetPlot2.canvas.axis2 = self.widgetPlot2.canvas.figure.add_subplot(111, facecolor='#a9f971')

            self.widgetPlot3.canvas.figure.clf()
            self.widgetPlot3.canvas.axis3 = self.widgetPlot3.canvas.figure.add_subplot(421, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis4 = self.widgetPlot3.canvas.figure.add_subplot(422, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis5 = self.widgetPlot3.canvas.figure.add_subplot(423, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis6 = self.widgetPlot3.canvas.figure.add_subplot(424, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis7 = self.widgetPlot3.canvas.figure.add_subplot(425, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis8 = self.widgetPlot3.canvas.figure.add_subplot(426, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis9 = self.widgetPlot3.canvas.figure.add_subplot(427, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis10 = self.widgetPlot3.canvas.figure.add_subplot(428, facecolor='#a9f971')

            cb_reg = CatBoostRegressor(random_state=100)
            self.perform_regression(cb_reg, self.X_final, self.y_final, self.X_train, self.y_train, self.X_test, self.y_test, self.X_val, self.y_val, "Catboost Regression", "Adj Close", self.widgetPlot1.canvas.axis1, self.widgetPlot2.canvas.axis2, self.widgetPlot3.canvas.axis3, self.widgetPlot3.canvas.axis4, self.widgetPlot3.canvas.axis5, self.widgetPlot3.canvas.axis6, self.widgetPlot3.canvas.axis7, self.widgetPlot3.canvas.axis8, self.widgetPlot3.canvas.axis9, self.widgetPlot3.canvas.axis10)
            self.widgetPlot3.canvas.figure.tight_layout()
            self.widgetPlot2.canvas.figure.tight_layout()
            self.widgetPlot1.canvas.figure.tight_layout()
            self.widgetPlot3.canvas.draw()
            self.widgetPlot2.canvas.draw()
            self.widgetPlot1.canvas.draw()


        #Support Vector Regression
        elif strCB == "SVR Regression":
            self.widgetPlot1.canvas.figure.clf()
            self.widgetPlot1.canvas.axis1 = self.widgetPlot1.canvas.figure.add_subplot(111, facecolor='#a9f971')

            self.widgetPlot2.canvas.figure.clf()
            self.widgetPlot2.canvas.axis2 = self.widgetPlot2.canvas.figure.add_subplot(111, facecolor='#a9f971')

            self.widgetPlot3.canvas.figure.clf()
            self.widgetPlot3.canvas.axis3 = self.widgetPlot3.canvas.figure.add_subplot(421, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis4 = self.widgetPlot3.canvas.figure.add_subplot(422, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis5 = self.widgetPlot3.canvas.figure.add_subplot(423, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis6 = self.widgetPlot3.canvas.figure.add_subplot(424, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis7 = self.widgetPlot3.canvas.figure.add_subplot(425, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis8 = self.widgetPlot3.canvas.figure.add_subplot(426, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis9 = self.widgetPlot3.canvas.figure.add_subplot(427, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis10 = self.widgetPlot3.canvas.figure.add_subplot(428, facecolor='#a9f971')

            svm_reg = SVR()
            self.perform_regression(svm_reg, self.X_final, self.y_final, self.X_train, self.y_train, self.X_test, self.y_test, self.X_val, self.y_val, "SVR Regression", "Adj Close", self.widgetPlot1.canvas.axis1, self.widgetPlot2.canvas.axis2, self.widgetPlot3.canvas.axis3, self.widgetPlot3.canvas.axis4, self.widgetPlot3.canvas.axis5, self.widgetPlot3.canvas.axis6, self.widgetPlot3.canvas.axis7, self.widgetPlot3.canvas.axis8, self.widgetPlot3.canvas.axis9, self.widgetPlot3.canvas.axis10)
            self.widgetPlot3.canvas.figure.tight_layout()
            self.widgetPlot2.canvas.figure.tight_layout()
            self.widgetPlot1.canvas.figure.tight_layout()
            self.widgetPlot3.canvas.draw()
            self.widgetPlot2.canvas.draw()
            self.widgetPlot1.canvas.draw()


        #Multi-Layer Perceptron Regression
        elif strCB == "MLP Regression":
            self.widgetPlot1.canvas.figure.clf()
            self.widgetPlot1.canvas.axis1 = self.widgetPlot1.canvas.figure.add_subplot(111, facecolor='#a9f971')

            self.widgetPlot2.canvas.figure.clf()
            self.widgetPlot2.canvas.axis2 = self.widgetPlot2.canvas.figure.add_subplot(111, facecolor='#a9f971')

            self.widgetPlot3.canvas.figure.clf()
            self.widgetPlot3.canvas.axis3 = self.widgetPlot3.canvas.figure.add_subplot(421, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis4 = self.widgetPlot3.canvas.figure.add_subplot(422, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis5 = self.widgetPlot3.canvas.figure.add_subplot(423, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis6 = self.widgetPlot3.canvas.figure.add_subplot(424, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis7 = self.widgetPlot3.canvas.figure.add_subplot(425, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis8 = self.widgetPlot3.canvas.figure.add_subplot(426, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis9 = self.widgetPlot3.canvas.figure.add_subplot(427, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis10 = self.widgetPlot3.canvas.figure.add_subplot(428, facecolor='#a9f971')

            mlp_reg = MLPRegressor(random_state=100, max_iter=1000)
            self.perform_regression(mlp_reg, self.X_final, self.y_final, self.X_train, self.y_train, self.X_test, self.y_test, self.X_val, self.y_val, "MLP Regression", "Adj Close", self.widgetPlot1.canvas.axis1, self.widgetPlot2.canvas.axis2, self.widgetPlot3.canvas.axis3, self.widgetPlot3.canvas.axis4, self.widgetPlot3.canvas.axis5, self.widgetPlot3.canvas.axis6, self.widgetPlot3.canvas.axis7, self.widgetPlot3.canvas.axis8, self.widgetPlot3.canvas.axis9, self.widgetPlot3.canvas.axis10)
            self.widgetPlot3.canvas.figure.tight_layout()
            self.widgetPlot2.canvas.figure.tight_layout()
            self.widgetPlot1.canvas.figure.tight_layout()
            self.widgetPlot3.canvas.draw()
            self.widgetPlot2.canvas.draw()
            self.widgetPlot1.canvas.draw()


        #Lasso Regression
        elif strCB == "Lasso Regression":
            self.widgetPlot1.canvas.figure.clf()
            self.widgetPlot1.canvas.axis1 = self.widgetPlot1.canvas.figure.add_subplot(111, facecolor='#a9f971')

            self.widgetPlot2.canvas.figure.clf()
            self.widgetPlot2.canvas.axis2 = self.widgetPlot2.canvas.figure.add_subplot(111, facecolor='#a9f971')

            self.widgetPlot3.canvas.figure.clf()
            self.widgetPlot3.canvas.axis3 = self.widgetPlot3.canvas.figure.add_subplot(421, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis4 = self.widgetPlot3.canvas.figure.add_subplot(422, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis5 = self.widgetPlot3.canvas.figure.add_subplot(423, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis6 = self.widgetPlot3.canvas.figure.add_subplot(424, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis7 = self.widgetPlot3.canvas.figure.add_subplot(425, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis8 = self.widgetPlot3.canvas.figure.add_subplot(426, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis9 = self.widgetPlot3.canvas.figure.add_subplot(427, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis10 = self.widgetPlot3.canvas.figure.add_subplot(428, facecolor='#a9f971')

            lasso_reg = LassoCV(n_alphas=1000, max_iter=3000, random_state=0)
            self.perform_regression(lasso_reg, self.X_final, self.y_final, self.X_train, self.y_train, self.X_test, self.y_test, self.X_val, self.y_val, "Lasso Regression", "Adj Close", self.widgetPlot1.canvas.axis1, self.widgetPlot2.canvas.axis2, self.widgetPlot3.canvas.axis3, self.widgetPlot3.canvas.axis4, self.widgetPlot3.canvas.axis5, self.widgetPlot3.canvas.axis6, self.widgetPlot3.canvas.axis7, self.widgetPlot3.canvas.axis8, self.widgetPlot3.canvas.axis9, self.widgetPlot3.canvas.axis10)
            self.widgetPlot3.canvas.figure.tight_layout()
            self.widgetPlot2.canvas.figure.tight_layout()
            self.widgetPlot1.canvas.figure.tight_layout()
            self.widgetPlot3.canvas.draw()
            self.widgetPlot2.canvas.draw()
            self.widgetPlot1.canvas.draw()


        #Ridge Regression
        elif strCB == "Ridge Regression":
            self.widgetPlot1.canvas.figure.clf()
            self.widgetPlot1.canvas.axis1 = self.widgetPlot1.canvas.figure.add_subplot(111, facecolor='#a9f971')

            self.widgetPlot2.canvas.figure.clf()
            self.widgetPlot2.canvas.axis2 = self.widgetPlot2.canvas.figure.add_subplot(111, facecolor='#a9f971')

            self.widgetPlot3.canvas.figure.clf()
            self.widgetPlot3.canvas.axis3 = self.widgetPlot3.canvas.figure.add_subplot(421, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis4 = self.widgetPlot3.canvas.figure.add_subplot(422, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis5 = self.widgetPlot3.canvas.figure.add_subplot(423, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis6 = self.widgetPlot3.canvas.figure.add_subplot(424, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis7 = self.widgetPlot3.canvas.figure.add_subplot(425, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis8 = self.widgetPlot3.canvas.figure.add_subplot(426, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis9 = self.widgetPlot3.canvas.figure.add_subplot(427, facecolor='#a9f971')
            self.widgetPlot3.canvas.axis10 = self.widgetPlot3.canvas.figure.add_subplot(428, facecolor='#a9f971')

            ridge_reg = RidgeCV(gcv_mode='auto')
            self.perform_regression(ridge_reg, self.X_final, self.y_final, self.X_train, self.y_train, self.X_test, self.y_test, self.X_val, self.y_val, "Ridge Regression", "Adj Close", self.widgetPlot1.canvas.axis1, self.widgetPlot2.canvas.axis2, self.widgetPlot3.canvas.axis3, self.widgetPlot3.canvas.axis4, self.widgetPlot3.canvas.axis5, self.widgetPlot3.canvas.axis6, self.widgetPlot3.canvas.axis7, self.widgetPlot3.canvas.axis8, self.widgetPlot3.canvas.axis9, self.widgetPlot3.canvas.axis10)
            self.widgetPlot3.canvas.figure.tight_layout()
            self.widgetPlot2.canvas.figure.tight_layout()
            self.widgetPlot1.canvas.figure.tight_layout()
            self.widgetPlot3.canvas.draw()
            self.widgetPlot2.canvas.draw()
            self.widgetPlot1.canvas.draw()

    def extract_data(self, X):
        #Extracts output and input variables
        y=X["daily_returns"]
        y=np.array([1 if i>0 else 0 for i in y])

        #Drops irrelevant column
        X = X.drop(["daily_returns", "Day", "Week", "Month", "Year", "Quarter"], axis=1)
        return X, y
    

    def split_data(self):
        #Extracts input and output variables
        X, y = self.extract_data(self.df)

        #Resample data using SMOTE
        sm=SMOTE(random_state=42)
        X, y = sm.fit_resample(X, y.ravel())

        #Splits the data into training and testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2021, shuffle=True, stratify=y)

        #Normalises data with robust scaler 
        rob_scaler = RobustScaler()
        X_train_rob = X_train.copy()
        X_test_rob = X_test.copy()
        self.y_train_rob = y_train.copy()
        self.y_test_rob = y_test.copy()
        self.X_train_rob = rob_scaler.fit_transform(X_train_rob)
        self.X_test_rob = rob_scaler.transform(X_test_rob)

        #Saves into pkl files
        joblib.dump(self.X_train_rob, "X_train_rob.pkl")
        joblib.dump(self.X_test_rob, "X_test_rob.pkl")
        joblib.dump(self.y_train_rob, "y_train_rob.pkl")
        joblib.dump(self.y_test_rob, "y_test_rob.pkl")

        #Normalises data with MinMax Scaler
        X_train_norm = X_train.copy()
        X_test_norm = X_test.copy()
        self.y_train_norm = y_train.copy()
        self.y_test_norm = y_test.copy()

        norm = MinMaxScaler()
        self.X_train_norm = norm.fit_transform(X_train_norm)
        self.X_test_norm = norm.transform(X_test_norm)

        #Saves into pkl files
        joblib.dump(self.X_train_norm, "X_train_norm.pkl")
        joblib.dump(self.X_test_norm, "X_test_norm.pkl")
        joblib.dump(self.y_train_norm, "y_train_norm.pkl")
        joblib.dump(self.y_test_norm, "y_test_norm.pkl")

        #Normalises data with Standard Scaler 
        X_train_stand = X_train.copy()
        X_test_stand = X_test.copy()
        self.y_train_stand = y_train.copy()
        self.y_test_stand = y_test.copy()
        scaler = StandardScaler()
        self.X_train_stand = scaler.fit_transform(X_train_stand)
        self.X_test_stand = scaler.transform(X_test_stand)

        #Saves into pkl files
        joblib.dump(self.X_train_stand, "X_train_stand.pkl")
        joblib.dump(self.X_test_stand, "X_test_stand.pkl")
        joblib.dump(self.y_train_stand, "y_train_stand.pkl")
        joblib.dump(self.y_test_stand, "y_test_stand.pkl")


    def split_data_ML(self):
        '''Loads or creates robust scaled, minmax scaled and standard scaled train and test sets'''
        if os.path.isfile('X_train_rob.pkl'):
            self.X_train_rob, self.X_test_rob, self.y_train_rob, self.y_test_rob = self.load_rob_files()
            self.X_train_norm, self.X_test_norm, self.y_train_norm, self.y_test_norm = self.load_norm_files()
            self.X_train_stand, self.X_test_stand, self.y_train_stand, self.y_test_stand = self.load_stand_files()
        else:
            self.split_data()

        #Prints each shape
        print('X train ROB: ', self.X_train_rob.shape)
        print('X test ROB: ', self.X_test_rob.shape)
        print('Y train ROB: ', self.y_train_rob.shape)
        print('Y test ROB: ', self.y_test_rob.shape)

        #Prints each shape
        print('X train NORM: ', self.X_train_norm.shape)
        print('X test NORM: ', self.X_test_norm.shape)
        print('Y train NORM: ', self.y_train_norm.shape)
        print('Y test NORM: ', self.y_test_norm.shape)

        #Prints each shape
        print('X train STAND: ', self.X_train_stand.shape)
        print('X test STAND: ', self.X_test_stand.shape)
        print('Y train STAND: ', self.y_train_stand.shape)
        print('Y test STAND: ', self.y_test_stand.shape)


    def load_rob_files(self):
        X_train_rob = joblib.load('X_train_rob.pkl')
        X_test_rob = joblib.load('X_test_rob.pkl')
        y_train_rob = joblib.load('y_train_rob.pkl')
        y_test_rob = joblib.load('y_test_rob.pkl')
        return X_train_rob, X_test_rob, y_train_rob, y_test_rob
    

    def load_norm_files(self):
        X_train_norm = joblib.load('X_train_norm.pkl')
        X_test_norm = joblib.load('X_test_norm.pkl')
        y_train_norm = joblib.load('y_train_norm.pkl')
        y_test_norm = joblib.load('y_test_norm.pkl')
        return X_train_norm, X_test_norm, y_train_norm, y_test_norm
    

    def load_stand_files(self):
        X_train_stand = joblib.load('X_train_stand.pkl')
        X_test_stand = joblib.load('X_test_stand.pkl')
        y_train_stand = joblib.load('y_train_stand.pkl')
        y_test_stand = joblib.load('y_test_stand.pkl')
        return X_train_stand, X_test_stand, y_train_stand, y_test_stand


    def train_model_ML(self):
        self.split_data_ML()

        #Turns on two widgets
        self.cbData.setEnabled(True)
        self.cbClassifier.setEnabled(True)

        #Turns off pbTrainML
        self.pbTrainML.setEnabled(False)

        #Turns on three radio buttons
        self.rbRobust.setEnabled(True)
        self.rbRobust.setChecked(True)
        self.rbMinMax.setEnabled(True)
        self.rbStandard.setEnabled(True)


    def plot_real_pred_val(self, Y_pred, Y_test, widget, title):
        '''Calculate Metrics'''
        acc=accuracy_score(Y_test, Y_pred)

        #Output plot
        widget.canvas.figure.clf()
        widget.canvas.axis1 = widget.canvas.figure.add_subplot(111, facecolor='steelblue')
        widget.canvas.axis1.scatter(range(len(Y_pred)), Y_pred, color='yellow', lw=5, label='Predictions')
        widget.canvas.axis1.scatter(range(len(Y_test)), Y_test, color='red', label='Actual')
        widget.canvas.axis1.set_title("Prediction Values vs Real Values of " + title, fontsize=10)
        widget.canvas.axis1.set_xlabel("Accuracy: " + str(round((acc*100), 3)) + "%")
        widget.canvas.axis1.legend()
        widget.canvas.axis1.grid(True, alpha=0.75, lw=1, ls='-.')
        widget.canvas.axis1.yaxis.set_ticklabels(["", "Negative Daily Returns", "", "", "", "", "Positive Daily Returns"]);
        widget.canvas.draw()


    def plot_cm(self, Y_pred, Y_test, widget, title):
        cm=confusion_matrix(Y_test, Y_pred)
        widget.canvas.figure.clf()
        widget.canvas.axis1 = widget.canvas.figure.add_subplot(111)
        class_label = ["Negative Daily Returns", "Positive Daily Returns"]
        df_cm = pd.DataFrame(cm, index=class_label, columns=class_label)
        sns.heatmap(df_cm, ax=widget.canvas.axis1, annot=True, cmap='plasma', linewidths=2, fmt='d')
        widget.canvas.axis1.set_title("Confusion Matrix of " + title, fontsize=10)
        widget.canvas.axis1.set_xlabel("Predicted")
        widget.canvas.axis1.set_ylabel("True")
        widget.canvas.axis1.xaxis.set_ticklabels(["Negative Daily Returns", "Positive Daily Returns"]);
        widget.canvas.axis1.yaxis.set_ticklabels(["Negative Daily Returns", "Positive Daily Returns"]);
        widget.canvas.draw()



    def plot_learning_curve(self, estimator, title, X, y, widget, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
        widget.canvas.axis1.set_title(title)
        if ylim is not None:
            widget.canvas.axis1.set_ylim(*ylim)
        widget.canvas.axis1.set_xlabel("Training examples")
        widget.canvas.axis1.set_ylabel("Score")

        train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, return_times=True)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        #Plot learning curve
        widget.canvas.axis1.grid()
        widget.canvas.axis1.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='r')
        widget.canvas.axis1.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
        widget.canvas.axis1.plot(train_sizes, train_scores_mean, 'o-', color='r', label="Training score")
        widget.canvas.axis1.plot(train_sizes, test_scores_mean, 'o-', color='g', label="Cross-validation score")
        widget.canvas.axis1.legend(loc="best")


    def plot_scalability_curve(self, estimator, title, X, y, widget, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
        widget.canvas.axis1.set_title(title, fontweight='bold', fontsize=15)
        if ylim is not None:
            widget.canvas.axis1.set_ylim(*ylim)
        widget.canvas.axis1.set_xlabel("Training examples")
        widget.canvas.axis1.set_ylabel("Score")

        train_sizes, train_scores, test_scores, fit_times, _ =  learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, return_times=True)
        fit_times_mean = np.mean(fit_times, axis=1)
        fit_times_std = np.std(fit_times, axis=1)

        #Plot n_samples vs fit_times
        widget.canvas.axis1.grid()
        widget.canvas.axis1.plot(train_sizes, fit_times_mean, 'o-')
        widget.canvas.axis1.fill_between(train_sizes, fit_times_mean - fit_times_std, fit_times_mean + fit_times_std, alpha=0.1)
        widget.canvas.axis1.set_xlabel("Training examples")
        widget.canvas.axis1.set_ylabel('fit_times')

    
    def plot_performance_curve(self, estimator, title, X, y, widget, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
        widget.canvas.axis1.set_title(title, fontweight='bold', fontsize=15)
        if ylim is not None:
            widget.canvas.axis1.set_ylim(*ylim)
        widget.canvas.axis1.set_xlabel("Training examples")
        widget.canvas.axis1.set_ylabel("Score")

        train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, return_times=True)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        fit_times_mean = np.mean(fit_times, axis=1)

        #Plot n_samples vs fit_times
        widget.canvas.axis1.grid()
        widget.canvas.axis1.plot(fit_times_mean, test_scores_mean, 'o-')
        widget.canvas.axis1.fill_between(fit_times_mean, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1)
        widget.canvas.axis1.set_xlabel('fit_times')
        widget.canvas.axis1.set_ylabel("Score")


    def plot_decision(self, df, cla, feat1, feat2, widget, title=""):
        X,y = self.extract_data(df)

        #Plots decision boundary of two features 
        X_feature = np.array(X.iloc[:, 13:14])
        X_train_feature, X_test_feature, y_train_feature, y_test_feature = train_test_split(X_feature, y, test_size=0.3, random_state=42)
        cla.fit(X_train_feature, y_train_feature)

        plot_decision_regions(X_test_feature, y_test_feature.ravel(), clf=cla, legend=2, ax=widget.canvas.axis1)
        widget.canvas.axis1.set_title(title, fontweight='bold', fontsize=15)
        widget.canvas.axis1.set_xlabel(feat1)
        widget.canvas.axis1.set_ylabel(feat2)
        widget.canvas.figure.tight_layout()
        widget.canvas.draw()


    #Training model and predicting 
    def train_model(self, model, X, y):
        model.fit(X, y)
        return model
    

    def predict_model(self, model, X, proba=False):
        if ~proba:
            y_pred = model.predict(X)
        else:
            y_pred_proba = model.predict_proba(X)
            y_pred = np.argmax(y_pred_proba, axis=1)
        
        return y_pred
    

    def run_model(self, name, scaling, model, X_train, X_test, y_train, y_test, train=True, proba=True):
        if train == True:
            model = self.train_model(model, X_train, y_train)
        y_pred = self.predict_model(model, X_test, proba)

        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        print('accuracy: ', accuracy)
        print('recall: ', recall)
        print('precision: ', precision)
        print('f1: ', f1)
        print(classification_report(y_test, y_pred))

        self.widgetPlot1.canvas.figure.clf()
        self.widgetPlot1.canvas.axis1 = self.widgetPlot1.canvas.figure.add_subplot(111, facecolor='#fbe7dd')
        self.plot_cm(y_pred, y_test, self.widgetPlot1, name + " -- " + scaling)
        self.widgetPlot1.canvas.figure.tight_layout()
        self.widgetPlot1.canvas.draw()

        self.widgetPlot2.canvas.figure.clf()
        self.widgetPlot2.canvas.axis1 = self.widgetPlot2.canvas.figure.add_subplot(111, facecolor='#fbe7dd')
        self.plot_real_pred_val(y_pred, y_test, self.widgetPlot2, name + " -- " + scaling)
        self.widgetPlot1.canvas.figure.tight_layout()
        self.widgetPlot1.canvas.draw()

        self.widgetPlot3.canvas.figure.clf()
        self.widgetPlot3.canvas.axis1 = self.widgetPlot3.canvas.figure.add_subplot(221, facecolor="#fbe7dd")
        self.plot_decision(self.df, model, 'Open_Close', 'High_Low', self.widgetPlot3, title="The decision boundaaries of " + name + " -- " + scaling)

        self.widgetPlot3.canvas.axis1 = self.widgetPlot3.canvas.figure.add_subplot(222, facecolor='#fbe7dd')
        self.plot_learning_curve(model, 'Learning Curve' + ' -- ' + scaling, X_train, y_train, self.widgetPlot3)
        self.widgetPlot3.canvas.figure.tight_layout()

        self.widgetPlot3.canvas.axis1 = self.widgetPlot3.canvas.figure.add_subplot(223, facecolor='#fbe7dd')
        self.plot_scalability_curve(model, 'Scalability of ' + name + ' -- ' + scaling, X_train, y_train, self.widgetPlot3)
        self.widgetPlot3.canvas.figure.tight_layout()

        self.widgetPlot3.canvas.axis1 = self.widgetPlot3.canvas.figure.add_subplot(224, facecolor='#fbe7dd')
        self.plot_performance_curve(model, 'Performance of ' + name + ' -- ' + scaling, X_train, y_train, self.widgetPlot3)
        self.widgetPlot3.canvas.figure.tight_layout()
        self.widgetPlot3.canvas.draw()


    #Logistic Regression Classifier
    def build_train_lr(self):
        if os.path.isfile('logregRob.pkl'):
            #Loads model
            self.logregRob = joblib.load('logregRob.pkl')
            self.logregNorm = joblib.load('logregNorm.pkl')
            self.logregStand = joblib.load('logregStand.pkl')

            if self.rbRobust.isChecked():
                self.run_model('Logistic Regression', 'Robust', self.logregRob, self.X_train_rob, self.X_test_rob, self.y_train_rob, self.y_test_rob)

            elif self.rbMinMax.isChecked():
                self.run_model('Logistic Regression', 'MinMax', self.logregNorm, self.X_train_norm, self.X_test_norm, self.y_train_norm, self.y_test_norm)

            elif self.rbStandard.isChecked():
                self.run_model('Logistic Regression', 'Standard', self.logregStand, self.X_train_stand, self.X_test_stand, self.y_train_stand, self.y_test_stand)

        else:
            #Builds and trains Logistic Regression
            self.logregRob = LogisticRegression(solver='lbfgs',max_iter=2000, random_state=2021)
            self.logregNorm = LogisticRegression(solver='lbfgs',max_iter=2000, random_state=2021)
            self.logregStand = LogisticRegression(solver='lbfgs',max_iter=2000, random_state=2021)

            if self.rbRobust.isChecked():
                self.run_model('Logistic Regression', 'Robust', self.logregRob, self.X_train_rob, self.X_test_rob, self.y_train_rob, self.y_test_rob)

            elif self.rbMinMax.isChecked():
                self.run_model('Logistic Regression', 'MinMax', self.logregNorm, self.X_train_norm, self.X_test_norm, self.y_train_norm, self.y_test_norm)

            elif self.rbStandard.isChecked():
                self.run_model('Logistic Regression', 'Standard', self.logregStand, self.X_train_stand, self.X_test_stand, self.y_train_stand, self.y_test_stand)

            
            #Saves model
            joblib.dump(self.logregRob, 'logregRob.pkl')
            joblib.dump(self.logregNorm, 'logregNorm.pkl')
            joblib.dump(self.logregStand, 'logregStand.pkl')


    def build_train_svm(self):
            if os.path.isfile('SVMRob.pkl'):
                #Loads model
                self.SVMRob = joblib.load('SVMRob.pkl')
                self.SVMNorm = joblib.load('SVMNorm.pkl')
                self.SVMStand = joblib.load('SVMStand.pkl')

                if self.rbRobust.isChecked():
                    self.run_model('SVM', 'Robust', self.SVMRob, self.X_train_rob, self.X_test_rob, self.y_train_rob, self.y_test_rob)

                elif self.rbMinMax.isChecked():
                    self.run_model('SVM', 'MinMax', self.SVMNorm, self.X_train_norm, self.X_test_norm, self.y_train_norm, self.y_test_norm)

                elif self.rbStandard.isChecked():
                    self.run_model('SVM', 'Standard', self.SVMStand, self.X_train_stand, self.X_test_stand, self.y_train_stand, self.y_test_stand)

            else:
            #Builds and trains SVM model
                self.SVMRob = SVC(random_state=2021, probability=True)
                self.SVMNorm = SVC(random_state=2021, probability=True)
                self.SVMStand = SVC(random_state=2021, probability=True)

                if self.rbRobust.isChecked():
                    self.run_model('SVM', 'Robust', self.SVMRob, self.X_train_rob, self.X_test_rob, self.y_train_rob, self.y_test_rob)

                elif self.rbMinMax.isChecked():
                    self.run_model('SVM', 'MinMax', self.SVMNorm, self.X_train_norm, self.X_test_norm, self.y_train_norm, self.y_test_norm)

                elif self.rbStandard.isChecked():
                    self.run_model('SVM', 'Standard', self.SVMStand, self.X_train_stand, self.X_test_stand, self.y_train_stand, self.y_test_stand)

            
                #Saves model
                joblib.dump(self.SVMRob, 'SVMRob.pkl')
                joblib.dump(self.SVMNorm, 'SVMNorm.pkl')
                joblib.dump(self.SVMStand, 'SVMStand.pkl')


    def build_train_dt(self):
        if os.path.isfile('DTRob.pkl'):
            #Loads model
            self.DTRob = joblib.load('DTRob.pkl')
            self.DTNorm = joblib.load('DTNorm.pkl')
            self.DTStand = joblib.load('DTStand.pkl')

            if self.rbRobust.isChecked():
                self.run_model('DT', 'Robust', self.DTRob, self.X_train_rob, self.X_test_rob, self.y_train_rob, self.y_test_rob)

            elif self.rbMinMax.isChecked():
                self.run_model('DT', 'MinMax', self.DTNorm, self.X_train_norm, self.X_test_norm, self.y_train_norm, self.y_test_norm)

            elif self.rbStandard.isChecked():
                self.run_model('DT', 'Standard', self.DTStand, self.X_train_stand, self.X_test_stand, self.y_train_stand, self.y_test_stand)

        else:
            #Builds and trains Decision Tree
            dt_cla = DecisionTreeClassifier()
            parameters = {
                'max_depth':np.arange(1, 20, 1), 'random_state':[2021]
            }
            self.DTRob = GridSearchCV(dt_cla, parameters)
            self.DTNorm = GridSearchCV(dt_cla, parameters)
            self.DTStand = GridSearchCV(dt_cla, parameters)

            if self.rbRobust.isChecked():
                self.run_model('DT', 'Robust', self.DTRob, self.X_train_rob, self.X_test_rob, self.y_train_rob, self.y_test_rob)

            elif self.rbMinMax.isChecked():
                self.run_model('DT', 'MinMax', self.DTNorm, self.X_train_norm, self.X_test_norm, self.y_train_norm, self.y_test_norm)

            elif self.rbStandard.isChecked():
                self.run_model('DT', 'Standard', self.DTStand, self.X_train_stand, self.X_test_stand, self.y_train_stand, self.y_test_stand)

        
            #Saves model
            joblib.dump(self.DTRob, 'DTRob.pkl')
            joblib.dump(self.DTNorm, 'DTNorm.pkl')
            joblib.dump(self.DTStand, 'DTStand.pkl')


    def build_train_knn(self):
        if os.path.isfile('KNNRob.pkl'):
            #Loads model
            self.KNNRob = joblib.load('KNNRob.pkl')
            self.KNNNorm = joblib.load('KNNNorm.pkl')
            self.KNNStand = joblib.load('KNNStand.pkl')

            if self.rbRobust.isChecked():
                self.run_model('KNN', 'Robust', self.KNNRob, self.X_train_rob, self.X_test_rob, self.y_train_rob, self.y_test_rob)

            elif self.rbMinMax.isChecked():
                self.run_model('KNN', 'MinMax', self.KNNNorm, self.X_train_norm, self.X_test_norm, self.y_train_norm, self.y_test_norm)

            elif self.rbStandard.isChecked():
                self.run_model('KNN', 'Standard', self.KNNStand, self.X_train_stand, self.X_test_stand, self.y_train_stand, self.y_test_stand)

        else:
            #Builds and trains K-Nearest Neighbor
            self.KNNRob = KNeighborsClassifier(n_neighbors=10)
            self.KNNNorm = KNeighborsClassifier(n_neighbors=10)
            self.KNNStand = KNeighborsClassifier(n_neighbors=10)

            if self.rbRobust.isChecked():
                self.run_model('KNN', 'Robust', self.KNNRob, self.X_train_rob, self.X_test_rob, self.y_train_rob, self.y_test_rob)

            elif self.rbMinMax.isChecked():
                self.run_model('KNN', 'MinMax', self.KNNNorm, self.X_train_norm, self.X_test_norm, self.y_train_norm, self.y_test_norm)

            elif self.rbStandard.isChecked():
                self.run_model('KNN', 'Standard', self.KNNStand, self.X_train_stand, self.X_test_stand, self.y_train_stand, self.y_test_stand)

    
            #Saves model
            joblib.dump(self.KNNRob, 'KNNRob.pkl')
            joblib.dump(self.KNNNorm, 'KNNNorm.pkl')
            joblib.dump(self.KNNStand, 'KNNStand.pkl')

    

    def build_train_rf(self):
        if os.path.isfile('RFRob.pkl'):
            #Loads model
            self.RFRob = joblib.load('RFRob.pkl')
            self.RFNorm = joblib.load('RFNorm.pkl')
            self.RFStand = joblib.load('RFStand.pkl')

            if self.rbRobust.isChecked():
                self.run_model('RF', 'Robust', self.RFRob, self.X_train_rob, self.X_test_rob, self.y_train_rob, self.y_test_rob)

            elif self.rbMinMax.isChecked():
                self.run_model('RF', 'MinMax', self.RFNorm, self.X_train_norm, self.X_test_norm, self.y_train_norm, self.y_test_norm)

            elif self.rbStandard.isChecked():
                self.run_model('RF', 'Standard', self.RFStand, self.X_train_stand, self.X_test_stand, self.y_train_stand, self.y_test_stand)

        else:
            #Builds and trains Random Forest
            self.RFRob = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=2021)
            self.RFNorm = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=2021)
            self.RFStand = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=2021)

            if self.rbRobust.isChecked():
                self.run_model('RF', 'Robust', self.RFRob, self.X_train_rob, self.X_test_rob, self.y_train_rob, self.y_test_rob)

            elif self.rbMinMax.isChecked():
                self.run_model('RF', 'MinMax', self.RFNorm, self.X_train_norm, self.X_test_norm, self.y_train_norm, self.y_test_norm)

            elif self.rbStandard.isChecked():
                self.run_model('RF', 'Standard', self.RFStand, self.X_train_stand, self.X_test_stand, self.y_train_stand, self.y_test_stand)

    
            #Saves model
            joblib.dump(self.RFRob, 'RFRob.pkl')
            joblib.dump(self.RFNorm, 'RFNorm.pkl')
            joblib.dump(self.RFStand, 'RFStand.pkl')


    def build_train_gb(self):
        if os.path.isfile('GBRob.pkl'):
            #Loads model
            self.GBRob = joblib.load('GBRob.pkl')
            self.GBNorm = joblib.load('GBNorm.pkl')
            self.GBStand = joblib.load('GBStand.pkl')

            if self.rbRobust.isChecked():
                self.run_model('GB', 'Robust', self.GBRob, self.X_train_rob, self.X_test_rob, self.y_train_rob, self.y_test_rob)

            elif self.rbMinMax.isChecked():
                self.run_model('GB', 'MinMax', self.GBNorm, self.X_train_norm, self.X_test_norm, self.y_train_norm, self.y_test_norm)

            elif self.rbStandard.isChecked():
                self.run_model('GB', 'Standard', self.GBStand, self.X_train_stand, self.X_test_stand, self.y_train_stand, self.y_test_stand)

        else:
            #Builds and trains Gradient Boosting
            self.GBRob = GradientBoostingClassifier(n_estimators=50, max_depth=10, subsample=0.8, max_features=0.2, random_state=2021)
            self.GBNorm = GradientBoostingClassifier(n_estimators=50, max_depth=10, subsample=0.8, max_features=0.2, random_state=2021)
            self.GBStand = GradientBoostingClassifier(n_estimators=50, max_depth=10, subsample=0.8, max_features=0.2, random_state=2021)

            if self.rbRobust.isChecked():
                self.run_model('GB', 'Robust', self.GBRob, self.X_train_rob, self.X_test_rob, self.y_train_rob, self.y_test_rob)

            elif self.rbMinMax.isChecked():
                self.run_model('GB', 'MinMax', self.GBNorm, self.X_train_norm, self.X_test_norm, self.y_train_norm, self.y_test_norm)

            elif self.rbStandard.isChecked():
                self.run_model('GB', 'Standard', self.GBStand, self.X_train_stand, self.X_test_stand, self.y_train_stand, self.y_test_stand)

    
            #Saves model
            joblib.dump(self.GBRob, 'GBRob.pkl')
            joblib.dump(self.GBNorm, 'GBNorm.pkl')
            joblib.dump(self.GBStand, 'GBStand.pkl')



    def build_train_nb(self):
        if os.path.isfile('NBRob.pkl'):
            #Loads model
            self.NBRob = joblib.load('NBRob.pkl')
            self.NBNorm = joblib.load('NBNorm.pkl')
            self.NBStand = joblib.load('NBStand.pkl')

            if self.rbRobust.isChecked():
                self.run_model('NB', 'Robust', self.NBRob, self.X_train_rob, self.X_test_rob, self.y_train_rob, self.y_test_rob)

            elif self.rbMinMax.isChecked():
                self.run_model('NB', 'MinMax', self.NBNorm, self.X_train_norm, self.X_test_norm, self.y_train_norm, self.y_test_norm)

            elif self.rbStandard.isChecked():
                self.run_model('NB', 'Standard', self.NBStand, self.X_train_stand, self.X_test_stand, self.y_train_stand, self.y_test_stand)

        else:
            #Builds and trains Naive Bayes
            self.NBRob = GaussianNB()
            self.NBNorm = GaussianNB()
            self.NBStand = GaussianNB()

            if self.rbRobust.isChecked():
                self.run_model('NB', 'Robust', self.NBRob, self.X_train_rob, self.X_test_rob, self.y_train_rob, self.y_test_rob)

            elif self.rbMinMax.isChecked():
                self.run_model('NB', 'MinMax', self.NBNorm, self.X_train_norm, self.X_test_norm, self.y_train_norm, self.y_test_norm)

            elif self.rbStandard.isChecked():
                self.run_model('NB', 'Standard', self.NBStand, self.X_train_stand, self.X_test_stand, self.y_train_stand, self.y_test_stand)

    
            #Saves model
            joblib.dump(self.NBRob, 'NBRob.pkl')
            joblib.dump(self.NBNorm, 'NBNorm.pkl')
            joblib.dump(self.NBStand, 'NBStand.pkl')


    def build_train_ada(self):
        if os.path.isfile('ADARob.pkl'):
            #Loads model
            self.ADARob = joblib.load('ADARob.pkl')
            self.ADANorm = joblib.load('ADANorm.pkl')
            self.ADAStand = joblib.load('ADAStand.pkl')

            if self.rbRobust.isChecked():
                self.run_model('ADA', 'Robust', self.ADARob, self.X_train_rob, self.X_test_rob, self.y_train_rob, self.y_test_rob)

            elif self.rbMinMax.isChecked():
                self.run_model('ADA', 'MinMax', self.ADANorm, self.X_train_norm, self.X_test_norm, self.y_train_norm, self.y_test_norm)

            elif self.rbStandard.isChecked():
                self.run_model('ADA', 'Standard', self.ADAStand, self.X_train_stand, self.X_test_stand, self.y_train_stand, self.y_test_stand)

        else:
            #Builds and trains Adaboost
            self.ADARob = AdaBoostClassifier(n_estimators=50, learning_rate=0.01)
            self.ADANorm = AdaBoostClassifier(n_estimators=50, learning_rate=0.01)
            self.ADAStand = AdaBoostClassifier(n_estimators=50, learning_rate=0.01)

            if self.rbRobust.isChecked():
                self.run_model('ADA', 'Robust', self.ADARob, self.X_train_rob, self.X_test_rob, self.y_train_rob, self.y_test_rob)

            elif self.rbMinMax.isChecked():
                self.run_model('ADA', 'MinMax', self.ADANorm, self.X_train_norm, self.X_test_norm, self.y_train_norm, self.y_test_norm)

            elif self.rbStandard.isChecked():
                self.run_model('ADA', 'Standard', self.ADAStand, self.X_train_stand, self.X_test_stand, self.y_train_stand, self.y_test_stand)

    
            #Saves model
            joblib.dump(self.ADARob, 'ADARob.pkl')
            joblib.dump(self.ADANorm, 'ADANorm.pkl')
            joblib.dump(self.ADAStand, 'ADAStand.pkl')


    def build_train_xgb(self):
        if os.path.isfile('XGBRob.pkl'):
            #Loads model
            self.XGBRob = joblib.load('XGBRob.pkl')
            self.XGBNorm = joblib.load('XGBNorm.pkl')
            self.XGBStand = joblib.load('XGBStand.pkl')

            if self.rbRobust.isChecked():
                self.run_model('XGB', 'Robust', self.XGBRob, self.X_train_rob, self.X_test_rob, self.y_train_rob, self.y_test_rob)

            elif self.rbMinMax.isChecked():
                self.run_model('XGB', 'MinMax', self.XGBNorm, self.X_train_norm, self.X_test_norm, self.y_train_norm, self.y_test_norm)

            elif self.rbStandard.isChecked():
                self.run_model('XGB', 'Standard', self.XGBStand, self.X_train_stand, self.X_test_stand, self.y_train_stand, self.y_test_stand)

        else:
            #Builds and trains XGB classifier
            self.XGBRob = XGBClassifier(n_estimators=50, max_depth=10, random_state=2021, use_label_encoder=False, eval_metric='mlogloss')
            self.XGBNorm = XGBClassifier(n_estimators=50, max_depth=10, random_state=2021, use_label_encoder=False, eval_metric='mlogloss')
            self.XGBStand = XGBClassifier(n_estimators=50, max_depth=10, random_state=2021, use_label_encoder=False, eval_metric='mlogloss')

            if self.rbRobust.isChecked():
                self.run_model('XGB', 'Robust', self.XGBRob, self.X_train_rob, self.X_test_rob, self.y_train_rob, self.y_test_rob)

            elif self.rbMinMax.isChecked():
                self.run_model('XGB', 'MinMax', self.XGBNorm, self.X_train_norm, self.X_test_norm, self.y_train_norm, self.y_test_norm)

            elif self.rbStandard.isChecked():
                self.run_model('XGB', 'Standard', self.XGBStand, self.X_train_stand, self.X_test_stand, self.y_train_stand, self.y_test_stand)

    
            #Saves model
            joblib.dump(self.XGBRob, 'XGBRob.pkl')
            joblib.dump(self.XGBNorm, 'XGBNorm.pkl')
            joblib.dump(self.XGBStand, 'XGBStand.pkl')


    def build_train_lgbm(self):
        if os.path.isfile('LGBMRob.pkl'):
            #Loads model
            self.LGBMRob = joblib.load('LGBMRob.pkl')
            self.LGBMNorm = joblib.load('LGBMNorm.pkl')
            self.LGBMStand = joblib.load('LGBMStand.pkl')

            if self.rbRobust.isChecked():
                self.run_model('LGBM', 'Robust', self.LGBMRob, self.X_train_rob, self.X_test_rob, self.y_train_rob, self.y_test_rob)

            elif self.rbMinMax.isChecked():
                self.run_model('LGBM', 'MinMax', self.LGBMNorm, self.X_train_norm, self.X_test_norm, self.y_train_norm, self.y_test_norm)

            elif self.rbStandard.isChecked():
                self.run_model('LGBM', 'Standard', self.LGBMStand, self.X_train_stand, self.X_test_stand, self.y_train_stand, self.y_test_stand)

        else:
            #Builds and trains LGBM classifier
            self.LGBMRob = LGBMClassifier(n_estimators=50, max_depth=10, random_state=2021, subsample=0.8)
            self.LGBMNorm = LGBMClassifier(n_estimators=50, max_depth=10, random_state=2021, subsample=0.8)
            self.LGBMStand = LGBMClassifier(n_estimators=50, max_depth=10, random_state=2021, subsample=0.8)

            if self.rbRobust.isChecked():
                self.run_model('LGBM', 'Robust', self.LGBMRob, self.X_train_rob, self.X_test_rob, self.y_train_rob, self.y_test_rob)

            elif self.rbMinMax.isChecked():
                self.run_model('LGBM', 'MinMax', self.LGBMNorm, self.X_train_norm, self.X_test_norm, self.y_train_norm, self.y_test_norm)

            elif self.rbStandard.isChecked():
                self.run_model('LGBM', 'Standard', self.LGBMStand, self.X_train_stand, self.X_test_stand, self.y_train_stand, self.y_test_stand)

    
            #Saves model
            joblib.dump(self.LGBMRob, 'LGBMRob.pkl')
            joblib.dump(self.LGBMNorm, 'LGBMNorm.pkl')
            joblib.dump(self.LGBMStand, 'LGBMStand.pkl')


    def build_train_extra(self):
        if os.path.isfile('ExtraRob.pkl'):
            #Loads model
            self.ExtraRob = joblib.load('ExtraRob.pkl')
            self.ExtraNorm = joblib.load('ExtraNorm.pkl')
            self.ExtraStand = joblib.load('ExtraStand.pkl')

            if self.rbRobust.isChecked():
                self.run_model('Extra', 'Robust', self.ExtraRob, self.X_train_rob, self.X_test_rob, self.y_train_rob, self.y_test_rob)

            elif self.rbMinMax.isChecked():
                self.run_model('Extra', 'MinMax', self.ExtraNorm, self.X_train_norm, self.X_test_norm, self.y_train_norm, self.y_test_norm)

            elif self.rbStandard.isChecked():
                self.run_model('Extra', 'Standard', self.ExtraStand, self.X_train_stand, self.X_test_stand, self.y_train_stand, self.y_test_stand)

        else:
            #Builds and trains Gaussian Mixture classifier
            self.ExtraRob = ExtraTreesClassifier(n_estimators=200, random_state=100)
            self.ExtraNorm = ExtraTreesClassifier(n_estimators=200, random_state=100)
            self.ExtraStand = ExtraTreesClassifier(n_estimators=200, random_state=100)

            if self.rbRobust.isChecked():
                self.run_model('Extra', 'Robust', self.ExtraRob, self.X_train_rob, self.X_test_rob, self.y_train_rob, self.y_test_rob)

            elif self.rbMinMax.isChecked():
                self.run_model('Extra', 'MinMax', self.ExtraNorm, self.X_train_norm, self.X_test_norm, self.y_train_norm, self.y_test_norm)

            elif self.rbStandard.isChecked():
                self.run_model('Extra', 'Standard', self.ExtraStand, self.X_train_stand, self.X_test_stand, self.y_train_stand, self.y_test_stand)

    
            #Saves model
            joblib.dump(self.ExtraRob, 'ExtraRob.pkl')
            joblib.dump(self.ExtraNorm, 'ExtraNorm.pkl')
            joblib.dump(self.ExtraStand, 'ExtraStand.pkl')





    
    def choose_ML_model(self):
        strCB = self.cbClassifier.currentText()

        if strCB == "Logistic Regression":
            self.build_train_lr()

        elif strCB == "Support Vector Machine":
            self.build_train_svm()

        elif strCB == "Decision Tree":
            self.build_train_dt()

        elif strCB == "K-Nearest Neighbor":
            self.build_train_knn()

        elif strCB == "Random Forest":
            self.build_train_rf()

        elif strCB == "Gradient Boosting":
            self.build_train_gb()

        elif strCB == "Naive Bayes":
            self.build_train_nb()

        elif strCB == "Adaboost":
            self.build_train_ada()

        elif strCB == "XGB Classifier":
            self.build_train_xgb()

        elif strCB == "LGBM Classifier":
            self.build_train_lgbm()

        elif strCB == "Extra Trees Classifier":
            self.build_train_extra()



if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    ex = DemoGUI_Yahoo()
    ex.show()
    sys.exit(app.exec_())


