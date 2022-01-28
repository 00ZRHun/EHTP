#---------------------------------#
# Import library
import pandas as pd
import numpy as np
import math   # math.isnan(x)
from scipy.stats import pearsonr   # cc, r

## improved heatmaps
from matplotlib import pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 7,7
import seaborn as sns
sns.set(color_codes=True, font_scale=1.2)

'''%matplotlib inline
%config InclineBackend.figure_format = 'retina'
%load_ext autoreload
%autoreload 2  # ???'''

# === 2 ===
# !pip install heatmapz

# import the 2 methods from heatmap library
from heatmap import heatmap, corrplot


#---------------------------------#
# Reusable function

## Get time serial
# def getTimeSeries(df = y, colName = 'Net International Investment Position (RM million)'):
def getTimeSeries(df, colName):
    # 1. var declaration
    l_niip = list(df[colName])
    pos_1st = 0
    ts = list()


    # 2. get available time series

    # 2.1 find 1st pos
    # if l_niip[0] == np.nan:
    if math.isnan(l_niip[0]):
        for i in range(len(l_niip)):
            if not math.isnan(l_niip[i]):
                pos_1st = i
                break

    # 2.2 append avaiable cell in ts
    for ii in range(pos_1st, len(l_niip)):
        if not math.isnan(l_niip[ii]):
            ts.append(l_niip[ii])
        else:
            break
    
    return ts
    
    
    ### DEBUG PURPOSE ###
    # l_niip   # OUT: [nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,-137625.0,-139786.0,-141993.0,-127221.0,-74929.0,-23378.0,-15011.0,104789.0,105794.0,12940.0,36553.0,-17762.0,-47069.0,-17475.0,109178.0,7077.0,122699.0,-262076.0,-142691.0]
    # type(l_niip[0])   # OUT: float
    # print(pos_1st)   # OUT: 31

    # type(ts)   # OUT: list
    # print(ts)   # OUT: [-137625.0, -139786.0, -141993.0, -127221.0, -74929.0, -23378.0, -15011.0, 104789.0, 105794.0, 12940.0, 36553.0, -17762.0, -47069.0, -17475.0, 109178.0, 7077.0, 122699.0, -262076.0, -142691.0]
    # print(len(ts))   # OUT: 19/

## Get excel cols name
'''def getColsName(df = y):
    """cols = list(df.columns)
    return cols"""
    return list(df.columns)'''

## Redundant x cols name
'''def redundantColsName(old_l = [1,2,3], siz = 3):
    new_l = []

    for i in old_l:
        for _ in range(siz):
            new_l.append(i) 
    
    return new_l
'''

## Merge for the funcs of getColsName && redundantColsName
### Get x & y cols name & time series
'''def getXnYColsNameNTimeSeries(x = x2, y = y2):
    xColsName, yColsName = list(x.columns), list(y.columns)
    xTimeSeries, yTimeSeries = [], []
    
    xRedundantColsName, yRedundantColsName, xRedundantTimeSeries, yRedundantTimeSeries = [], [], [], []
    
    # x, y time series 
    for xSingleColName in xColsName:
        """print(xSingleColName)
        print(getTimeSeries(x, xSingleColName))"""
        xTimeSeries.append(getTimeSeries(x, xSingleColName))
        
    for ySingleColName in yColsName:
        yTimeSeries.append(getTimeSeries(y, ySingleColName))

    
    
    # make X & Y ColsName && TimeSeries redundant
    
    # print("xColsName", xColsName, "\n\n\n", "xTimeSeries", xTimeSeries, "\n\n\n")
    
    """for xSingleColName, xSingleTimeSeries in (xColsName, xTimeSeries):
        print(xSingleColName, xSingleTimeSeries)
        for _ in range(len(yColsName)):
            xRedundantColsName.append(xSingleColName)
            xRedundantTimeSeries.append(x, xSingleTimeSeries)"""
    for xIdx in range(len(xColsName)):
        for _ in range(len(yColsName)):
            xRedundantColsName.append(xColsName[xIdx])
            xRedundantTimeSeries.append(xTimeSeries[xIdx])
        
    for _ in range(len(xColsName)):
        # yRedundantColsName.append(yColsName)
        yRedundantColsName += yColsName
        yRedundantTimeSeries += yTimeSeries
        

    # print(f"xRedundantColsName: {xRedundantColsName} \n\n\n yRedundantColsName: {yRedundantColsName} \n\n\n xRedundantTimeSeries: {xRedundantTimeSeries} \n\n\n yRedundantTimeSeries: {yRedundantTimeSeries}")
    return xRedundantColsName, yRedundantColsName, xRedundantTimeSeries, yRedundantTimeSeries
'''

## Get x & y data
# def getXnYData(x = x2, y = y2):
def getXnYData(x, y):
    xColsName, yColsName = list(x.columns), list(y.columns)
    xRedundantColsName, yRedundantColsName = [], []
  
    
    # make X & Y ColsName redundant
    
    # print("xColsName", xColsName, "\n\n\n", "xTimeSeries", xTimeSeries, "\n\n\n")
    
    """for xSingleColName, xSingleTimeSeries in (xColsName, xTimeSeries):
        print(xSingleColName, xSingleTimeSeries)
        for _ in range(len(yColsName)):
            xRedundantColsName.append(xSingleColName)
            xRedundantTimeSeries.append(x, xSingleTimeSeries)"""
    for xIdx in range(len(xColsName)):
        for _ in range(len(yColsName)):
            xRedundantColsName.append(xColsName[xIdx])

        
    for _ in range(len(xColsName)):
        # yRedundantColsName.append(yColsName)
        yRedundantColsName += yColsName

        

    # print(f"xRedundantColsName: {xRedundantColsName}; \n\n\n yRedundantColsName: {yRedundantColsName}")
    return xRedundantColsName, yRedundantColsName

# Pearson correlation coefficient
def calCC(x=[1,2,3,4,5], y=[50,60,70, 100,110]):
    return pearsonr(x, y)


# Data cleaning - truncate the non-null data only
# def getStartNEnd(serial = "Total of Employed persons(000)", dataframe = x2):
def getStartNEnd(serial, dataframe):
    col_l = list(dataframe[serial])
    # start, end = 0, len(col_l)-1
    start, end = 0, 0
        
    # start
    #if list(dataframe[serial])[0].isnan():
    if math.isnan(col_l[0]):
        for i in range(len(col_l)):
            if not math.isnan(col_l[i]):
                start = i
                break
                
    # end
    for j in range(start, len(col_l)):
        if not math.isnan(col_l[j]):
            end = j
        else:
            break
        '''if math.isnan(col_l[j]):  # LOGIC ERROR -> coz it may be NOT NULL until the last element
            print("here = {j}")
            end = j-1
            break'''

    return start, end


    
    # print(dataframe[serial] )
    # print(list(dataframe[serial])[0])
    
# def getCCnR(df = newDf):
def getCCnR(df, x2, y2):
    # var declaration
    xTS_l, yTS_l = [], []
    cc_l, r_l = [], []
    
    for i in range(len(df)):
        # show current indicator for x & y
        xIndicatorName = df.x[i]
        yIndicatorName = df.y[i]
        
        xStart, xEnd = getStartNEnd(xIndicatorName, x2)
        yStart, yEnd = getStartNEnd(yIndicatorName, y2)

        start = xStart if xStart>yStart else yStart   # choose greater (/higher)
        end = xEnd if xEnd<yEnd else yEnd             # choose smaller (/lower)
        
        #print(start, end)

        # time serial for x & y
        for year in range(start, end+1):
            # xTS_l.append(list(x2.xIndicatorName)[year])   # <- WRONG
            xTS_l.append(list(x2[xIndicatorName])[year])
            yTS_l.append(list(y2[yIndicatorName])[year])
        # print(f"xTS_l = {xTS_l}, yTS_l = {yTS_l}")  # DEBUG PURPOSE
        cc, r = pearsonr(xTS_l, yTS_l)
        cc_l.append(cc)
        r_l.append(r)
        
        xTS_l, yTS_l = [], []
    
    return cc_l, r_l
    
    # print(df)
    # print(len(df))
    # print(curX, curY)
    # print(xTS_l)
    
    


#---------------------------------#
# Main application
def displayModifiedHeatmap(selected_infra_indi_df, selected_eco_indi_df):
    # read csv files
    # x = pd.read_csv("infra_indi_data/Infrastructure - dev_infra.csv")  # hardcode
    # y = pd.read_csv("eco_indi_data/Economic - financial_dev.csv")  # hardcode
    x = selected_infra_indi_df
    y = selected_eco_indi_df

    # Set year col as df index   # ***
    x2 = x.set_index('Year')
    y2 = y.set_index('Year')

    # === 2 ===
    # === get x (infrastructure indicator) && y (economy indicator) ===
    # xCol, yCol = getXnYColsName(x2, y2)
    # xColName, yColName, xTimeSeries, yTimeSeries = getXnYColsName(x2, y2)
    xColData, yColData = getXnYData(x2, y2)
    # print(f"xColName: {xColName} \n\n\n yColName: {yColName}")

    # d = {"year":, "xColName": type(xColName), "yColName": type(yColName), "xTimeSeries": type(xTimeSeries), "yTimeSeries": type(yTimeSeries)}
    d = {"x": xColData, "y": yColData}

    newDf = pd.DataFrame(data=d)


    # === 3 ===
    df = newDf
    xTS_l, yTS_l = [], []

    xIndicatorName = df.x[0]
    yIndicatorName = df.y[0]

    xStart, xEnd = getStartNEnd(xIndicatorName, x2)
    yStart, yEnd = getStartNEnd(yIndicatorName, y2)

    start = xStart if xStart>yStart else yStart   # choose greater (/higher)
    end = xEnd if xEnd<yEnd else yEnd             # choose smaller (/lower)

    for year in range(start, end+1):
        # xTS_l.append(list(x2.xIndicatorName)[year])   # <- WRONG
        xTS_l.append(list(x2[xIndicatorName])[year])
        yTS_l.append(list(y2[yIndicatorName])[year])

        
    # DEBUG PURPOSE
    '''print("len:", len(df))
    print("xIndicatorName:", xIndicatorName)
    print("yIndicatorName:", yIndicatorName)
    print("xTS_l:", xTS_l)
    print("yTS_l:", yTS_l)

    print(start, end)'''


    # === 4 ===
    # === get cc && r ===
    cc_l, r_l = getCCnR(newDf, x2, y2)
    newDf['cc'], newDf['r'] = cc_l, r_l


    newDf

    # === 5 ===
    # display heatmap
    plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]

    # plt.figure(figsize=(5,5))
    # plt.figure(figsize=[350, 350])
    plt.figure(figsize=[75, 75])
    # plt.figure(figsize=[100, 100])

    heatmap(
        x=newDf['x'],   # <=<=<=   # col. to use as horizontal dimension
        y=newDf['y'],   # col. to use as vertical dimension
        size_scale=7000,   # OPTION: 7900, 7000 # change this to see how it affects the plot
        size=newDf['cc'],   # OPTION: newDf['cc']*5 # Values to map to color, here we use number of items in each bucket
        #x_order=[]
        color=newDf['cc'],
        palette=sns.cubehelix_palette(128)[::1]   # we'll use black->red palette
    )

    # plt.savefig("result_path", dpi=100)
    plt.savefig("heatmap-diagram")  # *** result_path123.jpg

    newDf.rename(columns={"x": "x (infrastructure indicator(s))", "y": "y (economic indicator(s))", "cc": "cc (Pearson`s correlation coefficient)", "r": "r (2-tailed p-value)"}, inplace=True)  # *** -> escape the apostrophe"'""  # rename column names  # *** -> (+) "_"(s)

    # plt.show()
    return newDf, plt


# displayModifiedHeatmap()
