# import numbers
from operator import index
import sys
import os
import platform
import csv
import re
# from turtle import shape
# from typing import Final
# from turtle import shape
# from cv2 import normalize
import pandas as pd
from datetime import datetime
import pytz
from pytz import timezone
import numpy as np
import time
# import datetime
import pyhrv.frequency_domain as fd
# from pyhrv.hrv import hrv
import pyhrv.time_domain as td
import pyhrv.nonlinear as nl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import warnings
# import gc
# from memory_profiler import profile
from hrv.filters import moving_median, quotient, threshold_filter, moving_average 
from hrv.rri import RRi
from sklearn.model_selection import train_test_split
from scipy import stats
from keras.models import Sequential
from keras.layers import LSTM, Dense, Flatten, Dropout
from keras.models import load_model

# filter or parameters 
# Whether to generate plots 

plot_bool = 0 

num_std = 3
remove_outlier = 1 
thresFilt = 0




# files processed 


uOS = platform.system()
metrics = ['hr_mean', 'hr_min', 'hr_max', 'hr_std', 'nni_diff_mean', 'nni_diff_min', 'nni_diff_max', 'rmssd', 'sdsd', 'sd1', 'sd2', 'sd_ratio', 'fft_peak_vlf', 'fft_peak_lf', 'fft_peak_hf', 'fft_abs_vlf', 'fft_abs_lf', 'fft_abs_hf', 'fft_rel_vlf', 'fft_rel_lf', 'fft_rel_hf', 'fft_log_vlf', 'fft_log_lf', 'fft_log_hf', 'fft_ratio', 'fft_total', 'fft_norm_LF', 'fft_norm_HF']
pyStatsDict = {}
filtered = [0]

# RR-specific window parsing
#@profile
def rrParse(window, retDF, subjectID, dataCol, curWin, start, end, subPlotPath, df2 = 0 ):

  
    # Filter by RR_interval quality - Maybe create a function ... however it is too recursive already ... 


# number of lines in a window

    numLines_initial =window.shape[0]

    result = pd.merge(window,df2,on='Relative_Time')

    # values to filter - 0 clean - 1 noisy , 128 - unrealiable, check: 129 - noisy and unreliable 

    #save back to the window dataframe --- need to uniformize it
    
    window = result[result['RR_interval_quality'].isin(filtered)]

    

# number of windows after filtered

    numLines =window.shape[0] 

    # end of filtering 


    if (numLines == 0):
        row = ["9999"] * len(metrics)
        retDF.loc[len(retDF.index)] = row
        # print("empty DF")
        return retDF
     

    # end of filtering


     # check for zero line RR interval 

    rrinterval = window['RR_Interval'].values
    if (rrinterval[0] == 0):
        rrinterval = np.delete(rrinterval, 0)

    n_zeros = np.count_nonzero(rrinterval==0)
    if n_zeros > 0:
        print("\nERROR: This file has " + str(n_zeros) + " VALUE(S) AS ZERO\n")
        rrinterval = rrinterval[rrinterval != 0]

# instead of three values, changed the test to be at lest 10% of the window and at least 10 seconds - HRV indices obtained with time windows as
# short as 10–30 s are as reliable - pg.2 - Pham et al, 2021

    if (len(rrinterval) < (numLines_initial * 0.10) or (len(rrinterval) < 10)):
        row = ["9999"] * len(metrics)
        retDF.loc[len(retDF.index)] = row
        return retDF


    nni_results = td.nni_differences_parameters(rrinterval)
    nni_results2 = fd.welch_psd(rrinterval, show=False)
    nni_results3 = td.rmssd(rrinterval)
    nni_results4 = td.sdsd(rrinterval)
    nni_results5 = td.hr_parameters(rrinterval)
    nni_results6 = nl.poincare(rrinterval, show=False)

    if (plot_bool == 1):

        title = "Subject id# " + str(subjectID) +  " - " + str(dataCol) +  " - Window # " +  str(curWin) +  "\n Start Time: " + convert(start) + "   -  End Time: " + convert(end) + " s \n"
        nni_results6['poincare_plot'].suptitle(title, fontweight ="bold")             
        filenaming = os.path.join(subPlotPath, "poincare_" + str(subjectID) + "_" + str(dataCol) + "_" + str(curWin) + ".svg")
        nni_results6['poincare_plot'].savefig(filenaming)

        title = "Subject id# " + str(subjectID) +  " - " + str(dataCol) +  " - Window # " +  str(curWin) +  "\n Start Time: " + convert(start) + "   -  End Time: " + convert(end) + " s \n"
        nni_results2['fft_plot'].suptitle(title, fontweight ="bold")    
        filenaming = os.path.join(subPlotPath, "Welch_PSD_" + str(subjectID) + "_" + str(dataCol) + "_" + str(curWin) + ".svg")
        nni_results2['fft_plot'].savefig(filenaming)
    
    # close all the figures - this helps the memory
    plt.close('all') 

    # change so this only has to be run once for all the files during a batch processing run 

    bands = []
    for k in nni_results2['fft_bands'].keys() :
        if (nni_results2['fft_bands'][k] != None):
            bands.append(k)
    final_dict = {}
    for x in nni_results2.keys():
        if (type(nni_results2[x]) == tuple and len(bands) == len(nni_results2[x])):
                count = 0
                for z in bands:
                    newkey = x + '_' + z
                    final_dict[newkey] = nni_results2[x][count]
                    count= count + 1
        if (isinstance(nni_results2[x], np.floating)):
                final_dict[x] = nni_results2[x]
    final_dict['fft_norm_LF'] = nni_results2['fft_norm'][0]
    final_dict['fft_norm_HF'] = nni_results2['fft_norm'][1]
    par = ['sd1','sd2' , 'sd_ratio']
    nni_results6_filtered = {}
    for x in par:
        nni_results6_filtered[x] = nni_results6[x]
    hrv_dict = {**nni_results5, **nni_results, **nni_results3 , **nni_results4 ,  **nni_results6_filtered, **final_dict}

    hrv_dict = { metric: hrv_dict[metric] for metric in metrics }
    row = []
    for metric in metrics:
        row.append(hrv_dict[metric])
    # change the way I add new files 
    # retDF.loc[0] = row
    retDF.loc[len(retDF.index)] = row



    # delete objects 
    # del(window, result, rrinterval,nni_results, nni_results2,nni_results3, nni_results4 , nni_results5 , nni_results6)
    # del()
    # gc.collect()
    # print(gc.get_stats())

    return retDF

# Cadence-specific window parsing
def cadParse(window, retDF, fileType):
    if (window['cadence'].sum() == 0):
        pass
        # print("\nERROR: Cadence window all 0\n")
    else:
        window = window[window.cadence != 0]
    retDF.loc[len(retDF.index)] = window[fileType].mean()
    return retDF

# Alternative HR from HR.CSV -specific window parsing
def altParse(window, retDF, fileType):
    if (window['HR'].sum() == 0):
        print("\nERROR: all HR values for this window are 0\n")
    else:
        window = window[window.HR != 0]

    retDF.loc[len(retDF.index)] = [window[fileType].mean() , window[fileType].max() , window[fileType].min()]
    return retDF

def stepParse(window, retDF, fileType):

    if (window['steps'].sum() == 0):
        # print("\nERROR: No Steps were detected\n")
        # print(window['steps'].head())
        retDF.loc[len(retDF.index)] = 0
        # return retDF
    else:
        retDF.loc[len(retDF.index)] = window['steps'].iat[-1] - window['steps'].iat[0]
        # return retDF
        # print("number of steps" + str(total_steps))
       


    # retDF.loc[len(retDF.index)] = [window[fileType].mean() , window[fileType].max() , window[fileType].min()]
    return retDF

# RR interval quality specific window parsing
def rrQual(window, retDF):
    counts = window['RR_interval_quality'].value_counts(normalize=True)
    for val, cnt in (counts.iteritems()):
        if (val == 0):
            retDF.loc[len(retDF.index)] = cnt
            return retDF
    retDF.loc[len(retDF.index)] = 0.0
    return retDF

# Splits the data up into windows, calling the respective parsing function on the data window

def generalParse(df, retDF, rPath, winSize, winCount, fileType, subjectID, dataCol, subPlotPath, df2 = 0, baseStart = 0, baseEnd = 0):
    
    numLines = df.shape[0]
    # print DF  
    # adding a check for zero line RR Interval 

    if (numLines == 0):
        print ("Nothing to return - Zero line file")
        return retDF

    timeRemaining = numLines % winSize

    if (dataCol == 0):
        window = df.loc[(df['Relative_Time'] >= baseStart) & (df['Relative_Time'] <= baseEnd)]
        if (fileType == "cadence"):
            retDF = cadParse(window, retDF, fileType)
        elif (fileType == "HR"):
            retDF = altParse(window, retDF, fileType)
        elif (fileType == "steps"):
            retDF = stepParse(window, retDF, fileType)
        elif (fileType == "RR_interval"):
            retDF = rrParse(window, retDF, subjectID, dataCol, curWin, start, end, subPlotPath,df2)
        elif (fileType == "RR_interval_quality"):
            retDF =  rrQual(window, retDF)
    else:
        start = 0
        curWin = 1
        print("\n(", rPath, ") has", winCount, str(winSize) +
            "-second slices, with", timeRemaining, "seconds remaining\n")
        if (winCount > 0):
            end = winSize
            while(start<(winSize*winCount)):
                window = df.loc[(df['Relative_Time'] > start) & (df['Relative_Time'] < end)]
                if (fileType == "cadence"):
                    retDF = cadParse(window, retDF, fileType)
                elif (fileType == "HR"):
                    retDF = altParse(window, retDF, fileType)
                elif (fileType == "steps"):
                    retDF = stepParse(window, retDF, fileType)
                elif (fileType == "RR_interval"):
                    retDF = rrParse(window, retDF, subjectID, dataCol, curWin, start, end, subPlotPath,df2)
                elif (fileType == "RR_interval_quality"):
                    retDF =  rrQual(window, retDF)
                
                    
                start = end
                end += winSize
                curWin += 1
    return retDF

# RR quality analysis entry point, called for parsing RR quality data
def rr_int_qual(root, winSize, winCount, subjectID, dataCol, subPlotPath, baseStart, baseEnd):
    retDF = pd.DataFrame(columns=["RR Interval Clear Data Quality (%)"])
    rPath = os.path.join(root, "RR_interval_quality.csv")
    df = pd.read_csv(rPath)
    df.columns =['Relative_Time', 'RR_interval_quality']
    #df.astype(float)

    return generalParse(df, retDF, rPath, winSize, winCount, "RR_interval_quality", subjectID, dataCol, subPlotPath, 0, baseStart, baseEnd)

    

# RR analysis entry point function, called for parsing the RR data
# @profile
def rr_int(root, winSize, winCount, subjectID, dataCol, subPlotPath, baseStart, baseEnd):
    
    retDF = pd.DataFrame(columns=metrics)
    rPath = os.path.join(root, "RR_interval.csv")
    df = pd.read_csv(rPath)
    df.columns =['Relative_Time', 'RR_Interval']
    # df.astype(float)
    df['RR_Interval'] = df['RR_Interval'] / 256 


# added the filtering - removing extreme values - participant-wide removal at 3 std deviations - keep 99.9% of data  
    if (remove_outlier == 1):
        outlier_low = df['RR_Interval'].mean() - num_std * df['RR_Interval'].std()
        outlier_high = df['RR_Interval'].mean() + num_std * df['RR_Interval'].std()
 
        num_original = len(df.index)

        df = df.loc[(df['RR_Interval'] > (outlier_low)) & (df['RR_Interval'] < (outlier_high))]
        num_final = num_original - len(df.index)  
        # add this to the statistics file - for each processed file  
        # print ("removed" + str(num_final) )

    rrArr = []
    # for rr in df["RR_Interval"]:
    #     if (rr > 0):
    #         rrArr.append(rr)
    
    # rrinterval = df['RR_Interval'].values
    # if (rrinterval[0] == 0):
    #     rrinterval = np.delete(rrinterval, 0)

    # n_zeros = np.count_nonzero(rrinterval==0)
    # if n_zeros > 0:
        # print("\nERROR: This file has " + str(n_zeros) + " VALUE(S) AS ZERO\n")

    df = df.loc[(df["RR_Interval"] > 0)]  
    filtRR = RRi(df["RR_Interval"].values) 
    filtRR2 = filtRR
    # try: 
    #     filtRR2 = threshold_filter(filtRR, threshold='low', local_median_size=5)
    # except:
    # filtRR2 = moving_median(filtRR, order=3)

# use the threshold filter on high

    # Select Filtering method: Moving Average (1); Moving Median (2); Threshold (3); Quotient (4)

    if (filterType == 1):
        print("\nUtilizing Moving Average filter\n")
        filtRR2 = moving_average(filtRR, order=3)
    elif (filterType == 2):
        print("\nUtilizing Moving Median filter\n")
        filtRR2 = moving_median(filtRR, order=3)
    elif (filterType == 3):
        print("\nUtilizing Threshold filter\n")
        try: 
            filtRR2 = threshold_filter(filtRR, threshold='very strong', local_median_size=5)
        except:
            print("\nERROR: Threshold filter exception, fallback to Moving Median\n")
            filtRR2 = moving_median(filtRR, order=3)
    elif (filterType == 4):
        print("\nUtilizing Quotient filter\n")
        filtRR2 = quotient(filtRR)
        
    rrArr = filtRR2.values
    if (filterType == 4):
        df["RR_Interval"] =  df["RR_Interval"] * 1000
        temp_rr = df["RR_Interval"].values
        position_new = 0
        position_old = 0
        found = 0 
        indexes = []
        for i in rrArr:
            while (found == 0):
                if (rrArr[position_new] == temp_rr[position_old]):
                    indexes.append(position_old)
                    position_new += 1
                    position_old += 1
                    found = 1
                else:
                    position_old += 1
            found = 0 
        df = df.filter(items = indexes, axis=0)
    else:
        df["RR_Interval"] = rrArr
# added the plot bool condition check to speed up testing ...

    if (plot_bool == 1):
    
    #### Added to generate the outlier plots 
        rr_trimmed = df.loc[(df['Relative_Time'] > int(winSize)) & (df['Relative_Time'] <df['Relative_Time'].max()-int(winSize))]
        
        plt.figure(figsize=(20, 7))
        plt.title("Distribution of RR-intervals for " + subjectID + " during " + dataCol )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore") # ignore FutureWarning 
            rr_series = pd.Series(rr_trimmed['RR_Interval']*1000)
            # print(rr_series)
            sns.kdeplot(data=rr_series, label="rr-intervals", color="#A651D8", shade=True)

        # outlier_low = np.mean(rr_series) - 2 * np.std(rr_series)
        # outlier_high = np.mean(rr_series) + 2 * np.std(rr_series)

        plt.axvline(x=outlier_low)
        plt.axvline(x=outlier_high, label="outlier boundary")
        plt.text(outlier_low - 370, 0.004, "outliers low (< mean - 2 sigma)")
        plt.text(outlier_high + 20, 0.004, "outliers high (> mean + 2 sigma)")

        plt.xlabel("RR-interval (ms)")
        plt.ylabel("Density")
        plt.legend()
        filenaming = 'distribution/distribution_' + subjectID + "_"+ dataCol +'.svg'
        plt.savefig(filenaming)
        
        plt.close('all') 


 # Filter by RR_interval quality - Maybe create a function ... however it is too recursive already ... 

    rPath2 = os.path.join(root, "RR_interval_quality.csv")
    df2 = pd.read_csv(rPath2)
    df2.columns =['Relative_Time', 'RR_interval_quality']
    df2.astype(float)

    #filtering has been implemented in the next function - general parse, but exports the quality in df2 - end of filtering

    return generalParse(df, retDF, rPath, winSize, winCount, "RR_interval", subjectID, dataCol, subPlotPath, df2, baseStart, baseEnd)

# Cadence analysis entry point function, called for parsing cadence data
def cadence(root, winSize, winCount, subjectID, dataCol, subPlotPath, baseStart, baseEnd):
    retDF = pd.DataFrame(columns=["Mean Cadence"])
    rPath = os.path.join(root, "cadence.csv")
    df = pd.read_csv(rPath, header=None)
    df.columns = ['Absolute_Time', 'Relative_Time', 'cadence']

    return generalParse(df, retDF, rPath, winSize, winCount, "cadence", subjectID, dataCol, subPlotPath, 0, baseStart, baseEnd)


def steps(root, winSize, winCount, subjectID, dataCol, subPlotPath, baseStart, baseEnd):
    retDF = pd.DataFrame(columns=["Number of Steps"])
    rPath = os.path.join(root, "step.csv")
    # remove the headerHexoskin Raw Data\Data Collection 4
    df = pd.read_csv(rPath, header=0)
    df.columns = ['Relative_Time', 'steps']
    # df.astype(float)
    # print(df)

    return generalParse(df, retDF, rPath, winSize, winCount, "steps", subjectID, dataCol, subPlotPath, 0, baseStart, baseEnd)




# alternative HR entry point function, called for parsing cadence data
def alt_hr(root, winSize, winCount, subjectID, dataCol, subPlotPath, baseStart, baseEnd):
    retDF = pd.DataFrame(columns=["Mean HR", "Max HR", "Min HR" ] )
    rPath = os.path.join(root, "heart_rate.csv")
    df = pd.read_csv(rPath, header=None)
    df.columns = ['Absolute_Time', 'Relative_Time', 'HR']

    return generalParse(df, retDF, rPath, winSize, winCount, "HR", subjectID, dataCol, subPlotPath, 0, baseStart, baseEnd)



# Base analysis file generator
def baseDF(subjectID, dataCol, winCount, startTime, winSize, baseStart, baseEnd):
    if (dataCol == "Baseline"):
        dataCol = 0
        winCount = 1
        subArr = [subjectID]
        dataArr = [dataCol]
        winArr = [winCount]
        absTimeArr = [startTime + baseStart]
        absTimeReadArr = []
        relTimeArr = [baseStart]
        originalTZ = pytz.timezone('America/Los_Angeles')
        absTimeStr = datetime.fromtimestamp(absTimeArr[0], tz=originalTZ)
        absTimeStr = absTimeStr.astimezone(timezone('US/Eastern'))
        absTimeStr = absTimeStr.strftime("%c %Z")
        absTimeReadArr.append(absTimeStr)
    else:
        if (dataCol == "Data Collection 1"):
            dataCol = 1
        elif (dataCol == "Data Collection 2"):
            dataCol = 2
        elif (dataCol == "Data Collection 3"):
            dataCol = 3
        elif (dataCol == "Data Collection 4"):
            dataCol = 4
        subArr = [subjectID] * winCount
        dataArr = [dataCol] * winCount
        winArr = list(range(1, winCount + 1))
        absTimeArr = []
        absTimeReadArr = []
        relTimeArr = []
        curAbsTime = startTime
        curRelTime = 0
        for i in range(winCount):
            absTimeArr.append(curAbsTime)
            relTimeArr.append(curRelTime)
            originalTZ = pytz.timezone('America/Los_Angeles')
            absTimeStr = datetime.fromtimestamp(curAbsTime, tz=originalTZ)
            absTimeStr = absTimeStr.astimezone(timezone('US/Eastern'))
            absTimeStr = absTimeStr.strftime("%c %Z")
            absTimeReadArr.append(absTimeStr)
            curRelTime += winSize
            curAbsTime += winSize
    retDF = pd.DataFrame()
    retDF = retDF.assign(Subject_ID=subArr, Data_Phase=dataArr, Window_Count=winArr, Absolute_Time=absTimeArr, Absolute_Time_Readable=absTimeReadArr, Relative_Time=relTimeArr)
    return retDF

# Extracts subject ID and Data collection from user input path
def pathExtract(root, slash):
    folder = root.split(slash)[-1]
    subjectID = re.search("^\d{4}_", folder)
    if (subjectID != None):
        subjectID = subjectID.group(0).strip("_")
    else:
        print("Error: Could not get subject ID from folder (" + root + ")\n")
        return None
    if (folder[-1] == ")"):
        collectionPart = folder.split(" ")
        collectionPart = collectionPart[-2] + " " + collectionPart[-1]
        subjectID = subjectID + " " + collectionPart

    dataCol = root.split(slash)[-2]

    return subjectID, dataCol

# Extracts releveant information from statistics file in record currently being analyzed

def statsExtract(statsPath, winTime):
    startTime = 0
    endTime = 0
    startFlag = False
    endFlag = False
    with open(statsPath, mode='r') as stats:
        statsReader = csv.DictReader(stats)
        line_count = 0
        for row in statsReader:
            if ((row["start"] is not None) and (not row["start"] == "na") and (row["start"].isnumeric)):
                startTime = row["start"]
                startFlag = True
            if ((row["end"] is not None) and (not row["end"] == "na") and (row["end"].isnumeric)):
                endTime = row["end"]
                endFlag = True
            line_count += 1
            if (startFlag and endFlag):
                break
    startTime = float(float(startTime) / 256)
    endTime = float(float(endTime) / 256)
    duration = endTime - startTime
    
    

    winCount = int(float(duration) / float(winTime))

    # 
    # print("The extracted time is " + str(winCount))

    return startTime, endTime, duration, winCount

def convert(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
      
    return "%d:%02d:%02d" % (hour, minutes, seconds)

def baseBestWindow(root, winSize):
    rPath2 = os.path.join(root, "RR_interval_quality.csv")
    df2 = pd.read_csv(rPath2)
    df2.columns =['Relative_Time', 'RR_interval_quality']
    df2.astype(float)

    #filtering has been implemented in the next function - general parse, but exports the quality in df2 - end of filtering

    # provided by the argument --- winSize = 237 

    startTime = (df2['Relative_Time'].min())
    endTime = (df2['Relative_Time'].max())
    # print(startTime)
    duration = endTime - startTime
    # print(duration)


    timeRemaining = duration >= winSize

    # check it the df contains at least one window. 

    winCount = 1 
    # the number of windows is equal 1 as default for the baseline procedures - removed theint(float(duration) / float(winSize))


    start = 0
    curWin = 1

    max = 0 
    start_max = 0 
    end_max = 0 

    if (winCount > 0):
        end = winSize
        while(start<(winSize*winCount)):
            window = df2.loc[(df2['Relative_Time'] >= start) & (df2['Relative_Time'] <= end)]

            counts = window['RR_interval_quality'].value_counts(normalize=True)
            for val, cnt in (counts.iteritems()):
                if (val == 0):
                    # print(cnt)
                    if (cnt > max):
                        max = cnt
                        start_max = start
                        end_max = end
                        # print("new maximum quality")
                        # print("located between" + str(start) + " and " +  str(end) )
                        


            start_index = (window[window['Relative_Time']>start].index.values)[0] 
            # print(start_index)
            # print(indexes_larger[0])
            
            
            # start = start + 0.5 
            start = window['Relative_Time'][start_index]
            # print("start time is " + str(start))
            end = start + winSize
            curWin += 1
            # print(curWin)
            #print(window)

    return start_max, end_max

def acclActivity(root, winSize, winCount, subjectID, dataCol, subPlotPath, baseStart, baseEnd, model):
    activities = ['Walking', 'Jogging', 'Upstairs', 'Downstairs', 'Sitting', 'Standing']
    retDF = pd.DataFrame(columns=["Physical Activity Classification"])
    totalDF = pd.DataFrame(columns=["Relative_Time", "AcclX", "AcclY", "AcclZ"])
    rPathX = os.path.join(root, "acceleration_X.csv")
    rPathY = os.path.join(root, "acceleration_Y.csv")
    rPathZ = os.path.join(root, "acceleration_Z.csv")
    dfX = pd.read_csv(rPathX)
    dfY = pd.read_csv(rPathY)
    dfZ = pd.read_csv(rPathZ)
    dfX.columns =['Absolute_Time', 'Relative_Time', 'AcclX']
    dfY.columns =['Absolute_Time', 'Relative_Time', 'AcclY']
    dfZ.columns =['Absolute_Time', 'Relative_Time', 'AcclZ']
    totalDF["Relative_Time"] = dfX["Relative_Time"]
    totalDF["AcclX"] = dfX["AcclX"]
    totalDF["AcclY"] = dfY["AcclY"]
    totalDF["AcclZ"] = dfZ["AcclZ"]
    # Convert dataframe into array of XYZ axis points
    axis = totalDF[["AcclX","AcclY","AcclZ"]].to_numpy()
    # Find out how many axis point sub arrays of length 50 can be made, as the model predicts in batches of 50 axis points
    length = axis.shape[0] / 50
    # Checking if we have any remaining axis points if we were to cut into subarrays
    dec = length % 1
    # If we do, remove extra axis points so we can evenly cut into subarrays of 50 points
    if (dec != 0):
        # Calculate how many axis points left over
        cutoff = (int) (dec * 50)
        # Slice off extra axis points
        axis = axis[:-cutoff]
        # Remove decimal from length
        length = length // 1
    # Split axis array into array of arrays of 50 sub arrays
    arr = np.array(np.split(axis,length))
    # Input into model for prediction
    classes = model.predict(arr)

    # Model predicts in batches of 50, if using windows of 300, 6 batches make one window
    # so we add the batches together to get the prediction for the window
    i = 0
    length = (int) (len(classes))
    while i+6 <= length:
        # Extract 6 batches from model return
        chunk = classes[i:i+6]
        # Add batches together into one array
        temp = np.add(chunk[0], chunk[1])
        temp = np.add(temp, chunk[2])
        temp = np.add(temp, chunk[3])
        temp = np.add(temp, chunk[4])
        temp = np.add(temp, chunk[5])
        # Find index of max predicted activity
        index = np.argmax(temp)
        # Decode activity from index, then add to return dataframe
        act = activities[index]
        retDF.loc[len(retDF.index)] = [act]
        i+=6

    return retDF

# Recursively analyze all records in subdirectories of user-input path
def recordAnalysis(dPath, debug):

    ## Create a new empty CSV file and append to it after each file.  
    # with open(os.path.join(dPath, "pyStats.csv"), 'w') as f: 
    
    
    first_Run = 1
    finalAnalysisDF = None
    slash = "\\"
    if (uOS == "Darwin" or uOS == "Linux"):
        slash = "/"
    winSize = "1"
    while (winSize.isalnum()):
        if debug:
            winSize = "300"
        else:
            winSize = input("\nInput window size in seconds, or return to folder select (f)\n>> ")
        if (winSize == "f"):
            return
        elif (winSize.isnumeric()):
            break
        else:
            print("\nERROR: Window size must be an integer, please retry")
    scriptStartTime = time.time()
    winSize = int(winSize)
    pyStatsDict["Folder Path"] = dPath
    pyStatsDict["Window Size (Seconds)"] = winSize
    if (filterType == 1):
        pyStatsDict["Filter Type"] = "Moving Average"
    elif (filterType == 2): 
        pyStatsDict["Filter Type"] = "Moving Median"
    elif (filterType == 3): 
        pyStatsDict["Filter Type"] = "Threshold"
    elif (filterType == 4): 
        pyStatsDict["Filter Type"] = "Quotient"
    plotPath = os.path.join(dPath, "plot_output")
    if (not os.path.isdir(plotPath)):
        os.mkdir(plotPath)

    modPath = input("\nInput path to model for classification\n>> ")
    if debug:
        modPath = r"C:\Users\ishaanghosh\Box\Code\Sensorlab-Yale Hexoskin Repository\Analysis\Acceleration Model\HexoModel.h5"
    while(not os.path.isfile(modPath)):
        print("\nERROR: Unknown directory, please retry\n")
        modPath = input("\nInput path to model for classification\n>> ")
    model = load_model(modPath)
    print(model.summary())

    for root, dirs, files in os.walk(dPath):
        if ("plot_output" in root):
            continue

        statsFilePath = os.path.join(root, "statistics.csv")

        if (not os.path.exists(statsFilePath)):
            print("\nCould not find stats file (" + statsFilePath +
                  ") in working directory, continuing to subdirectories\n")
            continue
        
        startTime, endTime, duration, winCount = statsExtract(
            statsFilePath, winSize)
        subjectID, dataCol = pathExtract(root, slash)

        subPlotPath = os.path.join(plotPath, subjectID)
        if (not os.path.isdir(subPlotPath)):
            os.mkdir(subPlotPath)        

        if (dataCol == "Baseline"):
            baseStart, baseEnd = baseBestWindow(root, winSize)
        else:
            baseStart = 0
            baseEnd = 0
        finalDF = baseDF(subjectID, dataCol, winCount, startTime, winSize, baseStart, baseEnd)
        rrQualDF = rr_int_qual(root, winSize, winCount, subjectID, dataCol, subPlotPath, baseStart, baseEnd)
        actiDF = acclActivity(root, winSize, winCount, subjectID, dataCol, subPlotPath, baseStart, baseEnd, model)
        altDF = alt_hr(root, winSize, winCount, subjectID, dataCol, subPlotPath, baseStart, baseEnd)
        stepDF = steps(root, winSize, winCount, subjectID, dataCol, subPlotPath, baseStart, baseEnd)
        cadDF = cadence(root, winSize, winCount, subjectID, dataCol, subPlotPath, baseStart, baseEnd)
        rrDF = rr_int(root, winSize, winCount, subjectID, dataCol, subPlotPath, baseStart, baseEnd)
        

        #add steps
        
        
        # added Alt_Df
        
        finalDF = finalDF.join(rrQualDF)
        finalDF = finalDF.join(actiDF)
        finalDF = finalDF.join(stepDF)
        finalDF = finalDF.join(altDF)
        finalDF = finalDF.join(cadDF)
        finalDF = finalDF.join(rrDF) 
        
        

        ## gc.collect()
        
        if (type(finalAnalysisDF) != type(finalDF)):
            finalAnalysisDF = finalDF
        else:
            finalAnalysisDF = pd.concat([finalAnalysisDF, finalDF], ignore_index=True)
        
        ## save the file to CSV instead of PANDAS DF ... might have a better memory management 
        ## if (first_Run == 1):
        ##    finalDF.to_csv("existing.csv", mode="w", index=False, header=True)
        ##    first_Run = 0
            
            # print("Number of lines of the final file" + str(finalAnalysisDF.shape[0]))
        #else:
            # print(finalDF)
        #    finalDF.to_csv("existing.csv", mode="a", index=False, header=False)
            # print("memory usage")
            # print(finalDF.memory_usage(deep=False))
            #release memory 
            # del(finalDF,rrQualDF,stepDF,altDF,cadDF,rrDF)
            # gc.collect()
            
    
    # record the file as a CSV file - One option would be to re-open the file and add the columns

    analysisPath = os.path.join(dPath, "pyAnalysis.csv")
    finalAnalysisDF.to_csv(analysisPath, index=False)

    
    scriptDuration = (time.time() - scriptStartTime)
    value = datetime.fromtimestamp(scriptStartTime)
    pyStatsDict["Script Start Time"] = value.strftime('%Y-%m-%d %H:%M:%S')
    pyStatsDict["Analysis Runtime (Seconds)"] = scriptDuration
    pyStatsDict["Filtered"] = filtered
    if (plot_bool == 1):
        pyStatsDict["Generate Plots"] = True
    else:
        pyStatsDict["Generate Plots"] = False
    if (remove_outlier == 1):
        pyStatsDict["Outliers Removed"] = True
    else:
        pyStatsDict["Outliers Removed"] = False
    pyStatsDict["Number of Standard Deviations"] = num_std
    with open(os.path.join(dPath, "pyStats.txt"), 'w') as f: 
        for key, value in pyStatsDict.items(): 
            f.write('%s: %s\n' % (key, value))
    return


def main():
    while (True):
        print(
                "\n<-------------------------------------------------------------------->\n")
        dPath = input("\nInput path to directory for analysis, or exit (e)\n>> ")
        
        # add this conditional to the top of the file 
        # in the future change the number of consecutive data points - from 2 to 5 

        debug = False
        if (dPath.lower() == "t"):
            dPath = r"C:\Users\ishaanghosh\Box\Code\Sensorlab-Yale Hexoskin Repository\data\Baseline\1612_03062021"
            debug = True
        if (dPath.lower() == "e"):
            print("\nGoodBye\n")
            print(
                "\n<-------------------------------------------------------------------->\n")
            break
        elif(os.path.isdir(dPath)):
            global filterType
            filterType = ""
            if debug:
                filterType = "0"
            while(filterType != "0" and filterType != "1" and filterType != "2" and filterType != "3" and filterType != "4"):
                filterType = input("\nSelect Filtering method: No Filter (0); Moving Average (1); Moving Median (2); Threshold (3); Quotient (4)\n>> ")
            filterType = int(filterType)
            if not debug:
                process_consecutive = input("Consolidate Results for consecutive data points \n>> ")
            recordAnalysis(dPath, debug)
        else:
            print("\nERROR: Unknown directory, please retry\n")

    return


if __name__ == "__main__":
    sys.exit(main())