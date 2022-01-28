# -*- coding: utf-8 -*-
# coding: utf-8
from functools import reduce
import os 
import random
import numpy as np
from scipy.stats.stats import kendalltau, pearsonr, spearmanr
from deap import base, creator, tools
# from bokeh.tests.test_driving import offset
from bokeh import *   # conda install bokeh
from bokeh.plotting import figure, show

# UI
import streamlit as st
# Save data of result
import pickle
# LOG
from streamlit.report_thread import REPORT_CONTEXT_ATTR_NAME
from threading import current_thread
from contextlib import contextmanager
from io import StringIO
import sys
import logging
import time
@contextmanager
def st_redirect(src, dst):
    placeholder = st.empty()
    output_func = getattr(placeholder, dst)

    with StringIO() as buffer:
        old_write = src.write

        def new_write(b):
            if getattr(current_thread(), REPORT_CONTEXT_ATTR_NAME, None):
                buffer.write(b + '')
                output_func(buffer.getvalue() + '')
            else:
                old_write(b)

        try:
            src.write = new_write
            yield
        finally:
            src.write = old_write


@contextmanager
def st_stdout(dst):
    "this will show the prints"
    with st_redirect(sys.stdout, dst):
        yield


@contextmanager
def st_stderr(dst):
    "This will show the logging"
    with st_redirect(sys.stderr, dst):
        yield

def demo_function():
    """
    Just a sample function to show how it works.
    :return:
    """
    for i in range(10):
        # logging.warning(f'Counting... {i}')
        # time.sleep(2)
        # print('Time out...')
        logging.warning(i)
        time.sleep(2)
        if i>2:
            print("Continue?")


'''try:
    from Tkinter import *
    import Tkinter as tk
    import ttk
    import tkFileDialog as filedialog
except:
    from tkinter import *
    import tkinter as tk
    from tkinter import ttk
    from tkinter import filedialog'''

import datetime
import csv
import shutil

'''import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.animation as animation
from matplotlib import style

from matplotlib import pyplot as plt'''


# global variable (global namespace)
# POP_NUM = None  # PREV: 50
# CXPB = None
# MUTPB = None

# headers = []
# compare_headers = []
# data_matrix = []
# compare_data_matrix = []

doLogging = True
minInt = float("-inf")
#set default value
# maxYear - minYear = size of arrays
minYear = 1970
maxYear = 2019  # PREV: 16, 00 -> DON'T KNOW WHY NOT WORK  
#mininum number of indicators in a chromosome
minIndicators = 1


# CXPB  is the probability with which two individuals
#       are crossed
#
# MUTPB is the probability for mutating an individual
# CXPB, MUTPB = 0.5, 0.2  # LATER

#set some font style, later for tkinter
'''LARGE_FONT= ("Verdana", 12)
NORM_FONT = ("Helvetica", 10)
SMALL_FONT = ("Helvetica", 8)
style.use("ggplot")'''

'''#plot 3 graph 
f = Figure(figsize=(10,6), dpi=100)
a = f.add_subplot(311)
b = f.add_subplot(312)
c = f.add_subplot(313)

#plot single graph
fig = Figure(figsize=(12,5.5), dpi=100)
ax = fig.add_subplot(111)

figure_graph2 = Figure(figsize=(12,5.5), dpi=100)
ax2 = figure_graph2.add_subplot(111)

figure_graph3 = Figure(figsize=(12,5.5), dpi=100)
ax3 = figure_graph3.add_subplot(111)'''


# prev_path = "/Users/zrhun/FYP Topic - Data Science (Analysis M'sia Finance)/FYPsystem/previous/"
prev_path = os.getcwd() + "/previous_computed_result/"
if not os.path.exists(prev_path):
    os.makedirs(prev_path)
    open(prev_path+"ALL_previous.csv", 'w+').close()
    open(prev_path+"evolution1_previous.csv", 'w+').close()
    open(prev_path+"evolution2_previous.csv", 'w+').close()
    open(prev_path+"evolution3_previous.csv", 'w+').close()


def popupmsg(msg):
    print(msg)





'''def animate(i):
    xar = []
    yar = []
    zar = []

    file_size = os.stat(r"/Users/zrhun/FYP Topic - Data Science (Analysis M'sia Finance)/FYPsystem/previous/evolution1_previous.csv").st_size

    if file_size != 0:
        try:
            with open("/Users/zrhun/FYP Topic - Data Science (Analysis M'sia Finance)/FYPsystem/previous/evolution1_previous.csv", "r") as f:
                # reader = csv.reader(f, skipinitialspace=TRUE)  # *** -> can work?
                reader = csv.reader(f, skipinitialspace=True)
                #skip the header
                next(reader)
                for row in reader:
                    xar.append(int(row[0]))
                    yar.append(float(row[1]))
                    zar.append(float(row[5]))
        except IOError:
            print("File not found!")

    a.clear()  
    a.plot(xar,yar, color="purple", marker="o", label = "Average Fitness")
    a.plot(xar,zar, 'bo-', label = "Max Fitness")
    a.legend()
    a.set_ylabel('Fitnesses')

#def animate1(i):
    xar2 = []
    yar2 = []
    zar2 = []

    file_size2 = os.stat(r"/Users/zrhun/FYP Topic - Data Science (Analysis M'sia Finance)/FYPsystem/previous/evolution2_previous.csv").st_size

    if file_size2 != 0:
        try:
            with open("/Users/zrhun/FYP Topic - Data Science (Analysis M'sia Finance)/FYPsystem/previous/evolution2_previous.csv", "r") as f2:
                reader2 = csv.reader(f2, skipinitialspace=TRUE)
                #skip the header
                next(reader2)
                for row in reader2:
                    xar2.append(int(row[0]))
                    yar2.append(float(row[1]))
                    zar2.append(float(row[5]))
        except IOError:
            print("File not found!") 

    b.clear()
    b.plot(xar2,yar2, color="orange", marker="o", label = "Average Fitness")
    b.plot(xar2,zar2, 'bo-', label = "Max Fitness")
    b.legend()
    b.set_ylabel('Fitnesses')

#def animate2(i):
    xar3 = []
    yar3 = []
    zar3 = []

    file_size3 = os.stat(r"/Users/zrhun/FYP Topic - Data Science (Analysis M'sia Finance)/FYPsystem/previous/evolution3_previous.csv").st_size

    if file_size3 != 0:
        try:
            with open("/Users/zrhun/FYP Topic - Data Science (Analysis M'sia Finance)/FYPsystem/previous/evolution3_previous.csv", "r") as f3:
                reader3 = csv.reader(f3, skipinitialspace=TRUE)
                #skip the header
                next(reader3)
                for row in reader3:
                    xar3.append(int(row[0]))
                    yar3.append(float(row[1]))
                    zar3.append(float(row[5]))
        except IOError:
            print("File not found!")

    c.clear()
    c.plot(xar3,yar3, color="green", marker="o", label = "Average Fitness")
    c.plot(xar3,zar3, 'bo-', label = "Max Fitness")
    c.legend()
    c.set_xlabel('Generation')
    c.set_ylabel('Fitnesses')

# def intialGA():
'''   
   

def runGeneticAlgorithm(
    infra_indi_folder_path=os.getcwd() + '/infra_indi_data/', eco_indi_folder_path=os.getcwd() + '/eco_indi_data/', stats_folder_path=os.getcwd() + '/stats_data/', eco_indi_folder_row_header='indicatorList', infra_indi_folder_row_header='indicatorList',  # ???
    selection_operator = "selRoulette (Roulette Wheel)",
    pop_num=50, 
    cxpb=0.5, 
    mutpb=0.2,
    reduce=0.0,
    cc="Pearson correlation coefficient"):  # default value required to be provided if one of them is provided
    
    # DEBUG PURPOSE
    # print(f'infra_indi_folder_path: {infra_indi_folder_path}')
    # print(f'eco_indi_folder_path: {eco_indi_folder_path}')
    # print(f'stats_folder_path: {stats_folder_path}')
    # print(f'eco_indi_folder_row_header: {eco_indi_folder_row_header}')
    # print(f'infra_indi_folder_row_header: {infra_indi_folder_row_header}')

    # move GA declaration for FitnessMax & Individual to TOP
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # Input Options
        # LATER
    path = infra_indi_folder_path
    compare_path = eco_indi_folder_path
    stats_path = stats_folder_path
    if not os.path.exists(stats_path):  # create new folder if not exists
        os.makedirs(stats_path)
    
    # eco_indi_folder_row_header = 'Indicator list'
    if eco_indi_folder_row_header == "Indicator List":
        inputRowHeader = "indicatorList"
    # else:  # LATER
    # infra_indi_folder_row_header = 'Indicator list'
    if infra_indi_folder_row_header == "Indicator List":
        compareRowHeader = "indicatorList"
    # else:  # LATER

    global POP_NUM, CXPB, MUTPB, REDUCE, CC
    POP_NUM = pop_num  # *** add to sidebar
    CXPB = cxpb  # PREV: 0.5  # *** add to sidebar
    MUTPB = mutpb  # PREV: 0.5  # *** add to sidebar
    REDUCE = reduce
    CC = cc

    print(POP_NUM)
    print(type(POP_NUM))
    
    


    doLogging = True  # *** add to sidebar
    # minYear = 1970  # *** add to sidebar
    # maxYear = 2019  # PREV: 16, 00 -> DON'T KNOW WHY NOT WORK  # *** add to sidebar
    minIndicators = 1  # *** add to sidebar
    # [START not_use_coz_have_set_default_value]
    """
    if not directory1Select.folder_path:
        popupmsg("Please select input path for Select Input Folder ! ")  # *** -> warning
    else:
        path = "%s/"%(directory1Select.folder_path)
    
    if not directory2Select.folder_path:
        popupmsg("Please select compare path for Select Compare Folder ! ")
    else:
        compare_path = "%s/"%(directory2Select.folder_path)

    if not directory3Select.folder_path:
        popupmsg("Please select output path for Select Output Folder ! ")
    else:
        stats_path = "%s/"%(directory3Select.folder_path)
    """
    # [END not_use_coz_have_set_default_value]

    # [START get_the_user_input_from_sidebar]
    """global inputRowHeader
    inputRowHeader = str(v.get())

    global compareRowHeader
    compareRowHeader = str(v1.get())

    global POP_NUM

    global CXPB

    global MUTPB"""
    # [END get_the_user_input_from_sidebar]

    
    if(doLogging):
        logFile = open(stats_path + "logs.txt", "w")
        simpleLog = open(stats_path +"logs_simplified.txt","w")
    
    
    getFileExtension(path)
    getFileExtension(compare_path)
    #Validate row header inside file
    getHeadersbyYear(path,inputRowHeader)
    getHeadersbyYear(compare_path,inputRowHeader)

    # creates a giant 2D array to lookup operands == data in time series
    # there's no year in the first column
    headers = getHeaders(path, inputRowHeader)
    
    files = os.listdir(compare_path) 
    
    # num of runs controls how many times we are going to run ga on
    # the same compare indicator so that we can collect some stats
    # in collectiveStats file
    num_of_runs = 3
    # collectiveStats_headers = "mean, standard deviation, minimum CC, indicators, maximum CC, indicators\n"  # with space
    collectiveStats_headers = "Mean,Std,Min Fitness,Min Infrastructure Indicator,Max Fitness,Max Infrastructure Indicator\n"
    
    
    for filename in files:    #loop each of the files

        #collectiveStats records the best individual from individual ga runs
        #for each compare_indicators
        collectiveStats = open(stats_path + "ALL_"+filename, "w")
        collectiveStats.write(collectiveStats_headers)
        for i in range(num_of_runs):
            compare_data_matrix = []    
    
            statsFile = open(stats_path + "evolution"+str(i+1) + "_" + filename, "w")
            #writes the first line in stats file for column headers
            # statsFile.write('"Generation", "Average", "Std", "Min_Fitness", "Min_Individual", "Max_Fitness", "Max_Individual"\n')  # Beta Testing -> to get rid of the extra initial white spaces => NOT WORKING with space & double quote ("A", "B")
            statsFile.write("Generation,Mean,Std,Min Fitness,Min Infrastructure Indicator,Max Fitness,Max Infrastructure Indicator\n")
            
            #row by column matrix
            data_matrix =  [[0 for x in range(len(headers))] for y in range(maxYear - minYear + 1)]
            
            if(str(inputRowHeader) == "indicatorList"):
                # popupmsg(f"Bfr) Path: {path} \n Data Matrix: {data_matrix} \n Header Length: {len(headers)}")
                parseDataFile(path, data_matrix, len(headers), False)
                # popupmsg(f"Aft) Path: {path} \n Data Matrix: {data_matrix} \n Header Length: {len(headers)}")
            elif(str(inputRowHeader) == "yearList"):
                parseDataFilebyYear(path, data_matrix, inputRowHeader, len(headers), False)

            #gets the comparison data
            compare_headers = getHeadersFromFile(compare_path, filename, compareRowHeader)
            compare_data_matrix = [[0 for x in range(len(compare_headers))] for y in range(maxYear - minYear + 1)]

            
            
            if(str(inputRowHeader) == "indicatorList"):
                parseSingleDataFile(compare_path, filename, compare_data_matrix, len(compare_headers), True)    
            elif(str(inputRowHeader) == "yearList"):
                parseSingleDataFilebyYear(compare_path, filename, compare_data_matrix, compareRowHeader, len(compare_headers), True)

            # [START run_startGA_function]
            #generate random chromosomes
            line = startGA(headers, compare_headers, data_matrix, compare_data_matrix, maxYear - minYear + 1, len(headers), selection_operator, logFile, simpleLog, statsFile)
            collectiveStats.write(line)
            # [END run_startGA_function]
        collectiveStats.write("\n* Infrastructure Indicators = (randomly picked by Genetic Algorithm & showed in the columns of Infrastructure Indicator)")
        collectiveStats.write("\n* Economic Indicator = GDP(LCU); GDP(USD); Inflation Rate; Reserves assets; Net International Investment Position (RM million); Stock Share Commodity Brokers and Foreign Exchange Services Salaries & Wages Paid RM ('000); Broad money (% of GDP); Domestic credit provided by financial sector (% of GDP); Domestic credit to private sector (% of GDP); Stocks traded turnover ratio of domestic shares (%); Stocks traded total value (% of GDP); Leading Index(2005=100); Coincident Index(2005=100&2005=100); Lagging Index(2005=100)")  # add economic indicator into collective stats file
        collectiveStats.close()
    if(doLogging):
        logFile.close()
        simpleLog.close()
    source = r"%s" % stats_path
    destination = r"/Users/zrhun/FYP Topic - Data Science (Analysis M'sia Finance)/FYPsystem/previous/"  # ***
    consolidate(source,destination)
    print("## All done!")
    # st.write("All done!")


def resolve_path(filename, destination_dir, fileExt):
    string1 = str(filename)
    substring1 = "ALL_"
    substring2= "evolution1_"
    substring3= "evolution2_"
    substring4= "evolution3_"
    newName = ""
    if string1.find(substring1) == 0:
        newName = substring1 + "previous" + fileExt

    elif string1.find(substring2) == 0:
        newName = substring2 + "previous" + fileExt

    elif string1.find(substring3) == 0:
        newName = substring3 + "previous" + fileExt

    elif string1.find(substring4) == 0:
        newName = substring4 + "previous" + fileExt

    else:
        newName = filename

    dest = os.path.join(destination_dir, newName)
    return dest

def consolidate(source, destination):
    if not os.path.exists(destination):
        os.makedirs(destination)
    for root, dirs, files in os.walk(source):
        for f in files:
            #if f.lower().endswith(extension):
            source_path = os.path.join(root, f)
            fileExtension = os.path.splitext(source_path)[1]
            destination_path = resolve_path(f, destination, fileExtension)
            shutil.copyfile(source_path, destination_path)

def getFileExtension(folder):
    for root, dirs, files in os.walk(folder):
        for filename in files:
            source_path = os.path.join(root, filename)
            fileExtension = os.path.splitext(source_path)[1]
            if(str(fileExtension) != ".csv"):
                popupmsg("%s is not a type of csv file! " % source_path)

def startGA(headers, compare_headers, matrix, compare_matrix, rowNum, colNum, selection_operator, logFile, simpleLog, statsFile):
    global localCXPB
    localCXPB = CXPB

    line_to_write = ""
    # To assure reproductibility, the RNG seed is set prior to the items
    # dict initialization. It is also seeded in main().
    random.seed(64)
    
    #weights=(1.0,) indicates that we only seek to maximize the only 1 objective function
        # move GA declaration for FitnessMax & Individual to TOP
    '''creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)'''
    
    '''
    /Users/zrhun/opt/anaconda3/lib/python3.8/site-packages/deap/creator.py:138: RuntimeWarning: A class named 'FitnessMax' has already been created and it will be overwritten. Consider deleting previous creation of that class or rename it.
    warnings.warn("A class named '{0}' has already been created and it "

    /Users/zrhun/opt/anaconda3/lib/python3.8/site-packages/deap/creator.py:138: RuntimeWarning: A class named 'Individual' has already been created and it will be overwritten. Consider deleting previous creation of that class or rename it.
    warnings.warn("A class named '{0}' has already been created and it "
    '''
    
    toolbox = base.Toolbox()
    
    #max number of possible indicator combinations being colNum/2
    toolbox.register("chromosome", initIndividual, creator.Individual, colNum, minIndicators, int(colNum/2))  # *** (last 4 -> modified parameters)   
    toolbox.register("population", tools.initRepeat, list, toolbox.chromosome)  
    
    #all of the evaluation functions and operators
    toolbox.register("evaluate", evaluateInd)

    # register the crossover operator
    toolbox.register("mate", tools.cxTwoPoint)
    # a crossover may potentially introduce duplicate indicator, so
    # we need to remove it from an individual
    toolbox.register("mate_correction", mateCorrectionFunc)

    # register a mutation operator with a probability to
    # flip each attribute/gene of 0.05
    #toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("mutate", mutationFunc, colNum)

    if selection_operator == "selRoulette (Roulette Wheel)":
        toolbox.register("select", tools.selRoulette)
        
    elif selection_operator == "selTournament (Elitism)":
        # operator for selecting individuals for breeding the next
        # generation: each individual of the current generation
        # is replaced by the 'fittest' (best) of three individuals
        # drawn randomly from the current generation.
        toolbox.register("select", tools.selTournament, tournsize=3)  # tournsize=6, 9

    elif selection_operator == "selBest":
        toolbox.register("select", tools.selBest)  # *** -> is it better than selTournament

    # Variable keeping track of the number of generations
    g = 0

    #run GA
    pop = toolbox.population(n=POP_NUM)
    
    print("Start of evolution\n")
    # st.write("Start of evolution")
    # Evaluate the entire population
    #fitnesses = list(map(toolbox.evaluate, pop))
    fitnesses = [toolbox.evaluate(ind, matrix, compare_matrix, headers, compare_headers,logFile,g) for ind in pop]

    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    # print("  Evaluated %i individuals\n" % len(pop))  # LONG SYNTAX
    print(f"  Evaluated {len(pop)} individuals\n")
    # st.write("  Evaluated %i individuals" % len(pop))
    logFile.write("  Evaluated %i individuals \n" % len(pop))

    # Extracting all the fitnesses of 
    fits = [ind.fitness.values[0] for ind in pop]
   
    length = len(pop)
    mean = sum(fits) / length
    sum2 = sum(x*x for x in fits)
    std = abs(sum2 / length - mean**2)**0.5
    index_min = np.argmin(fits)
    index_max = np.argmax(fits)
   
    logFile.write("min fitness: " + str(min(fits)) + "     max fitness: " + str(max(fits)) + "     mean fitness: " + str(mean) + "\n")
    if(doLogging):
        logResult(headers, simpleLog, pop, g, min(fits), max(fits), mean)

    # line_to_write = "%s, %s, %s, %s,\"%s\", %s,\"%s\"\n" % (g, mean, std, min(fits), pop[index_min], max(fits), pop[index_max])
    line_to_write = "%s, %s, %s, %s,\"%s\", %s,\"%s\"\n" % (g, mean, std, min(fits), convertToNames(headers, pop[index_min]), max(fits), convertToNames(headers, pop[index_max]))  # Convert to name
    statsFile.write(line_to_write)


    # Begin the evolution
    while max(fits) < 100 and g < POP_NUM:  # Stop Criteria?; 100? ***
        # TESTING - to reduce the pb gradually
        '''# CXPB *= 0.9
        # MUTPB *= 0.9                
        CXPB = CXPB * 0.9
        # MUTPB = MUTPB * 0.9

        # newCXPB = CXPB
        print(f"CXPB = {CXPB}")
        # newCXPB = newCXPB * 0.9
        print(f"CXPB * 0.9 = {CXPB}")'''
        # localCXPB = localCXPB*0.9  # hardcode
        localCXPB = localCXPB*(1-REDUCE)
        # print(f"localCXPB = {localCXPB}")  # DEBUG PURPOSE


        # A new generation
        g = g + 1
        # print("-- Generation %i --" % g)  # LONG SYNTAX
        print(f"-- Generation {g} --\n")
        # st.write("-- Generation %i --" % g)
        
        # Select the next generation individuals
        if selection_operator == "selRoulette (Roulette Wheel)":
            offspring = toolbox.select(pop, len(pop))  # 5  # selRoulette
        elif selection_operator == "selTournament (Elitism)":
            offspring = toolbox.select(pop, len(pop))  # selTournament
        elif selection_operator == "selBest":
            offspring = toolbox.select(pop, len(pop))  # 5  # selBest

        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
        
            # cross two individuals with probability CXPB
            if random.random() < localCXPB:
                if(len(child1) > 1 and len(child2) > 1):
                    toolbox.mate(child1, child2)
                    child1 = toolbox.mate_correction(child1)
                    child2 = toolbox.mate_correction(child2)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

        fitnesses = [toolbox.evaluate(ind, matrix, compare_matrix, headers, compare_headers,logFile,g) for ind in pop]
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        print(f"  Evaluated {len(invalid_ind)} individuals\n")
        # st.write("  Evaluated %i individuals" % len(invalid_ind))
        logFile.write("  Evaluated %i individuals" % len(invalid_ind))
        #The population is entirely replaced by the offspring
        pop[:] = offspring
        
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        index_min = np.argmin(fits)
        index_max = np.argmax(fits)
        
        #outputs min max individuals to the stats file
        line_to_write = "%s, %s, %s, %s,\"%s\", %s,\"%s\"\n" % (g, mean, std, min(fits), convertToNames(headers, pop[index_min]), max(fits), convertToNames(headers, pop[index_max]))
        statsFile.write(line_to_write)

        logFile.write("\n min fitness: " + str(min(fits)) + "     max fitness: " + str(max(fits)) + "     mean fitness: " + str(mean) + "\n")
        if(doLogging):
            logResult(headers, simpleLog, pop, g, min(fits), max(fits), mean)

        print(f"  Min = {min(fits)}\n")
        print(f"  Max = {max(fits)}\n")
        print(f"  Avg = {mean}\n")
        print(f"  Std = {std}\n")
        ''' st.write(f"""Min {min(fits)}
                    Max {max(fits)}
                    Avg (mean)
                    Std {std}""") '''
    
    print("-- End of (successful) evolution --\n")
    # st.write("-- End of (successful) evolution --")
    logFile.write("-- End of (successful) evolution --")
    simpleLog.write("-- End of (successful) evolution --")

    best_ind_l = tools.selBest(pop, 1)[0]

    # print(f"***best_ind_l = {headers[ind] for ind in best_ind_l:}")  # WHY NOT WORK
    # for ind in best_ind_l:  # DEBUG PURPOSE
    #     print(f"***best_ind_l = {headers[ind]}\n")
        
    writeIntoHalloffame(best_ind_l, headers)
    
    # print("Best individual is best_ind_l, best_ind_l.fitness.values\n")  # DEBUG purpose
    # st.write("Best individual is %s, %s" % (best_ind_l, best_ind_l.fitness.values))
    # logFile.write("Best individual is %s, %s \n" % (convertToNames(headers, idx) for idx in best_ind_l), best_ind_l.fitness.values)  # best ind -> convert to name from idx
    # logFile.write("Best individual is %s, %s \n" % (convertToNames(headers, best_ind_l)), (best_ind_l.fitness.values))  # best ind -> convert to name from idx
    logFile.write(f"Best individual is {convertToNames(headers, best_ind_l)}, {best_ind_l.fitness.values} \n")  # best ind -> convert to name from idx  # WHY NOT WORK


    # simpleLog.write("Best individual is %s, %s \n" % (convertToNames(headers, idx) for idx in best_ind_l), best_ind_l.fitness.values)  # best ind -> convert to name from idx
    # simpleLog.write("Best individual is %s, %s \n" % convertToNames(headers, best_ind_l), best_ind_l.fitness.values)  # best ind -> convert to name from idx
    simpleLog.write(f"Best individual is {convertToNames(headers, best_ind_l)}, {best_ind_l.fitness.values} \n")  # best ind -> convert to name from idx
    
    statsFile.write("\n* Infrastructure Indicators = (randomly picked by Genetic Algorithm & showed in the columns of Infrastructure Indicator)")
    statsFile.write("\n* Economic Indicator = GDP(LCU); GDP(USD); Inflation Rate; Reserves assets; Net International Investment Position (RM million); Stock Share Commodity Brokers and Foreign Exchange Services; Salaries & Wages Paid RM ('000); Broad money (% of GDP); Domestic credit provided by financial sector (% of GDP); Domestic credit to private sector (% of GDP); Stocks traded turnover ratio of domestic shares (%); Stocks traded total value (% of GDP); Leading Index(2005=100); Coincident Index(2005=100&2005=100); Lagging Index(2005=100)")  # add economic indicator into stats file
    statsFile.close()
    # line_to_write = "%s, %s, %s,\"%s\", %s,\"%s\"\n" % (mean, std, min(fits), pop[index_min], max(fits), pop[index_max])
    line_to_write = "%s, %s, %s,\"%s\", %s,\"%s\"\n" % (mean, std, min(fits), convertToNames(headers, pop[index_min]), max(fits), convertToNames(headers, pop[index_max]))
    return line_to_write

def convertToNames(headers, offset_list):
    result = ""
    for offset in offset_list:
        result += headers[offset] + ";"
    """print(f"offset_list: {offset_list}")  # DEBUG PURPOSE
    print(f"headers: {headers}")
    print(f"result: {result}")"""
    return result[:-1]  # eliminate the last ";"

def writeIntoHalloffame(bestFittestIndL, headers):
    # halloffame = {ii1 : {"frequency": 123, "max_fitness": [0.6574]}, 
    #               ii2 : {"frequency": 123, "max_fitness": [0.6574]}}
    if "ranking.pkl" not in os.listdir():
        halloffame = {}
        with open("ranking.pkl", "wb") as f:
            pickle.dump(halloffame, f)  # create an empty dict in the newly created file
    else:
        with open ("ranking.pkl", "rb") as f:
            halloffame = pickle.load(f)

    # sepate d ind from list - WRONG concept
    # for bestFittestInd in bestFittestIndL:
    #     if headers[bestFittestInd] not in halloffame:
    #         #halloffame[headers[bestFittestInd]] = 0
    #         halloffame[headers[bestFittestInd]] = {"frequency": 0, "max_fitness": []}

    #     halloffame[headers[bestFittestInd]]["frequency"] += 1
    #     halloffame[headers[bestFittestInd]]["max_fitness"].append(bestFittestIndL.fitness.values[0])

    # don't sepate d ind from list
    bestFittestIndNameL = convertToNames(headers, bestFittestIndL)
    if bestFittestIndNameL not in halloffame:
        #halloffame[headers[bestFittestInd]] = 0
        halloffame[bestFittestIndNameL] = {"frequency": 0, "max_fitness": []}

    halloffame[bestFittestIndNameL]["frequency"] += 1
    halloffame[bestFittestIndNameL]["max_fitness"].append(bestFittestIndL.fitness.values[0])
    
    # halloffame = dict(sorted(halloffame.items(), key=lambda item: item[1]["frequency"], reverse=True))  # frequency
    halloffame = dict(sorted(halloffame.items(), key=lambda item: item[1]["max_fitness"], reverse=True))  # max_fitness
    # *** -> both frequency & max_fitness
    
    with open("ranking.pkl", "wb") as f:
        pickle.dump(halloffame, f)
    
    # DEBUG purpose
    """with open ("ranking.pkl", "rb") as f:
        halloffame = pickle.load(f)"""
    
    # print(f"halloffame = {halloffame}\n")  # DEBUG purpose

# a crossover may potentially introduce duplicate indicator, so
# we need to remove it from an individual
def mateCorrectionFunc(individual):
    # the list cannot be created directly since it has the fitness attribute
    # so we will create an individual then assign fitness to it
    fitness = individual.fitness
    individual = creator.Individual(list(set(individual)))
    individual.fitness = fitness
    return individual

def mutationFunc(maxRange, individual):
    offset = random.randint(0, len(individual)-1)
    val = random.randint(0, maxRange-1)
    while(any(val == elem for elem in individual)):
        val = random.randint(0, maxRange-1)
    individual[offset] = val    
    return 

def evaluateInd(individual, l_matrix, compare_data_matrix,headers,compare_headers,logFile,generationNumber):
    total = 0.0
    total_two_tail = 0.0
    compareDataMatrixColNum = len(compare_data_matrix[0])
    individualAverage = 0
    PvalueAverage = 0
   
    logFile.write("---- Generation: " + str(generationNumber) + " ----\n")
    # logFile.write("---- Individual: " + str(individual) + " ----\n")
    logFile.write("---- Individual: " + convertToNames(headers, individual) + " ----\n")  # Convert to name
    
    #iterate through the genes in an individual
    for columnOffset in individual:
        # gets the list from matrix and find the start and end year where data isn't 0 as start and end
        start_row_index, end_row_index = findStartAndEndIndex(l_matrix, columnOffset)
        
        seriesData = getSubseriesFromMatrix(l_matrix, start_row_index, end_row_index, columnOffset)
        currentIndicatorAverage = 0
        total = 0
        total_two_tail = 0
        for i in range(compareDataMatrixColNum):
            subseries = getSubseriesFromMatrix(compare_data_matrix, start_row_index, end_row_index, i)
            
            if CC == "Pearson correlation coefficient":
                #use the start and end year to compute pearson CC on all series in compare_data
                cc, two_tail = pearsonr(seriesData, subseries)
            if CC == "Spearman correlation coefficient":  # Spearman correlation coefficient rank-order correlation coefficient.
                cc, two_tail = spearmanr(seriesData, subseries)
            if CC == "Kendall correlation coefficient":  # Kendall correlation coefficient’s tau, a correlation measure for ordinal data.
                cc, two_tail = kendalltau(seriesData, subseries)


            #NOTE: Debug use only
            #print(seriesData)
            #print(subseries)
            #if(two_tail == 1.0):
            #    print(str(start_row_index) + " " + str(end_row_index) + " " + str(columnOffset))
            '''print(f"columnOffset = {columnOffset}\n")  # DEBUG PURPOSE
            print(f"seriesData = {seriesData}\n")
            print(f"subseries for {i} = {subseries}\n\n")
            print(f"cc, two_tail = {cc, two_tail}")'''

            if(doLogging):
                maxYeardiff = len(l_matrix) - end_row_index
                logPopulation(logFile, headers, compare_headers, individual, columnOffset, (i), seriesData, subseries, start_row_index, maxYeardiff, cc, two_tail)

            total += cc
            total_two_tail += two_tail
        currentIndicatorAverage = total / (compareDataMatrixColNum)
        individualAverage += currentIndicatorAverage

        currentPvalue = total_two_tail / (compareDataMatrixColNum)
        PvalueAverage += currentPvalue
        
    #get the average
    individualAverage = individualAverage/len(individual)
    logFile.write("Average fitness for current population: " + str(individualAverage) + "\nAverage p_value: " + str(PvalueAverage) + "\n")
    return individualAverage, #the comma is VERY IMPORTANT!!!


def getSubseriesFromMatrix(l_matrix, startRIndex, endRIndex, columnOffset):
    l_list = []
    for i in range(startRIndex,endRIndex+1):
        l_list.append(l_matrix[i][columnOffset])
    return l_list


def findStartAndEndIndex(l_matrix, columnOffset):
    startIndex = 0
    endIndex = len(l_matrix) - 1
    for i in range(len(l_matrix)):
        if(l_matrix[i][columnOffset] != minInt):
            startIndex = i
            break
    for i in range(len(l_matrix)-1,0,-1):
        if(l_matrix[i][columnOffset] != minInt):
            endIndex = i
            break
    return startIndex, endIndex


def initIndividual(icls, totalOffsets, minLength, maxLength):
    ind = creator.Individual(np.random.choice(totalOffsets,random.randint(minLength,maxLength), replace = False))
    return ind    


def parseSingleDataFile(folder, file, l_matrix, col_num, ignoreZero):
    #col_num needs to be +1 since the getHeaders() ignores the year column
    total_col_num = col_num + 1
    col = 0
    col_increment = 0
    
    count = 0
    handle = open(folder+file,"r")
    line = handle.readline().replace("\n","")
    try:
        while line:
            if(len(line) == 0):
                break
            #ignore sthe first row as it names the columns (which has been read
            # by getHeaders()))
            if(count != 0):
                #ignores the first column since it's year, but we use it as index into
                #matrix just in case the year for this file doesn't start with minYear        
                array = line.split(",")
                col_increment = len(array)
                for idx,item in enumerate(array):
                    if(idx == 0): #year
                        index = int(array[idx]) - minYear                   
                    elif(idx < total_col_num):
                        if(array[idx] != None and len(array[idx]) > 0):
                            l_matrix[index][(idx-1)] = float(array[idx])
                        else:
                            if(ignoreZero):
                                l_matrix[index][(idx-1)] = 0
                            else:
                                l_matrix[index][(idx-1)] = minInt
            count = count + 1        
            line = handle.readline().replace("\n","")
        col = col + col_increment
        handle.close()
    except:
            popupmsg("1. Data format is incorrect!\nPlease ensure only number format is allowed!\n*Please remove comma within number")
          
        
def parseDataFile(folder, l_matrix, col_num, ignoreZero):
    #col_num needs to be +1 since the getHeaders() ignores the year column
    total_col_num = col_num + 1
    col = 0
    col_increment = 0
    #reads all .csv files, ignores the first column since it's YEAR
    files = os.listdir(folder)

    for file in files:
        count = 0
        handle = open(folder+file,"r")
        line = handle.readline().replace("\n","")
        #replace() string.replace(old, new, count)
        
        try:
            while line:
                if(len(line) == 0):
                    break
                #ignores the first row as it names the columns (which has been read
                # by getHeaders()))
                if(count != 0):
                    #ignores the first column since it's year, but we use it as index into
                    #matrix just in case the year for this file doesn't start with minYear        
                    array = line.split(",")
                    col_increment = len(array)
                    for idx,item in enumerate(array):
                        if(idx == 0): #year
                            index = int(array[idx]) - minYear                     
                        elif(idx < total_col_num):
                            if(array[idx] != None and len(array[idx]) > 0):
                                l_matrix[index][(idx-1)] = float(array[idx])  # <<<--- IndexError: list index out of range
                            else:
                                if(ignoreZero):
                                    l_matrix[index][(idx-1)] = 0
                                else:
                                    l_matrix[index][(idx-1)] = minInt
                count = count + 1        
                line = handle.readline().replace("\n","")
            col = col + col_increment
            handle.close()
        except:
            popupmsg("2. Data format is incorrect!\nPlease ensure only number format is allowed!\n*Please remove comma within number")
            


def parseSingleDataFilebyYear(folder, file, compare_matrix, rowHeader, col_num, ignoreZero):
    total_col_num = col_num + 1
    handle = open(folder+file,"r")
    year = []
    count = 0
    with handle as f:
        reader = csv.reader(f, skipinitialspace=True)
        for row in reader:
            #skip the first column as it is Year in str
            #the rest of the column will be year in int
            year = row[1:]
            break
        for row2 in reader:
            array = row2[1:]
            index = int(year[count]) - minYear

            try:
                for idx,item in enumerate(array):
                    if(idx < total_col_num):
                        if(array[idx] != None and len(array[idx]) > 0):
                            compare_matrix[idx][index] = float(array[idx])
                        else:
                            if(ignoreZero):
                                compare_matrix[idx][index] = 0
                            else:
                                compare_matrix[idx][index] = minInt
                count += 1
            except:
                popupmsg("3. Data format is incorrect!\nPlease ensure only number format is allowed!\n*Please remove comma within number")
          
    handle.close()


def parseDataFilebyYear(folder, l_matrix, rowHeader, col_num, ignoreZero):
    total_col_num = col_num + 1
    files = os.listdir(folder)
    for file in files:
        handle = open(folder+file,"r")
        year = []
        count = 0
        with handle as f:
            reader = csv.reader(f, skipinitialspace=True)
            for row in reader:
                #skip the first column as it is Year in str
                #the rest of the column will be year in int
                year = row[1:]
                break
            for row2 in reader:
                array = row2[1:]
                index = int(year[count]) - minYear

                try:
                    for idx,item in enumerate(array):
                        if(idx < total_col_num):
                            if(array[idx] != None and len(array[idx]) > 0):
                                l_matrix[idx][index] = float(array[idx])
                            else:
                                if(ignoreZero):
                                    l_matrix[idx][index] = 0
                                else:
                                    l_matrix[idx][index] = minInt
                    count += 1
                except:
                    popupmsg("4.Data format is incorrect!\nPlease ensure only number format is allowed!\n*Please remove comma within number")
        handle.close()


def getHeadersFromFile(folder, file, rowHeader):
    l_headers = []
    handle=open(folder + file, "r")
    if(str(rowHeader) == "indicatorList"):
        #ignores the first column since it's year
        array = handle.readline().split(",")[1:]
        for item in array:
            l_headers.append(item.replace("\n",""))
    if(str(rowHeader) == "yearList"):
        with handle as f:
            reader = csv.reader(f, skipinitialspace=True)
            #skip the header
            next(reader)
            for row in reader:
                l_headers.append(row[0])
    handle.close()
    return l_headers        

def getHeaders(folder,rowHeader):
    l_headers = []
    files = os.listdir(folder)
    for file in files:
        handle = open(folder + file, "r")
        if(str(rowHeader) == "indicatorList"):
            #ignores the first column since it's year
            array = handle.readline().split(",")[1:]
            
            for item in array:
                l_headers.append(item.replace("\n",""))
        
        if(str(rowHeader) == "yearList"):
            with handle as f:
                reader = csv.reader(f, skipinitialspace=True)
                #skip the header
                next(reader)
                for row in reader:
                    l_headers.append(row[0])
        handle.close()
    return l_headers

def getHeadersbyYear(folder,rowHeader):
    global minYear
    global maxYear
    files = os.listdir(folder)
    for file in files:
        handle = open(folder + file, "r")
        l_headers_year = []
        if(str(rowHeader) == "indicatorList"):
            with handle as f:
                
                reader = csv.reader(f, skipinitialspace=True)
                #skip the header
                next(reader)
                for row in reader:
                    l_headers_year.append(row[0])
        if(str(rowHeader) == "yearList"):
            #ignores the first column since it's year
            array = handle.readline().split(",")[1:]
            for item in array:
                l_headers_year.append(item.replace("\n",""))

        try:
            minYear = int(l_headers_year[0])       
            maxYear = int(l_headers_year[-1])

        except:
            popupmsg("Please check your row header! ")

        """
        if(int(l_headers_year[0]) != minYear):
            popupmsg("This is not Min Year! ")
        if(int(l_headers_year[-1]) != maxYear):
            popupmsg("This is not Max Year! ")
        """
        handle.close()
    return l_headers_year


def logPopulation(logFile, headers, compare_headers, ind, offset, compareidx, input_matrix, compare_matrix, start, end, cc, pvalue):
        logFile.write("input subset: ['" + str(headers[offset]) + "']     compare subset: ['" + str(compare_headers[compareidx]) + "']     min year: ['" + str(minYear+start) + "']     max year: ['" + str(maxYear-end+1) + "']     cc: ['" + str(cc) + "']     p_value: ['" + str(pvalue) + "']\n")
    
def logResult(headers, simpleLog, pop, generation,minf,maxf,meanf):
    simpleLog.write("---- Generation: " + str(generation) + " ----\n")
    for ind in pop:
        # simpleLog.write(str(ind) + " Fitness:"+ str(ind.fitness) + "\n")
        simpleLog.write(convertToNames(headers, ind) + " Fitness:"+ str(ind.fitness) + "\n")  # convert to name
    simpleLog.write("Min fitness: " + str(minf) + "\nMax fitness: " + str(maxf) + "\nMean fitness: " + str(meanf) + "\n")

# runGeneticAlgorithm()

# # 


# # 