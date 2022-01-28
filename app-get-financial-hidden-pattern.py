# This app is for educational purpose only. Insights gained is not financial advice. Use at your own risk!

#---------------------------------#
# Import lib
import streamlit as st  # web app framework
from PIL import Image  # image
from io import BytesIO
import pickle  # Save data of result
# ===
Image.MAX_IMAGE_PIXELS = None  # *** BETA ???TESTING -> load images larger than MAX_IMAGE_PIXELS with PIL
import pandas as pd  # dataframe (maybe for web scrap [table] LATER)
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.set_option.html
# pd.set_option('display.max_colwidth', -1)
# st.set_option('display.max_colwidth', None)
# ===
import altair as alt  # *** -> build line chart with >2 values on y axis and sorted X axis according string
import plotly.express as px  # plotly -> data visualization (EG: multi line graph)
# ===
# [START import_libraries_for_heatmap]
import matplotlib.pyplot as plt  # // subplot
# # Sets the MPLBACKEND environment variable to Agg inside Streamlit to prevent Python crashing.  # BETA TESTING: not sure useful/- => streamlit UserWarning: Starting a Matplotlib GUI outside of the main thread will likely fail. ==>> https://github.com/streamlit/streamlit/issues/469
# # Default: true
# fixMatplotlib = True
# ===
# import seaborn as sns  # NOT USE FOR NOW
# import numpy as np  # NOT USE FOR NOW
# [END import_libraries_for_heatmap]
# [START add_on_other_library_and_need_to_update_in_the_md]
import os

#---------------------------------#
# Import my custom lib
## run genetic algorithm (GA)
from genetic_algorithm import runGeneticAlgorithm, demo_function  # -> GOT BUG
    # intialGA  # *** -> CHANGE BETTER NAME
from modules.modified_heatmap import displayModifiedHeatmap  # display heatmap
from modules.streamlit_custom_function import *  # streamlit custom function

#---------------------------------#
# Side note
## abbreviation for var name
# infra -> infrastructure
# eco -> economic
# stats -> statistics
# indi -> indicator
# POP -> population
# NUM -> number
# cxpb -> crossover probability
# mutpb -> mutation probability

## reminder add_on_other_library_and_need_to_update_in_the_md
# container VS BETA container 
# # analysis here -> 122-126
# BETA WARNING: 
    # st.expander has graduated out of beta. On 2021-11-02, the beta_ version will be removed.
        # WARNING: st.beta_expander("About")
    
    # st.container has graduated out of beta. On 2021-11-02, the beta_ version will be removed. \n\n Before then, update your code from st.beta_container to st.container.

    # st.columns has graduated out of beta. On 2021-11-02, the beta_ version will be removed. \n\n Before then, update your code from st.beta_columns to st.columns.
        # WARNING: st.beta_columns(2) -> 

# StreamlitAPIException
    # streamlit.errors.StreamlitAPIException: `set_page_config()` can only be called once per app, and must be called as the first Streamlit command in your script.
## New feature (make sure to upgrade your streamlit library)
# pip install --upgrade streamlit 
    # pip (/pip3 [python3] for Mac)

# wording
    # - Graph 1 VS Genetic Algorithm Evolutionary Cycle 1

# RuntimeError: Data of size 154.4MB exceeds write limit of 50.0MB
    # readonly


#---------------------------------#
# Log GA Info (LOG)    # NOT USE FOR NOW
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

# def demo_function():
#     """
#     Just a sample function to show how it works.
#     :return:
#     """
#     for i in range(10):
#         # logging.warning(f'Counting... {i}')
#         # time.sleep(2)
#         # print('Time out...')
#         logging.warning(i)
#         time.sleep(2)
#         if i>2:
#             print("Continue?")

#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(layout="wide", page_title="EHTP",
page_icon="Financial.ico")  # OPTION: initial_sidebar_state="expanded"

## Divide page to 3 columns (st.sidebar = sidebar, st and st = page contents)  # *** -> useless
# st.sidebar = st.sidebar  # *** -> useless

#---------------------------------#
# Global variable scope / namespace ??? (suit term)
selected_infra_indi_df = None

#---------------------------------#
# Cover Page
    # st.beta_container()
with st.container():
    ## Poster
    image = Image.open("Financial.ico")  # import image file
    image = image.convert('LA')  # grayscale
    st.image(image, width=450, caption="Extracting Hidden Trends and Patterns From Financial Data Series")  # set image width to 500  # *** -> size & position

    ## Title
    st.title('Extracting Hidden Trends and Patterns from Financial Dataseries')

    ## Description
    st.markdown("""
    This app help us to find extracting hidden trends and patterns from financial dataseries in fast speed!!!

    This app retrieve the data from [eStatistik](https://newss.statistics.gov.my/newss-portalx/ep/epProductFreeDownloadSearch.seam)
    """)

#---------------------------------#
# About
with st.container():
    # expander_bar = st.expander("About")
    expander_bar = st.expander(label="About", expanded=False)
    with expander_bar:
        st.markdown("<center><iframe width='560' height='315' src='https://www.youtube.com/embed/Drej348tZjc' title='YouTube video player' frameborder='0' allow='accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture' allowfullscreen></iframe></center>", unsafe_allow_html=True)
        
        expander_bar.markdown("""
        * **Python libraries:** streamlit, Image, pandas, base64, matplotlib, seaborn, numpy
        * **Data source:** [eStatistik](https://newss.statistics.gov.my/newss-portalx/ep/epProductFreeDownloadSearch.seam)
        * **Formula of Pearson correlation coefficient:**
        """)
        # * **Credit:** Dr. Mike Ong Teong Joo and Ang Sze Sin  # LATER
        # Web scraper adapted from the Medium article *[Web Scraping Crypto Prices With Python](https://towardsdatascience.com/web-scraping-crypto-prices-with-python-41072ea5b5bf)* written by [Bryan Feng](https://medium.com/@bryanf). 

        # SOURCE: Pearson correlation coefficient -> 
        # https://www.google.com/search?q=pearson+correlation+coefficient&oq=pearson+correlation+&aqs=chrome.0.35i39j69i57j35i39j69i59j0i512j0i131i433i512j69i60l2.7583j0j7&sourceid=chrome&ie=UTF-8
            # https://www.google.com/search?q=formula+of+pearson+correlation+coefficient&oq=Formula+of+Pearson+correlation+coefficient&aqs=chrome.0.0i512j0i22i30l2j0i390l5.2959j0j7&sourceid=chrome&ie=UTF-8
        st.latex(r"""
            r =\frac{\sum\left(x_{i}-\bar{x}\right)\left(y_{i}-\bar{y}\right)}{\sqrt{\sum\left(x_{i}-\bar{x}\right)^{2} \sum\left(y_{i}-\bar{y}\right)^{2}}}
            """)
        # st.latex("""  =   correlation coefficient x_{i}   =   values of the x-variable in a sample \bar{x}    =   mean of the values of the x-variable y_{i}  =   values of the y-variable in a sample \bar{y}    =   mean of the values of the y-variable
        # """)  # BETA TESTING -> eliminate 'r' & copy from Wiki directly
        st.latex(r"r    =   correlation coefficient")
        # st.write("=   correlation coefficient")  # ***
        st.latex(r"x_{i}    =   values of the x-variable in a sample")
        st.latex(r"\bar{x}  =   mean of the values of the x-variable")
        st.latex(r"y_{i}    =   values of the y-variable in a sample")
        st.latex(r"\bar{y}  =   mean of the values of the y-variable")

#---------------------------------#
# Sidebar + Main panel
with st.container():
    ## Sidebar - header
    # st.sidebar.title('Input Options')  # ***
    st.sidebar.header('Input Options')

    ## path - infrastructure indicator dataset folder path
    # Text: Upload infrastructure indicator files
    # Text: folder / files ? LATER
    # infra_indi_folder_path = st.sidebar.selectbox('Upload Input Folder', [os.getcwd
    # infra_indi_folder_path = st.sidebar.selectbox('Upload Input Folder', [os.getcwd() + '/infra_indi_data/'])  # *** ->  # LATER -> upload file & python os function auto add / => os.path.join(root, f)
    infra_indi_folder_path = os.getcwd() + '/infra_indi_data/'
    # st.write(os.path.join('input_data' + 'input_data'))  # *** -> WHY NOT WORKING

    ## path - economic indicator dataset folder path
        # Text: Upload economic indicator files
    # eco_indi_folder_path = st.sidebar.selectbox('Upload Compare Folder', [os.getcwd() + '/eco_indi_data/'])  # LATER
    eco_indi_folder_path = os.getcwd() + '/eco_indi_data/'
    # st.sidebar.write("***")  # ***

    ## path - output statistics folder path
    stats_folder_path = os.getcwd() + '/stats_data/'  # -> user can optionally download output files
    
    
    # analysis here
    ## Sidebar - file uploader
    # all_combined_infra_indi_df = False
    # if not all_combined_infra_indi_df:
    # uploaded_infra_indi_csv_file = st.sidebar.file_uploader("Upload your CSV file of infrastructure indicator", type=["csv"])
    # uploaded_infra_indi_csv_file = None  # REDUNDANT
    # if uploaded_infra_indi_csv_file == st.sidebar.file_uploader("Upload your CSV file of infrastructure indicator", type=["csv"]):  # REDUNDANT
    uploaded_infra_indi_csv_file = st.sidebar.file_uploader("Upload your CSV file of infrastructure indicator", type=["csv"])
    # st.sidebar.markdown("(Sample file)[]")
    # if uploaded_infra_indi_csv_file != None:  # REDUNDANT
    st.sidebar.markdown(get_binary_file_downloader_html(os.getcwd() + "/infra_indi_data/Infrastructure - dev_infra.csv", "Sample File"), unsafe_allow_html=True)  # SAME: st.sidebar.write
    # if uploaded_infra_indi_csv_file is not None:  # REDUNDANT
    if uploaded_infra_indi_csv_file:
        # *** read dataframe bfr override bfr override by the real file path
            # *** handle by adjust the parameter of runGeneticAlgorithm()
        selected_infra_indi_df = pd.read_csv(uploaded_infra_indi_csv_file, index_col=['Year'], sep=r',', skipinitialspace=True)  # *** -> handle error if the file has no year column (NOT a timeseries dataset)

        # save file to local storage
        # file_details = {"FileName": uploaded_infra_indi_csv_file.name, "FileType": uploaded_infra_indi_csv_file.type}
        # infra_indi_folder_path = uploaded_infra_indi_csv_file  # pass to runGeneticAlgorithm()  # NOT WORK -> require path for the runGeneticAlgorithm()
        infra_indi_folder_path = save_uploaded_file(uploaded_infra_indi_csv_file)




    uploaded_eco_indi_csv_file = st.sidebar.file_uploader("Upload your CSV file of economic indicator", type=["csv"])
    st.sidebar.write(get_binary_file_downloader_html(os.getcwd() + "/eco_indi_data/Economic - financial_dev.csv", "Sample File"), unsafe_allow_html=True)

    if uploaded_eco_indi_csv_file:
        eco_indi_folder_path = uploaded_eco_indi_csv_file  # pass to runGeneticAlgorithm()
        selected_eco_indi_df = pd.read_csv(uploaded_eco_indi_csv_file, index_col=['Year'], sep=r',', skipinitialspace=True)

    # analysis here

    # multiselect
    if not uploaded_infra_indi_csv_file:
        # index_col='Year'  # WORK for single column only
        all_combined_infra_indi_df = pd.read_csv(infra_indi_folder_path + 'Infrastructure - dev_infra.csv', index_col=['Year'], sep=r',', skipinitialspace=True)
        ### display filtered (user selected) infrastructure indicator
        infra_indi_option = sorted(list(all_combined_infra_indi_df.columns))  # *** -> unique
        selected_infra_indi = st.sidebar.multiselect("Select Infrastructure Indicator", infra_indi_option, infra_indi_option[:])  #10])

        # selected_infra_indi += 'year'  # NOT WORK for list; This ['year'] work, BUT append at last  # insert -> put at the specify place & put the back element back; append -> put the last of the last element place
        # selected_infra_indi.insert(0, "Year")  # *** <= index_col=['Year']
        selected_infra_indi_df = all_combined_infra_indi_df[selected_infra_indi]    

    if not uploaded_eco_indi_csv_file:
        all_combined_eco_indi_df = pd.read_csv(eco_indi_folder_path + 'Economic - financial_dev.csv', index_col=['Year'], sep=r',', skipinitialspace=True)
        ### display filtered (user selected) economic indicator
        eco_indi_option = sorted(list(all_combined_eco_indi_df.columns))  # *** -> unique
        selected_eco_indi = st.sidebar.multiselect("Select Economic Indicator", eco_indi_option, eco_indi_option[:])  #10])
        
        selected_eco_indi_df = all_combined_eco_indi_df[selected_eco_indi]

    ## Sidebar - row header for infrastructure indicator dataset folder path
    # st.sidebar.header('Folder Path')  # ***
    # st.sidebar.sidebar.<widget>
    infra_indi_folder_row_header = st.sidebar.radio('Row Header for Input Folder (Infrastructure Indicator)',['Indicator List','Year List'])  # -> radio button  # *** -> can't work with st.sidebar.sidebar.radio  # MAYBE COZ bug ([l]ist -> [L]ist)
    # a = st.sidebar.selectbox('R:',[1,2])  # -> dropdown menu

    ## Sidebar - row header for economic indicator dataset folder path
    eco_indi_folder_row_header = st.sidebar.radio('Row Header for Compare Folder (Economic Indicator)',['Indicator List','Year List'])
    # st.sidebar.write("***")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.sidebar.header('Genetic Algoritm Parameters')  # OPTIONL: GA Parameters

    ## Sidebar - population number
    ## Sidebar - correlation coefficient
    cc = st.sidebar.radio('Fitness Evaluation', ['Pearson correlation coefficient', 'Spearman correlation coefficient', 'Kendall correlation coefficient'])  # cc = st.sidebar.radio('correlation coefficient', ['Pearson', 'Spearman', 'Kendall'])

    ## Sidebar - selection operator
    selection_operator = st.sidebar.radio('Selection Operator',['selTournament (Elitism)','selRoulette (Roulette Wheel)','selBest'])  # *** -> set default to elitism
    
    # pop_num = st.sidebar.slider('Population Number', 0, 100, 100)  # OPTION: 50)  # LATER -> middle (all 3 align to the center) & const (/constant)  # change naming case from CAP -> low
    pop_num = st.sidebar.slider('Population Number', 1, 300, 100)  # OPTION: 150, †300) # SUIT?
    
    ## Sidebar - crossover probability
    cxpb = st.sidebar.slider('Crossover Probability', 0.0, 1.0, 0.5)  # LATER -> 0.0-1.0 (// 0.0-0.9) & const (/constant)

    ## Sidebar - mutation probability
    mutpb = st.sidebar.slider('Mutation Probability', 0.0, 1.0, 0.2)  # OPTION: 0.5)  # LATER -> 0.0-1.0 (// 0.0-0.9) & const (/constant)

    ## Sidebar - reduce crossover and mutation probability
    reduce = st.sidebar.slider('Reduce Crossover and Mutation Probability by Epoch', 0.0, 1.0, 0.0)
    # /INPUT OPTION

# COPY FROM HERE
#---------------------------------#
# display both of dataframes (infrastructure VS economic indicator)
## infrastructure indicator
with st.container():        
    # st.header("Infrastructure Indicator")
    st.markdown("<h1><center>Infrastructure Indicator</center></h1><br>", unsafe_allow_html=True)  # OPTION: center,h2,h2,center; h2,center,center,h2; h2,center,h2,center
    # col1_one_to_one, col2_one_to_one = st.columns(2)  # col1_one_to_one : col2_one_to_one = 1:1
    # col1_one_to_one, space_center, col2_one_to_one = st.columns([10,1,10])  # col1_one_to_one : col2_one_to_one = 1:1
    col1_one_to_one, space_center, col2_one_to_one = st.columns((10,1,10))  # col1_one_to_one : col2_one_to_one = 1:1

    try:
        with col1_one_to_one:  # dataframe
            selected_infra_indi_df = selected_infra_indi_df.interpolate(method='linear', limit_direction='both', axis=0)  # interpolate df (fill in null value)

            # st.write(type(all_combined_infra_indi_df[selected_infra_indi]))  # *** -> WHY NOT WORK
            ### download filtered (user selected) infrastructure indicator
            # selected_infra_indi_df = selected_infra_indi_df.reset_index(level=0)  # SAME as below / ALTERNATIVE to below
            selected_infra_indi_df.reset_index(level=0, inplace=True)  # convert dataframe index to column
            st.markdown(get_binary_file_downloader_html(selected_infra_indi_df, "Infrastructure - dev_infra.csv"), unsafe_allow_html=True)  # CANNOT without file extension (EG: csv)

            # st.dataframe(selected_infra_indi_df.style.highlight_max(axis=0))  # OPTION: , height=240  # *** -> sf.style  # PREV: st.dataframe(all_combined_infra_indi_df)
            # df = 
            st.dataframe(selected_infra_indi_df.set_index('Year').style.highlight_max(axis=0, color='#5429a3').format(precision=4), height=400)  # OPTION: , height=240, .format("{:.0}")  # *** -> sf.style  # PREV: st.dataframe(all_combined_infra_indi_df)
            # df.round(1)
            # df
            # axis=1  # ???
            # st.dataframe(all_combined_infra_indi_df[all_combined_infra_indi_df.columns.isin(selected_infra_indi)])  # WHY NOT WORK
            # st.dataframe(all_combined_infra_indi_df[all_combined_infra_indi_df.columns.isin(['Balance of trade', 'PPID', 'IPIC'])])  # WHY NOT WORK

            

        with col2_one_to_one:  # Multiple Line Graph (line chart)
            # REFERENCE of streamlit build line chart with two variables: https://discuss.streamlit.io/t/how-to-build-line-chart-with-two-values-on-y-axis-and-sorded-x-axis-acording-string/9490/3
            ### build line chart with >2 values on y axis and sorted X axis according string
            selected_infra_indi_df_melted = selected_infra_indi_df.melt("Year", var_name="Infrastructure Indices", value_name="Price")  # *** -> value_vars= // more suit wording (Price)?

            # selected_infra_indi_df_melted = selected_infra_indi_df_melted.dropna()
            # st.dataframe(selected_infra_indi_df_melted)  # DEBUG PURPOSE

            ## altair
            # chart = alt.Chart(selected_infra_indi_df_melted).mark_line().encode(
            #     x=alt.X("Year:N"),
            #     y=alt.Y("Price:Q"),
            #     color=alt.Color("Infrastructure Indices:N"),
            #     tooltip="Infrastructure Indices"
            # ).properties(title="Infrastructure Indices").interactive()
            # st._arrow_altair_chart(chart, use_container_width=True)
            
            ## streamlit - syntax sugar
            # st.line_chart(selected_infra_indi_df.set_index("Year"))  # *** -> year format?  # SAME: _arrow_line_chart

            ## plotly
            fig = px.line(selected_infra_indi_df.set_index("Year"))
            fig.update_layout(
                showlegend=False,
                autosize=False,
                # width=1500,
                # height=325,
                margin=dict(  # *** -> other new way
                    l=1,
                    r=1,
                    b=45,
                    t=45
                ),
                # font=dict(
                #     # color="#383635",
                #     # size=15
                # ),
                # paper_bgcolor=background_color,
                # paper_bgcolor="#000000",
            )
            fig.update_xaxes(type="category")

            # st.write(fig)
            st.plotly_chart(fig, use_container_width=True)
    
    except IndexError as e:  # IndexError: list index out of range
        # if selected_infra_indi == None:
        if not selected_infra_indi:
            st.sidebar.warning("**Infrastructure indicator** is not selected!")
            st.warning("""**Infrastructure indicator** is not selected! \n
            Steps to proceed:
            1. click the upper left sidebar icon to expand sidebar
            2. scroll to the dropdown menu of "Select Infrastructure Indicator"
            3. select your interest infrastructure indicator
            """)

## economic indicator
with st.container():        
    st.markdown("<h1><center>Economic Indicator</center></h1><br>", unsafe_allow_html=True)  # OPTION: center,h2,h2,center; h2,center,center,h2; h2,center,h2,center
    col1_one_to_one, space_center, col2_one_to_one = st.columns((10,1,10))  # col1_one_to_one : col2_one_to_one = 1:1

    try:
        with col1_one_to_one:  # dataframe
            selected_eco_indi_df = selected_eco_indi_df.interpolate(method='linear', limit_direction='both', axis=0)  # interpolate df (fill in null value)

            selected_eco_indi_df.reset_index(level=0, inplace=True)  # convert dataframe index to column
            st.markdown(get_binary_file_downloader_html(selected_eco_indi_df, "Economic - financial_dev.csv"), unsafe_allow_html=True)  # CANNOT without file extension (EG: csv)

            st.dataframe(selected_eco_indi_df.set_index('Year').style.highlight_max(axis=0, color='#5429a3').format(precision=4), height=400)  # OPTION: , height=240  # *** -> sf.style  # PREV: st.dataframe(all_combined_infra_indi_df)
            

        with col2_one_to_one:  # Multiple Line Graph (line chart)
            # REFERENCE of streamlit build line chart with two variables: https://discuss.streamlit.io/t/how-to-build-line-chart-with-two-values-on-y-axis-and-sorded-x-axis-acording-string/9490/3
            ### build line chart with >2 values on y axis and sorted X axis according string
            selected_eco_indi_df_melted = selected_eco_indi_df.melt("Year", var_name="Economic Indices", value_name="Price")  # *** -> value_vars= // more suit wording (Price)?

            ## plotly
            fig = px.line(selected_eco_indi_df.set_index("Year"))
            fig.update_layout(
                showlegend=False,
                autosize=False,
                margin=dict(  # *** -> other new way
                    l=1,
                    r=1,
                    b=45,
                    t=45
                ),
            )
            fig.update_xaxes(type="category")

            st.plotly_chart(fig, use_container_width=True)
    
    except IndexError as e:  # IndexError: list index out of range
        # if selected_infra_indi == None:
        if not selected_infra_indi:
            st.sidebar.warning("**Economic indicator** is not selected!")
            st.warning("""**Economic indicator** is not selected! \n
            Steps to proceed:
            1. click the upper left sidebar icon to expand sidebar
            2. scroll to the dropdown menu of "Select Economic Indicator"
            3. select your interest Economic indicator
            """)
    

## economic indicator 00
# with st.container():
#     st.markdown("<h1><center>Economic Indicator</center></h1></br>", unsafe_allow_html=True)
#     # col1_one_to_one, col2_one_to_one = st.columns(2)  # col1_one_to_one : col2_one_to_one = 1:
#     col1_one_to_one, space_center, col2_one_to_one = st.columns([10,1,10])  # col1_one_to_one : col2_one_to_one = 1:1

#     if not selected_eco_indi:
#         st.sidebar.warning("**Economic indicator** is not selected!")
#         st.warning("""**Economic indicator** is not selected! \n
#         Steps to proceed:
#         1. click the upper left sidebar icon to expand sidebar
#         2. scroll to the dropdown menu of "Select Economic Indicator"
#         3. select your interest economic indicator
#         """)
#     else:
#         with col1_one_to_one:  # dataframe
#             selected_eco_indi_df = selected_eco_indi_df.interpolate(method='linear', limit_direction='both', axis=0)  # interpolate df (fill in null value)

#             st.dataframe(selected_eco_indi_df.style.highlight_max(axis=0), height=240)
#             selected_eco_indi_df = selected_eco_indi_df.reset_index(level=0)  # convert dataframe index to column
#             ### download filtered (user selected) s
#             st.markdown(get_binary_file_downloader_html(selected_eco_indi_df, "Economic - financial_dev.csv"), unsafe_allow_html=True)

#         with col2_one_to_one:  # Multiple Line Graph (line chart)
#             ### build line chart with >2 values on y axis and sorted X axis according string 

#             #### alt ####
#             selected_eco_indi_df_melted = selected_eco_indi_df.melt("Year", var_name="Economic Indices", value_name="Price")

#             # chart = alt.Chart(selected_eco_indi_df_melted).mark_line().encode(
#             #     x=alt.X("Year:N"),
#             #     y=alt.Y("Price:Q"),
#             #     color=alt.Color("Economic Indices")
#             # ).properties(title="Economic Indicesabc")
#             # st.altair_chart(chart, use_container_width=True)   

#             #### Plotly ####
#             fig = px.line(selected_eco_indi_df.set_index("Year"))
#             fig.update_layout(
#                 autosize=False,
#                 width=500,
#                 # height=325,
#                 # margin=dict(  # *** -> other new way
#                 # ),
#                 # font=dict(
#                 #     # color="#383635",
#                 #     # size=15
#                 # ),
#                 # paper_bgcolor=background_color,
#                 # paper_bgcolor="#000000",
#             )
#             fig.update_xaxes(type="category")

#             # st.write(fig)
#             st.plotly_chart(fig, use_container_width=True)

#         st.write("---")


# file uploader

# ===
# OLD PLACE for Select Infrastructure and Economic Indicator
# ===

#---------------------------------#
# button - run algo & visualize data

## run genetic algorithm (GA)
with st.container():
    # if st.button('Start GA'):
    ga_btn = st.button('Run Computation by Genetic Algorithm')  # X->Computing
    if ga_btn:  # all st element: default value =  False, after get user input = True
        # st.header("Please wait for Genetic Algorithm running...") 
            # DEBUG PURPOSE
        # st.write(f"infra_indi_folder_path: {infra_indi_folder_path}")
        # st.write(f"eco_indi_folder_path: {eco_indi_folder_path}")
        # st.write(f"stats_folder_path: {stats_folder_path}")
        # st.write(f"eco_indi_folder_row_header: {eco_indi_folder_row_header}")
        # st.write(f"infra_indi_folder_row_header: {infra_indi_folder_row_header}")
        # st.write(f"pop_num: {pop_num}")
        # st.write(f"cxpb: {cxpb}")
        # st.write(f"mutpb: {mutpb}")
        
        # ga_log_text_area = st.text_area("Genetic Algorithm log")

        with st_stdout("success"), st_stderr("code"):
            """print(f'infra_indi_folder_path: {infra_indi_folder_path}')  # DEBUG PURPOSE
            print(f'eco_indi_folder_path: {eco_indi_folder_path}')
            print(f'stats_folder_path: {stats_folder_path}')
            print(f'eco_indi_folder_row_header: {eco_indi_folder_row_header}')
            print(f'infra_indi_folder_row_header: {infra_indi_folder_row_header}')"""
            # demo_function()
            runGeneticAlgorithm(infra_indi_folder_path, eco_indi_folder_path, stats_folder_path, eco_indi_folder_row_header, infra_indi_folder_row_header, selection_operator, pop_num, cxpb, mutpb, reduce, cc)
        #st.header('All Done!')  # popup msg
        # st.success('All Done!')

    #---------------------------------#
    # OLD PLACE for display heatmap
    #---------------------------------#

    #---------------------------------#
    # OLD PLACE for Custom Function
    #---------------------------------#

    #---------------------------------#
    # === 3 GA graphes - PyPlot two subplots ===
    ## Graph 1
    # graph1_df = pd.read_csv(os.getcwd() + "/stats_data/evolution1_Economic - financial_dev.csv", sep=r'\s*,\s*',header=0, encoding='ascii', engine='python')
    graph1_df = pd.read_csv(os.getcwd() + "/stats_data/evolution1_Economic - financial_dev.csv", sep=r',', skipinitialspace=True, skipfooter=2, engine="python")  # to get rid of the extra initial white spaces  #PREV: skipfooter=1  # *** -> sep
    ### Plot graph
    GA_graph1_fig = plt.figure()
    # *** -> line graph
        # ''' title='simple line example',
        # x_axis_label='x',
        # y_axis_label='y' '''
    # plt.subplot(211)  # ORI
    # ax = GA_graph1_fig.add_subplot(511)  # IN STREAMLIT
    ax = GA_graph1_fig.add_subplot(2,1,1)  # IN STREAMLIT
    #### Graph - Max Fitness
    # ax.plot(graph1_df['Generation'], graph1_df['Max Fitness'], color='blue')
    # ax.plot(graph1_df['Generation'], graph1_df['Max Fitness'], color='tab:blue', marker='o', label='Maximum Fitness')  # *** -> tab:
    ax.plot(graph1_df['Generation'], graph1_df['Max Fitness'], 'bo-', label='Maximum Fitness')  # mo-
    #### Graph - Average
    # ax.plot(graph1_df['Generation'], graph1_df['Mean'], color='tab:purple')   # OPTION: linestyle='--' (dash-line style)
    # ax.plot(graph1_df['Generation'], graph1_df['Mean'], color='tab:purple', marker='o')
    # ax.plot(graph1_df['Generation'], graph1_df['Mean'], color='r', marker='o', label='Average Fitness')
    ax.plot(graph1_df['Generation'], graph1_df['Mean'], 'go-', label='Average Fitness')  # ro-
    #### Graph - Min Fitness
    ax.plot(graph1_df['Generation'], graph1_df['Min Fitness'], 'ro-', label='Minimum Fitness')  # mo-
    ax.legend()  # display the label for subplot
    # my_legend()  # show the label along the line
    ax.set_title('Graph 1')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')
    # GA_graph1_fig

    ## Graph 2
    graph2_df = pd.read_csv(os.getcwd() + "/stats_data/evolution2_Economic - financial_dev.csv", sep=r',', skipinitialspace=True, skipfooter=2, engine="python")
    ### Plot graph
    GA_graph2_fig = plt.figure()
    ax = GA_graph2_fig.add_subplot(211)  # IN STREAMLIT
    #### Graph - Max Fitness
    ax.plot(graph2_df['Generation'], graph2_df['Max Fitness'], 'bo-', label='Maximum Fitness')
    #### Graph - Average
    ax.plot(graph2_df['Generation'], graph2_df['Mean'], 'go-', label='Average Fitness')
    #### Graph - Min Fitness
    ax.plot(graph2_df['Generation'], graph1_df['Min Fitness'], 'ro-', label='Minimum Fitness')

    ax.legend()  # display the label for subplot
    ax.set_title('Graph 2')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')

    ## Graph 3
    graph3_df = pd.read_csv(os.getcwd() + "/stats_data/evolution3_Economic - financial_dev.csv", sep=r',', skipinitialspace=True, skipfooter=2, engine="python")
    ### Plot graph
    GA_graph3_fig = plt.figure()
    ax = GA_graph3_fig.add_subplot(211)  # IN STREAMLIT
    #### Graph - Max Fitness
    ax.plot(graph3_df['Generation'], graph3_df['Max Fitness'], 'bo-', label='Maximum Fitness')
    #### Graph - Average
    ax.plot(graph3_df['Generation'], graph3_df['Mean'], 'go-', label='Average Fitness')
    #### Graph - Min Fitness
    ax.plot(graph2_df['Generation'], graph1_df['Min Fitness'], 'ro-', label='Minimum Fitness')
    ax.legend()  # display the label for subplot
    ax.set_title('Graph 3')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')

    ### Display all 3 GA graph
    # plt.show()  # ORI
    # st.write(GA_graph1_fig, GA_graph2_fig, GA_graph3_fig)  # IN STREAMLIT
    # st.write("Previous Computing Result") if ga_btn else st.write("Current Computing Result")
    if ga_btn:  # after click ga_btn
        # st.write("Current Computing Result") 
        st.write("<h2><center>Current Computed Result</center></h2></br>", unsafe_allow_html=True)

        
    
    else:  # before click ga_btn
        # Previous Computing Result
        st.write("<h2><center>Previous Computed Result</center></h2></br>", unsafe_allow_html=True) 

    # col_small_left, col_big_center, col_small_right = st.columns([1,5,1])
    col1_one_to_one, space_center, col2_one_to_one = st.columns([10,1,10])  # col1_one_to_one : col2_one_to_one = 1:1
    # with col1_one_to_one:
    #     st.dataframe(graph1_df)
    #     st.dataframe(graph2_df)
    #     st.dataframe(graph3_df)

    # with col2_one_to_one:
    #     st.write(GA_graph1_fig)
    #     st.write(GA_graph2_fig)
    #     st.write(GA_graph3_fig)
    st.write(" ")
    with col1_one_to_one:
        # pd.set_option('display.max_colwidth', None)
        # pd.set_option('display.max_columns', None)
        # pd.DataFrame(graph1_df)
        graph1_df = pd.DataFrame(graph1_df)
        # st.table(graph1_df)
        # st.dataframe(graph1_df)
        st.dataframe(graph1_df.style.hide_index())
        # st.dataframe(graph1_df.set_index("Generation"))
        # st.dataframe(graph1_df.to_string(index=False))
    with col2_one_to_one:
        st.write(GA_graph1_fig, use_container_width=True)
        # st.caption("<center>evolution1_Economic - financial_dev</center>", unsafe_allow_html=True)
        # st.caption("evolution1_Economic - financial_dev")
        st.caption("Genetic Algorithm Evolutionary Cycle 1")

    col1_one_to_one, space_center, col2_one_to_one = st.columns([10,1,10])  # col1_one_to_one : col2_one_to_one = 1:1
    st.write(" ")
    with col1_one_to_one:
        st.dataframe(graph2_df)
    with col2_one_to_one:
        st.write(GA_graph2_fig, use_container_width=True)
        st.caption("Genetic Algorithm Evolutionary Cycle 2")

    col1_one_to_one, space_center, col2_one_to_one = st.columns([10,1,10])  # col1_one_to_one : col2_one_to_one = 1:1
    st.write(" ")
    with col1_one_to_one:
        st.dataframe(graph3_df)
    with col2_one_to_one:
        st.write(GA_graph3_fig, use_container_width=True)
        st.caption("Genetic Algorithm Evolutionary Cycle 3")

    # read the logs simplified file
    with st.container():
        try:
            #with open (os.getcwd() + "/stats_data/logs_simplified.txt", "r") as logs_simplified:
                #logs_simplified_content = logs_simplified.read()
            with open (os.getcwd() + "/ranking.pkl", "rb") as f:
                ranking_content = pickle.load(f)

            print(f)  # GOT BUG

            #print(f"logs_simplified_content = {logs_simplified_content}")

            content = ""
            contentIdx = 1
            for k, vs in ranking_content.items():
                #content += f"{contentIdx}) {k} -> {v}\n"
                content += f"{contentIdx}) {k}\n"

                for k, v in vs.items():
                    content += f"\t{k}: {(list(set(v))[0] if len(set(v)) == 1 else set(v)) if type(v) == list else v}\t\t"
                content += "\n"

                contentIdx += 1
            
        except FileNotFoundError:
            content = "No ranking yet \n\n   It is your first run! 😁"

        ranking_text_area = st.text_area("Hall of Fame (Ranking of Infrastructure Indicator)", content, height=250)  # OPTION: Simplified logs file

        ranking_text_area = ranking_text_area.splitlines()
        # 
        ranking_text_area = "".join(ranking_text_area)
        
        # # RuntimeError: Data of size 154.4MB exceeds write limit of 50.0MB
        # # read the logs file
        # with open (os.getcwd() + "/stats_data/logs.txt", "r") as logs_simplified:
        #     logs_simplified_content = logs_simplified.read()
        
        # logs_text_area = st.text_area("Logs file", logs_simplified_content, height=250)

        # logs_text_area = logs_text_area.splitlines()
        # # 
        # logs_text_area = "".join(logs_text_area)
        

    # st.write('''<center>GA_graph1_fig</center>''', unsafe_allow_html=True)

    ### Download all 3 GA graph CSV file
    st.write("Evolutions of Economic in CSV file:")
    col1_one_to_one_to_one, col2_one_to_one_to_one, col3_one_to_one_to_one = st.columns([1,1,1])  # col1_one_to_one_to_one : col2_one_to_one_to_one : col3_one_to_one_to_one = 1:1:1
    # st.markdown("""<div style="text-align:center"><span style="float:left">Text Left BlahBlahBlahBlah</span><span style="float:right">Text Right BlahBlahBlahBlah</span><span>Center Text BlahBlahBlahBlah</span></div>""", unsafe_allow_html=True)  # NOT BETTER
        # display: inline-block  # NOT WORK
    with col1_one_to_one_to_one:    
        st.write("<span style='float: left;'>" + get_binary_file_downloader_html(os.getcwd() + '/stats_data/evolution1_Economic - financial_dev.csv', 'evolution1_Economic - financial_dev.csv') + "</span>", unsafe_allow_html=True)
    with col2_one_to_one_to_one: 
        st.write("<span style='float: left;'>" + get_binary_file_downloader_html(os.getcwd() + '/stats_data/evolution2_Economic - financial_dev.csv', 'evolution2_Economic - financial_dev.csv') + "</span>", unsafe_allow_html=True)
    with col3_one_to_one_to_one:
        st.write("<span style='float: right;'>" + get_binary_file_downloader_html(os.getcwd() + '/stats_data/evolution3_Economic - financial_dev.csv', 'evolution3_Economic - financial_dev.csv') + "</span>", unsafe_allow_html=True)

    # st.markdown('<div style="text-align:center"><span style="float:left">' + 'Text Left' +'</span><span style="float:right">' + 'Text Right' +'</span><span>' + 'Center Text' +'</span> </div>', unsafe_allow_html=True)
    # st.markdown('''<div style="text-align:center"><span style="float:left">' + 'Text Left' +'</span><span style="float:right">' + 'Text Right' +'</span><span>' + 'Center Text' +'</span> </div>''', unsafe_allow_html=True)
    # st.markdown("<div style='text-align:center'><span style='float:left'>" + get_binary_file_downloader_html(os.getcwd() + '/stats_data/evolution1_Economic - financial_dev.csv', 'evolution1_Economic - financial_dev.csv') + "</span><span>" + get_binary_file_downloader_html(os.getcwd() + '/stats_data/evolution2_Economic - financial_dev.csv', 'evolution2_Economic - financial_dev.csv') + "</span><span style='float:right'>" + get_binary_file_downloader_html(os.getcwd() + '/stats_data/evolution3_Economic - financial_dev.csv', 'evolution3_Economic - financial_dev.csv') + "</span></div>", unsafe_allow_html=True)  # CANNOT split into 5 parts & MUST in 1 single line, OR ELSE get compilation error

    ### Download 2 log files
    #### logs_simplified text file
    ##### MORE LIGHT WEIGHT ???
    # st.markdown(f'<a download="stats_data/logs_simplified.txt">Download</a>', unsafe_allow_html=True)
    # st.markdown(f'<a href="/Users/zrhun/Desktop/BoSE - FYP/stats_data/logs_simplified.txt" download>Download</a>', unsafe_allow_html=True)
    # st.markdown('<a href="' + os.getcwd()+'/stats_data/logs_simplified.txt"' + ' download>Download</a>', unsafe_allow_html=True)
    # st.markdown('<a href="' + 'file:///'+os.getcwd()+'/stats_data/logs_simplified.txt' + ' download>Download</a>', unsafe_allow_html=True)
    # st.markdown('''<a href="' + os.getcwd()+'/stats_data/logs_simplified.txt"' + ' download>Download</a>''', unsafe_allow_html=True)
    ##### MORE LIGHT WEIGHT ???
    st.write("<span style='float: left'>" + "Logs in txt file:" + "</span>", unsafe_allow_html=True)

    # col1_one_to_one, space_center, col2_one_to_one = st.columns([10,1,10])  # col1_one_to_one : col2_one_to_one = 1:1
    col1_one_to_one, col2_one_to_one = st.columns([10,10])  # col1_one_to_one : col2_one_to_one = 1:1

    with col1_one_to_one:
        st.markdown("<span style='float: left'>" + get_binary_file_downloader_html(os.getcwd() + '/stats_data/logs_simplified.txt', 'logs_simplified.txt') + "</span>", unsafe_allow_html=True)
    with col2_one_to_one:
        # Save large size file into streamlit download folder
        # original = 
        # target = 
        # shutil.copyfile(original, target)


        # st.markdown("<span style='float: right'>" + get_binary_file_downloader_html(os.getcwd() + '/stats_data/logs.txt', '??? logs_simplified.txt ???') + "</span>", unsafe_allow_html=True)
        st.markdown("""<span style='float: right'> [DOWNLOAD: logs.txt](downloads/logs.txt) </span>""", unsafe_allow_html=True)
        # st.markdown("<span style='text-align: right'>" + get_binary_file_downloader_html(os.getcwd() + '/stats_data/logs_simplified.txt', 'logs_simplified.txt') + "</span>", unsafe_allow_html=True)
    # st.markdown(get_binary_file_downloader_html(os.getcwd() + '/stats_data/logs.txt', 'logs.txt'), unsafe_allow_html=True)  # ***
    # st.markdown(get_binary_file_downloader_html(os.getcwd() + '/stats_data/logs_simplified.txt', 'logs_simplified.txt') + " &nbsp; " + '<br>' + get_binary_file_downloader_html(os.getcwd() + '/stats_data/logs_simplified.txt', 'logs_simplified.txt'), unsafe_allow_html=True)  # &nbsp; -> space in HTML

    #### logs text file

    export_as_pdf="Yes"
    def create_download_link(val, filename):
        b64 = base64.b64encode(val)  # val looks like b'...'
        return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download={filename}.pdf">Download file</a>'

    #   if export_as_pdf=="Yes":
    #         pdf = FPDF()
    #         for fig in figs:
    #             pdf.add_page()
    #             with NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
    #                     fig.savefig(tmpfile.name,bbox_inches="tight")#)
    #                     pdf.image(tmpfile.name)
    #         html = create_download_link(pdf.output(dest="S").encode("latin-1"), "Resultatfil")
    #         st.markdown(html, unsafe_allow_html=True)

    st.write("---")

## display heatmap

with st.container():
    if st.button('Intercorrelation Matrix Heatmap'):  # *** -> refer back to t1
        # if not (selected_infra_indi or uploaded_infra_indi_csv_file) and not (selected_eco_indi or uploaded_eco_indi_csv_file):

        # *** -> CHECK LATER
        # if selected_infra_indi_df != None and selected_eco_indi_df != None:
        #     # st.markdown("<center> **Infrastructure indicator** and **economic indicator** are not selected! </center>", unsafe_allow_html=True)
        #     # st.markdown("<center><b>Infrastructure indicator</b> and <b>economic indicator</b> are not selected! </center>", unsafe_allow_html=True)
        #     st.warning("**Infrastructure indicator** and **economic indicator** are not selected!")
        # # elif not (selected_infra_indi or uploaded_infra_indi_csv_file):
        # elif selected_infra_indi_df != None:
        #     st.warning("**Infrastructure indicator** is not selected!")
        # elif selected_eco_indi_df != None:
        #     st.warning("**Economic indicator** is not selected!")
        # else:
        # *** -> CHECK LATER
        heatmap_df, heatmap_plt = displayModifiedHeatmap(selected_infra_indi_df, selected_eco_indi_df)
        st.dataframe(heatmap_df)
        st.markdown(get_binary_file_downloader_html(heatmap_df, "heatmap_df.csv"), unsafe_allow_html=True)
        
        st.pyplot(heatmap_plt)
        
        if st.button("DOWNLOAD: heatmap"):
            heatmap_plt.savefig("result_path123.jpg")
        # st.markdown(create_download_link(heatmap_plt, "heatmap_plt"), unsafe_allow_html=True)

        # REDUNDANT