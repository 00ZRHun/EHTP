import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage  # Inline labels in Matplotlib
import base64  # encode & decode -> user can download the CSV file
import os
import pandas as pd


#---------------------------------#
# Custom Function
## Inline labels in Matplotlib
def my_legend(axis = None):
    
    if axis == None:
        axis = plt.gca()

    N = 32
    Nlines = len(axis.lines)
    # print(Nlines)  # DEBUG PURPOSE

    xmin, xmax = axis.get_xlim()
    ymin, ymax = axis.get_ylim()

    # the 'point of presence' matrix
    pop = np.zeros((Nlines, N, N), dtype=np.float)    

    for l in range(Nlines):
        # get xy data and scale it to the NxN squares
        xy = axis.lines[l].get_xydata()
        xy = (xy - [xmin,ymin]) / ([xmax-xmin, ymax-ymin]) * N
        xy = xy.astype(np.int32)
        # mask stuff outside plot        
        mask = (xy[:,0] >= 0) & (xy[:,0] < N) & (xy[:,1] >= 0) & (xy[:,1] < N)
        xy = xy[mask]
        # add to pop
        for p in xy:
            pop[l][tuple(p)] = 1.0

    # find whitespace, nice place for labels
    ws = 1.0 - (np.sum(pop, axis=0) > 0) * 1.0 
    # don't use the borders
    ws[:,0]   = 0
    ws[:,N-1] = 0
    ws[0,:]   = 0  
    ws[N-1,:] = 0  

    # blur the pop's
    for l in range(Nlines):
        pop[l] = ndimage.gaussian_filter(pop[l], sigma=N/5)

    for l in range(Nlines):
        # positive weights for current line, negative weight for others....
        w = -0.3 * np.ones(Nlines, dtype=np.float)
        w[l] = 0.5

        # calculate a field         
        p = ws + np.sum(w[:, np.newaxis, np.newaxis] * pop, axis=0)
        plt.figure()
        plt.imshow(p, interpolation='nearest')
        plt.title(axis.lines[l].get_label())

        pos = np.argmax(p)  # note, argmax flattens the array first 
        best_x, best_y =  (pos / N, pos % N) 
        x = xmin + (xmax-xmin) * best_x / N       
        y = ymin + (ymax-ymin) * best_y / N       


        axis.text(x, y, axis.lines[l].get_label(), 
                  horizontalalignment='center',
                  verticalalignment='center')

# SOURCE: 
    # How to download file in streamlit -> https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806/26 // (23/28)
        # - download df / text file
        # How to download local folder? -> https://discuss.streamlit.io/t/how-to-download-local-folder/3717

## Download file (csv, txt) ?
## Download file (csv)
# def filedownload(df):
'''
'''

def get_binary_file_downloader_html01(bin_file=None, file_label='File', df=None):
    if df != None:  # convert df
        csv = df.to_csv(index=False)
        data = csv.encode()
    else:  # download local stored file
        with open(bin_file, 'rb') as f:
            data = f.read()

    b64 = base64.b64encode(data).decode()  # bin_str: strings <-> bytes conversions
    # href = f'<a href="data:file/csv;base64,{b64}"'  # ***
    # href = f'<a href="data:file/csv;base64, {b64}" download="XXX.csv">Download {file_label}</a>' if df else f'<a href="data:application/octet-stream;base64,{b64}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'  # ***
    # return href
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{os.path.basename(bin_file)}">DOWNLOAD: {file_label}</a>'


def get_binary_file_downloader_html(bin_file_or_df, file_label='File'):
    if isinstance(bin_file_or_df, pd.DataFrame):  # NOT CSV file type (pandas DataFrame) -> convert df
        csv = bin_file_or_df.to_csv(index=True)  # False -> maybe coz bug
        data = csv.encode()
    else:  # download local stored file
        with open(bin_file_or_df, 'rb') as f:
            data = f.read()

    b64 = base64.b64encode(data).decode()  # bin_str: strings <-> bytes conversions
    return f'<a href="data:file/csv;base64, {b64}" download="{file_label}">DOWNLOAD: {file_label}</a>' if isinstance(bin_file_or_df, pd.DataFrame) else f'<a href="data:application/octet-stream;base64,{b64}" download="{os.path.basename(bin_file_or_df)}">DOWNLOAD: {file_label}</a>'

# def get_binary_file_downloader_html(bin_file_or_df=None, file_label='File'):
def get_binary_file_downloader_html(bin_file_or_df, file_label='File'):  # BETA TESTING -> can miss 1 default value if other have
    # if bin_file_or_df.split('.')[-1].lower() != "csv":  # NOT WORK -> df don't have '.' (WORK for file with extension)
    if isinstance(bin_file_or_df, pd.DataFrame):  # NOT CSV file type (pandas DataFrame) -> convert df
        csv = bin_file_or_df.to_csv(index=False)
        data = csv.encode()
    else:  # download local stored file
        with open(bin_file_or_df, 'rb') as f:  # *** -> VS below wb
            data = f.read()

    b64 = base64.b64encode(data).decode()  # bin_str: strings <-> bytes conversions
    # href = f'<a href="data:file/csv;base64,{b64}"'  # ***
    # href = f'<a href="data:file/csv;base64, {b64}" download="XXX.csv">Download {file_label}</a>' if df else f'<a href="data:application/octet-stream;base64,{b64}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'  # ***
    # return href
    # return f'<a href="data:application/octet-stream;base64,{bin_file_or_df}" download="{os.path.basename(bin_file_or_df)}">
    # download="XXX.csv"
    return f'<a href="data:file/csv;base64, {b64}" download="{file_label}">DOWNLOAD: {file_label}</a>' if isinstance(bin_file_or_df, pd.DataFrame) else f'<a href="data:application/octet-stream;base64,{b64}" download="{os.path.basename(bin_file_or_df)}">DOWNLOAD: {file_label}</a>'
    

    
def save_uploaded_file(uploaded_file, path="TESTING/tempdDir/"):
    uploaded_file_path = os.path.join(path, uploaded_file.name)
    with open(uploaded_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        st.success("Saved File:{} to {}".format(uploaded_file.name, path))
        return path