# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 06:37:35 2023

@author: pmartins 
Based on Dr Robert Dzudzar's Distribution Analyser: https://rdzudzar-distributionanalyser-main-45cc69.streamlit.app/
Thank you!
"""

# Streamlit
import streamlit as st
# Add pages
from page_introduction import page_introduction
from page_analyse import page_analyse
# from page_validate import page_validate
from page_predict import page_predict


# Set the default elements on the sidebar
st.set_page_config(page_title='NAGpkin')

st.sidebar.markdown("<h2 style='text-align: center;'>NAGpkin</h2>", 
            unsafe_allow_html=True)
st.sidebar.success('**N**ucleation-**An**d-**G**rowth **KIN**etics\
                   during **P**rotein phase separation')

def main_nag():
    """
    Register pages to Explore and Fit:
        page_introduction - contains page with images and brief explanations
        page_analyse - contains various functions that allows user to upload
                    data as a .ods file and fit parameters.
       page_validate - contains various functions that allows user to upload
                   data as a .ods file and validate results from page_analyse.             
       page_predict - contains various functions that allows user to predict
       particle size distributions from fitted parameters.   
    """
    pages = {
        "INTRODUCTION": page_introduction,
        "ANALYSE phase separation kinetics": page_analyse,
        # "VALIDATE the kinetic mechanism": page_validate,
        "PREDICT particle size distributions": page_predict,
    }

    st.sidebar.title("Main options")

    # Radio buttons to select desired option
    page = st.sidebar.radio("Select:", tuple(pages.keys()))
                                
    # Display the selected page with the session state
    pages[page]()

  
if __name__ == "__main__":
    main_nag()