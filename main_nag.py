# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 06:37:35 2023

@author: pmartins 
Based on Dr Robert Dzudzar's Distribution Analyser: https://rdzudzar-distributionanalyser-main-45cc69.streamlit.app/
Thank you!
"""
# Needed for google analytics
from bs4 import BeautifulSoup
import shutil
import pathlib
import logging
# Streamlit
import streamlit as st

# google analytics
def add_analytics_tag():
    
    analytics_js = """
    <!-- Global site tag (gtag.js) - Google Analytics -->
    <script async src="https://www.googletagmanager.com/gtag/js?id=G-TVHC4G4TZB"></script>
    <script>
        window.dataLayer = window.dataLayer || [];
        function gtag(){dataLayer.push(arguments);}
        gtag('js', new Date());
        gtag('config', 'G-TVHC4G4TZB');
    </script>
    <div id="G-TVHC4G4TZB"></div>
    """
    analytics_id = "G-TVHC4G4TZB"

    
    # Identify html path of streamlit
    index_path = pathlib.Path(st.__file__).parent / "static" / "index.html"
    logging.info(f'editing {index_path}')
    soup = BeautifulSoup(index_path.read_text(), features="html.parser")
    if not soup.find(id=analytics_id): # if id not found within html file
        bck_index = index_path.with_suffix('.bck')
        if bck_index.exists():
            shutil.copy(bck_index, index_path)  # backup recovery
        else:
            shutil.copy(index_path, bck_index)  # save backup
        html = str(soup)
        new_html = html.replace('<head>', '<head>\n' + analytics_js) 
        index_path.write_text(new_html) # insert analytics tag at top of head


# Add pages
from page_introduction import page_introduction
from page_analyse import page_analyse
# from page_validate import page_validate
from page_about import page_about
# from dotenv import load_dotenv
# import os
# load_dotenv(".env")    


# Set the default elements on the sidebar
st.set_page_config(page_title='NAGpkin')

# analytics
# streamlit_analytics.start_tracking()


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
        "ANALYSE": page_analyse,
        # "VALIDATE the kinetic mechanism": page_validate,
        "ABOUT": page_about,
    }

    st.sidebar.title("Main options")

    # Radio buttons to select desired option
    page = st.sidebar.radio("Select:", tuple(pages.keys()))
                                
    # Display the selected page with the session state
    pages[page]()

  
if __name__ == "__main__":
    main_nag()
    
# my_password = os.getenv('Password')
# streamlit_analytics.stop_tracking(my_password)