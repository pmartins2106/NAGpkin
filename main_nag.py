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

# # google analytics
# def inject_ga():
#     GA_ID = "google_analytics"


#     GA_JS = """
#     <!-- Google tag (gtag.js) -->
#     <script async src="https://www.googletagmanager.com/gtag/js?id=G-10ZHSZ60NN"></script>
#     <script>
#       window.dataLayer = window.dataLayer || [];
#       function gtag(){dataLayer.push(arguments);}
#       gtag('js', new Date());
    
#       gtag('config', 'G-10ZHSZ60NN');
#     </script>>
#     """

#     # Insert the script in the head tag of the static template inside your virtual
#     index_path = pathlib.Path(st.__file__).parent / "static" / "index.html"
#     logging.info(f'editing {index_path}')
#     soup = BeautifulSoup(index_path.read_text(), features="html.parser")
#     if not soup.find(id=GA_ID): 
#         bck_index = index_path.with_suffix('.bck')
#         if bck_index.exists():
#             shutil.copy(bck_index, index_path)  
#         else:
#             shutil.copy(index_path, bck_index)  
#         html = str(soup)
#         new_html = html.replace('<head>', '<head>\n' + GA_JS)
#         index_path.write_text(new_html)
# inject_ga()

#Theming
CURRENT_THEME = "Dark"
IS_DARK_THEME = True

# Add pages
from page_introduction import page_introduction
from page_analyse import page_analyse
# from page_validate import page_validate
from page_about import page_about


# Set the default elements on the sidebar
st.set_page_config(page_title='NAGPKin')


st.sidebar.markdown("<h2 style='text-align: center;'>NAGPKin</h2>", 
            unsafe_allow_html=True)
st.sidebar.success('**N**ucleation-**A**nd-**G**rowth **P**arameters from the **Kin**etics\
                   of protein phase separation')

def main_nag():
    """
    Register pages to Explore and Fit:
        page_introduction - contains page with images and brief explanations
        page_analyse - contains various functions that allows user to upload
                    data as a .ods file and fit parameters.
       page_about - about               
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
