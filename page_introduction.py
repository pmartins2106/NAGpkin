# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 06:41:39 2023

@author: pmartins
"""

import streamlit as st

def page_introduction():
    
    # Space so that 'About' box-text is lower
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    
    # st.markdown("<h2 style='text-align: center;'> NAGkpin Guidelines </h2>", 
    #             unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center;'>Guidelines</h1>", 
                unsafe_allow_html=True)
     

    st.info("""
            Write Something Here
            """)
    st.info("""
            - Write Something Here
            - ...
            """)


    # image1 = "https://raw.githubusercontent.com/rdzudzar/DistributionAnalyser/main/images/Dist1.png?token=AIAWV2ZQOGWADUFWZM3ZWBLAN3CD6"
    # image2 = "https://raw.githubusercontent.com/rdzudzar/DistributionAnalyser/main/images/Dist2.png?token=AIAWV27IFN4ZLN3EAONHMVLAN3BNS"
    # image3 = "https://raw.githubusercontent.com/rdzudzar/DistributionAnalyser/main/images/Dist3.png?token=AIAWV25DCGRPJRFLDPQIWN3AN3BPA"
    # image4 = "https://raw.githubusercontent.com/rdzudzar/DistributionAnalyser/main/images/Fit1.png?token=AIAWV2ZVPX4HJL77ZQRTIBDAN3BQK"
    # image5 = "https://raw.githubusercontent.com/rdzudzar/DistributionAnalyser/main/images/Fit2.png?token=AIAWV27QFQIAEOQSRDQVC3DAN3BRQ"
    # image6 = "https://raw.githubusercontent.com/rdzudzar/DistributionAnalyser/main/images/Fit3.png?token=AIAWV265V2EQ24SLCTLEHOTAN3BSQ"


    
    def make_line():
        """ Line divider between images. """
            
        line = st.markdown('<hr style="border:1px solid gray"> </hr>',
                unsafe_allow_html=True)

        return line    


    # Images and brief explanations.
    st.error('Write Something Here')
    feature1, feature2 = st.columns([0.5,0.4])
    # with feature1:
        # st.image(image1, use_column_width=True)
    with feature2:
        st.warning('Write Something Here')
        st.info("""
                - Write Something Here
            
                """)
    
    make_line()
     
    return