# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 06:41:04 2023

@author: pmartins
"""

# Package imports
import streamlit as st

def page_about():
        
    
    st.markdown("<h2 style='text-align: center;'>About</h2>", 
                unsafe_allow_html=True)
    st.markdown('''
    **NAGpkin** is a free-to-use web tool for characterizing the mechanisms of protein aggregation, crystallization or liquid-liquid phase separation.
    Users can upload raw data of mass-increase or size-increase over time and then check what is the nucleation-and-growth mechanism that best describes their results.
    Poor numerical fits mean that other mechanims of protein self-assembly are at play or that the experimental data is of insufficient quality.\n
    ''')
    st.markdown('Please <a href="mailto:nagpkin@gmail.com">let us know</a> if you have any questions or need more in-depth analysis.', unsafe_allow_html=True)
    
    # Lower next markdowns
    st.write("")

   
    def make_line():
        """ Line divider between images. """
            
        line = st.markdown('<hr style="border:1px solid gray"> </hr>',
                unsafe_allow_html=True)

        return line    
   
    make_line()
    
    # Lower next markdowns
    st.write("")

    
    st.markdown("<h2 style='text-align: center;'>Disclaimer</h2>", 
                unsafe_allow_html=True)
    st.markdown('''
    This webtool is free to use for all non-commercial purposes. This webtool is provided by the copyright holders and contributors "as is" and any express or implied warranties, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed. In no event shall the copyright owner or contributors be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to, procurement of substitute goods or services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of the use of this webtool, even if advised of the possibility of such damage.
    ''')
    
# page_about()
