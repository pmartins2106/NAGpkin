# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 06:41:04 2023

@author: pmartins
"""

# Package imports
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
# curve-fit() function imported from scipy
from scipy.optimize import curve_fit
import math
# List of GMMS
from gmm_list import gmm_list_names, creating_dictionaries


def page_validate():
    """
    The find page in this app is made with Streamlit for fitting a GMM
    a GMM mechanism to the User imported data.
    """
    # load the name and urls of all possible mechanisms
    name_proper_dict, name_url_dict = creating_dictionaries()
       
    st.sidebar.info("""
                Import enyme kinetics data and discover the mechanism of enzyme 
                inhibiton/activation.
                """)
    st.sidebar.text(" ")
    
    # Input enzyme concentration
    Enzyme = st.sidebar.number_input('Enzyme concentration [E]', format="%2.2e", value = 1e0,min_value = 1e-20)

    # Run full analysis or step-by-step analysis
    st.sidebar.markdown("**Type of Analysis:**")
    analysis_mode = st.sidebar.radio(' ', ('Run All', 'Step-by-Step'))   
    st.sidebar.text(" ")
    st.sidebar.text(" ")

    st.markdown("<h1 style='text-align: center;'> Find the General Modifier Mechanism </h1>", 
                unsafe_allow_html=True)

    def load_csv():
        """ Get the loaded .csv into Pandas dataframe. """
        
        # df_load = pd.read_csv(input, sep=',' , engine='python',
        #                       nrows=25, skiprows=1, encoding='utf-8')
        df_load = pd.read_excel(input, nrows=25, skiprows=5,  engine="odf")
        return df_load
    

    input = st.file_uploader('')
 
    # The run_example variable is session state and is set to False by default
    # Therefore, loading an example is possible anytime after running an example.  
    st.session_state.run_example = False
    st.session_state.run_example = st.checkbox('Run a prefilled example') 
        
    # Ask for upload (if not yet) or run example.       
    if input is None:
        st.write('Upload your data, or:')
        # Get the template  
        download_sample = st.checkbox("Download template")
      
    try:
        if download_sample:
            with open("datasets//GMM_Finder_template.ods", "rb") as fp:
                st.download_button(
                label="Download",
                data=fp,
                file_name="GMM_Finder_template.ods",
                mime="application/vnd.ms-excel"
                )
            st.markdown("""**fill the template; save the file (keeping the .ods format); upload the file into the GMM finder*""")
           
            # if run button is pressed
        if st.session_state.run_example:
            input = "datasets//GMM_Finder_example.ods"
      
            df = load_csv()
            # Replace inf/-inf with NaN and remove NaN if present
            df = df.loc[:, ~df.columns.str.contains('Unnamed')]
            df = df.replace([np.inf, -np.inf], np.nan).dropna() #[data_col]
            
            #convert to [v]/[E]
            cols = df.columns[1:]
            df[cols] = df[cols] / Enzyme
               
            st.info('Uploaded Data')
            st.dataframe(df)
         
    except:
        # If the user imports file - parse it
       if input:
           with st.spinner('Loading data...'):
                    df = load_csv()
                    # Replace inf/-inf with NaN and remove NaN if present
                    df = df.replace([np.inf, -np.inf], np.nan).dropna() #[data_col]
                    st.info('Uploaded Data')
                    st.dataframe(df)
            
                
    #plot experimental and fitted data
    def plot_fit(p_x, p_xfit, p_y, p_yfit, i, colors):
        plt.style.use("dark_background")
        plt.scatter(p_x, p_y, color=colors[i], label=df.columns[i+1])
        plt.plot(p_xfit, p_yfit, color=colors[i])
        plt.xlabel('[S]')
        plt.ylabel('[v] / [E]')
        

    # Michaelis-Menten (MM) equation
    def MMeq(x, p_kcat, p_Km):
        y = p_kcat*x/(p_Km+x)
        return y
    
    # Fit of MM equation to data
    @st.cache_data()
    def fit_data(df):
        """ 
        Fit data:
        """
        
        fig_fit = plt.figure()
        
        NX = df.shape[1]-1 #number of modifier concentrations
        
        # Check for nan/inf and remove them
        num_bins = round(math.sqrt(len(df)))
        hist, bin_edges = np.histogram(df, num_bins, density=True)
        
        results = {}
        xdata = df.iloc[:, 0]
        fit_x = np.linspace(0,max(xdata),200)
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for modifier in range(NX):
            # Go through each [X]
            ydata = df.iloc[:, modifier+1]
            parameters, covariance = curve_fit(MMeq, xdata, ydata)
        
            # standard error https://education.molssi.org/python-data-analysis/03-data-fitting/index.htmlhttps://education.molssi.org/python-data-analysis/03-data-fitting/index.html
            SE = np.sqrt(np.diag(covariance))
            kcat = parameters[0]
            SE_kcat = SE[0]
            Km = parameters[1]
            SE_Km = SE[1]
            fit_y = MMeq(fit_x, kcat, Km)
            plot_fit(xdata, fit_x, ydata, fit_y, modifier, colors)
            plt.legend(title='[X]')
        
            # array of results
            results[modifier] = [df.columns[modifier+1], kcat, 
                                 SE_kcat, Km, SE_Km]
        
        # Parse resuls
        results = pd.DataFrame(results,
                               index = ['[X]','kcat', 'SE kcat', 'Km', 'SE Km'])
        results = results.transpose()
        return results, colors, fig_fit
    
    
    #Definttions of kcat_app and Km_app
    def fun_kcat(x, p_k2, p_Kca, p_K1):
        """
        Kca = Kx*alpha
        K1 = beta/Kca
        """
        y = p_k2 * (1 + x*p_K1) / (1 + x/p_Kca)
        return y
    
    def fun_Km(x, p_Km, p_Kx, p_Kca):
        """
        Kca = Kx*alpha
        """
        y = p_Km * (1 + x/p_Kx) / (1 + x/p_Kca)
        return y
    
    # Fit the deffinitons of apparent constants to data
    @st.cache_data()
    def fit_fingerprints(df):
        """ 
        """
        # Modifier concentrations [X]
        xdata = df.iloc[:, 0].astype(float)
        
        # kcat vs. [X]
        ydata0 = df.iloc[:,1]
        #index of minimum [X]
        imin = np.argmin(xdata)
        
        # initial guess of k2, Kca and K1
        ig = np.asarray([1,ydata0[imin],np.mean(xdata)])
        parameters0, covariance0 = curve_fit(fun_kcat, xdata, ydata0, p0=ig, maxfev=10000)
        k2 = parameters0[0]
        Kca = parameters0[1]
        K1 = parameters0[2]
        
        # Kmvs. [X]
        ydata1 = df.iloc[:,3]
        # initial guess of Km, Kx and Ka
        ig = np.asarray([ydata1[imin],np.mean(xdata),np.mean(xdata)])
        parameters1, covariance1 = curve_fit(fun_Km, xdata, ydata1, ig, maxfev=10000)
        Km = parameters1[0]
        Kx = parameters1[1]
        Kca2 = parameters1[2]
        
        # two subplots
        fig_fgpts, axs = plt.subplots(2)
        fig_fgpts.suptitle('Dependencies of the apparent parameters on the modifier concentration')
        plt.style.use("dark_background")
 
        x_fit = np.linspace(0,max(xdata),200)
        
        # First subplot
        axs[0].scatter(xdata, ydata0)
        fit_y = fun_kcat(x_fit, k2, Kca, K1)
        axs[0].plot(x_fit, fit_y)
        axs[0].set(ylabel = 'kcat_app')
       
        # Second subplot
        axs[1].scatter(xdata, ydata1)
        fit_y = fun_Km(x_fit, Km, Kx, Kca2)
        axs[1].plot(x_fit, fit_y)
        plt.xlabel('[X]')
        plt.ylabel('Km_app')
        
        return k2, Km, Kx, fig_fgpts
    
    
    # Global fit
    @st.cache_data()
    def fun_eq_global(XDATA,alpha,beta,Kx):
        """
        # Global Fitting with fixed alpha and beta
        Based on https://stackoverflow.com/questions/28372597/python-curve-fit-with-multiple-independent-variables
        and on Eq (1) of https://www.enzyme-modifier.ch/raw-data-new/
        """
        S = XDATA[0:NS]
        X = XDATA[NS:NS+NX]
        vE = np.zeros((len(S),len(X)))
        
        for i in range(NS):
           vE [:][i] = k2 * (1 + beta * X / alpha / Kx) * S[i] / (Km * (1 + X / Kx) + S[i] * ( 1+ X / Kx / alpha))
        
        vE = vE.flatten()
        return vE
    
   
    # Decision tree to select GMM     
    def selector(p_alpha,p_beta,p_Km,p_Kx):
        """
        Decision tree to select the GMM mechanism from the fitted parameters.
        """
        
        SF = 0.05  # Sensitivity factor
        
        # Evaluating the Specific/Catalytic/Mixed nature of the modifier (alpha)
        alpha_round = round(p_alpha * (10**2)) * (10**-2)
        if (alpha_round < 1+SF) and (alpha_round > 1-SF):  # alpha = 1
            flag_alpha = 'Balanced'
        elif alpha_round >= 1+SF:  # alpha > 1
            flag_alpha = 'Specific'
        elif alpha_round <= 1-SF:  # alpha < 1
            flag_alpha = 'Catalytic'
        
        # Evaluating alpha & beta to determine modifying mechanism
        beta_round = round(p_beta * (10**2)) * (10**-2)
        
        if beta_round <= 0.05: # beta = 0
            if (alpha_round < 1+SF) and (alpha_round > 1-SF):  # alpha = 1
                Mechanism = 'LMx(Sp=Ca)I'
            elif alpha_round >= 1+SF:  # alpha > 1
                if alpha_round > 20:  # alpha --> +Inf
                    Mechanism = 'LSpI'
                else:
                    Mechanism = 'LMx(Sp>Ca)I'
            elif (alpha_round <= 1-SF) and (alpha_round >= SF*2):  # alpha < 1
                Mechanism = 'LMx(Sp<Ca)I'
            # elif alpha_round > 20:  # alpha --> +Inf
            #     Mechanism = 'LSpI'
            elif alpha_round < SF*2 and p_Kx > p_Km:  # alpha --> 0, Kx > Km in the absence of modifier
                Mechanism = 'LCaI'   
            elif (beta_round < 1+SF) and (beta_round > 1-SF):
                if alpha_round >= 1+SF:  # alpha > 1
                    Mechanism = 'HSpI'
                elif alpha_round <= 1-SF:  # alpha < 1
                    Mechanism = 'HSpA'
            else: # Unsuccessful determination
                Mechanism ='Unable to successfully determine mechanism.'
        elif (beta_round <= 1-SF) and (beta_round > SF):     # beta < 1
            if alpha_round >= 1+SF:  # alpha > 1
                Mechanism = 'HMx(Sp>Ca)I'
            elif alpha_round <= 1-SF:  # alpha < 1
                if (abs(alpha_round - beta_round) < 0.3):  # alpha = beta
                    Mechanism = 'HCaI'
                elif alpha_round > beta_round:  # alpha > beta
                    Mechanism = 'HMx(Sp<Ca)I'
                elif beta_round > alpha_round:  # beta > alpha
                    Mechanism = 'HMxD(A/I)'
            elif (alpha_round < 1+SF) and (alpha_round > 1-SF):  # alpha = 1
                Mechanism = 'HMx(Sp=Ca)I'
        elif beta_round >= 1 + SF:    # beta > 1
            if alpha_round <= 1 - SF: # alpha < 1
                if (abs(alpha_round - beta_round) < 0.3): # alpha = beta
                    Mechanism = 'HCaA'
                # elif alpha_round > beta_round: # alpha > beta
                #     Mechanism = 'HMxD(I/A)'
                elif beta_round > alpha_round: # beta > alpha
                    Mechanism = 'HMx(Sp>Ca)A'
            elif alpha_round >= 1 + SF: # alpha > 1
                if (abs(alpha_round - beta_round) < 0.3): # alpha = beta
                    Mechanism = 'HCaA'
                elif alpha_round < beta_round: # alpha < beta
                    Mechanism = 'HMx(Sp<Ca)A'
                elif alpha_round > beta_round: # alpha > beta
                    Mechanism = 'HMxD(I/A)'
            elif (alpha_round < 1 + SF) and (alpha_round > 1 - SF): # alpha = 1
                Mechanism = 'HMx(Sp<Ca)A'
        else: # Unsuccessful determination
            Mechanism = 'Unable to successfully determine mechanism.'
        return Mechanism

    # Key to run all if this option is selected
    key_run = True if analysis_mode == 'Run All' else False
    # Display resulsts    
    if input:
        # st.write("Determine apparent kcat and Km values:")
        data_fit_chk = st.checkbox("Determine apparent values of kcat and Km", value=key_run)
        if data_fit_chk:
            with st.spinner("Fitting... Please wait a moment."):
                results, colors, fig = fit_data(df)
                st.pyplot(fig)
                st.info('Fitted parameters and values of standard\
                        error (SE)')
                st.dataframe(results)
                
                data_comp_chk = st.checkbox("Compute derived parameters", value=key_run)
                if data_comp_chk:
                    results = results.assign(inv_cat = 1 / results.kcat,
                                             kcat_Km = results.kcat / results.Km,
                                             Km_kcat = results.Km / results.kcat)
                    st.dataframe(results)
                    
                    k2, Km, Kx_e, fig_fgpts = fit_fingerprints(results)
                    st.pyplot(fig_fgpts)
                    
                    # st.info('First estimates of parameters k2 and Km')
                    # results_fgrprt_0 = pd.DataFrame([ [k2, Km ] ],
                    #                               index = ['Values'],
                    #                               columns = ['k2','Km'])
                    # st.dataframe(results_fgrprt_0)
                    
                    final_polish = st.checkbox("Final polish and GMM identification", value=key_run)
                    if final_polish:
                       
                        # Global Fitting with fixed alpha and beta
                        y = (df.iloc[:,1:].to_numpy())
                        y = y.flatten()
                        # y = np.array(y)
                        S_exp = df.iloc[:,0].astype(float) #.to_numpy()
                        X_exp = results.iloc[:, 0].astype(float)
                        NS = len(S_exp)
                        NX = len(X_exp)
                        XDATA = np.concatenate((S_exp, X_exp), axis=0)
                       
                        # initial guesses for alpha, beta, Kx
                        p0 = 1, 1, Kx_e
                        # y1 = fun_eq_global(XDATA,2.4,0.35,39.77)
                        # st.write(y1)
                        # XDATA=np.column_stack([S,X])
                        
                        popt, pcov = curve_fit(fun_eq_global,XDATA, y, p0, maxfev=10000)
                        alpha = popt[0]
                        beta= popt[1]
                        Kx = popt[2]
                        
                        st.info('Global Fit')     
                        # plot global fit
                        fig_fit2 = plt.figure()
                        xdata = S_exp
                        fit_x = np.linspace(0,max(xdata),200)
                        for modifier in range(NX):
                            # Go through each [X]
                            ydata = df.iloc[:, modifier+1]
                            Kca = Kx*alpha
                            kcat =  fun_kcat(X_exp[modifier], k2, Kca, beta/Kca)
                            Km = fun_Km(X_exp[modifier], Km, Kx, Kca)
                            fit_y = MMeq(fit_x, kcat, Km)
                       
                            plot_fit(xdata, fit_x, ydata, fit_y, modifier, colors)
                            plt.legend(title='[X]')
                        plt.xlabel('[S]')
                        plt.ylabel('[v] / [E]')
                        st.pyplot(fig)
                        results_fgrprt = pd.DataFrame([ [alpha, beta, k2, Km, Kx ]],
                                                      index = ['Values'],
                                                      columns = ['alpha','beta','k2','Km', 'Kx'])
                        st.dataframe( results_fgrprt)
                        
                        # selector(p_alpha, p_beta, p_Km, p_Kx)                      
                        Mechanism = selector(alpha, beta, Km, Kx )
                        if Mechanism != 'Unable to successfully determine mechanism.':
                            st.success('Success!')
                            # Goodness of  https://stackoverflow.com/questions/19189362/getting-the-r-squared-value-using-curve-fit
                            y1 = fun_eq_global(XDATA,alpha, beta, Kx)
                            residuals = y - y1
                            ss_res = np.sum(residuals**2)
                            ss_tot = np.sum((y-np.mean(y))**2)
                            r_squared = 1 - (ss_res / ss_tot)  
                            st.write(':green[The kinetic mechanism of this modifier is:]  ', name_proper_dict[Mechanism])
                            st.write(':green[Acronym:]', Mechanism)
                            st.write(':green[Goodness of fit (r^2):]','%.5f' % r_squared)
                            st.write(':green[For more information click:]  ', name_url_dict[Mechanism])
                           
                        else:
                            st.warning(Mechanism)