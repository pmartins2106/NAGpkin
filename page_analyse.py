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
# import scipy.stats
# curve-fit() function imported from scipy
from scipy.optimize import curve_fit
# ode solver
from scipy.integrate import odeint


def page_analyse():
    """
    The analyse page is made with Streamlit for fitting nucleation-and-growth
    parameters to data uploaded by the User.
    """
    # load the name and urls of all possible mechanisms
    
    st.sidebar.info("""
                - Upload size-based or mass-based progress curves
                - Characterize the phase separation mechanismn
                - Find kinetic measurables
                """)
    st.sidebar.markdown("#")

    # Choose mass-based or size-based analysis
    # st.sidebar.markdown("**What type of progress curves do you have?**")
    analysis_mode = st.sidebar.radio('What type of progress curves do you have?', ('Mass-based', 'Size-based'))   
    st.sidebar.markdown("#")
  
    st.markdown("<h2 style='text-align: center;'> Analyse phase separation kinetics </h2>", 
                unsafe_allow_html=True)
    
    def load_odf():
        """ Get the loaded .csv into Pandas dataframe. """
        
        # df_load = pd.read_csv(input, sep=',' , engine='python',
        #                       nrows=25, skiprows=1, encoding='utf-8')
        df_load = pd.read_excel(input, nrows=25, skiprows=5,  engine="odf")
        return df_load
    
    
    input = st.file_uploader('')
 
    # The run_example variable is session state and is set to False by default
    # Therefore, loading an example is possible anytime after running an example.  
    st.write('Upload your data') 
    st.write('---------- OR ----------')
    st.session_state.run_example = False
    st.session_state.run_example = st.checkbox('Run a prefilled example') 
        
    # Ask for upload (if not yet) or run example.       
    if input is None:
        # Get the template  
        download_sample = st.checkbox("Download template")
      
    try:
        if download_sample:
            with open("datasets//NAGpkin_template.ods", "rb") as fp:
                st.download_button(
                label="Download",
                data=fp,
                file_name="NAGpkin_template.ods",
                mime="application/vnd.ms-excel"
                )
            st.markdown("""**fill the template; save the file (keeping the .ods format); upload the file into NAGpkin*""")
           
            # if run button is pressed
        if st.session_state.run_example:
            input = "datasets//NAGpkin_example.ods"
      
            df = load_odf()
            # Remove empty columns
            df.dropna(how='all', axis=1, inplace=True)
            
            st.info('Uploaded Data')
            st.dataframe(df)
         
    except:
        # If the user imports file - parse it
       if input:
           with st.spinner('Loading data...'):
                    df = load_odf()
                    # Remove empty columns
                    df.dropna(how='all', axis=1, inplace=True)

                                  
                    st.info('Uploaded Data')
                    st.dataframe(df)
        
                
    #plot experimental and fitted data
    def plot_fit(p_x, p_xfit, p_y, p_yfit, i, flag):
        # flag to know the color of p_yfit (red in case of global fit)
        plt.style.use("dark_background")
        cy = colors[i] if flag == 1 else 'g'
        plt.plot(p_xfit, p_yfit, color=cy, linewidth=1)
        plt.scatter(p_x, p_y, color=colors[i], label=df.columns[i+1])
        plt.xlabel('Time')
        if analysis_mode == 'Mass-based':
            plt.ylabel('Normalized Mass')
        else:
            plt.ylabel('Normalized Size')
                
        
    def plot_scale(p_x, p_xfit, p_y, p_yfit):
        plt.style.use("dark_background")
        plt.scatter(p_x, p_y, label='Experimental')
        plt.plot(p_xfit, p_yfit, 'g-', label='Fitted',linewidth=1)
        plt.xlabel('[P]')
        plt.ylabel('t$_{50}$')
        
    def extract_curves(dfi):
        # remove empty cells
        dfi = dfi.dropna()
        ydata = dfi.iloc[:, 1]
        xdata = dfi.iloc[:, 0]
        return xdata, ydata
    
    def goodness(y_teo, y_exp):
        residuals = y_exp - y_teo
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_exp-np.mean(y_exp))**2)
        r_2= 1 - (ss_res / ss_tot) 
        return r_2
    

    # Progress curves assuming STEs=0
    def first_fit(x, param0, param1, param2, param3):
        ka = param0
        if analysis_mode == 'Mass-based':
            kb = param1
            F0 = param2
            Fmax = param3
            y = F0 + (Fmax-F0) * (1-1 / (kb * (np.exp(ka*x) - 1) + 1))
        else:
            #betai >= 1 assuming 2ry nucleation rates = 0
            betai = 1/param1
            R1 = param2
            # assuming 2ry nucleation rates = 0
            y = betai * R1 /(1 + (betai-1) * np.exp(-ka*x))
        
        return y

    
    # Assume STEs = 0, perform first estimates of kinetic parameters, normlize data
    def fit_data(df):
        """ 
        Fit data:
        """
        fig_fit = plt.figure()
        
        results = {}
        for curve in range(Ncurves):
            # select each curve
            dfi = df.iloc[:, [0,curve+1]]
            xdata, ydata = extract_curves(dfi)
            # generate time variable to be used in the fitted curves 
            fit_x = np.linspace(0,max(xdata),200)
            # curve fitting
            # initial guesses
            tmax = np.max(xdata)
            ymin = np.min(ydata)
            ymax = np.max(ydata)
            # initial guesses of ka, kb, Fmin and Fmax
            ig = np.asarray([1/tmax,.1,ymin,ymax])
            parameters, covariance = curve_fit(first_fit, xdata, ydata, p0=ig, maxfev=10000)
                       
            # Normalize data
            cols = df.columns[curve + 1]
            if analysis_mode == 'Mass-based':
                # alpha-normalization (between 0 and 1)
                df[cols] = (df[cols] - parameters[2]) / (parameters[3] - parameters[2])
                #definitions of half-life coordiantes
                v50 = 1/4 * (parameters[1] + 1) * parameters[0]
                t50 = np.log(1 + 1/parameters[1]) / parameters[0]
                amplitude = parameters[3] - parameters[2]
            else:
                # beta-normalization (between 1 and betai)
                df[cols] = df[cols] / parameters[2]
                #definitions of half-life coordiantes
                v50 = 1/4 * parameters[0]
                t50 = np.log(1/parameters[1] - 1) / parameters[0]
                amplitude = 1/parameters[1]
            
            # generate fitted curve
            fit_y = first_fit(fit_x, parameters[0], parameters[1], parameters[2], parameters[3])
            plot_fit(xdata, fit_x, ydata, fit_y, curve, 1)
            
            # array of results
            results[curve] = [df.columns[curve+1], t50, v50, amplitude]
        plt.legend(title='[P]')
        # Parse resuls
        ampl_name = '[M]' if analysis_mode == 'Mass-based' else 'Rf/R1'
        results = pd.DataFrame(results,
                                index = ['[P]','t50', 'v50', ampl_name])
        # results = results.transpose()
        return results, fig_fit
    
    
    
    # Progress curves assuming STES
    def second_fit(x, param0, param1, param3):
        
        NP = len(x)
        t50_dic = np.zeros((NP,))
        
        for i in range(NP):
            kaD = param0 * (x[i] - ci) # = kac *DeltaC_0
            kb = param1
            cc = param3
            ac = (x[i] - cc) / (x[i] - ci)
            
            tau = 2 / (kaD * np.sqrt(1 - 4*kb*ac*(1-ac)))
            t1 = tau * np.arctanh(kaD*tau/2*(1-2*kb*ac)) 
            tc = t1 + tau * np.arctanh(kaD*tau/2*(2*ac-1))
            # st.write(tau, t1, tc)
            if analysis_mode == 'Mass-based':
                t50_dic[i] = np.piecewise(t1, [t1 < tc, t1 >= tc],
                                 [lambda z: t1-1/2*tau * np.log((np.tanh(t1/tau) + 1 - 0.5*(1-kb))\
                                                               /(-np.tanh(t1/tau) + 1 + 0.5*(1-kb))), lambda z: tc - np.log(ac/(1-ac)) ])
            else:
                betai = 1 + (1-ac)/ac * np.exp(kaD*tc)
                # assuming 2ry nucleation rates = 0
                t50_dic[i] = t1
        # t50 = np.fromiter(t50_dic.values(), dtype=float)
        t50 = t50_dic.reshape(-1)
        # st.write(t50_dic.shape)
        return t50
    
    # Ask for STEs, perform final estimates of kinetic parameters
    # @st.cach1e_data()
    def fit_data2(results):
        """ 
        """
        fig_scale = plt.figure()
        # values of [P]
        xdata = results.iloc[0]
        xdata = xdata.transpose()
        # values of t50
        ydata = results.iloc[1] #results.iloc[[1]].to_numpy()
        ydata = ydata.transpose()
       
        # curve fitting
        if cc_mode == 'Unknown':
            # initial guesses of ka kb cc
            ig = np.array([.1, .1, cc0])
            bds =  ([0, 0, cc0], [np.inf, np.inf, np.min(xdata)])
            parameters, covariance = curve_fit(second_fit, xdata, ydata, p0=ig, bounds = bds, maxfev=10000)
        else:
            # initial guesses of ka kb
            ig = np.array([.1, .1])
            bds = (0, np.inf)
            fun = lambda xdata, a, b: second_fit(xdata, a, b, cc0)
            parameters, covariance = curve_fit(fun, xdata, ydata, p0=ig, bounds = bds, maxfev=10000)
            
        kac = parameters[0]
        kbeta = parameters[1]
        cc = cc0 if cc_mode == 'Known' else parameters[2]
        # Goodness of fit:  https://stackoverflow.com/questions/19189362/getting-the-r-squared-value-using-curve-fit
        y_teo = second_fit(xdata, kac, kbeta, cc)
        r_squared = goodness(y_teo, ydata)
            # # standard error https://education.molssi.org/python-data-analysis/03-data-fitting/index.html
            # SE = np.sqrt(np.diag(covariance))
            # kcat = parameters[0]
            # SE_kcat = SE[0]
            # Km = parameters[1]
            # SE_Km = SE[1]
        # generate time variable to be used in the fitted curves 
        fit_x = np.linspace(min(xdata),max(xdata),100)
        fit_y = second_fit(fit_x, kac, kbeta, cc)
        plot_scale(xdata, fit_x, ydata, fit_y)
        plt.legend()
        # st.pyplot(fig_scale)
            
        # Parse resuls
        # results2 = pd.DataFrame(results2,
        #                         index = ['Curve','param0', 'param1'])
        # results2 = results2.transpose()
        return kac, kbeta, cc, r_squared, fig_scale
    
    
    # Progress curves assuming STES
    def sode (y, time, kaD, kb, cc, ac):
        if analysis_mode == 'Mass-based':
            dydt = kaD * (kb * (0.5*(abs(ac-y) + ac-y))**2 + (1 - y)*y)
        else:
            dydt = kaD * (kb * (0.5*(abs(ac-y) + ac-y))**2 + (1 - y)*y)
                
        # tau = 2 / (kaD * np.sqrt(1 - 4*kb*ac*(1-ac)))
        # t1 = tau * np.arctanh(kaD*tau/2*(1-2*kb*ac)) 
        # tc = t1 + tau * np.arctanh(kaD*tau/2*(2*ac-1))
        
        return dydt
    
    
    def third_fit(XDATA, param0, param1, param3, PDATA, NDATA):
        kb = param1
        cc = param3
        YSOL = []
        y0 = 0
        # cumulative number of the each progress curve (0, len(curve1), len(curve1) + len(curve2), ...))
        Ncount = 0
        for curve in range(Ncurves):
            x = PDATA[curve]
            kaD = param0 * (x - ci) # = kac *DeltaC_0
            ac = (x - cc) / (x - ci)
            time = XDATA[Ncount : Ncount + 1 + NDATA[curve]]
            sol = odeint(sode, y0, time, args=(kaD, kb, cc, ac))
            Ncount = Ncount + 1 + NDATA[curve]
            YSOL = np.concatenate((YSOL, sol[:,0]), axis=0)
        return YSOL
        
    
    def fit_data3(df, results, kac, kbeta, cc):
        """ 
        """
        fig_scale = plt.figure()
        
        XDATA = []
        YDATA = []
        PDATA = np.zeros(Ncurves)
        NDATA = np.zeros(Ncurves, dtype=int)
        for curve in range(Ncurves):
           # select each curve
           dfi = df.iloc[:, [0,curve+1]]
           t_exp , a_exp = extract_curves(dfi)
           # size of each progress curve
           NDATA[curve] = len(t_exp)
           # concatenated progress curve adding intial condition (0,0)
           XDATA = np.concatenate((XDATA, [0], t_exp), axis=0)
           YDATA = np.concatenate((YDATA, [0], a_exp), axis=0)
           PDATA[curve] = df.columns[curve+1]
        # solution = third_fit(XDATA, kac, kbeta, cc, PDATA, NDATA)
           
        # curve fitting
        if cc_mode == 'Unknown':
            # initial guesses of ka kb cc
            ig = np.array([kac, kbeta, cc0])
            bds =  ([0, 0, cc0], [np.inf, np.inf, np.min(PDATA)])
            fun = lambda XDATA, a, b, c: third_fit(XDATA, a, b, c, PDATA, NDATA)
            parameters, covariance = curve_fit(fun, XDATA, YDATA, p0=ig, bounds = bds, maxfev=10000)
        else:
            # initial guesses of ka kb
            ig = np.array([kac, kbeta])
            bds = (0, np.inf)
            fun = lambda XDATA, a, b: third_fit(XDATA, a, b, cc0, PDATA, NDATA)
            parameters, covariance = curve_fit(fun, XDATA, YDATA, p0=ig, bounds = bds, maxfev=10000)

        kac_g = parameters[0]
        kbeta_g = parameters[1]
        cc_g = cc0 if cc_mode == 'Known' else parameters[2]
            
        # Goodness of fit:  https://stackoverflow.com/questions/19189362/getting-the-r-squared-value-using-curve-fit
        y_teo = third_fit(XDATA, kac_g, kbeta_g, cc_g, PDATA, NDATA)
        r_squared_g = goodness(y_teo, YDATA)
        xdata = []
        y0 = 0
        for curve in range(Ncurves):
            # select each curve
            dfi = df.iloc[:, [0,curve+1]]
            t_exp , a_exp = extract_curves(dfi)
            fit_x = np.linspace(0,max(t_exp),200)
            x = PDATA[curve]
            kaD = kac_g * (x - ci) 
            ac_g = (x - cc_g) / (x - ci)
            fit_y = odeint(sode, y0, fit_x, args=(kaD, kbeta_g, cc_g, ac_g))
            plot_fit(t_exp, fit_x, a_exp, fit_y, curve, 2)
        plt.legend(title='[P]')    
        # fit_x = np.linspace(min(xdata),max(xdata),100)
        # fit_y = second_fit(fit_x, kac, kbeta, cc)
        # plot_scale(xdata, fit_x, ydata, fit_y)
        # plt.legend()
        
        return kac_g, kbeta_g, cc_g, r_squared_g, fig_scale
   
    # Display resulsts    
    if input:
        # number of curves
        Ncurves = df.shape[1]-1
        # define color in plots
        # colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colormap = plt.cm.coolwarm 
        colors = [colormap(i) for i in np.linspace(0, 1, Ncurves)]
        
        data_fit_chk = st.checkbox("Determine half-life coordinates")
        if data_fit_chk:
            with st.spinner("Fitting... Please wait a moment."):
                results, fig = fit_data(df)
                st.pyplot(fig)
                st.info('Fitting results')
                st.dataframe(results)
                
                st.write("Thermodynamic parameters:")                
                col1, col2 = st.columns(2)
                with col1:
                    ci = st.number_input("Solubility, $c_{"+(u"\u221e")+"}$", format="%2.2f", value = 0.,min_value = 0.)
                   
                with col2:
                    cc_mode = st.radio("Critical solubility, $c_c$", ('Unknown', 'Known'))
                    if cc_mode == 'Unknown':
                        cc0 = 1 # initial guess
                    else:
                        cc0 = st.number_input("Enter value", format="%2.2f", value = ci, min_value = ci)
                
              
                data_comp_chk = st.checkbox("Scaling of t50 with [P]")
                if data_comp_chk:
                    kac, kbeta, cc, r_squared, fig  = fit_data2(results)
                    st.pyplot(fig)
                    st.info('Fitting results')
                    st.write(":green[Autocatalytic rate, $k_a/c_{"+(u"\u221e")+"} = $]", kac)
                    st.write(":green[Dimensionless nucleation rate, $k_{"+(u"\u03b2")+"} = $]", kbeta)
                    st.write("Critical solubility, $c_c :$", '%.2f' % cc)
                    st.write('Goodness of fit, $r^2 :$','%.5f' % r_squared)
       
                
                    data_valid_chk = st.checkbox("Check global fit")
                    if data_valid_chk:
                        Ncurves = df.shape[1]-1
                        kac_g, kbeta_g, cc_g, r_squared_g, fig = fit_data3(df, results, kac, kbeta, cc)
                        st.write(kac_g, kbeta_g, cc_g, r_squared_g)
                        st.pyplot(fig)
                        st.write('Goodness of fit, $r^2 :$','%.5f' % r_squared_g)
                    
                
                    
# page_analyse()