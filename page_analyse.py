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
import io
from scipy.optimize import curve_fit
from scipy.integrate import odeint


def page_analyse():
    """
    The analyse page is made with Streamlit for fitting nucleation-and-growth
    parameters to data uploaded by the User.
    """
    
    # function to make expander-selection
    def make_expanders(expander_name, sidebar=True):
        """ Set up Figure Mode expander. """
        if sidebar:         
            try:
                return st.sidebar.expander(expander_name)
            except:
                return st.sidebar.beta_expander(expander_name)
    
    # Choose mass-based or size-based analysis
    # Figure display properties expander
    with make_expanders("**Type of progress curves:**"):
        st.markdown("**type of progress curves:**")
        analysis_mode = st.radio("Options", ('Mass-based', 'Size-based'))
    
    # analysis_mode = st.sidebar.radio('**What type of progress curves do you have?**', ('Mass-based', 'Size-based'))   
    # st.sidebar.markdown("#")
  
    st.markdown("<h2 style='text-align: center;'> Analysis of phase separation kinetics </h2>", 
                unsafe_allow_html=True)
        
    st.info("""
                - Upload mass-based or size-based progress curves
                - Find kinetic measurables
                - Characterize the phase separation mechanismn
                """)
    st.markdown("#")
    
    
    # Figure display properties expander
    with make_expanders("Select Figure Mode:"):
        st.markdown("Select Figure Mode:")
        plot_mode = st.radio("Options", ('Dark Mode', 'Light Mode'))
    
    def load_odf():
        """ Get the loaded .odf into Pandas dataframe. """
        try:
            df_load = pd.read_excel(input, skiprows=5,  engine="odf", header = None) #, header = None
            # allow duplicate column names (https://stackoverflow.com/questions/50083583/allow-duplicate-columns-in-pandas)
            df_load.columns = df_load.iloc[0]  # replace column with first row
            df_load = df_load.drop(0)  # remove the first row
            # Remove empty columns
            df_load.dropna(how='all', axis=1, inplace=True)
            # number of curves
            Ncurves = df_load.shape[1]-1
            # Curve1, Curve2...
            colist = ['Curve'+str(i) for i in range(1, Ncurves+1)]
            P = df_load.columns[1:].values.astype(np.float64)
            # reshape to disply
            Pdisp = P.reshape(1, -1)
            dfP = pd.DataFrame(Pdisp, columns=colist, index=['[P]'])
            st.info('Uploaded Data')
            st.write(dfP)
            df_load.columns =  np.concatenate((['Time'], colist), axis=0)
            st.dataframe(df_load)
            flag_return = 0
        except:
            st.write('Error - check if the uploaded file is in the right format')
            df_load = 0
            Ncurves  = 0
            P = ''
            colist = 0
            flag_return = 1
            

        return df_load, Ncurves, P, colist, flag_return
    
    
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
            st.markdown("""**fill the template; save the file (keeping the .ods format); upload the file into NAGPKin*""")
           
            # if run option is selected
        if st.session_state.run_example:
            input = "datasets//NAGpkin_example.ods"
            df, Ncurves, P, colist, flag_return = load_odf()
    except:
        # If the user imports file - parse it
       if input:
           with st.spinner('Loading data...'):
                df, Ncurves, P, colist, flag_return = load_odf()
                    
     
    # Error Handling
    def errhand(err):
        if err == 0:
            st.write("Error - curve fit failed")
            st.write("Check if the uploaded .ods file is properly filled")
        elif err == 1:
            st.write("Error - curve fit failed")
            
            
    #plot experimental and fitted data
    def plot_fit(p_x, p_xfit, p_y, p_yfit, i, flag_legend):
        if plot_mode == 'Dark Mode':
            plt.style.use("dark_background")
        else:
            plt.style.use("default")
        y_legend_m = 'Mass-based progress curve(s)' if flag_legend == 1 else 'Normalized Mass: [M] / [M]inf' 
        y_legend_s = 'Mean size (R)'
        if i < 10:
            plt.scatter(p_x, p_y, color=colors[i], label=df.columns[i+1])
        elif i == 11:
            plt.scatter(p_x, p_y, color=colors[i], label = '...')
        else:
            plt.scatter(p_x, p_y, color=colors[i])
                
        plt.plot(p_xfit, p_yfit, color='g', linewidth=1)
        plt.xlabel('Time')
        if analysis_mode == 'Mass-based':
            plt.ylabel(y_legend_m)
        else:
            plt.ylabel(y_legend_s)
              
        
    def plot_scale(p_x, p_xfit, p_y, p_yfit):
        if plot_mode == 'Dark Mode':
            plt.style.use("dark_background")
        else:
            plt.style.use("default")
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
    
    #https://stackoverflow.com/a/37899817
    def goodness(y_teo, y_exp):
        residuals = y_exp - y_teo
        ss_res = np.sum(residuals**2)
        tot = y_exp-np.mean(y_exp)
        ss_tot = np.sum(tot**2)
        r_2 = 1 - (ss_res / ss_tot) 
        return r_2
    
    def AB_size(kb, kk2, R1):
        A = kb/(1-kb)**2 * (1 - kk2 * R1**3)
        B = kb/(1-kb) * (1 - kk2/kb * R1**3)
        return A, B
    
    def size_func(x, ka, kb, R1, kk2):
        alphat = 1-1 / (kb * (np.exp(ka*x) - 1) + 1)
        A, B = AB_size(kb, kk2, R1)
        y = A * (np.log(1-alphat) + ka*x)/alphat - B
        return y
    
    # Progress curves assuming STEs=0
    def first_fit(x, param0, param1, param2, param3):
        ka = param0
        kb = param1
        if analysis_mode == 'Mass-based':
            F0 = param2
            Fmax = param3
            y = F0 + (Fmax-F0) * (1-1 / (kb * (np.exp(ka*x) - 1) + 1))
        else:
            R1 = param2
            kk2 = param3
            y1 = x.apply(lambda xx: 1 if xx == 0 else size_func(xx, ka, kb, R1, kk2))
            y = R1 * (1/y1)**(1/3)
        return y

    
    # Assume STEs = 0, perform first estimates of kinetic parameters, normlize data
    def fit_data(df):
        """ 
        Fit and plot data:
        """
        fig_fit = plt.figure(facecolor=backc)
        
        results_lst = {}
        results_curve = np.array([], dtype=np.int64).reshape(200,0)
        R1lst = np.zeros(Ncurves)
        flag_return = 0
        for curve in range(Ncurves):
            # select each curve
            try:
                dfi = df.iloc[:, [0,curve+1]].astype(float)
                xdata, ydata = extract_curves(dfi)
                # generate time variable to be used in the fitted curves 
                fit_x = np.linspace(0,max(xdata),200)
                # curve fitting
                # initial guesses
                tmax = np.max(xdata)
                ymin = np.min(ydata)
                ymax = np.max(ydata)
                # initial guesses of ka, kb, (Fmin and Fmax) or (R1 and kk2)
                ig = np.asarray([1/tmax,1e-2,ymin,ymax]) if analysis_mode == 'Mass-based'\
                    else np.asarray([1/tmax,.01,ymin,0])
                    
                try:
                    ymin_lim = min(ymin,0)
                    bds =  ([0, 0, ymin_lim , 0], [np.inf, 100, np.inf,  np.inf])
                    parameters, covariance = curve_fit(first_fit, xdata, ydata, p0=ig,\
                                                   bounds=bds, xtol=1e-20*tmax, ftol=1e-20*ymax, maxfev=10000)
                    kac0 = parameters[0]
                    kbeta0 = parameters[1]
                    # Normalize data
                    cols = df.columns[curve + 1]
                    #definitions of half-life coordiantes
                    v50 = 1/4 * (parameters[1] + 1) * parameters[0]
                    t50 = np.log(1 + 1/parameters[1]) / parameters[0]
                    if analysis_mode == 'Mass-based':
                        # alpha-normalization (between 0 and 1)
                        df[cols] = (df[cols] - parameters[2]) / (parameters[3] - parameters[2])
                        amplitude = parameters[3] - parameters[2]
                        baseline = parameters[2]
                    else:
                        kb = parameters[1]
                        kk2 = parameters[3]
                        R1 = parameters[2]
                        A, B = AB_size(kb, kk2, R1)
                        y1 = A * np.log(1/kb) - B
                        amplitude = R1 * (1/y1)**(1/3) #Rf
                        baseline = R1
                        R1lst[curve] = baseline
                except RuntimeError:
                       flag_return = flag_return + 1  
                
                # generate fitted curve
                if flag_return == 0:
                    fit_x = pd.Series(fit_x) # need a pd series to feed first_fit function
                    fit_y = first_fit(fit_x, parameters[0], parameters[1], parameters[2], parameters[3])
                    plot_fit(xdata, fit_x, ydata, fit_y, curve, 1)
             
                    # array of results
                    if analysis_mode == 'Mass-based':
                        results_lst[curve] = [P[curve], t50, v50, amplitude]
                    else:
                        results_lst[curve] = [P[curve], t50, baseline, amplitude]
                    
                    plt.legend()
                    
                    # Parse resuls
                    if analysis_mode == 'Mass-based':
                        results = pd.DataFrame(results_lst,
                                                index = ['[P]','t50', 'v50', '[M]inf'])
                    else:
                        results = pd.DataFrame(results_lst,
                                                    index = ['[P]','t50', 'R1', 'Rinf'])
            except:
                # flag to terminate run
                flag_return = 1
                results = 0
                
            fit_xy = np.stack((fit_x.T, fit_y.T), axis = 1)
            results_curve = np.concatenate((results_curve, fit_xy), axis=1)
                
        if flag_return > 0:
            # error message and terminate run
             errhand(0)  
            
        # R1 values are used during size-based global fit
        return results, results_curve, kac0, kbeta0, R1lst, fig_fit, flag_return
    
    
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
            t50_dic[i] = np.piecewise(t1, [t1 < tc, t1 >= tc],
                         [lambda z: t1-1/2*tau * np.log((np.tanh(t1/tau) + 1 - 0.5*(1-kb)*kaD*tau)\
                                              /(-np.tanh(t1/tau) + 1 + 0.5*(1-kb)*kaD*tau)), lambda z: tc - np.log(ac/(1-ac)) / kaD ])
        t50 = t50_dic.reshape(-1)
        return t50
    
    # Ask about STEs, perform final estimates of kinetic parameters
    def fit_data2(results):
        """ 
        Fit and plot data
        """
        fig_scale = plt.figure(facecolor=backc)
        # values of [P]
        xdata = P
        # values of t50
        ydata = results.iloc[1] #results.iloc[[1]].to_numpy()
        ydata = ydata.transpose()
        
        # curve fitting
        if cc_mode == 'Unknown':
            # initial guesses of ka kb cc
            ig = np.array([.1, .1, cc0])
            try:
                bds =  ([0, 0, cc0], [np.inf, np.inf, np.nanmin(xdata)])
                parameters, covariance = curve_fit(second_fit, xdata, ydata, p0=ig, bounds = bds, maxfev=10000)
            except RuntimeError:
               # error message and proceed using ig for parameters
               errhand(1)
               parameters = ig
                
        else:
            # initial guesses of ka kb
            ig = np.array([.1, .1])
            bds = (0, np.inf)
            fun = lambda xdata, a, b: second_fit(xdata, a, b, cc0)
            try:
                parameters, covariance = curve_fit(fun, xdata, ydata, p0=ig, bounds = bds, maxfev=10000)
            except RuntimeError:
               # error message and proceed using ig for parameters
               errhand(1)
               parameters = ig
               
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
        fit_xy = np.stack((fit_x.T, fit_y.T), axis = 1)
        plot_scale(xdata, fit_x, ydata, fit_y)
        plt.legend()
        return kac, kbeta, cc, r_squared, fit_xy, fig_scale
    
    
    # Progress curves assuming STES
    def sode (y, time, kaD, kb, cc, ac, k2f):
        a, g = y
        dydt = [kaD * (kb * (0.5*(abs(ac-a) + ac-a))**2 + (1 - a)*a),
                    kaD * (kb * (0.5*(abs(ac-a) + ac-a))**2 + (1 - a)*a*k2f)
                    ]
        return dydt
    
    
    def third_fit(XDATA, param0, param1, param3, param4, PDATA, NDATA):
        kb = param1
        cc = param3
        k2f = param4
        YSOL = []
        y0 = [0, 0]
        # summing up the size of each progress curve (0, len(curve1), len(curve1) + len(curve2), ...))
        Ncount = 0
        for curve in range(Ncurves):
            x = PDATA[curve]
            kaD = param0 * (x - ci) # = kac *DeltaC_0
            ac = (x - cc) / (x - ci)
            time = XDATA[Ncount : Ncount + 1 + NDATA[curve]]
            sol = odeint(sode, y0, time, args=(kaD, kb, cc, ac, k2f))
            Ncount = Ncount + 1 + NDATA[curve]
            if analysis_mode == 'Mass-based':
                YSOL = np.concatenate((YSOL, sol[:,0]), axis=0)
            else:
                # zeroing the numerical solution for size(t)
                size_num = [0 for element in range(len(sol))]
                #initial condition
                size_num[0] = R1lst[curve]
                #size = R1 * (beta(t)) ** (1/3)
                size_num[1:len(sol)] = R1lst[curve] * (sol[1:,0] / sol[1:,1]) ** (1/3)
                YSOL = np.concatenate((YSOL, size_num), axis=0)
        return YSOL
        
    
    def fit_data3(df, results, kac, kbeta, cc):
        """ 
        """
        fig_scale = plt.figure(facecolor=backc)
        flag_return = 0
        XDATA = []
        YDATA = []
        PDATA = np.zeros(Ncurves)
        NDATA = np.zeros(Ncurves, dtype=int)
        results_global = np.array([], dtype=np.int64).reshape(200,0)
        for curve in range(Ncurves):
           # select each curve
           dfi = df.iloc[:, [0,curve+1]].astype(float)
           t_exp , a_exp = extract_curves(dfi)
           # concatenated progress curve adding intial condition (0,0)
           if 0 in t_exp.values:
                 XDATA = np.concatenate((XDATA, t_exp), axis=0)
                 # size of each progress curve
                 NDATA[curve] = len(t_exp)-1
           else:
                 XDATA = np.concatenate((XDATA, [0], t_exp), axis=0)
                 NDATA[curve] = len(t_exp)
           # XDATA = np.concatenate((XDATA, [0], t_exp), axis=0)
           
           PDATA[curve] = P[curve]
           
           if analysis_mode == 'Mass-based':
                if 0 in t_exp.values:
                    YDATA = np.concatenate((YDATA, a_exp), axis=0)
                else:
                    YDATA = np.concatenate((YDATA, [0], a_exp), axis=0)
                # YDATA = np.concatenate((YDATA, [0], a_exp), axis=0)
           else:
                if 0 in t_exp.values:
                    YDATA = np.concatenate((YDATA, a_exp), axis=0)
                else:
                    YDATA = np.concatenate((YDATA, [R1lst[curve]], a_exp), axis=0)
                # YDATA = np.concatenate((YDATA, [R1lst[curve]], a_exp), axis=0)
           
               
       
        # curve fitting
        if cc_mode == 'Unknown':
            # initial guesses of ka kb cc k2f
            ig = np.array([min(kac0,.9), min(kbeta0,.9), cc0, 1])
            bds =  ([0, 0, ci, 0], [np.inf, 10, np.min(PDATA), np.inf])
            fun = lambda XDATA, a, b, c, d: third_fit(XDATA, a, b, c, d, PDATA, NDATA)
           
            try:
                parameters, covariance = curve_fit(fun, XDATA, YDATA, p0=ig, bounds = bds, maxfev=10000)
                
            except RuntimeError:
                # error message and proceed using ig for parameters
                errhand(1)
                parameters = ig
                flag_return = -1
        else:
            # initial guesses of ka kb k2f
            ig = np.array([min(kac0,.9), min(kbeta0,.9), 1])
            bds =  ([0, 0, 0], [np.inf, 10, np.inf])
            fun = lambda XDATA, a, b, d: third_fit(XDATA, a, b, cc0, d, PDATA, NDATA)
            try:
                parameters, covariance = curve_fit(fun, XDATA, YDATA, p0=ig, bounds = bds, maxfev=10000)
            except RuntimeError:
                # error message and proceed using ig for parameters
                errhand(1)
                parameters = ig
                flag_return = -1

        kac_g = parameters[0]
        kbeta_g = parameters[1]
        cc_g = parameters[2] if cc_mode == 'Unknown' else cc0
        k2f_g = parameters[3] if cc_mode == 'Unknown' else parameters[2]
            
        # Goodness of fit:  https://stackoverflow.com/questions/19189362/getting-the-r-squared-value-using-curve-fit
        y_teo = third_fit(XDATA, kac_g, kbeta_g, cc_g, k2f_g, PDATA, NDATA)
        r_squared_g = goodness(y_teo, YDATA)
        Rmin = min(y_teo) #rough estimate of R2 for reporting analysis
        
        y0 = [0, 0]
        for curve in range(Ncurves):
            # select each curve
            dfi = df.iloc[:, [0,curve+1]].astype(float)
            t_exp , a_exp = extract_curves(dfi)
            fit_x = np.linspace(0,max(t_exp),200)
            x = PDATA[curve]
            kaD = kac_g * (x - ci) 
            ac_g = (x - cc_g) / (x - ci)
            sol = odeint(sode, y0, fit_x, args=(kaD, kbeta_g, cc_g, ac_g, k2f_g))
            if analysis_mode == 'Mass-based':
                fit_y = sol[:,0]
            else:
                fit_y = R1lst[curve] * (sol[:,0] / sol[:,1]) ** (1/3)
            plot_fit(t_exp, fit_x, a_exp, fit_y, curve, 2)
            
            fit_xy = np.stack((fit_x.T, fit_y.T), axis = 1)
            results_global = np.concatenate((results_global, fit_xy), axis=1)
        
        plt.legend()    
                
        return kac_g, kbeta_g, cc_g, k2f_g, Rmin, r_squared_g, results_global, fig_scale, flag_return
   
    # Display results    
    if input and flag_return == 0:
        # define color in plots
        # colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colormap = plt.cm.coolwarm 
        colors = [colormap(i) for i in np.linspace(0, 1, Ncurves)]
        #background color
        if plot_mode == 'Dark Mode':
            backc = 'black'
        else:
            backc = 'white'
           
      
        data_fit_chk = st.checkbox("Estimate kinetic measurables")
        if data_fit_chk:
            with st.spinner("Fitting... Please wait a moment."):
                results, results_curve, kac0, kbeta0, R1lst, fig, flag = fit_data(df)
                if flag == 0:
                    st.pyplot(fig)
                    
                    # # Save to memory and offer to download.
                    # fn = 'Half-Life.png'
                    # buf = io.BytesIO()
                    # plt.savefig(buf, format='png', dpi=600)
                    # btn = st.download_button(
                    #     label="Download image",
                    #     data=buf,
                    #     file_name=fn,
                    #     mime="image/png"
                    #     )
                    # buf.close()
        
                    st.info('Half-life coordinates and curve limits')
                    results.columns = colist
                    st.dataframe(results)
                    csvdf1 = pd.DataFrame(results).to_csv().encode('utf-8')
                    st.download_button(
                        "Fitted parameters",
                        csvdf1,
                        "Measurables.csv",
                        "text/csv",
                        key='download-csv') 
                    
                    pd.set_option('display.max_rows', None) #to avoid truncation
                    results_curve = pd.DataFrame(results_curve, columns =["Curve "+str(round(i - (i+i%2)/2 + i%2 )) for i in range(1, (Ncurves)*2+1)])
                    # st.dataframe(fit_x)
                    csvdf1 = pd.DataFrame(results_curve).to_csv().encode('utf-8')
                    st.download_button(
                        "Fitted Curves",
                        csvdf1,
                        "Curves.csv",
                        "text/csv",
                        key='download2-csv') 
    
                    
                    data_comp_chk = st.checkbox("t50 scaling with [P]")
                    if data_comp_chk:
                        # Different [P] values
                        Nunique = (len(set(P)))
                        st.write("Adjust thermodynamic parameters (if needed):")
                        col1, col2 = st.columns(2)
                        with col1:
                            ci = st.number_input("Solubility, $c_{"+(u"\u221e")+"}$", format="%2.2f", value = 0.,min_value = 0.)
                           
                        with col2:
                            cc_mode = st.radio("Critical solubility, $c_c$", ('Unknown', 'Known'))
                            if cc_mode == 'Unknown':
                                cc0 = ci + .1 # initial guess
                            else:
                                cc0 = st.number_input("Enter value", format="%2.2f", value = ci, min_value = ci, max_value = min(results.iloc[0])*.9999)
                        
                        kac, kbeta, cc, r_squared, fit_xy, fig  = fit_data2(results)
                        if Nunique > 3:    
                            st.pyplot(fig)
                            
                            # # Save to memory and offer to download.
                            # fn = 'Scaling.png'
                            # buf = io.BytesIO()
                            # plt.savefig(buf, format='png', dpi=600)
                            # btn = st.download_button(
                            #     label="Download image",
                            #     data=buf,
                            #     file_name=fn,
                            #     mime="image/png"
                            #     )
                            # buf.close()
                            
                            pd.set_option('display.max_rows', None) #to avoid truncation
                            results_scaling = pd.DataFrame(fit_xy, columns =['[P]', 't50'])
                            csvdf1 = pd.DataFrame(results_scaling).to_csv().encode('utf-8')
                            st.download_button(
                                "Fitted Curve",
                                csvdf1,
                                "Scaling.csv",
                                "text/csv",
                                key='download3-csv')
                            
                            st.info('Fitting results')
                            st.write(":green[Autocatalytic rate, $k_{"+(u"\u03b1")+"}/c_{"+(u"\u221e")+"} = $]", kac)
                            st.write(":green[Dimensionless nucleation rate, $k_{"+(u"\u03b2")+"} = $]", kbeta)
                            st.write(":green[Critical solubility, $c_c :$]", cc)
                            st.write(':green[Goodness of fit, $r^2 :$]','%.5f' % r_squared)
                        else:
                            st.warning('Not possible: at least 4 different [P] values are required')
           
                    
                        data_valid_chk = st.checkbox("Global fit")
                        if data_valid_chk:
                            Ncurves = df.shape[1]-1
                            kac_g, kbeta_g, cc_g, k2f_g, Rmin, r_squared_g, results_global, fig, flag = fit_data3(df, results, kac, kbeta, cc)
                            if flag == 0:
                                st.pyplot(fig)
                                
                                # # Save to memory and offer to download.
                                # fn = 'Global-Fit.png'
                                # buf = io.BytesIO()
                                # plt.savefig(buf, format='png', dpi=600)
                                # btn = st.download_button(
                                #     label="Download image",
                                #     data=buf,
                                #     file_name=fn,
                                #     mime="image/png"
                                #     )
                                # buf.close()
                                
                                pd.set_option('display.max_rows', None) #to avoid truncation                               
                                csvdf1 = pd.DataFrame(df).to_csv().encode('utf-8')
                                st.download_button(
                                    "Normalized Data",
                                    csvdf1,
                                    "Normalized.csv",
                                    "text/csv",
                                    key='download4-csv')
                                
                                pd.set_option('display.max_rows', None) #to avoid truncation
                                results_global = pd.DataFrame(results_global, columns =["Curve "+str(round(i - (i+i%2)/2 + i%2 )) for i in range(1, (Ncurves)*2+1)])
                                csvdf1 = pd.DataFrame(results_global).to_csv().encode('utf-8')
                                st.download_button(
                                    "Fitted Curves",
                                    csvdf1,
                                    "GlobalFit.csv",
                                    "text/csv",
                                    key='download5-csv')
                                
                                st.info('Global fitting results')
                                st.write(":green[Autocatalytic rate, $k_{"+(u"\u03b1")+"}/c_{"+(u"\u221e")+"} = $]", kac_g)
                                st.write(":green[Dimensionless nucleation rate, $k_{"+(u"\u03b2")+"} = $]", kbeta_g)
                                if analysis_mode == 'Size-based':
                                    st.write(":green[Dimensionless secondary nucleation rate, $k_2N_1/(k_{"+(u"\u03b1")+"}N_2) = $]", k2f_g)
                                st.write(":green[Critical solubility, $c_c :$]", cc_g)
                                st.write('Goodness of fit, $r^2 :$','%.5f' % r_squared_g)
                                # Report
                                st.info('Report')
                                if flag == 0:
                                    if r_squared_g > 0.9:
                                        st.write('This report is based on the global fit results:')
                                        if kbeta_g > 0.1:
                                            st.write('- **Primary Nucleation**: the dimensionless value of $k_{'+(u"\u03b2")+'} = $','%.2E' % kbeta_g, \
                                                     'is higher than 0.1 indicating fast primary nucleation rates compared with the autocatalytic processes of growth and secondary nucleation.')
                                        elif kbeta_g < 0.01:
                                            st.write('- **Primary Nucleation**: the dimensionless value of $k_{'+(u"\u03b2")+'} = $','%.2E' % kbeta_g, \
                                                     'is below than 0.01 indicating primary nucleation is the rate-limiting step. Compared with primary nucleation, \
                                                         the autocatalytic processes of growth and/or secondary nucleation are much faster.')
                                        else:
                                            st.write('- **Primary Nucleation**: the dimensionless value of $k_{'+(u"\u03b2")+'} = $','%.2E' % kbeta_g, \
                                                     'is between 0.01 and 0.1 indicating primary nucleation is the rate-limiting step. Compared with primary nucleation, \
                                                     the autocatalytic processes of growth and/or secondary nucleation are faster.')
                                        st.write('- **Autocatalytic processes**: the value of $k_{'+(u'\u03b1')+'}$ has units of $[time]^{-1}$ and corresponds to the sum\
                                                 of the rate constans for growth and sencondary nucleation: $k_{'+(u'\u03b1')+'} = k_{+} + k_2$.')
                                                 
                                        if analysis_mode == 'Size-based':
                                            N1N2lst = (R1lst/Rmin)**3
                                            N1N2 = min(N1N2lst) if min(N1N2lst > 1) else 223
                                            k2ka = k2f_g/N1N2                                            
                                            k2ka_perc = k2ka*100
                                            kgka_perc = 100 - k2ka_perc
                                            if N1N2 == 223:
                                                st.write('- **Secondary Nucleation**: the dimensionless value of $k_2N_1/(k_{'+(u'\u03b1')+'}N_2) = $','%.2E' % k2f_g, \
                                                     'corresponds to a $k_2/k_{'+(u'\u03b1')+'}$ ratio of','%.2E' % k2ka, 'if a $N_1/N_2 $', 'ratio of', '%.0f' %N1N2, '[is assumed](https://doi.org/10.1002/ange.201707345). This\
                                                         means that the percentage distribution (in mass) of autocatalytic processes is','%.4f' % k2ka_perc, '% secondary nucleation and','%.4f' % kgka_perc, '% growth.' )
                                            else:
                                                st.write('- **Secondary Nucleation**: the dimensionless value of $k_2N_1/(k_{'+(u'\u03b1')+'}N_2) = $','%.2E' % k2f_g, \
                                                     'corresponds to a $k_2/k_{'+(u'\u03b1')+'}$ ratio of','%.2E' % k2ka, 'if a $N_1/N_2 $', 'ratio of', '%.0f' %N1N2, 'is assumed. This\
                                                         means that the percentage distribution (in mass) of autocatalytic processes is','%.4f' % k2ka_perc, '% secondary nucleation and','%.4f' % kgka_perc, '% growth.' )
                                        else:
                                            st.write('- **Secondary Nucleation**: Mass-based analysis alone provide no accurate information about secondary nucleation parameters. \
                                                     For information about the relative importance of the autocatalytic processes of growth and secondary nucleation consider\
                                                         [measuring size-based](https://doi.org/10.1002/ange.201707345) progress curves.')
                                                         
                                        
                                        if r_squared_g < 0.95:
                                            if r_squared > 0.95:
                                                st.write('- **Parallel processes**: The global fitting result is not great ($r^2 < $0.95). A possible cause is the occurrence of paralell processes such as coalescence and off-pathway aggregation (OPA).',\
                                                         'Since the t50 scaling with [P] provided better fitting results ($r^2 > $0.95) the occurrence of some parallel process is very likely.\
                                                             Note: t50 vs. [P] scaling laws are [less sensitive to OPA](https://doi.org/10.1074/jbc.M115.699348) than the global fit analysis.')
                                            else:
                                                st.write('- **Parallel processes**: The global fitting result is not great ($r^2 < $0.95). A possible cause is the occurrence of paralell processes such as coalescence and off-pathway aggregation (OPA).',\
                                                         'The t50 scaling with [P] [is less sensitive to OPA](https://doi.org/10.1074/jbc.M115.699348) than the global fit analysis. In the present case, however, no additional insight was provided by the t50 scaling with [P].')
                                        else:
                                            if r_squared - r_squared_g < 0.03:
                                                st.write('- **Parallel processes**: Since the global fitting result is good ($r^2 > $0.95), the occurrence of very extensive parallel processes such as coalescence and off-pathway aggregation (OPA) is not likely.',\
                                                     'Note: t50 scaling with [P] [is less sensitive to OPA](https://doi.org/10.1074/jbc.M115.699348) than the global fit analysis.')
                                            else:
                                                st.write('- **Parallel processes**: Since the global fitting result is good ($r^2 > $0.95), the occurrence of very extensive parallel processes such as coalescence and off-pathway aggregation (OPA) is not likely.',\
                                                         'Since the t50 scaling with [P] is better than the global fit, the occurrenc of some parallel process is likely. Note: the t50 scaling with [P] [is less sensitive to OPA](https://doi.org/10.1074/jbc.M115.699348) than the global fit analysis')
                                        
                                        if cc == ci:
                                            st.write('- **Surface Tension Effects** (STEs): Since $c_{'+(u'\u221e')+'} = c_c$, no STEs are present.')
                                        else:
                                            ste = 1 - (np.max(P) - cc_g) / (np.max(P) - ci)
                                            st.write('- **Surface Tension Effects** (STEs): Based on the values of $c_{'+(u'\u221e')+'}$ and $c_c$, the relative importance of STEs is ','%.2f' % ste,' at the least (calculated using [P] = ', '%.1f' % np.max(P),\
                                                     '). [The STEs scale](https://doi.org/10.1101/2022.11.23.517626) is between 0 (no STEs) and 1 (very strong STEs).') 
                                        if Nunique < 4:
                                            st.write('**Warning**: The fitted values are merely indicative. For better results, more progress curves are required measured using \
                                                     different concentrations of protein.')
                                        if r_squared_g < 0.95:
                                            st.write('**Warning**: The global fitting is not great ($r^2 < $0.95). For better results, try to [improve data reproducibility](https://doi.org/10.3389/fnmol.2020.582488)')
                                                     
                                        st.write('Note: The occurence of [Off-Pathway Aggregation](https://doi.org/10.3390/biom8040108), [Surface Tension Effects](https://doi.org/10.1101/2022.11.23.517626) and coalescence can be checked using complementary analytical methods.')        
                                            
                                                 
                                    else:
                                        st.write('Not available. Check if the quality of the experimental results can be improved and if the model assumptions (found [here](https://doi.org/10.1074/jbc.M112.375345) and [here](https://doi.org/10.1002/anie.201707345)) apply to your system.')
    
# page_analyse()
