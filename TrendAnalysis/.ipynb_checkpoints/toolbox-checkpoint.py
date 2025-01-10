#!/usr/bin/python3

# import statements
import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats

def ENSO_QBO_remover(anomalies_ts, enso_ts, qbo_eof1, qbo_eof2):
    """
    This function takes in an array of timeseries and removes interannual variations
    induced by the QBO and ENSO using a multiple linear regression model.
    ===============================================================================
    anomalies_ts: the input is typically a zonal mean (i.e., heightxlatitude)
    timeseries which has already had the climatology removed and goes from 2002 to 2022.
    
    enso_ts: ENSO MEIv2 timeseries with 3-month lag to account for influence on stratosphere
    
    qbo_eof1: First eof of QBO winds
    
    qbo_eof2: Second eof of QBO winds
    """
    
    timeseries_interannual_removed = []
    for height_index in range(np.shape(anomalies_ts)[1]):
        inter_var_removed = []
        for lat_index in range(np.shape(anomalies_ts)[2]):
            
            # select anomaly timeseries
            anom_ts = anomalies_ts[:,height_index, lat_index]
            
            # find non nan values
            anom_non_nans = anom_ts[~np.isnan(anom_ts)]
            qbo_eof1_non_nans = qbo_eof1[~np.isnan(anom_ts)]
            qbo_eof2_non_nans = qbo_eof2[~np.isnan(anom_ts)]
            enso_non_nans = enso_ts[~np.isnan(anom_ts)]

            # create data frame for MLR
            predictors_and_target = [qbo_eof1_non_nans, qbo_eof2_non_nans, 
                                     enso_non_nans, anom_non_nans]
            df = pd.DataFrame(np.transpose(predictors_and_target), 
                              columns=['qbo1', 'qbo2', 'enso', 'anom'])
            try:
                # Predictors are time, qbo, and enso, target is anomaly timeseries
                Preds = df[['qbo1', 'qbo2', 'enso']]
                Target = df['anom']

                # fit the multiple linear regression
                lm = pg.linear_regression(Preds, Target)
            
                # get coefficients of regression
                intercept = lm.coef[0]
                qbo1 = lm.coef[1]
                qbo2 = lm.coef[2]
                enso = lm.coef[3]

                # recreation with QBO and ENSO
                recreation = intercept + qbo1*qbo_eof1 + qbo2*qbo_eof2 + enso*enso_ts

                # find residual after removing the enso+qbo variability
                residual = anom_ts - recreation

                # append the enso_qbo_variance removed timeseries
                inter_var_removed.append(residual)

            except:
                # if an error occurs, just return nan timeseries
                interannual_var_removed = np.empty(252)
                interannual_var_removed[:] = np.NaN
                inter_var_removed.append(interannual_var_removed)
        timeseries_interannual_removed.append(inter_var_removed)
    return(np.array(timeseries_interannual_removed))

def annual_trend_finder(anomalies_ts):
    """
    This function takes in an array of timeseries and fits a linear trend to the data.
    ===============================================================================
    anomalies_ts: the input is typically a zonal mean (i.e., heightxlatitude)
    timeseries which has already had the climatology removed and goes from 2002 to 2022.
    """
    
    # define time overwhich to take trends
    time = np.arange(2002, 2023, 1/12)/10

    map_of_trends = []
    for height_index in range(np.shape(anomalies_ts)[1]):
        trends_by_lat = []
        for lat_index in range(np.shape(anomalies_ts)[2]):
            
            # select timeseries at height and lat,
            anom_ts = anomalies_ts[:,height_index, lat_index]
            
            # remove nan values from timeseries, and the time array
            anom_non_nans = anom_ts[~np.isnan(anom_ts)]
            time_non_nans = time[~np.isnan(anom_ts)]
            
            try:
                # find linear trend
                trend = stats.linregress(time_non_nans, anom_non_nans)
                anom_trend = trend[0]
                anom_linear_trend = trend[1] + anom_trend*time_non_nans
                
                # get errors for significance testing
                errors = anom_non_nans - anom_linear_trend
                time_errors = time_non_nans - np.nanmean(time_non_nans)
                
                # find degrees of freedom with autocorrelation
                N = len(anom_non_nans)
                r1_autocorrelation = stats.pearsonr(anom_non_nans[1:], anom_non_nans[:-1])[0]
                N_star = N*((1-r1_autocorrelation)/(1+r1_autocorrelation))
        
                # find squared error in x and y
                sum_squared_error_res = np.sum(np.square(errors))
                var_errors = (1/N_star)*sum_squared_error_res
                time_squared_error = np.sum(np.square(time_errors))
                
                # find t-statistic
                simga_slope_squared = var_errors/time_squared_error
                sa = np.sqrt(simga_slope_squared)
                t_stat_calculated = anom_trend/sa

                # get critical t value
                tcrit = stats.t.ppf(1-0.025, N_star)
                
                # test for significance in trend
                if np.abs(t_stat_calculated) > tcrit:
                    significance = 1
                else:
                    significance = 0
                
            except:
                # if an error occurs, set trend to nan
                anom_trend = np.nan
                significance = 0
            trends_by_lat.append([anom_trend, significance])
        map_of_trends.append(trends_by_lat)
    return(np.array(map_of_trends))


def detrender(time, ts):
    """
    This function takes in a timeseries and removes the linear trend.
    ===============================================================================
    time: the timesteps relevant to the timeseries of interest
    
    ts: timeseries which will be detrended
    """
    
    slope = stats.linregress(time, ts)[0]
    intercept = stats.linregress(time, ts)[1]
    recreation = intercept + slope*time
    ts_detrended = ts - recreation
    return(ts_detrended)

def annual_corr_finder(anomalies_ts, anomalous_temp_in_box):
    """
    This function takes in an array of anomaly timeseries, and finds correlation
    with the AWLS region temperature anomalies. Designed for using data from all
    months
    ===============================================================================
    anomalies_ts: anomalies to find correlation with AWLS
    
    anomalous_temp_in_box: AWLS timeseries 
    """
    
    # define time
    time = np.arange(2002, 2023, 1/12)

    # define correlation and signficance maps
    r_map = []
    sig_map = []
    for height_index in range(np.shape(anomalies_ts)[1]):
        
        # define correlation and significance rows
        r_x = []
        sig_x = []
        for lat_index in range(np.shape(anomalies_ts)[2]):
            
            # select  anomaly timeseries
            anom_ts = anomalies_ts[:,height_index, lat_index]
            
            # remove nans
            anom_non_nans = anom_ts[~np.isnan(anom_ts)]
            temp_in_box_non_nans = anomalous_temp_in_box[~np.isnan(anom_ts)]
            time_non_nans = time[~np.isnan(anom_ts)]
            try:
                # detrend the data for correlation so that you don't inflate correlation
                anom_detrend = detrender(time_non_nans, anom_non_nans)
                temp_in_box_detrend = detrender(time_non_nans, temp_in_box_non_nans)

                # get N-samples and correlation coefficient
                N = len(anom_detrend)
                r = stats.pearsonr(anom_detrend, temp_in_box_detrend)[0]
                
                # find significance based on correlation (and auto-correlation)
                r1_autocorrelation = stats.pearsonr(anom_detrend[1:], anom_detrend[:-1])[0]
                r2_autocorrelation = stats.pearsonr(temp_in_box_non_nans[1:],
                                                    temp_in_box_non_nans[:-1])[0]
                N_star = N*((1-r1_autocorrelation*r2_autocorrelation)/
                            (1+r1_autocorrelation*r2_autocorrelation))
                tcrit = stats.t.ppf(1-0.025, N_star)
                t = (r*np.sqrt(N_star - 2))/(np.sqrt(1 - r**2))
                
                # test for significance
                if abs(t) - tcrit > 0:
                    significant = 1
                else:
                    significant = 0
            except:
                r = np.NaN
                significant = 0
            r_x.append(r)
            sig_x.append(significant)
        r_map.append(r_x)
        sig_map.append(sig_x)
    return(np.transpose(r_map), np.transpose(sig_map))

def annual_circ_regr(anomalies_ts, anomalous_temp_in_box):
    """
    This function takes in all months anomaly timeseries and preforms circulation
    regression against the anomalous_temp_in_box timeseries
    ===============================================================================
    anomalies_ts: anomalies to find correlation with AWLS
    
    anomalous_temp_in_box: AWLS timeseries 
    """
    
    # define time
    time = np.arange(2002, 2023, 1/12)/10
    
    # define maps of regressed variability and significance
    map_mlr = []
    map_sig = []
    for height_index in range(np.shape(anomalies_ts)[1]):
        mlr_by_lat = []
        sig_by_lat = []
        for lat_index in range(np.shape(anomalies_ts)[2]):
            
            # define timeseries
            anom_ts = anomalies_ts[:,height_index, lat_index]
            
            # find non nan values
            anom_non_nans = anom_ts[~np.isnan(anom_ts)]
            time_non_nans = time[~np.isnan(anom_ts)]
            tbox_non_nans = anomalous_temp_in_box[~np.isnan(anom_ts)]
            
            # if there is more than one value, circulation regression can continue
            if len(anom_non_nans) > 1:
                anom_detrend = detrender(time_non_nans, anom_non_nans)
                temp_in_box_detrend = detrender(time_non_nans, tbox_non_nans)
                
                # find regression coefficients on non-nan interannual variability
                circ_proj = stats.linregress(temp_in_box_detrend, anom_detrend)

                # find errors for significance testing
                reg_coef = circ_proj[0]
                anom_linear_trend = circ_proj[1] + reg_coef*temp_in_box_detrend
                errors = anom_detrend - anom_linear_trend
                time_errors = temp_in_box_detrend - np.nanmean(temp_in_box_detrend)

                # find degrees of freedom with autocorrelation
                N = len(anom_detrend)
                r1_autocorrelation = stats.pearsonr(anom_detrend[1:], anom_detrend[:-1])[0]
                N_star = N*((1-r1_autocorrelation)/
                            (1+r1_autocorrelation))
        
                # find squared error in x and y
                sum_squared_error_res = np.sum(np.square(errors))
                var_errors = (1/N_star)*sum_squared_error_res
                time_squared_error = np.sum(np.square(time_errors))
                simga_slope_squared = var_errors/time_squared_error
                sa = np.sqrt(simga_slope_squared)
                t_stat_calculated = reg_coef/sa

                # get critical t value
                tcrit = stats.t.ppf(1-0.025, N_star)
                
                # test significance
                if np.abs(t_stat_calculated) > tcrit:
                    significance = 1
                else:
                    significance = 0

                # recreate over whole timeseries
                circ_recreation = circ_proj[1] + circ_proj[0]*anomalous_temp_in_box


                # set nan values to original timeseries
                nan_locations = np.isnan(anom_ts)
                circ_recreation[nan_locations] = np.nan
                mlr_by_lat.append(circ_recreation)
                sig_by_lat.append(significance)
                
            else:
                mlr_by_lat.append(np.repeat(np.nan, 252))
                sig_by_lat.append(0)
        
        map_mlr.append(mlr_by_lat)
        map_sig.append(sig_by_lat)
    return(np.array(map_mlr), np.array(map_sig))

def monthly_trend_finder(time_non_nans, anom_non_nans):
    """
    This function takes in a timeseries representative of data for just one month
    ===============================================================================
    anom_non_nans: timeseries for one month after nans have been removed
    
    time_non_nans: time after nans have been removed
    """
    
    # fit trend
    trend = stats.linregress(time_non_nans, anom_non_nans)
    anom_trend = trend[0]
    
    # reconstruct timeseries based on linear trend
    anom_linear_trend = trend[1] + anom_trend*time_non_nans
    
    # get errors
    errors = anom_non_nans - anom_linear_trend
    time_errors = time_non_nans - np.nanmean(time_non_nans)

    # find degrees of freedom with autocorrelation
    N = len(anom_non_nans)
    r1_autocorrelation = stats.pearsonr(anom_non_nans[1:], anom_non_nans[:-1])[0]
    N_star = N*((1-r1_autocorrelation)/
                (1+r1_autocorrelation))

    # find squared error in x and y
    sum_squared_error_res = np.sum(np.square(errors))
    var_errors = (1/N_star)*sum_squared_error_res
    time_squared_error = np.sum(np.square(time_errors))
    simga_slope_squared = var_errors/time_squared_error
    sa = np.sqrt(simga_slope_squared)
    t_stat_calculated = anom_trend/sa

    # get critical t value
    tcrit = stats.t.ppf(1-0.025, N_star)
    
    # test significance
    if np.abs(t_stat_calculated) > tcrit:
        significance = 1
    else:
        significance = 0
    return([anom_trend, significance])

def monthly_circ_regr(anomalies_ts, anomalous_temp_in_box_, s1):
    """
    This function takes in an array of timeseries and preforms circulation regression
    technique on just one month
    ===============================================================================
    anomalies_ts: array of anomaly timeseries
    
    anomalous_temp_in_box_: AWLS timeseries
    
    s1: month in which circulation regression will be applied
    """
    
    # create array to append data to
    map_mlr = []
    
    for height_index in range(np.shape(anomalies_ts)[1]):
        mlr_by_lat = []
        for lat_index in range(np.shape(anomalies_ts)[2]):
            
            # select timeseries
            anom_ts = anomalies_ts[:, height_index, lat_index]
            
            # create time and calendars
            time = np.arange(2002, 2023, 1/12)/10
            time_cal = np.reshape(time, (21,12))
            anom_cal = np.reshape(anom_ts, (21,12))
            anomalous_temp_in_box_cal = np.reshape(anomalous_temp_in_box_, (21,12))

            # create seasonal timeseries
            time = np.transpose([time_cal[:, s1]]).ravel()
            anom_ts = np.transpose([anom_cal[:, s1]]).ravel()
            anomalous_temp_in_box = np.transpose([anomalous_temp_in_box_cal[:, s1]]).ravel()

            # find non nan values
            anom_non_nans = anom_ts[~np.isnan(anom_ts)]
            time_non_nans = time[~np.isnan(anom_ts)]
            anomalous_temp_in_box = anomalous_temp_in_box[~np.isnan(anom_ts)]

            try:
                # get raw trend
                raw_trend_sig = monthly_trend_finder(time_non_nans, anom_non_nans)

                # get circulation trend
                #######################################################
                # first detrend data
                anom_non_nans_detrend = detrender(time_non_nans, anom_non_nans)
                temp_in_box_detrend = detrender(time_non_nans, anomalous_temp_in_box)

                # then get projection and circulation
                circ_proj = stats.linregress(temp_in_box_detrend, anom_non_nans_detrend)
                circ_recreation = circ_proj[1] + circ_proj[0]*anomalous_temp_in_box
                circ_trend = stats.linregress(time_non_nans, circ_recreation)[0]
                _, circ_significance = monthly_trend_finder(temp_in_box_detrend, 
                                                            anom_non_nans_detrend)
                circ_trend_sig = [circ_trend, circ_significance]
                #######################################################

                # get residual trend
                residual = anom_non_nans - circ_recreation
                res_trend_sig = monthly_trend_finder(time_non_nans, residual)

                # append to mlr
                mlr_by_lat.append([raw_trend_sig, circ_trend_sig, res_trend_sig])
           
            except:

                # create nan array
                nan_trend_sig = [np.NaN, np.NaN]

                # append nans
                mlr_by_lat.append([nan_trend_sig, nan_trend_sig, nan_trend_sig])
        
        # append to mlr map
        map_mlr.append(mlr_by_lat)
        
    return(np.array(map_mlr))

def monthly_corr_map(anomalies_ts, anomalous_temp_in_box_, s1):
    """
    This function takes in an array of timeseries and obtains the correlation map
    between these timeseries and the AWLS timeseries, for just one month
    ===============================================================================
    anomalies_ts: array of anomaly timeseries
    
    anomalous_temp_in_box_: AWLS timeseries
    
    s1: month in which circulation regression will be applied
    """
    # set map to append data
    r_map = []
    for height_index in range(np.shape(anomalies_ts)[1]):
        r_by_lat = []
        for lat_index in range(np.shape(anomalies_ts)[2]):
            
            # select timeseries
            anom_ts = anomalies_ts[:,height_index, lat_index]
            
            # create time and calendars
            time = np.arange(2002, 2023, 1/12)/10
            time_cal = np.reshape(time, (21,12))
            anom_cal = np.reshape(anom_ts, (21,12))
            anomalous_temp_in_box_cal = np.reshape(anomalous_temp_in_box_, (21,12))

            # create seasonal timeseries
            time = np.transpose([time_cal[:,s1]]).ravel()
            anom_ts = np.transpose([anom_cal[:,s1]]).ravel()
            anomalous_temp_in_box = np.transpose([anomalous_temp_in_box_cal[:,s1]]).ravel()

            # find non nan values
            anom_non_nans = anom_ts[~np.isnan(anom_ts)]
            time_non_nans = time[~np.isnan(anom_ts)]
            anomalous_temp_in_box = anomalous_temp_in_box[~np.isnan(anom_ts)]

            if len(anom_non_nans) > 2:
                # get raw trend
                raw_trend_sig = monthly_trend_finder(time_non_nans, anom_non_nans)

                # get correlations for month trend
                #######################################################
                # detrend the data for correlation so that you don't inflate correlation
                anom_non_nans_detrend = detrender(time_non_nans, anom_non_nans)
                temp_in_box_detrend = detrender(time_non_nans, anomalous_temp_in_box)

                # get N-samples and correlation coefficient
                N = len(anom_non_nans_detrend)
                r = stats.pearsonr(anom_non_nans_detrend, temp_in_box_detrend)[0]

                # find significance based on correlation (and auto-correlation)
                r1_autocorrelation = stats.pearsonr(anom_non_nans_detrend[1:], 
                                                    anom_non_nans_detrend[:-1])[0]
                r2_autocorrelation = stats.pearsonr(temp_in_box_detrend[1:], 
                                                    temp_in_box_detrend[:-1])[0]
                N_star = N*((1-r1_autocorrelation*r2_autocorrelation)/
                            (1+r1_autocorrelation*r2_autocorrelation))
                
                # find critical t-values
                tcrit = stats.t.ppf(1-0.025, N_star)
                t = (r*np.sqrt(N_star - 2))/(np.sqrt(1 - r**2))

                # test for significance
                if abs(t) - tcrit > 0:
                    significant = 1
                else:
                    significant = 0
                #######################################################

                # append to mlr
                r_by_lat.append([r, significant])
                
            else:
            
                # create and append nan array
                r_by_lat.append([np.NaN, np.NaN])
                
        r_map.append(r_by_lat)
    return(np.array(r_map))

def monthly_regr_map(anomalies_ts, anomalous_temp_in_box_, s1):
    """
    This function takes in an array of timeseries and obtains the regression map
    between these timeseries and the AWLS timeseries, for just one month
    ===============================================================================
    anomalies_ts: array of anomaly timeseries
    
    anomalous_temp_in_box_: AWLS timeseries
    
    s1: month in which circulation regression will be applied
    """
    
    r_map = []
    for height_index in range(np.shape(anomalies_ts)[1]):
        r_by_lat = []
        for lat_index in range(np.shape(anomalies_ts)[2]):
            
            # select timeseries
            anom_ts = anomalies_ts[:,height_index, lat_index]
            
            # create time and calendars
            time = np.arange(2002, 2023, 1/12)/10
            time_cal = np.reshape(time, (21,12))
            anom_cal = np.reshape(anom_ts, (21,12))
            anomalous_temp_in_box_cal = np.reshape(anomalous_temp_in_box_, (21,12))

            # create seasonal timeseries
            time = np.transpose([time_cal[:,s1]]).ravel()
            anom_ts = np.transpose([anom_cal[:,s1]]).ravel()
            anomalous_temp_in_box = np.transpose([anomalous_temp_in_box_cal[:,s1]]).ravel()

            # find non nan values
            anom_non_nans = anom_ts[~np.isnan(anom_ts)]
            time_non_nans = time[~np.isnan(anom_ts)]
            anomalous_temp_in_box = anomalous_temp_in_box[~np.isnan(anom_ts)]

            if len(anom_non_nans) > 2:
                # get regression coefficents that are used in circulation regression
                #######################################################
                # first detrend data
                anom_non_nans_detrend = detrender(time_non_nans, anom_non_nans)
                temp_in_box_detrend = detrender(time_non_nans, anomalous_temp_in_box)

                # then get projection and circulation
                regr = stats.linregress(temp_in_box_detrend, anom_non_nans_detrend)
                regr_coef = regr[0]
                _, significance = monthly_trend_finder(temp_in_box_detrend, 
                                                            anom_non_nans_detrend)
                [regr_coef, significance]
                #######################################################

                # append to mlr
                r_by_lat.append([regr_coef, significance])
            else:
            
                # create and append nan array
                r_by_lat.append([np.NaN, np.NaN])
                
        r_map.append(r_by_lat)
    return(np.array(r_map))