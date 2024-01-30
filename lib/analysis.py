# coding: utf-8

# Copyright 2023 Sony Group Corporation - All Rights Reserved.
# Subject to the terms and conditions contained in LICENSE.txt accompanying this file, you may use this file.

import os
import numpy as np
import pickle
import datetime
import matplotlib
matplotlib.use("Agg") # use Anti-Grain Geometry (AGG, a non-interactive backend that can only write to files) backend to prevent memory overuse
import matplotlib.pyplot as plt
from scipy import fft
import glob
import math
from lib import utils
import config_by_gui
import config
from lib import common



window = np.ones(config.NUM_TRACK_TO_AVERAGE) / config.NUM_TRACK_TO_AVERAGE # window function for filtering
track_offset = config.NUM_TRACK_TO_AVERAGE//2

plt.rcParams['lines.linewidth'] = config.GRAPH_LINE_WIDTH



class CoorFFT():
    def __init__(self):
        self.fs = config.coor_fft_fs # sampling freq. default: 250 Hz (4ms period)

    def process(self, png_header, pre_str, cluster, x_history, y_history):
        # FFT of x/y coordinates
        plt_path = fig_path + png_header + str(cluster.ID)
        custom_fft(self.fs, plt_path+"_x" + pre_str + ".png", cluster, x_history, "X", bool_coor=1)
        custom_fft(self.fs, plt_path+"_y" + pre_str + ".png", cluster, y_history, "Y", bool_coor=1)



class EventFFT():
    def __init__(self):
        self.fs = config.event_fft_fs # sampling freq. default: 1000 Hz (1ms period)

    def process(self, png_header, cluster, all_event_history, pos_event_history):
        plt_path = fig_path + png_header + str(cluster.ID)

        # FFT of positive and negative events
        custom_fft(self.fs, plt_path+"_all.png", cluster, all_event_history, "All events", bool_coor=0)
        #plt.show()

        # FFT of positive events
        custom_fft(self.fs, plt_path+"_pos.png", cluster, pos_event_history, "Pos events", bool_coor=0)

        # FFT of negative events
        neg_event_history = all_event_history - pos_event_history
        custom_fft(self.fs, plt_path+"_neg.png", cluster, neg_event_history, "Neg events", bool_coor=0)



def custom_fft(fs, plt_path, cluster, array, ylabel_name, bool_coor):
    L = len(array)

    if bool_coor:
        t = config.track_step_ms * np.array( range(L) ) # time in [ms]. "track_step_ms" default: 4

        # subtract a linear function from "array" to make "array" start and end at coordinate 0s
        head_coor = array[0]
        tail_coor = array[L-1]
        diff_step = (tail_coor - head_coor) / L
        linear_coors = []
        for i in range(L):
            linear_coors.append( head_coor + i*diff_step )

        np_linear_coors = np.array( linear_coors )
        windowed_array = array - np_linear_coors # rectangular window (= no window function)
    else:
        t = config.cluster_step_ms * np.array( range(L) ) # time in [ms]. "cluster_step_ms" default: 4
        windowed_array = array

    # calc FFT
    #NFFT = 2**nextpow2(fs) * 2  # for faster calculation speed??? -> resulted in blunt peaks
    #fft_amp = fftpack.fft(windowed_array, NFFT)
    #fft_fq = fftpack.fftfreq(NFFT, d=1.0/fs)
    fft_amp = fft.fft(windowed_array)
    fft_fq = fft.fftfreq(L, d=1.0/fs)

    # extract positive area only
    fft_fq = fft_fq[ 0 : int( len(fft_fq)/2 ) ] # len(fft_fq) is supposed to be always "L"
    abs_fft_amp = abs( fft_amp[ 0 : int(len(fft_amp)/2 ) ] ) # len(fft_amp) is supposed to be always "L"
    #fft_amp_nozero = np.where(abs_fft_amp > 1.0e-10, abs_fft_amp, 1.0e-10) # delete values less than 1.0e-10
    fft_amp = 20 * np.log10( abs_fft_amp )  # complex num -> dB. This line generates "division by zero" warning, which does not halt the entire process

    # plot graph at the top
    plt.figure(figsize=(6, 4))
    plt.subplots_adjust(hspace=0.4) # margin for x axis label
    plt.subplot(2, 1, 1)
    plt.plot(t, windowed_array)
    plt.xlabel("Time [ms]")
    plt.ylabel(ylabel_name)
    plt.grid()

    # detect FFT peaks
    increase_cnter = 0
    last_amp_point = fft_amp[0]
    list_peak_idx = [] # for storing detected coordinate/event peaks to draw figures
    for i,amp_point in enumerate(fft_amp):
        if amp_point > last_amp_point:
            # when larger than previous
            increase_cnter += 1
        elif increase_cnter >= 1:
            # when smaller than previous, search in the 2-octave-range for a larger peak
            fq = fft_fq[i-1] # freq of "last_amp_point"
            top_fq = fq*2
            bottom_fq = fq/2
            peak_flg = 1

            avg_amp = 0. # average amplitude of the 2-octave-range
            tmp_cnter = 0
            for j,fq_point in enumerate(fft_fq):
                if fq_point <= bottom_fq:
                    # when smaller than the 2-octave-range
                    continue
                elif top_fq < fq_point:
                    # when larger than the 2-octave-range
                    increase_cnter = 0 # this line is required
                    break
                elif i == j:
                    # skip the freq of "last_amp_point"
                    continue

                if fft_amp[j] > last_amp_point:
                    # when a larger peak is found in the 2-octave-range
                    peak_flg = 0
                    if j < i:
                        # when the larger peak has smaller freq, forget "last_amp_point"
                        increase_cnter = 0
                    break

                avg_amp += fft_amp[j]
                tmp_cnter += 1

            if peak_flg:
                # when "last_amp_point" is confirmed to be a peak, see if it is acute enough
                avg_amp /= tmp_cnter
                amp_diff = last_amp_point - avg_amp
                if (amp_diff > 15.) and (amp_diff != np.inf): # [dB]. if the peak value is over the background level by 15 dB or more (but not "np.inf", which is the result of division by zero)
                    list_peak_idx.append(i-1)

                    # write to "fft_peaks.txt"
                    if (len(cluster.coor_fft_peaks) == 0) and (len(cluster.event_fft_peaks) == 0):
                        f_output_txt_fft_peak.write( f"\n{cluster.ID} from " )

                        # write "first_track_t" to features.csv
                        cur_sec = cluster.first_track_t//1000
                        cur_min = cur_sec//60
                        cur_hour = cur_min//60
                        f_output_txt_fft_peak.write( str(cur_hour).zfill(2) + ':' + str(cur_min%60).zfill(2) + ':' + str(cur_sec%60).zfill(2) )

                        dur_sec = (cluster.last_track_t - cluster.first_track_t)
                        f_output_txt_fft_peak.write( f" for {dur_sec} msec.\n" )

                    if bool_coor:
                        (cluster.coor_fft_peaks).append( [fft_fq[i-1], amp_diff] ) # store as a new peak
                        f_output_txt_fft_peak.write( f"Coordinate peak: {amp_diff:.2f} dB , over ave: {avg_amp:.2f} dB , @ {fft_fq[i-1]:.2f} Hz\n" )
                    else:
                        (cluster.event_fft_peaks).append( [fft_fq[i-1], amp_diff] ) # store as a new peak
                        f_output_txt_fft_peak.write( f"Event peak: {amp_diff:.2f} dB , over ave: {avg_amp:.2f} dB , @ {fft_fq[i-1]:.2f} Hz\n" )

                    increase_cnter = 0 # this line is required

        last_amp_point = amp_point # update "last_amp_point"

    # plot graph at the bottom
    plt.subplot(2, 1, 2)
    for peak_idx in list_peak_idx:
        #plt.plot(fft_fq[peak_idx], fft_amp[peak_idx]+10., "go") # add green circle at detected peaks
        #plt.plot(fft_fq[peak_idx], fft_amp[peak_idx], "rx") # add red cross at detected peaks
        plt.plot(fft_fq[peak_idx], fft_amp[peak_idx]+10., "rv") # add red aroowhead at detected peaks
    plt.plot(fft_fq, fft_amp)
    plt.xscale("log")
    plt.xlim(0.1, fs/2)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude [dB]')
    plt.grid()
    plt.savefig( plt_path, dpi=100, facecolor="white", bbox_inches="tight" )
    plt.close()

    # below are remaining codes from the past when a shifting fixed-duration-window was adopted for FFT
    #array = array[fs//2:]
    #while_cnter += 1



coor_fft = CoorFFT()
event_fft = EventFFT()

fig_path = ""
f_output_txt_fft_peak = None

if config.FLG_INFERENCE:
    # read weight info
    try:
        w_i_h = np.loadtxt(config.W_I_H__PATH, delimiter=',')
    except OSError:
        print(f"Failed to read \"{config.W_I_H__PATH}\". Inference process has been aborted.")
        config.FLG_INFERENCE = 0

    try:
        w_h_o = np.loadtxt(config.W_H_O__PATH, delimiter=',')
    except OSError:
        print(f"Failed to read \"{config.W_H_O__PATH}\". Inference process has been aborted.")
        config.FLG_INFERENCE = 0

    try:
        b_i_h = np.loadtxt(config.B_I_H__PATH, delimiter=',')
    except OSError:
        print(f"Failed to read \"{config.B_I_H__PATH}\". Inference process has been aborted.")
        config.FLG_INFERENCE = 0

    try:
        b_h_o = np.loadtxt(config.B_H_O__PATH, delimiter=',')
    except OSError:
        print(f"Failed to read \"{config.B_H_O__PATH}\". Inference process has been aborted.")
        config.FLG_INFERENCE = 0

    b_i_h.shape += (1,)
    b_h_o.shape += (1,)


def main(target_nm):
    global fig_path, f_output_txt_fft_peak

    utils.set_analysis_path(target_nm, flg_subcluster=0)

    #print(f"\nAnalyzing the contents of {common.analysis_path}")
    fig_path = common.analysis_path + "fig/"
    os.makedirs(fig_path, exist_ok=True)

    list_f_clusters = sorted( glob.glob(common.analysis_path + "*s.pkl") )

    if config.FLG_CALC_FEATURES:
        # write fft peaks to .txt
        f_output_txt_fft_peak = open(common.analysis_path + "fft_peaks.txt", "w")

        # write features to .csv
        f_output_txt_feature = open(common.analysis_path + "features.csv", "w")
        f_output_txt_feature.write( "cluster ID,first track timestamp,last track timestamp,velocity ave,velocity variance,jump cnter,length,width,antenna weight")
        f_output_txt_feature.write( ",pos length ratio(sq),pos width ratio(sq),neg length ratio(sq),neg width ratio(sq),pos to neg length ratio(sq),pos to neg width ratio(sq)")
        f_output_txt_feature.write( ",event fft peaks(<1Hz),event fft peaks(<2.2Hz),event fft peaks(<4.6Hz),event fft peaks(<10Hz),event fft peaks(<22Hz),event fft peaks(>=22Hz)")
        f_output_txt_feature.write( ",coordinate fft peaks(<1Hz),coordinate fft peaks(<2.2Hz),coordinate fft peaks(<4.6Hz),coordinate fft peaks(>=4.6Hz)" )
        # f_output_txt_feature.write( ",inference,Ksel score,Csp score,Acat score,Pcon score,Pden score,Gsca score,Oova score,Gins score,Acar score,Cmar_a score,Cmar_m score,Hdim score,annotation\n" )
        f_output_txt_feature.write( ",inference,Passive score,Pcra_C score,Pcra_N score,Nfus score,Ppec score, annotation\n" )

    list_num_vs_time = [] # idx: minute, value: num of moving objects
    list_num_vs_width = [] # idx: width range, value: num of moving objects

    for f_clusters_path in list_f_clusters:
        print( f"Analyzing {f_clusters_path}" )

        start_pos = len(f_clusters_path) - len("00h00m00sTo00h00m00s.pkl")
        end_pos = start_pos + len("00h00m00sTo00h00m00s_")
        png_header = f_clusters_path[ start_pos : end_pos ]

        if config.FLG_CALC_FEATURES:
            # write to txt output files
            f_output_txt_feature.write( f"{f_clusters_path}\n" ) # first, write the path of the file of clusters
            f_output_txt_fft_peak.write( f"\n\n{f_clusters_path}\n" ) # first, write the path of the file of clusters

        with open(f_clusters_path, 'rb') as f_pkl_nm:
            clusters = pickle.load(f_pkl_nm)

        for cluster in clusters:
            # calc centroid track
            cluster.inference = "" # initialize
            cluster.track_g_all_history = [] # initialize
            cluster.track_g_pos_history = [] # initialize
            cluster.track_g_neg_history = [] # initialize
            corners = [] # points of "rectangle" corners of "cluster"

            for rectangles_at_a_time in cluster.track_r_history: # rectangles_at_a_time = [ [318, 324, 652, 658, 3], [323, 328, 648, 652, 4], [315, 325, 657, 658, 2] ] (for example)
                total_weight = 0
                center_x_x2 = 0 # (center x of centroid) * 2 * weight
                center_y_x2 = 0 # (center y of centroid) * 2 * weight
                fringe_flg = 0

                pos_total_weight = 0
                pos_center_x_x2 = 0 # (center x of positive events) * 2 * weight
                pos_center_y_x2 = 0 # (center y of positive events) * 2 * weight

                neg_total_weight = 0
                neg_center_x_x2 = 0 # (center x of negative events) * 2 * weight
                neg_center_y_x2 = 0 # (center y of negative events) * 2 * weight

                corners.append([])

                for rectangle in rectangles_at_a_time:
                    tmp_center_x_x2 = (rectangle[0] + rectangle[1]).item()
                    tmp_center_y_x2 = (rectangle[2] + rectangle[3]).item()

                    # calc centroids of all(positive and negative) events
                    weight = rectangle[4].item() # change to native python list (64bits) since numpy 16bits is not enough
                    total_weight += weight
                    center_x_x2 += tmp_center_x_x2 * weight
                    center_y_x2 += tmp_center_y_x2 * weight

                    # calc centroids of positive events
                    pos_weight = rectangle[5].item() # change to native python list (64bits) since numpy 16bits is not enough
                    pos_total_weight += pos_weight
                    pos_center_x_x2 += tmp_center_x_x2 * pos_weight
                    pos_center_y_x2 += tmp_center_y_x2 * pos_weight

                    # calc centroids of negative events
                    neg_weight = weight - pos_weight # change to native python list (64bits) since numpy 16bits is not enough
                    neg_total_weight += neg_weight
                    neg_center_x_x2 += tmp_center_x_x2 * neg_weight
                    neg_center_y_x2 += tmp_center_y_x2 * neg_weight

                    if ( rectangle[0] <= 50 ) or ( rectangle[1] >= (config.WIDTH-50) ) or ( rectangle[2] <= 50 ) or ( rectangle[3] >= (config.HEIGHT-50) ):
                        fringe_flg = 1

                    # register each "rectangle" corners to "corners" (just for the calc of length/width below, unrelated to centroids)
                    # 
                    corners[-1].extend( [ [rectangle[0], rectangle[2]] , [rectangle[1], rectangle[2]] , [rectangle[1], rectangle[3]], [rectangle[0], rectangle[3]] ] )

                # calc the final result of centroids
                total_weight *= 2 # in order to cancel out "_x2"
                center_x = center_x_x2/total_weight # division result is always a float number
                center_y = center_y_x2/total_weight
                (cluster.track_g_all_history).append( [ center_x, center_y, total_weight, fringe_flg ] ) # [centroid_x, centroid_y, total_num_pixel, proximity_to_fringe]

                if pos_total_weight:
                    pos_total_weight *= 2 # in order to cancel out "_x2"
                    pos_center_x = pos_center_x_x2/pos_total_weight
                    pos_center_y = pos_center_y_x2/pos_total_weight
                    (cluster.track_g_pos_history).append( [ pos_center_x, pos_center_y, pos_total_weight ] ) # [centroid_x, centroid_y, total_num_pos_pixel]
                elif len(cluster.track_g_pos_history):
                    # when there is last "track_g_pos_history" entry and there is no positive events at the moment
                    (cluster.track_g_pos_history).append( cluster.track_g_pos_history[-1] ) # copy the last coordinates
                    cluster.track_g_pos_history[-1][2] = 0 # make the weight zero
                else:
                    # when there is no last "track_g_pos_history" entry and there is no positive events at the moment
                    (cluster.track_g_pos_history).append( [0., 0., 0] )

                if neg_total_weight:
                    neg_total_weight *= 2 # in order to cancel out "_x2"
                    neg_center_x = neg_center_x_x2/neg_total_weight
                    neg_center_y = neg_center_y_x2/neg_total_weight
                    (cluster.track_g_neg_history).append( [ neg_center_x, neg_center_y, neg_total_weight ] ) # [centroid_x, centroid_y, total_num_neg_pixel]
                elif len(cluster.track_g_neg_history):
                    # when there is last "track_g_neg_history" entry and there is no negative events at the moment
                    (cluster.track_g_neg_history).append( cluster.track_g_neg_history[-1] ) # copy the last coordinates
                    cluster.track_g_neg_history[-1][2] = 0 # make the weight zero
                else:
                    # when there is no last "track_g_neg_history" entry and there is no negative events at the moment
                    (cluster.track_g_neg_history).append( [0., 0., 0] )

            # count how many entries at the head of "track_g_all_history" are proximal to the fringe
            head_at_fringe_cnter = 0
            for g_entry in cluster.track_g_all_history:
                if g_entry[3] == 1: # fringe_flg
                    head_at_fringe_cnter += 1
                else:
                    break

            if head_at_fringe_cnter == len(cluster.track_g_all_history):
                # when the clustr has always been proximal to fringe, skip to the next "cluster"
                continue

            # count how many entries at the tail of "track_g_all_history" are proximal to the fringe
            tail_at_fringe_cnter = 0
            for g_entry in reversed(cluster.track_g_all_history):
                if g_entry[3] == 1: # fringe_flg
                    tail_at_fringe_cnter += 1
                else:
                    break

            # extract valid "track_g_all_history" and "corners" eliminating entries close to the fringe
            if tail_at_fringe_cnter == 0:
                # when "cluster" died somewhere not close to the fringe
                valid_track_r_history = cluster.track_r_history[ head_at_fringe_cnter : ]
                valid_track_g_all_history = cluster.track_g_all_history[ head_at_fringe_cnter : ]
                valid_track_g_pos_history = cluster.track_g_pos_history[ head_at_fringe_cnter : ]
                valid_track_g_neg_history = cluster.track_g_neg_history[ head_at_fringe_cnter : ]
                valid_corners = corners[ head_at_fringe_cnter : ]
            else:
                # when "cluster" died somewhere close to the fringe
                valid_track_r_history = cluster.track_r_history[ head_at_fringe_cnter : -tail_at_fringe_cnter ]
                valid_track_g_all_history = cluster.track_g_all_history[ head_at_fringe_cnter : -tail_at_fringe_cnter ]
                valid_track_g_pos_history = cluster.track_g_pos_history[ head_at_fringe_cnter : -tail_at_fringe_cnter ]
                valid_track_g_neg_history = cluster.track_g_neg_history[ head_at_fringe_cnter : -tail_at_fringe_cnter ]
                valid_corners = corners[ head_at_fringe_cnter : -tail_at_fringe_cnter ]

            if len(valid_track_g_all_history) < config.NUM_TRACK_TO_AVERAGE:
                # when "valid_track_g_all_history" is too short, skip to the next "cluster"
                continue

            # extract centroid x and y histories separately
            center_x_history = np.array( [ row[0] for row in valid_track_g_all_history ] )
            center_y_history = np.array( [ row[1] for row in valid_track_g_all_history ] )

            # average centroid x and y histories with a window function (LPF)
            center_x_history_ave = np.convolve(center_x_history, window, mode="valid") # window = np.ones(config.NUM_TRACK_TO_AVERAGE) / config.NUM_TRACK_TO_AVERAGE
            center_y_history_ave = np.convolve(center_y_history, window, mode="valid")

            # calc x/y velocities. Velocity: (num_pxls) / (track_step_ms [ms] )
            velocity_x_history = center_x_history_ave[1:] - center_x_history_ave[:-1] # /track_step_ms [ms]. default: 4 [ms]
            velocity_y_history = center_y_history_ave[1:] - center_y_history_ave[:-1] # /track_step_ms [ms]. default: 4 [ms]

            # calc velocity in terms of Euclidean distance
            velocity_history = np.sqrt( np.add( np.square(velocity_x_history), np.square(velocity_y_history) ) )
            travel_distance = np.sum(velocity_history) # total travel distance

            if travel_distance > config.TRAVEL_DISTANCE_TH:
                # when total travel distance is longer than "TRAVEL_DISTANCE_TH" (default: 200 pixels)
                #cluster.velocity_ave = np.average( velocity_history ) # average velocity [ pixel / (track_step_ms * ms) ]. Old definition
                cluster.velocity_ave = np.average( velocity_history ) / config.track_step_ms # average velocity [ pixel / ms ]

                # calc variance of x/y velocities
                #velocity_x_history_var = np.var( velocity_x_history[np.abs(velocity_x_history) >= 0.7], ddof=1 )
                velocity_x_history_var = np.var( velocity_x_history, ddof=1 )
                velocity_y_history_var = np.var( velocity_y_history, ddof=1 )

                # calc variance of velocity history
                cluster.var_velocity = np.add( velocity_x_history_var, velocity_y_history_var ) / np.square( cluster.velocity_ave * config.track_step_ms )

                # update statistics of num_vs_time using "var_velocity"
                first_min = cluster.first_track_t // (1000*60) # unit: minutes
                last_min = cluster.last_track_t // (1000*60) # unit: minutes

                while len(list_num_vs_time) <= last_min:
                    list_num_vs_time.append([0,0,0]) # [num particle, num active, num very active]
                for idx in range(first_min, last_min+1):
                    list_num_vs_time[idx][0] += 1
                    if cluster.var_velocity > config.ACTIVE_VAR_TH:
                        list_num_vs_time[idx][1] += 1
                    if cluster.var_velocity > config.VERY_ACTIVE_VAR_TH:
                        list_num_vs_time[idx][2] += 1


                # calc characteristic features of "cluster"
                if config.FLG_CALC_FEATURES:
                    # calc Convex-Hull's length and width based on the cluster's moving direction
                    tmp_cnter = 0
                    accum_length = 0.
                    accum_width = 0.

                    tmp_var_pos_cnter = 0
                    tmp_var_neg_cnter = 0
                    accum_var_pos_length_ratio = 0.
                    accum_var_pos_width_ratio = 0.
                    accum_var_neg_length_ratio = 0.
                    accum_var_neg_width_ratio = 0.

                    for i,velocity_x in enumerate(velocity_x_history):
                        velocity_y = velocity_y_history[i]
                        velocity = np.array([velocity_x, velocity_y]) # calc velocity vector 
                        len_velocity = np.linalg.norm(velocity, ord=2) # calc length of velocity vector (L2 norm. Euclidean)
                        if len_velocity == 0.0:
                            continue

                        norm_velocity = velocity / len_velocity # normarized vector of velocity
                        norm_orth_velocity = np.array([velocity_y, -velocity_x]) / len_velocity # normarized vector orthogonal to the vector of velocity

                        plus_length = 0.
                        minus_length = 0.
                        plus_width = 0.
                        minus_width = 0.

                        i_shift = i + track_offset # track_offset = config.NUM_TRACK_TO_AVERAGE//2
                        for corner in valid_corners[i_shift]: # notice "center_x_history_ave" is shorter than "valid_track_g_all_history" by (NUM_TRACK_TO_AVERAGE-1) resulting from averaging
                            # for each corner, calc the inner dot with "norm_velocity" and "norm_orth_velocity"
                            vec_to_corner = np.array( [ corner[0] - center_x_history_ave[i], corner[1] - center_y_history_ave[i] ] ) # temporal vector from the centroid
                            tmp_length = np.dot(vec_to_corner, norm_velocity)
                            tmp_width = np.dot(vec_to_corner, norm_orth_velocity)

                            # calc max/min length along the cluster's moving direction
                            if tmp_length > plus_length:
                                plus_length = tmp_length
                            elif tmp_length < minus_length:
                                minus_length = tmp_length

                            # calc max/min width orthgonal to the cluster's moving direction
                            if tmp_width > plus_width:
                                plus_width = tmp_width
                            elif tmp_width < minus_width:
                                minus_width = tmp_width

                        tmp_cnter += 1
                        cur_length = (plus_length - minus_length)
                        #cur_length *= 2
                        cur_width = (plus_width - minus_width)
                        accum_length += cur_length
                        accum_width += cur_width


                        # calc positive rectagle center's variance
                        if valid_track_g_pos_history[i_shift][2]:
                            total_pos_weight = 0
                            accum_var_pos_length = 0.
                            accum_var_pos_width = 0.
                            for rectangle in valid_track_r_history[i_shift]:
                                if rectangle[5]:
                                    pos_weight = rectangle[5].item() # change to native python list (64bits) since numpy 16bits is not enough
                                    total_pos_weight += pos_weight
                                    pos_center_x = (rectangle[0] + rectangle[1]).item() / 2 # turn to a float number
                                    pos_center_y = (rectangle[2] + rectangle[3]).item() / 2

                                    vec_to_pos_center = np.array( [ pos_center_x - valid_track_g_pos_history[i_shift][0], pos_center_y - valid_track_g_pos_history[i_shift][1] ] ) # temporal vector from the centroid of positive events
                                    tmp_pos_length = np.dot(vec_to_pos_center, norm_velocity)
                                    tmp_pos_width = np.dot(vec_to_pos_center, norm_orth_velocity)

                                    accum_var_pos_length += (tmp_pos_length**2) * pos_weight # for positive events variance parallel to cluster's velocity vector
                                    accum_var_pos_width += (tmp_pos_width**2) * pos_weight # for positive events variance orthogonal to cluster's velocity vector

                            cur_var_pos_length = accum_var_pos_length / total_pos_weight # average variance at the moment
                            cur_var_pos_width = accum_var_pos_width / total_pos_weight

                            if cur_length != 0.:
                                accum_var_pos_length_ratio += cur_var_pos_length / (cur_length**2) # for positive events variance parallel to cluster's velocity vector
                            if cur_width != 0.:
                                accum_var_pos_width_ratio += cur_var_pos_width / (cur_width**2) # for positive events variance orthogonal to cluster's velocity vector
                            tmp_var_pos_cnter += 1

                        # calc negative rectagle center's variance
                        if valid_track_g_neg_history[i_shift][2]:
                            total_neg_weight = 0
                            accum_var_neg_length = 0.
                            accum_var_neg_width = 0.
                            for rectangle in valid_track_r_history[i_shift]:
                                tmp_neg_weight = rectangle[4] - rectangle[5]
                                if tmp_neg_weight:
                                    neg_weight = tmp_neg_weight.item() # change to native python list (64bits) since numpy 16bits is not enough
                                    total_neg_weight += neg_weight
                                    neg_center_x = (rectangle[0] + rectangle[1]).item() / 2 # turn to a float number
                                    neg_center_y = (rectangle[2] + rectangle[3]).item() / 2

                                    vec_to_neg_center = np.array( [ neg_center_x - valid_track_g_neg_history[i_shift][0], neg_center_y - valid_track_g_neg_history[i_shift][1] ] ) # temporal vector from the centroid of negative events
                                    tmp_neg_length = np.dot(vec_to_neg_center, norm_velocity)
                                    tmp_neg_width = np.dot(vec_to_neg_center, norm_orth_velocity)

                                    accum_var_neg_length += (tmp_neg_length**2) * neg_weight # for negative events variance parallel to cluster's velocity vector
                                    accum_var_neg_width += (tmp_neg_width**2) * neg_weight # for negative events variance orthogonal to cluster's velocity vector

                            cur_var_neg_length = accum_var_neg_length / total_neg_weight # average variance at the moment
                            cur_var_neg_width = accum_var_neg_width / total_neg_weight

                            if cur_length != 0.:
                                accum_var_neg_length_ratio += cur_var_neg_length / (cur_length**2) # for negative events variance parallel to cluster's velocity vector
                            if cur_width != 0.:
                                accum_var_neg_width_ratio += cur_var_neg_width / (cur_width**2) # for negative events variance orthogonal to cluster's velocity vector
                            tmp_var_neg_cnter += 1

                    # calc the final length/width of Convex-Hull
                    cluster.length = accum_length / tmp_cnter
                    cluster.width = accum_width / tmp_cnter

                    # update statistics of num_vs_width
                    cluster_width_int = int(cluster.width)
                    while len(list_num_vs_width) <= cluster_width_int:
                        list_num_vs_width.append(0)
                    list_num_vs_width[cluster_width_int] += 1

                    # calc the final positive/negative events variance parallel/orthogonal to cluster's velocity vector
                    if tmp_var_pos_cnter:
                        cluster.pos_length_ratio_sq = accum_var_pos_length_ratio / tmp_var_pos_cnter
                        cluster.pos_width_ratio_sq = accum_var_pos_width_ratio / tmp_var_pos_cnter
                    if tmp_var_neg_cnter:
                        cluster.neg_length_ratio_sq = accum_var_neg_length_ratio / tmp_var_neg_cnter
                        cluster.neg_width_ratio_sq = accum_var_neg_width_ratio / tmp_var_neg_cnter


                    # calc frontal corner weight of the cluster (this value is expected to be large if the cluster has antennae protruding from its head)
                    cluster.antenna_weight = 0.
                    tmp_cnter = 0
                    tmp_antenna_weight = 0.

                    # now "cluster.width" was defined, let's re-calculate "tmp_length" and "tmp_width"
                    for i,velocity_x in enumerate(velocity_x_history):
                        velocity_y = velocity_y_history[i]
                        velocity = np.array([velocity_x, velocity_y]) # calc velocity vector 
                        len_velocity = np.linalg.norm(velocity, ord=2) # calc length of velocity vector (L2 norm. Euclidean)
                        if len_velocity == 0.0:
                            continue

                        norm_velocity = velocity / len_velocity # normarized vector of velocity
                        norm_orth_velocity = np.array([velocity_y, -velocity_x]) / len_velocity # normarized vector orthogonal to the vector of velocity

                        for corner in valid_corners[i + track_offset]: # track_offset = config.NUM_TRACK_TO_AVERAGE//2
                            # for each corner, calc the inner dot with "norm_velocity" and "norm_orth_velocity"
                            vec_to_corner = np.array( [ corner[0] - center_x_history_ave[i], corner[1] - center_y_history_ave[i] ] ) # temporal vector from the centroid
                            tmp_length = np.dot(vec_to_corner, norm_velocity)
                            tmp_width = np.dot(vec_to_corner, norm_orth_velocity)

                            if abs(tmp_width)*2 > cluster.width:
                                tmp_cnter += 1
                                if tmp_length > 0.:
                                    tmp_antenna_weight += tmp_length
                                else:
                                    tmp_antenna_weight += tmp_length / config.ANTENNA_WEIGHT_BIAS # decrease tail's influence by division by "ANTENNA_WEIGHT_BIAS"

                    # calc the final frontal corner weight
                    if tmp_cnter:
                        cluster.antenna_weight = (tmp_antenna_weight/tmp_cnter) / cluster.length


                    # count jumps with prominently large acceleration
                    cluster.jump_cnter = 0

                    #if cluster.width >= 30:
                    vel_x_history = center_x_history[1:] - center_x_history[:-1] # momentary velocity without averaging. /track_step_ms [ms]. default 4 [ms]
                    vel_y_history = center_y_history[1:] - center_y_history[:-1] # momentary velocity without averaging. /track_step_ms [ms]. default 4 [ms]
                    acc_x_history = vel_x_history[1:] - vel_x_history[:-1] # acceleration. /track_step_ms [ms]. default 4 [ms]
                    acc_y_history = vel_y_history[1:] - vel_y_history[:-1] # acceleration. /track_step_ms [ms]. default 4 [ms]
                    sq_acc_history = np.add( np.square(acc_x_history), np.square(acc_y_history) ) # square of the length of acceleration vector
                    sq_acc_history_ave = np.average( sq_acc_history )
                    #(idxes_large_acc,) = np.where( (acc_history/cluster.width) > 0.1 )
                    (idxes_large_acc,) = np.where( sq_acc_history > (sq_acc_history_ave*config.SQ_JUMP_CNTER_TH) ) # idxes with prominently large acceleration

                    # count acceleration jump intermittently ignoring the head and tial of its trajectory
                    last_idx = 0
                    for idx in idxes_large_acc:
                        if ( len(sq_acc_history) - config.JUMP_CNTER_OFFSET ) <= idx:
                            # when the current "idx" is around the tail of the trajectory, break
                            break
                        if idx >= (last_idx + config.JUMP_CNTER_OFFSET): # if idxes_large_acc[0] = 0, it will be ignored
                            # when the current "idx" is not too close to the "last_idx" or not around the head of the trajectory, increment "jump_cnter"
                            cluster.jump_cnter += 1
                        last_idx = idx # update "last_idx"


                    # calc pos to neg distance divided by length/width
                    center_x_pos_history = np.array( [ row[0] for row in valid_track_g_pos_history ] )
                    center_y_pos_history = np.array( [ row[1] for row in valid_track_g_pos_history ] )

                    center_x_neg_history = np.array( [ row[0] for row in valid_track_g_neg_history ] )
                    center_y_neg_history = np.array( [ row[1] for row in valid_track_g_neg_history ] )

                    center_x_diff_history = center_x_pos_history - center_x_neg_history
                    center_y_diff_history = center_y_pos_history - center_y_neg_history
                    center_diff_history_sq = np.add( np.square(center_x_diff_history), np.square(center_y_diff_history) )
                    center_diff_sq_ave = np.average( center_diff_history_sq )

                    cluster.center_diff_length_ratio_sq = center_diff_sq_ave / (cluster.length**2)
                    cluster.center_diff_width_ratio_sq = center_diff_sq_ave / (cluster.width**2)


                    # calc FFT
                    cluster.coor_fft_peaks = [] # for storing detected coordinate FFT peaks
                    cluster.event_fft_peaks = [] # for storing detected event FFT peaks

                    if config.ENABLE_FFT:
                        print(f"FFT of cluster: {cluster.ID}")

                        # draw coordinate FFT figures
                        coor_fft.process( png_header, "_all", cluster, center_x_history, center_y_history )
                        coor_fft.process( png_header, "_pos", cluster, center_x_pos_history, center_y_pos_history )
                        coor_fft.process( png_header, "_neg", cluster, center_x_neg_history, center_y_neg_history )

                        # in order to draw event FFT figures, let's decode event history first
                        if (cluster.last_track_t - cluster.first_track_t) >= config.EVENT_FFT_TH: # tracked for "EVENT_FFT_TH" or more
                            all_event_history = [] # total of positive and negative enents
                            pos_event_history = [] # positive events
                            next_timestamp_ms = cluster.event_history[0][0] # initialize
                            for entry in cluster.event_history:
                                len_skip = entry[0] - next_timestamp_ms # detect jump in event history timestamp
                                for i in range(len_skip):
                                    # fill the jump in event history
                                    all_event_history.append(0)
                                    pos_event_history.append(0)

                                all_event_history.append(entry[1])
                                pos_event_history.append(entry[2])
                                next_timestamp_ms = entry[0] + config.cluster_step_ms # update "next_timestamp_ms"

                            # convert to numpy array and call the FFT function
                            np_all_event_history = np.array(all_event_history)
                            np_pos_event_history = np.array(pos_event_history)
                            event_fft.process( png_header, cluster, np_all_event_history, np_pos_event_history )


                    # set "active_level"
                    if cluster.velocity_ave > config.ACTIVE_VEL_TH:
                        # when "cluster" is fast enough to be active
                        if cluster.var_velocity > config.ACTIVE_VAR_TH:
                            # when "cluster" is active
                            if cluster.var_velocity > config.VERY_ACTIVE_VAR_TH:
                                # when "cluster" is very active
                                cluster.active_level = 2
                            else:
                                cluster.active_level = 1


                    # write the result to features.csv
                    f_output_txt_feature.write( f"{cluster.ID}," )

                    # write "first_track_t" to features.csv
                    cur_sec = cluster.first_track_t//1000
                    cur_min = cur_sec//60
                    cur_hour = cur_min//60
                    f_output_txt_feature.write( str(cur_hour).zfill(2) + ':' + str(cur_min%60).zfill(2) + ':' + str(cur_sec%60).zfill(2) )

                    f_output_txt_feature.write( "," )

                    # write "last_track_t" to features.csv
                    cur_sec = cluster.last_track_t//1000
                    cur_min = cur_sec//60
                    cur_hour = cur_min//60
                    f_output_txt_feature.write( str(cur_hour).zfill(2) + ':' + str(cur_min%60).zfill(2) + ':' + str(cur_sec%60).zfill(2) )

                    # write the rest of info
                    f_output_txt_feature.write( f",{cluster.velocity_ave},{cluster.var_velocity},{cluster.jump_cnter},{cluster.length},{cluster.width},{cluster.antenna_weight}" )
                    f_output_txt_feature.write( f",{cluster.pos_length_ratio_sq},{cluster.pos_width_ratio_sq},{cluster.neg_length_ratio_sq},{cluster.neg_width_ratio_sq}" )
                    f_output_txt_feature.write( f",{cluster.center_diff_length_ratio_sq},{cluster.center_diff_width_ratio_sq}" )

                    # prepare "feature_vec" for neural network input
                    feature_vec = np.array([ cluster.velocity_ave, cluster.var_velocity, cluster.jump_cnter, cluster.length, cluster.width, cluster.antenna_weight\
                                            , cluster.pos_length_ratio_sq, cluster.pos_width_ratio_sq, cluster.neg_length_ratio_sq, cluster.neg_width_ratio_sq\
                                            , cluster.center_diff_length_ratio_sq, cluster.center_diff_width_ratio_sq, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. ], dtype=np.float64)


                    # FFT peak statistics
                    list_num_event_peak = [0, 0, 0, 0, 0, 0]
                    for peak in cluster.event_fft_peaks:
                        for i,fq_border in enumerate(config.LIST_FQ_BORDER): # LIST_FQ_BORDER = [1., 2.15443469003188, 4.64158883361278, 10., 21.5443469003188] # [Hz]
                            if peak[0] < fq_border:
                                list_num_event_peak[i] += 1
                                break
                            elif i == len(list_num_event_peak) - 2: # up to 22 Hz
                                list_num_event_peak[i+1] += 1
                                break

                    list_num_coor_peak = [0, 0, 0, 0]
                    for peak in cluster.coor_fft_peaks:
                        for i,fq_border in enumerate(config.LIST_FQ_BORDER): # LIST_FQ_BORDER = [1., 2.15443469003188, 4.64158883361278, 10., 21.5443469003188] # [Hz]
                            if peak[0] < fq_border:
                                list_num_coor_peak[i] += 1
                                break
                            elif i == len(list_num_coor_peak) - 2: # up to 4.6 Hz
                                list_num_coor_peak[i+1] += 1
                                break

                    # write FFT peak statistics to features.csv and "feature_vec"
                    for i,num_event_peak in enumerate(list_num_event_peak):
                        f_output_txt_feature.write( f",{num_event_peak}" )
                        feature_vec[ config.COLUMN_IDX_FEATURE_OFFSET+9+i ] = np.float64(num_event_peak)
                    for i,num_coor_peak in enumerate(list_num_coor_peak):
                        f_output_txt_feature.write( f",{num_coor_peak}" )
                        feature_vec[ config.COLUMN_IDX_FEATURE_OFFSET+15+i ] = np.float64(num_coor_peak)


                    if config.FLG_INFERENCE: # if inference by neural network is enabled
                        feature_vec_norm = feature_vec / config.NP_NORM_CEIL # normalize the input vector
                        #feature_vec_norm[ feature_vec_norm > 1. ] = 1.
                        feature_vec_norm = np.where(feature_vec_norm > 1., 1., feature_vec_norm) # trim too large values
                        feature_vec_norm = np.where(feature_vec_norm < 0., 0., feature_vec_norm) # trim too small values
                        feature_vec_norm.shape += (1,)
                        #print(feature_vec)
                        #print(feature_vec.shape)

                        # input to hidden layer
                        #h_tmp = b_i_h + w_i_h @ feature_vec_norm.reshape(22, 1)
                        h_tmp = b_i_h + w_i_h @ feature_vec_norm
                        h = 1 / (1 + np.exp(-h_tmp))
                        # hidden to output layer
                        o_tmp = b_h_o + w_h_o @ h
                        o = 1 / (1 + np.exp(-o_tmp))
                        cluster.inference = config.Plankton( o.argmax() ).name
                        #print( config.Plankton(cluster.inference).name )

                        # write the inference result and the output vector(score) to features.csv
                        f_output_txt_feature.write( "," + cluster.inference )
                        for inference_score in o:
                            f_output_txt_feature.write( f",{inference_score[0]}" )
                        f_output_txt_feature.write( "," + cluster.annotation )
                    else:
                        # write empty data with the same number of comma
                        f_output_txt_feature.write( "," )
                        for i in range(config.NUM_CLASSIFICATION):
                            f_output_txt_feature.write( "," )
                        f_output_txt_feature.write( "," + cluster.annotation )

                    f_output_txt_feature.write( "\n" ) # end of .csv line


        # overwrite "*s.pkl"
        with open(f_clusters_path, 'wb') as f_pkl_nm:
            pickle.dump(clusters, f_pkl_nm)

    if config.FLG_CALC_FEATURES:
        f_output_txt_feature.close()
        f_output_txt_fft_peak.close()


    # write num_vs_minute.csv and num_vs_width.csv
    with open( common.analysis_path + "num_vs_minute.csv", 'w' ) as f_output_txt_num_vs_time:
        for idx,item in enumerate(list_num_vs_time):
            f_output_txt_num_vs_time.write( f"{idx+1},{item[0]},{item[1]},{item[2]}\n" )

    with open( common.analysis_path + "num_vs_width.csv", 'w' ) as f_output_txt_num_vs_width:
        for idx,item in enumerate(list_num_vs_width):
            f_output_txt_num_vs_width.write( f"{idx},{item}\n" )

    print(f"\nAnalyzing clusters finished at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")



if __name__ == "__main__":
    main()
