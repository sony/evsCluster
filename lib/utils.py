# coding: utf-8

# Copyright 2023 Sony Group Corporation - All Rights Reserved.
# Subject to the terms and conditions contained in LICENSE.txt accompanying this file, you may use this file.

import os
import cv2
import numpy as np
from numba import jit, i1, i2, i4, i8
from numba.experimental import jitclass
from scipy.spatial import ConvexHull
import copy
import pickle
import time
import glob
import config_by_gui
import config
from lib import common
#np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(threshold=np.inf) # needed to write out long txt file. DO NOT COMMENT OUT!!!


spec = [
    ("x", i2),
    ("y", i2),
    ("polarity", i1),
    ("timestamp_ms", i4),
    ("ID", i8),
    ("list_x", i2[:]),
    ("list_y", i2[:]),
    ("list_polarity", i1[:]),
    ("list_timestamp_ms", i4[:]),
    ("list_unique_x", i2[:]),
    ("list_unique_y", i2[:]),
    ("event_history", i4[:,:]),
    ("track_history", i2[:,:]),
    ("inactive_flg", i1),
    ("track_update_flg", i1),
    ("list_unique", i2[:]),
    ("a", i2)
]

@jitclass(spec)
class Subcluster:
    #subcluster_serial = 0

    def __init__(self, x, y, polarity, timestamp_ms, ID):
        #self.subcluster_serial += 1
        #self.ID = self.subcluster_serial
        self.ID = ID # ID values only larger than 0 are used
        #self.prev = None
        #self.next = None

        #self.list_xy = np.array( [[x, y]], dtype=np.int16 )
        self.list_x = np.array( [x], dtype=np.int16 )
        self.list_y = np.array( [y], dtype=np.int16 )
        self.list_polarity = np.array( [polarity], dtype=np.int8 )
        self.list_timestamp_ms = np.array( [timestamp_ms], dtype=np.int32 )

        self.list_unique_x = np.array( [x], dtype=np.int16 )
        self.list_unique_y = np.array( [y], dtype=np.int16 )

        self.event_history = np.array( [[ timestamp_ms, 1, polarity ]], dtype=np.int32 ) # event counter updated every 1 ms (by default) if a new event happens on the subcluster
        self.track_history = np.array( [[ x, x, y, y, 1, polarity, 0 ]], dtype=np.int16 ) # tracking history written every time "update_subcluster()" func is called
        self.inactive_flg = 0
        self.track_update_flg = 0

    def update_list_unique(self, list_unique, a):
        idx = np.searchsorted( list_unique , a )
        if idx == len( list_unique ): # if length is the same, append to the last element
            return np.append( list_unique , a )
        elif list_unique[idx] != a: # if a is not there yet
            #self.list_unique_x = np.insert( self.list_unique_x , idx, x )
            return np.concatenate( (list_unique[:idx], np.array([a], dtype=np.int16), list_unique[idx:]), axis=0 )
        return list_unique

    def append(self, x, y, polarity, timestamp_ms):
        #self.list_xy = np.append( self.list_xy, np.array([[x,y]]) , axis=0 ) # append to the pixel list
        self.list_x = np.append( self.list_x, x )
        self.list_y = np.append( self.list_y, y )
        self.list_polarity = np.append( self.list_polarity, polarity )
        self.list_timestamp_ms = np.append( self.list_timestamp_ms, timestamp_ms )

        self.list_unique_x = self.update_list_unique(self.list_unique_x , x)
        self.list_unique_y = self.update_list_unique(self.list_unique_y , y)

        # self.track_update_flg = 1 # "track_update_flg" is set after execution of this "append" func anyway

    def update_event_history(self, polarity, timestamp_ms):
        if self.event_history[-1][0] == timestamp_ms:
            self.event_history[-1][1] += 1 # num_new_pxls
            self.event_history[-1][2] += polarity # num_new_pos_pxls
        else:
            self.event_history = np.append( self.event_history, np.array([[ timestamp_ms, 1, polarity ]], dtype=np.int32), axis=0 ) # 3rd element calcs the sum of polarity in a subcluster. This equals to the number of positive pxls in the subcluster

    def update_track_history(self):
        if self.track_update_flg:
            # create temporal track entry
            new_track_entry = np.array([[ self.list_unique_x[0], self.list_unique_x[-1], self.list_unique_y[0], self.list_unique_y[-1], len(self.list_polarity), np.sum(self.list_polarity), 1 ]], dtype=np.int16)
            if np.array_equal( self.track_history[-1][0:6] , new_track_entry[0][0:6] ):
                # when no new track entry is to be appended to "track_history", increment the last entry's iteration counter
                self.track_history[-1][6] += 1
            else:
                self.track_history = np.append( self.track_history, new_track_entry, axis=0 )
            #if subcluster.ID == 8:
                #print( str(subcluster.ID) + " tracked at: " + str(timestamp_ms) + " where:" + str(new_track_entry[0]) + "," + str(new_track_entry[1]) )
            self.track_update_flg = 0
        else:
            # when no new track entry is to be appended to "track_history", increment the last entry's iteration counter
            self.track_history[-1][6] += 1

    def delete_obsolete(self, timestamp_ms):
        (old_idxes,) = np.where( ( timestamp_ms - self.list_timestamp_ms ) > config.SUBCLUSTER_DUR )
        if old_idxes.size:
            #self.list_xy = np.concatenate( [self.list_xy[:old_idxes], self.list_xy[old_idxes+1:]], axis=0 )
            #new_list_x = np.delete( self.list_xy[:,0:1], old_idxes )
            #new_list_y = np.delete( self.list_xy[:,1:2], old_idxes )
            #self.list_xy = np.vstack( (new_list_x, new_list_y) ).T

            self.list_x = np.delete( self.list_x, old_idxes )
            self.list_y = np.delete( self.list_y, old_idxes )
            self.list_polarity = np.delete( self.list_polarity, old_idxes )
            self.list_timestamp_ms = np.delete( self.list_timestamp_ms, old_idxes )
            self.track_update_flg = 1

            if len(self.list_polarity): # "list_polarity" is the easiest to count length
                # update the unique x/y lists
                #self.list_unique_x = np.unique( self.list_xy[:, 0:1].ravel() ) # sorted x_only
                #self.list_unique_y = np.unique( self.list_xy[:, 1:2].ravel() ) # sorted y_only
                self.list_unique_x = np.unique( self.list_x ) # sorted x_only
                self.list_unique_y = np.unique( self.list_y ) # sorted y_only
            else:
                self.inactive_flg = 1 # inactivate the subcluster
                return

        # inactivate subclusters with sparse pixels
        if ( len( self.list_unique_x )*1.2 < (self.list_unique_x[-1] - self.list_unique_x[0]) ) or ( len( self.list_unique_y )*1.2 < (self.list_unique_y[-1] - self.list_unique_y[0]) ):
            self.inactive_flg = 1 # inactivate the subcluster
            return

        # update tracking history
        self.update_track_history()

    def add_to_subcluster(self, x, y, polarity, timestamp_ms):
        # when the same pixel has already been regstered in the dominant subcluster, update its corresponding pixel's timestamp
        # note a pxl with the same x and y might be already in the subcluster (and overwritten by another subcluster). So be sure not to add a duplicate pxl with the same coordinates
        #(exactly_matched_idxes,) = np.where( np.sum( abs( sel_subcluster.list_xy - np.array([x,y], dtype=np.int16) ) , axis=1 ) == 0 )
        (matched_x_idxes,) = np.where( self.list_x == x )
        (exactly_matched_idx,) = np.where( self.list_y[matched_x_idxes] == y )
        #(matched_y_idxes,) = np.where( sel_subcluster.list_y == new_y )
        #exactly_matched_idxes = np.intersect1d(matched_x_idxes, matched_y_idxes)
        if exactly_matched_idx.size:
            if self.list_polarity[ matched_x_idxes[exactly_matched_idx] ] != polarity:
                # when the polarity of an existing pxl has changed
                self.list_polarity[ matched_x_idxes[exactly_matched_idx] ] = polarity
                self.track_update_flg = 1
            self.list_timestamp_ms[ matched_x_idxes[exactly_matched_idx] ] = timestamp_ms # simply update the timestamp of the same pixel postion
            self.update_event_history(polarity, timestamp_ms)
            return 1
        elif max( self.list_unique_x[-1] - self.list_unique_x[0] , self.list_unique_y[-1] - self.list_unique_y[0] ) < config.SUBCLUSTER_SIZE_LIMIT: # allow for each subcluster to grow up to 20(=19+1)) pxl length
            self.append(x, y, polarity, timestamp_ms)
            self.update_event_history(polarity, timestamp_ms)
            self.track_update_flg = 1
            return 1
        else:
            return 0


#@jitclass(spec)
class Subcluster_decoded:
    def __init__(self, ID, merged_idx, first_track_t, last_track_t, xy_limit, event_history, track_r_history):
        self.ID = ID # ID values only larger than 0 are used
        self.merged_idx = merged_idx # -2 means the subcluster is confined in a banned region. -1 means it has not been merged yet.
        self.list_brother_idx = [np.array([], dtype=np.int32), np.array([], dtype=np.int32)] # [list_older_brother_idx, list_younger_brother_idx]

        self.first_track_t = first_track_t # Unit = milliseconds
        self.last_track_t = last_track_t # Unit = milliseconds
        self.xy_limit = xy_limit # xy_limit = [xmin, xmax, ymin, ymax]

        self.event_history = np.array(event_history, dtype=np.int32)
        self.track_r_history = track_r_history # rectangle and "len_pxls" history. [[np.array([])], [np.array([])], [np.array([])]]


#@jitclass(spec)
class Cluster:
    def __init__(self, subcluster, subcluster_idx):
        self.ID = subcluster.ID # ID values only larger than 0 are used
        self.list_subcluster_idx = np.array([subcluster_idx], dtype=np.int32) # idx in the list of "all_subclusters"

        self.first_track_t = subcluster.first_track_t # Unit = milliseconds
        self.last_track_t = subcluster.last_track_t # Unit = milliseconds
        self.xy_limit = copy.deepcopy(subcluster.xy_limit) # xy_limit = [xmin, xmax, ymin, ymax]

        self.event_history = copy.deepcopy(subcluster.event_history)
        self.track_r_history = copy.deepcopy(subcluster.track_r_history) # rectangle and "len_pxls" history. [[np.array([])], [np.array([]), np.array([])], [np.array([])]]

        self.track_g_all_history = [] # centroid track of all events. ("g" means "gravity") [centroid_x, centroid_y, total_num_pixel, proximity_to_fringe]
        self.track_g_pos_history = [] # positive centroid track. [ centroid_x, centroid_y, total_weight ]
        self.track_g_neg_history = [] # negative centroid track. [ centroid_x, centroid_y, total_weight ]
        self.velocity_ave = 0. # average velocity. unit: [pixel/ms]
        self.var_velocity = 0. # variance of velocity (recognized as active if > 0.14 (by default) and as very active if >1.0 (by default) )
        self.jump_cnter = 0 # count jumps with prominently large acceleration
        self.length = 0 # average length of Convex-Hull (parallel to its velocity vector)
        self.width = 0 # average width of Convex-Hull (orthogonal to its velocity vector)
        self.pos_length_ratio_sq = 0. # positive variance parallel to its velocity vector divided by its length^2
        self.pos_width_ratio_sq = 0. # positive variance orthogonal to its velocity vector divided by its width^2
        self.neg_length_ratio_sq = 0. # negative variance parallel to its velocity vector divided by its length^2
        self.neg_width_ratio_sq = 0. # negative variance orthogonal to its velocity vector divided by its width^2
        self.center_diff_length_ratio_sq = 0. # squared distance between positive and negative centers divided by its length^2
        self.center_diff_width_ratio_sq = 0. # squared distance between positive and negative centers divided by its width^2
        self.antenna_weight = 0. # frontal corner weight of the cluster (this value is expected to be large if the cluster has antennae protruding from its head)

        self.coor_fft_peaks = [] # [ [ freq, diff_dB(= peak dB - average of 2-octave-range dB) ] , [freq, diff_dB], [freq, diff_dB] ,,, ]
        self.event_fft_peaks = [] # [ [ freq, diff_dB(= peak dB - average of 2-octave-range dB) ] , [freq, diff_dB], [freq, diff_dB] ,,, ]

        self.active_level = 0 # 0:inactive, 1:active, 2:very active
        self.inference = "" # inferred "Plankton" name
        self.annotation ="" # correct "Plankton" name given as teacher data

        self.draw_idx = 0 # for the ease of drawing

        self.inactive_flg = 0
        self.remerged_flg = 0
        self.insignificance = 0
        self.last_packed_track_idx = 0




def update_subclusters(timestamp_ms):
    tmp_cnter = 0 # needed to delete a subcluster avoiding index shifting
    for subcluster in common.subclusters[:]:
        # delete obsolete pixels in a subcluster, inactivating the subcluster if necessary
        subcluster.delete_obsolete(timestamp_ms)

        if subcluster.inactive_flg:
            # when the "subcluster" is inactive, write to the output file if its "track_history" is significant enough
            save_subcluster(subcluster)

            del common.subclusters[ tmp_cnter ]
            common.subcluster_cnter -= 1
            continue
        tmp_cnter += 1

    # update "img_subsubcluster_idxes" and subcluster info/history
    common.img_subcluster_idxes = np.copy(common.blank_img_idx) # initialize "common.img_subcluster_idxes" with 0s
    for i,subcluster in enumerate(common.subclusters):
        # update "common.img_subcluster_idxes". Notice only active subclusters remain here
        #for j,x in enumerate(subcluster.list_x):
            #common.img_subcluster_idxes[ subcluster.list_y[j] , x ] = i+1 # 0 means None. So "+1" is needed
        common.img_subcluster_idxes[ subcluster.list_y, subcluster.list_x ] = i+1 # 0 means None. So "+1" is needed


kernel_size_half = config.KERNEL_SIZE//2 # default: 2, since the default value of "KERNEL_SIZE" is 5
border_left = kernel_size_half+1
border_right = config.WIDTH-(kernel_size_half+2)
border_up = kernel_size_half+1
border_down = config.HEIGHT-(kernel_size_half+2)

def cluster_new_pxls():
    #new_pxls = np.array( list_new_pxls , dtype='int' )
    #new_pxls = np.array( list_new_pxls , dtype=np.int32 )
    #new_pxls = np.array( list_new_pxls , dtype={'names':['x','y','p','t'], 'formats':[np.uint16,np.uint16,np.uint8,np.uint64], 'offsets':[0,2,4,8], 'itemsize':16} )
    #new_xys = np.delete( new_pxls , slice(2,4) , axis=1 )

    for new_pxl in common.list_new_pxls:
        new_x, new_y = new_pxl[0], new_pxl[1]

        if (new_x < border_left) or (border_right < new_x) or (new_y < border_up) or (border_down < new_y): # this should be faster than "and"
            continue

        new_y_up = new_y-kernel_size_half
        new_y_down = new_y+(kernel_size_half+1)
        new_x_left = new_x-kernel_size_half
        new_x_right = new_x+(kernel_size_half+1)

        # check if neighbouring pixels have had a descent amount of events recently
        #binary_img_buf_neighbour_sum = binary_img_buf[ : , new_y-2 : new_y+3 , new_x-2 : new_x+3 ].sum()
        binary_img_buf_neighbour_sum = common.binary_img_buf[ : , new_y_up : new_y_down , new_x_left : new_x_right ].sum() - common.binary_img_buf[ : , new_y , new_x ].sum() # subtract the center
        if binary_img_buf_neighbour_sum >= config_by_gui.NEIGHBOUR_TH:
        #if (2 <= binary_img_buf_neighbour_sum) and (binary_img_buf_neighbour_sum < 5): # this was not good for fast moving clusters
            np_new_x, np_new_y, np_new_polarity, np_timestamp_ms = np.int16(new_x), np.int16(new_y), np.int8(new_pxl[2]), np.int32(common.timestamp_ms)
            #g_timestamp_ms = np.int32(timestamp_ms) # just for debug

            #fist, check if the new pixel should belong to an existing claster
            #neighbour_subcluster_idx_5x5 = common.img_subcluster_idxes[ new_y-2 : new_y+3 , new_x-2 : new_x+3 ]
            #neighbour_subcluster_idx_5x5 = common.img_subcluster_idxes[ new_y_up : new_y_down , new_x_left : new_x_right ]
            #(tmp_subcluster_idxes, counts) = np.unique( np.array( neighbour_subcluster_idx_5x5[ neighbour_subcluster_idx_5x5.nonzero() ] ) , return_counts=True ) # "np.unique( [] , return_counts=True )" returns values=[] and counts=[]
            (tmp_subcluster_idxes, counts) = np.unique( common.img_subcluster_idxes[ new_y_up : new_y_down , new_x_left : new_x_right ] , return_counts=True ) # "np.unique( [] , return_counts=True )" returns values=[] and counts=[]
            if tmp_subcluster_idxes[0] == 0:
                # when some of the neighbouring pixels did not belong to any subcluster yet, omit the corresponding item from the list
                tmp_subcluster_idxes = tmp_subcluster_idxes[1:]
                counts = counts[1:]
            if len(counts):
                # when there is a dominant subcluster in the area
                clustered_flg = 0
                #max1st_idx = np.argmax(counts)
                #dominant_subcluster_idx = tmp_subcluster_idxes[ max1st_idx ] - 1 # need "-1", because 0 of "tmp_subcluster_idxes" menas None

                # sort "idxes" of "subclusters" by dominance
                #sorted_subcluster_idxes = tmp_subcluster_idxes[ np.argsort(counts)[::-1] ] - 1
                sorted_subcluster_idxes = tmp_subcluster_idxes[ np.argsort(-counts) ] - 1
                for sorted_subcluster_idx in sorted_subcluster_idxes:
                    if common.subclusters[sorted_subcluster_idx].add_to_subcluster(np_new_x, np_new_y, np_new_polarity, np_timestamp_ms): # the more dominant a subcluster is, the earlier it gets selected.
                        # when the func returned 1, it means the "new_pxl" has been added to the subcluster
                        clustered_flg = 1
                        break

                if clustered_flg:
                    common.img_subcluster_idxes[ new_y , new_x ] = sorted_subcluster_idx + 1
                    continue

            # let the new pixel compose a new claster
            common.subcluster_id += 1
            common.subclusters.append( Subcluster( np_new_x, np_new_y, np_new_polarity, np_timestamp_ms, common.subcluster_id ) )
            common.subcluster_cnter += 1
            common.img_subcluster_idxes[ new_y , new_x ] = common.subcluster_cnter

    common.list_new_pxls.clear()
    common.binary_img_idx += 1 # rotate
    common.binary_img_idx %= config.NUM_BINARY_IMG_BUF # initialize if necessary
    common.binary_img_buf[common.binary_img_idx] = np.copy(common.blank_binary_img) # initialize with 0s


def add_new_pxl(x, y, polarity, flg_subcluster):
    if flg_subcluster:
        #common.list_new_pxls.append( [x, y, polarity, timestamp_ms] )
        common.list_new_pxls.append( [x, y, polarity] )
        common.binary_img_buf[common.binary_img_idx][ y, x ] += 1
    if config_by_gui.ENABLE_MONITORING or (not flg_subcluster):
        common.img[ y, x ] = config.CD_ON if polarity else config.CD_OFF #[0,0,63] #red #[63,0,0] #blue


frame_dur = 1000000 / config.FPS # [us]
font = cv2.FONT_HERSHEY_SIMPLEX
clip_fps = float(config.FPS)
#codec = cv2.VideoWriter_fourcc(*'mp4v')
codec = cv2.VideoWriter_fourcc(*'MJPG') # mjpg

def draw_img(timestamp, flg_subcluster):
    if flg_subcluster:
        for subcluster in common.subclusters:
            #if subcluster.active_flg:
                # when the subcluster is active, draw a bounding box
            cv2.rectangle( common.img, (subcluster.list_unique_x[0], subcluster.list_unique_y[0]), (subcluster.list_unique_x[-1], subcluster.list_unique_y[-1]), [255,191,0], thickness=1 )
    else:
        # draw isolated subclusters
        tmp_cnter = 0 # needed to delete an isolated subcluster avoiding index shifting
        for subcluster in common.isolated_subclusters[:]:
            # delete obsolete isolated subclusters
            if subcluster.last_track_t < common.timestamp_ms:
                del common.isolated_subclusters[ tmp_cnter ]
                continue
            tmp_cnter += 1

            if common.timestamp_ms < subcluster.first_track_t:
                # when the first timestamp is out of the timestamp range of the latest video frame
                break

            for a_rectangle_at_a_time in subcluster.track_r_history[:]:
                if subcluster.first_track_t <= common.timestamp_ms:
                    # draw a bounding box of the isolated subcluster
                    cv2.rectangle( common.img, 
                                  (a_rectangle_at_a_time[0][0], a_rectangle_at_a_time[0][2]), 
                                  (a_rectangle_at_a_time[0][1], a_rectangle_at_a_time[0][3]), [255,191,0], thickness=1 ) #191,127,0
                    subcluster.first_track_t += config.track_step_ms # update "first_track_t" until it gets bigger than "timestamp_ms"
                    del subcluster.track_r_history[0] # delete first track entry


        # draw clusters
        tmp_cnter = 0 # needed to delete a cluster avoiding index shifting
        for cluster in common.clusters[:]:
            if cluster.last_track_t < common.timestamp_ms:
                del common.clusters[ tmp_cnter ]
                continue
            tmp_cnter += 1

            if common.timestamp_ms < cluster.first_track_t:
                # when the first timestamp is out of the timestamp range of the latest video frame
                break

            # draw Convex-Hull of the cluster
            points = [] # points of cluster's "rectangle" corners within the timestamp range of the latest video frame. Used for Convex-Hull
            for rectangles_at_a_time in cluster.track_r_history[:]: # rectangles_at_a_time = [ [318, 324, 652, 658, 3], [323, 328, 648, 652, 4], [315, 325, 657, 658, 2] ] (for example)
                if cluster.first_track_t <= common.timestamp_ms:
                    for rectangle in rectangles_at_a_time:
                        # if the rectangle does not have length or width, skip
                        if rectangle[0] == rectangle[1]:
                            continue
                        if rectangle[2] == rectangle[3]:
                            continue

                        # register each bounding box's corners to "points"
                        points.extend( [ [rectangle[0], rectangle[2]] , [rectangle[1], rectangle[2]] , [rectangle[1], rectangle[3]] , [rectangle[0], rectangle[3]] ] )

                    cluster.first_track_t += config.track_step_ms # update "first_track_t" until it gets bigger than "timestamp_ms"
                    del cluster.track_r_history[0] # delete first track entry
                    cluster.draw_idx += 1 # update "draw_idx"
                else:
                    break

            if len(points):
                # calc Convex-Hull
                convex_hull = ConvexHull(points)
                hull_points = convex_hull.points
                hull_points_selected = hull_points[convex_hull.vertices]
                hull_points = np.vstack((hull_points_selected, hull_points_selected[0]))
                hull_points_int = hull_points.astype(int)

                # draw Convex-Hull
                hull_points_item = hull_points_int[-1]
                for hull_point in hull_points_int:
                    #cv2.line( img, hp_item, hp_point, COLORS[ cluster.ID % 16 ], thickness=1 )
                    if not config.FLG_INFERENCE:
                        cv2.line( common.img, hull_points_item, hull_point, [0,255,0], thickness=1 )
                    elif cluster.inference == config.Plankton(0).name: # Passive
                        cv2.line( common.img, hull_points_item, hull_point, [255,0,0], thickness=1 )
                    elif cluster.inference == config.Plankton(1).name: # Pcra_C
                        cv2.line( common.img, hull_points_item, hull_point, [0,255,255], thickness=1 )
                    elif cluster.inference == config.Plankton(2).name: # Pcra_N
                        cv2.line( common.img, hull_points_item, hull_point, [255,255,0], thickness=1 )
                    elif cluster.inference == config.Plankton(3).name: # Nfus
                        cv2.line( common.img, hull_points_item, hull_point, [0,0,255], thickness=1 )
                    elif cluster.inference == config.Plankton(4).name: # Ppec
                        cv2.line( common.img, hull_points_item, hull_point, [255,0,255], thickness=1 )
                    else:
                        cv2.line( common.img, hull_points_item, hull_point, [0,255,0], thickness=1 )

                    hull_points_item = hull_point


            # draw trajectory of the centroid
            if len(cluster.track_g_all_history) >= 2:
                # when "track_g_all_history" is long enough
                g_entry = cluster.track_g_all_history[0]
                for j,g_entry_next in enumerate(cluster.track_g_all_history[1:]):
                    if ( (cluster.draw_idx - config.dur_to_show_div) < j ) and ( j <= cluster.draw_idx ):
                        # when "j" is in the valid idx range to draw, draw trajectory lines
                        cv2.line( common.img, ( int(g_entry[0]), int(g_entry[1]) ), ( int(g_entry_next[0]), int(g_entry_next[1]) ), config.COLORS[ cluster.ID%13 ], thickness=1 ) #%16

                    # print cluster ID
                    if j == cluster.draw_idx:
                        if cluster.active_level == 2:
                            cv2.putText( common.img, str(cluster.ID), [ int(g_entry_next[0]), int(g_entry_next[1]) ], font, 0.6, (0,0,255), 1 , cv2.LINE_AA )
                        elif cluster.active_level == 1:
                            cv2.putText( common.img, str(cluster.ID), [ int(g_entry_next[0]), int(g_entry_next[1]) ], font, 0.6, (255,0,255), 1 , cv2.LINE_AA )
                        elif cluster.active_level == 0:
                            cv2.putText( common.img, str(cluster.ID), [ int(g_entry_next[0]), int(g_entry_next[1]) ], font, 0.6, (0,255,0), 1 , cv2.LINE_AA )

                    g_entry = g_entry_next

    cur_subsec = common.timestamp_ms//10
    cur_sec = cur_subsec//100
    cur_min = cur_sec//60
    cur_hour = cur_min//60
    cv2.putText( common.img, str(cur_hour).zfill(2) + ':' + str(cur_min%60).zfill(2) + ':' + str(cur_sec%60).zfill(2) + '.' + str(cur_subsec%100).zfill(2), (1040,715), font, 1.1, (0,0,255), 1, cv2.LINE_AA )

    #cv2.putText( common.img, str(common.subcluster_id), (100,720), font, 0.5, (0,0,255), 1 , cv2.LINE_AA )

    cv2.imshow('img', common.img)
    #cv2.imwrite("tmp/" + str(timestamp) + ".png", common.img)

    common.video.write(common.img)

    common.img = np.copy(common.blank_img)
    if cv2.waitKey(1) & 0xFF == ord('q'): # wait for 1 [ms] for key inputs
        return 1
    else:
        return 0


def detect_timestamp_jump(timestamp, last_timestamp):
    # detect timestamp jump over a certain length (default >500[us])
    if ( (timestamp - last_timestamp) < 0 ) or ( (timestamp - last_timestamp) > config.TIMESTAMP_JUMP_TH ):
        #print("Timestamp jump alart!")
        #print(raw_or_bin_path)
        #print("Jump from: " + str(last_timestamp) + str(" [us]"))
        print( f"Jump from: {last_timestamp/1000:.3f} [ms]" )

        #print("To  : " + str(timestamp) + str(" [us]"))
        #print("Diff: " + str(timestamp - last_timestamp) + str(" [us]\n"))
        print( f"Diff: { (timestamp-last_timestamp)/1000 :.3f} [ms]\n" )

        # abandon "list_new_pxls"
        common.list_new_pxls.clear()

        # calculate how many cycles of "UPDATING_SUBCLUSTERS_DUR" were jumped over and conpesate for that jump
        jump_cnter = ( timestamp - common.last_time_updating_subclusters ) // config.UPDATING_SUBCLUSTERS_DUR
        if jump_cnter:
            jump_cnter -= 1
        #print(jump_cnter)
        for i in range(jump_cnter):
            common.last_time_updating_subclusters += config.UPDATING_SUBCLUSTERS_DUR
            update_subclusters( common.last_time_updating_subclusters//1000 )
        #update_subclusters( common.last_time_updating_subclusters//1000 + 1000 ) # to chop subclusters at each timestamp jump


def display_elapsed_time(timestamp):
    # display the elapsed time to reach certain timestamp steps (default 1 minute steps)
    if timestamp >= common.timestamp_step:
        cur_time = time.time()
        print( f"{cur_time-common.last_time:.2f} [sec] elapsed until timestamp: {timestamp/60000000:.2f} [min]" )
        common.timestamp_step += config.SUBCLUSTER_TIMESTAMP_MILESTONE_STEP
        common.last_time = cur_time


def f_sub_operation(last_sec, cur_sec, cur_f_path, flg_rename):
    last_min = last_sec//60
    last_hour = last_min//60

    cur_min = cur_sec//60
    cur_hour = cur_min//60

    new_f_path = ( common.analysis_path + common.parent_dir_nm + '_' + common.target_basename + "th" + str(config_by_gui.NEIGHBOUR_TH) + '_'
                    + str(last_hour).zfill(2) + 'h' + str(last_min%60).zfill(2) + 'm' + str(last_sec%60).zfill(2) + 's'
                    + "To" + str(cur_hour).zfill(2) + 'h' + str(cur_min%60).zfill(2) + 'm' + str(cur_sec%60).zfill(2) + "s.sub" )

    if flg_rename:
        try:
            os.rename(cur_f_path, new_f_path)
        except FileExistsError:
            os.remove(new_f_path)
            os.rename(cur_f_path, new_f_path)

        if config_by_gui.ENABLE_MONITORING:
            try:
                os.rename(cur_f_path[:-4]+"_sub.avi", new_f_path[:-4]+"_sub.avi")
            except FileExistsError:
                os.remove(new_f_path[:-4]+"_sub.avi")
                os.rename(cur_f_path[:-4]+"_sub.avi", new_f_path[:-4]+"_sub.avi")
    else:
        common.f_sub_path = new_f_path
        common.f_sub = open(common.f_sub_path, "wb")


def save_subcluster(subcluster):
    if( len(subcluster.track_history) > 2 ): # to be written to the output file, a subcluster needs to have more than 2 history records of tracking
        (len_subcluster_max,) = subcluster.track_history[:,4:5].max(axis=0)
        if len_subcluster_max > 2: # to be written to the output file, a subcluster needs to have more than 2 "len_pxls" at its peak
            common.f_sub.write( ( subcluster.ID ).to_bytes(4, byteorder='little') ) # ID 1byte

            last_timestamp_ms = subcluster.event_history[0][0] # head of event history. numpy.int32(4 bytes)
            common.f_sub.write( ( last_timestamp_ms ).tobytes() ) # first time stamp. 4 bytes
            for event_entry in subcluster.event_history: # event history
                # record only the timestamp difference in milliseconds from the previous entry
                common.f_sub.write( ( event_entry[0]-last_timestamp_ms ).astype(np.int16).tobytes() ) # 2bytes
                last_timestamp_ms = event_entry[0] # update "last_timestamp_ms"
                common.f_sub.write( ( event_entry[1] ).astype(np.int16).tobytes() ) # num_new_pxls 2bytes
                common.f_sub.write( ( event_entry[2] ).astype(np.int16).tobytes() ) # num_new_pos_pxls 2bytes
            common.f_sub.write( b'\x00\x00' ).to_bytes(2, byteorder='little') # separator 2bytes

            if(subcluster.track_history[0][6] == 0):
                # when the num of iteration of the first entry is 0, avoid writing the first entry to the output file
                start_idx = 1
            else:
                start_idx = 0

            for rectangle in subcluster.track_history[start_idx:]: # track history
                common.f_sub.write( ( rectangle[0] ).tobytes() ) # xmin 2bytes
                common.f_sub.write( ( rectangle[1] ).tobytes() ) # xmax 2bytes
                common.f_sub.write( ( rectangle[2] ).tobytes() ) # ymin 2bytes
                common.f_sub.write( ( rectangle[3] ).tobytes() ) # ymax 2bytes
                common.f_sub.write( ( rectangle[4] ).tobytes() ) # len_pxls 2bytes
                common.f_sub.write( ( rectangle[5] ).tobytes() ) # len_pos_pxls 2bytes
                common.f_sub.write( ( rectangle[6] ).astype(np.int8).tobytes() ) # num_iteration 1byte
            common.f_sub.write( b'\xFF\xFF' ).to_bytes(2, byteorder='little') # separator 2bytes

            # if(subcluster.track_history[0][6] == 0):
            #     # when the num of iteration of the first entry is 0, avoid writing the first entry to the output file
            #     #common.f_output.write( str(subcluster.ID) + "\n" + str(subcluster.event_history) + "\n" + str(subcluster.track_history[1:]) + "\n\n" )
            #     common.f_output.write( f"{subcluster.ID}\n{subcluster.event_history}\n{subcluster.track_history[1:]}\n\n" )
            # else:
            #     common.f_output.write( f"{subcluster.ID}\n{subcluster.event_history}\n{subcluster.track_history}\n\n" )
            #if subcluster.ID == 320:
                #print( str(subcluster.ID) + " died at: " + str(timestamp_ms) )


def close_files(flg_subcluster):
    if config_by_gui.ENABLE_MONITORING or (not flg_subcluster):
        common.video.release()
        cv2.destroyAllWindows()

    if flg_subcluster:
        if not common.f_sub.closed:
            #if SALVAGE_LASTING_SUBCLUSTERS:
            for subcluster in common.subclusters[:]:
                save_subcluster(subcluster)

            common.f_sub.write( b'\x00\x00\x00\x00' ).to_bytes(4, byteorder='little') # marker of EOF
            common.f_sub.close()
            f_sub_operation(config_by_gui.LIST_BORDER_TIME[ common.border_time_idx-1 ], round(common.timestamp_ms/1000), cur_f_path=common.f_sub_path, flg_rename=1)

        if not common.f_input.closed:
            common.f_input.close()

        if common.f_bin_border != None:
            if not common.f_bin_border.closed:
                common.f_bin_border.close()
    else:
        common.isolated_subclusters.clear()
        common.clusters.clear()


def set_analysis_path(target_nm, flg_subcluster):
    if target_nm[-4:] == ".raw":
        common.raw_file_mode = 1
        common.target_basename = target_nm[:-4] + "_"
        analysis_path_body = f"{config_by_gui.DIR_PATH}/{common.target_basename}analysis"
        common.list_raw_or_bin_path = [config_by_gui.DIR_PATH + "/" + target_nm]
    else:
        common.raw_file_mode = 0
        common.target_basename = target_nm + "_"
        analysis_path_body = f"{config_by_gui.DIR_PATH}/{target_nm}/analysis"
        common.list_raw_or_bin_path = sorted( glob.glob(config_by_gui.DIR_PATH + "/" + target_nm + "/*.bin") )

    if flg_subcluster:
        # avoid overwriting exsiting "analyis*" folders
        i = 1
        while 1:
            common.analysis_path = f"{analysis_path_body}{i}/"
            if not os.path.exists( common.analysis_path ):
                break
            i += 1
        os.makedirs(common.analysis_path) # , exist_ok=True
        print(f"\n{common.analysis_path} has been created.")
    else:
        # search for the latest analysis* directory
        common.analysis_path = ""
        i = 1
        while 1:
            analysis_path_tmp = f"{analysis_path_body}{i}/"
            if os.path.exists( analysis_path_tmp ):
                common.analysis_path = analysis_path_tmp
            elif common.analysis_path != "":
                break
            
            if i>100:
                print(f"\n\"analysis1\" to \"analysis{i-1}\" were searched for but no analysis folder was found.")
                break
            i += 1
        print(f"\nProcessing the contents of {common.analysis_path}")


if config.HORIZONTAL_OPTICAL_FLIP:
    x_base_shift = -1
else:
    x_base_shift = 1

def process_data(flg_subcluster, flg_evs2video):
    common.timestamp_ms = 0
    common.border_time_idx = 1

    if flg_subcluster:
        if not common.raw_file_mode:
            common.f_bin_border = open(common.analysis_path + "bin_border.txt", "w")

        list_border_time_us = [item*1000000 for item in config_by_gui.LIST_BORDER_TIME]

        common.blank_binary_img = np.zeros((config.HEIGHT, config.WIDTH), dtype=np.int8)
        common.binary_img_buf = np.zeros((config.NUM_BINARY_IMG_BUF, config.HEIGHT, config.WIDTH), dtype=np.int8)
        common.binary_img_idx = 0

        common.img_subcluster_idxes = np.zeros((config.HEIGHT, config.WIDTH), dtype=np.int32)
        common.blank_img_idx = np.zeros((config.HEIGHT, config.WIDTH), dtype=np.int32)

        common.list_new_pxls = [] # [ [x, y, polarity, timestamp(TIME_HIGH + TIME_LOW) ] , [] ,,, [] ] -> [ [x, y, polarity] , [] ,,, [] ]

        # [   [ [pixel],[],[],,, ] , [min_x,,,max_x] , [min_y,,,max_y] , [ ID , ancestor_ID , descendant_ID , original_ID , best_score ] , [ [mid_x , mid_y , range_x , range_y , timestamp_of_the_latest_pixel , len_pixels ] ] , [motion_features] , [inference]  ]
        common.subclusters = []
        common.subcluster_cnter = 0 # for indexing of the "clusters" list
        common.subcluster_id = 0


        # create the basename of the output file
        list_slash_idx = [idx for idx,char in enumerate(config_by_gui.DIR_PATH) if char == '/' ]
        if len(list_slash_idx) >= 1:
            common.parent_dir_nm = config_by_gui.DIR_PATH[ list_slash_idx[-1]+1 : ]
        else:
            common.parent_dir_nm = config_by_gui.DIR_PATH[:-1] # omit semicolon from, for example, "C:"

        # open a file to store subclusters in binary compressed form: .sub
        f_sub_operation(config_by_gui.LIST_BORDER_TIME[0], config_by_gui.LIST_BORDER_TIME[1], cur_f_path="", flg_rename=0)

        if config_by_gui.ENABLE_MONITORING:
            last_frame_start_time = 0

            # Make empty black images of size(HEIGHT,WIDTH)
            common.img = np.zeros((config.HEIGHT, config.WIDTH, 3), dtype=np.uint8) #np.ones()
            common.blank_img = np.copy(common.img)

            #video_path = common.analysis_path + config_by_gui.DIR_PATH[-6:] + ".avi"
            video_path = common.f_sub_path[:-4] + "_sub.avi"
            common.video = cv2.VideoWriter(video_path, codec, clip_fps, (config.WIDTH, config.HEIGHT), isColor=True)
    else:
        list_f_clusters = sorted( glob.glob(common.analysis_path + "*s.pkl") )
        list_f_isolated_subclusters = sorted( glob.glob(common.analysis_path + "*_isolated.pkl") )

        # read new subclusters/clusters pkl files
        with open(list_f_clusters[0], 'rb') as f_pkl_nm:
            common.clusters = pickle.load(f_pkl_nm)
        print("0: Reading " + list_f_clusters[0] + " finished.")

        with open(list_f_isolated_subclusters[0] , 'rb') as f_pkl_nm:
            common.isolated_subclusters = pickle.load(f_pkl_nm)
        print("0: Reading " + list_f_isolated_subclusters[0] + " finished.")


        list_border_time_us = []

        # generate "list_video_path" and "list_border_time_us" at the same time
        list_video_path = []
        for i,f_cluster_path in enumerate(list_f_clusters):
            list_video_path.append( f_cluster_path[:-4] + ".avi" )

            if i == 0:
                # the first item defines the start time
                from_hour = int( f_cluster_path[-24:-22] )
                from_min = int( f_cluster_path[-21:-19] )
                from_sec = int( f_cluster_path[-18:-16] )
                from_us = ( ((from_hour*60) + from_min)*60 + from_sec )*1000000
                list_border_time_us.append(from_us)

            to_hour = int( f_cluster_path[-13:-11] )
            to_min = int( f_cluster_path[-10:-8] )
            to_sec = int( f_cluster_path[-7:-5] )
            to_us   = ( ((to_hour  *60) + to_min  )*60 + to_sec   )*1000000 # + 999999
            list_border_time_us.append(to_us)

        common.video = cv2.VideoWriter(list_video_path[0], codec, clip_fps, (config.WIDTH, config.HEIGHT), isColor=True)

        last_frame_start_time = 0

        # Make empty black images of size(HEIGHT,WIDTH)
        common.img = np.zeros((config.HEIGHT, config.WIDTH, 3), dtype=np.uint8) #np.ones()
        common.blank_img = np.copy(common.img)

    # local variables
    tmp_x = 0
    tmp_y = 0
    time_high = 4097 # values over 4096 are invalid
    timestamp = 0
    last_timestamp = 0
    polarity = True
    x_base = 0

    last_time_high = 0 # [us]
    time_high_1st = 0 # [us]
    timestamp_offset = 0 # [us]
    last_time_clustering_new_pxls = 0 # [us]
    common.last_time_updating_subclusters = 0 # [us]
    common.timestamp_step = config.SUBCLUSTER_TIMESTAMP_MILESTONE_STEP

    # cnter_time_low = 0
    # cnter_time_high = 0
    # cnter_cd_y = 0
    # cnter_x_pos = 0
    # cnter_x_base = 0

    common.last_time = time.time() # in order to measure elapsed time

    for raw_or_bin_path in common.list_raw_or_bin_path:
        common.f_input = open(raw_or_bin_path, "rb")

        if common.raw_file_mode:
            # read the ascii header
            for i in range(1000):
                byte = common.f_input.read(1)
                if byte == b'\x0A': # ASCII code for Line Feed
                    byte_1st = common.f_input.read(1)

                    if byte_1st != b'\x25': # ASCII code for %
                        byte_2nd = common.f_input.read(1)

                        if byte_2nd != b'\x20': # ASCII code for Space
                            break

        byte_1st = common.f_input.read(1)
        byte_2nd = common.f_input.read(1)

        # check if timestamp has been updated to valid range
        if (timestamp <= list_border_time_us[0]):
            while byte_1st: # while byte_1st != b'':
                #if ( int.from_bytes(byte_2nd, byteorder="big") & int.from_bytes(tmp_byte, byteorder="big") ) == int.from_bytes(tmp_byte, byteorder="big"):
                cur_event_type = byte_2nd[0] & config.MASK_EVENT_TYPE[0]

                if cur_event_type == config.EVENT_TYPE_TIME_LOW[0]:
                    # concatenate 4 + 8 bits
                    #timestamp = timestamp_offset + (time_high-time_high_1st)*4096 + ( byte_2nd[0] & config.MASK_VALUE_0F[0] )*256 + byte_1st[0] # 16777216 = 4096*4096
                    timestamp = time_high + ( byte_2nd[0] & config.MASK_VALUE_0F[0] )*256 + byte_1st[0]

                    display_elapsed_time(timestamp)

                    # if timestamp has been updated to valid range, break
                    if (timestamp >= list_border_time_us[0]) and (time_high != 4097):
                        last_timestamp = timestamp
                        if flg_subcluster and (not common.raw_file_mode):
                            common.f_bin_border.write(f"{ os.path.basename(common.f_sub_path)[:-4] } is from { os.path.basename(raw_or_bin_path) }\n")
                        break

                elif cur_event_type == config.EVENT_TYPE_TIME_HIGH[0]:
                    # concatenate 4 + 8 bits
                    if time_high == 4097:
                        time_high_1st = ( byte_2nd[0] & config.MASK_VALUE_0F[0] )*256 + byte_1st[0]
                        time_high = 0
                        last_time_high = time_high
                        last_time_high_byte_1st = byte_1st[0]

                    elif byte_1st[0] != last_time_high_byte_1st:
                        # when "byte_1st[0]" for "time_high" changed from the last time
                        time_high = ( ( byte_2nd[0] & config.MASK_VALUE_0F[0] )*256 + byte_1st[0] - time_high_1st + timestamp_offset ) * 4096

                        if time_high < last_time_high:
                            timestamp_offset += 4096
                            time_high = ( ( byte_2nd[0] & config.MASK_VALUE_0F[0] )*256 + byte_1st[0] - time_high_1st + timestamp_offset ) * 4096

                        last_time_high = time_high
                        last_time_high_byte_1st = byte_1st[0]

                # read the next 2 bytes
                byte_1st = common.f_input.read(1)
                byte_2nd = common.f_input.read(1)

        # "while loop" which lasts until the EOF
        while byte_1st:
            cur_event_type = byte_2nd[0] & config.MASK_EVENT_TYPE[0]

            if cur_event_type == config.EVENT_TYPE_X_POS[0]:
                #cnter_x_pos += 1

                # check Contrast Detection polarity
                polarity = bool(byte_2nd[0] & config.MASK_VALUE_08[0]) # bool(0x0800 or 0x0000)

                # concatenate 3 + 8 bits
                if config.HORIZONTAL_OPTICAL_FLIP:
                    tmp_x = config.WIDTH-1 - ( ( byte_2nd[0] & config.MASK_VALUE_07[0] )*256 + byte_1st[0] )
                else:
                    tmp_x = ( byte_2nd[0] & config.MASK_VALUE_07[0] )*256 + byte_1st[0]

                add_new_pxl( tmp_x, tmp_y, polarity, flg_subcluster )

            elif cur_event_type == config.EVENT_TYPE_CD_Y[0]:
                #cnter_cd_y += 1

                # concatenate 3 + 8 bits
                if config.VERTICAL_OPTICAL_FLIP:
                    tmp_y = config.HEIGHT-1 - ( ( byte_2nd[0] & config.MASK_VALUE_07[0] )*256 + byte_1st[0] )
                else:
                    tmp_y = ( byte_2nd[0] & config.MASK_VALUE_07[0] )*256 + byte_1st[0]

            elif cur_event_type == config.EVENT_TYPE_TIME_LOW[0]:
                #cnter_time_low += 1

                # concatenate 4 + 8 bits
                timestamp = time_high + ( byte_2nd[0] & config.MASK_VALUE_0F[0] )*256 + byte_1st[0]
                common.timestamp_ms = timestamp//1000

                detect_timestamp_jump(timestamp, last_timestamp)
                display_elapsed_time(timestamp)

                last_timestamp = timestamp

                if flg_subcluster and (not flg_evs2video):
                    if ( timestamp - last_time_clustering_new_pxls ) >= config.CLUSTERING_NEW_PXLS_DUR:
                        cluster_new_pxls()
                        last_time_clustering_new_pxls = timestamp - (timestamp % config.CLUSTERING_NEW_PXLS_DUR) # update with truncation

                    if ( timestamp - common.last_time_updating_subclusters ) >= config.UPDATING_SUBCLUSTERS_DUR:
                        update_subclusters(common.timestamp_ms)
                        common.last_time_updating_subclusters = timestamp - (timestamp % config.UPDATING_SUBCLUSTERS_DUR) # update with truncation



                if config_by_gui.ENABLE_MONITORING or (not flg_subcluster):
                    if ( timestamp - last_frame_start_time ) >= frame_dur:
                        if draw_img(timestamp, flg_subcluster):
                            # when 'q' key is pressed, exit the program
                            print("Interrupted by keyboard input")
                            return 1
                        last_frame_start_time = timestamp - (timestamp % frame_dur) # update with truncation

                if timestamp >= list_border_time_us[ common.border_time_idx ]:
                    if common.border_time_idx == len( list_border_time_us ) - 1:
                        # when the current timestamp exceeds "capture_end_time"
                        print("Finished up to the capture_end_time")
                        return 0

                    if flg_subcluster:
                        if not flg_evs2video:
                            for subcluster in common.subclusters[:]:
                                save_subcluster(subcluster)

                        common.f_sub.write( b'\x00\x00\x00\x00' ).to_bytes(4, byteorder='little') # marker of EOF
                        common.f_sub.close()

                        # reinitialize variables
                        common.subclusters.clear()
                        common.subcluster_cnter = 0
                        common.subcluster_id = 0 # ID values only larger than 0 are used

                        for idx in range( len(common.binary_img_buf) ):
                            common.binary_img_buf[idx] = np.copy(common.blank_binary_img)

                        common.img_subcluster_idxes = np.copy(common.blank_img_idx) # initialize "common.img_subcluster_idxes" with 0s

                        f_sub_operation(config_by_gui.LIST_BORDER_TIME[ common.border_time_idx ], config_by_gui.LIST_BORDER_TIME[ common.border_time_idx+1 ], cur_f_path="", flg_rename=0)

                        if not common.raw_file_mode:
                            common.f_bin_border.write(f"{ os.path.basename(common.f_sub_path)[:-4] } is from { os.path.basename(raw_or_bin_path) }\n")

                        if config_by_gui.ENABLE_MONITORING:
                            common.video.release()

                            video_path = common.f_sub_path[:-4] + "_sub.avi"
                            common.video = cv2.VideoWriter(video_path, codec, clip_fps, (config.WIDTH, config.HEIGHT), isColor=True)
                    else:
                        common.video.release()
                        common.isolated_subclusters.clear()
                        common.clusters.clear()

                        # read new "all_clusters"/"merged_clusters" pkl files
                        with open(list_f_clusters[ common.border_time_idx ], 'rb') as f_pkl_nm:
                            common.clusters = pickle.load(f_pkl_nm)
                        print( str(common.border_time_idx) + ": Reading " + list_f_clusters[ common.border_time_idx ] + " finished.")

                        with open(list_f_isolated_subclusters[ common.border_time_idx ] , 'rb') as f_pkl_nm:
                            common.isolated_subclusters = pickle.load(f_pkl_nm)
                        print( str(common.border_time_idx) + ": Reading " + list_f_isolated_subclusters[ common.border_time_idx ] + " finished.")

                        common.video = cv2.VideoWriter(list_video_path[ common.border_time_idx ], codec, clip_fps, (config.WIDTH, config.HEIGHT), isColor=True)
                    common.border_time_idx += 1


            elif cur_event_type == config.EVENT_TYPE_TIME_HIGH[0]:
                #cnter_time_high += 1

                # concatenate 4 + 8 bits
                if byte_1st[0] != last_time_high_byte_1st:
                    # when "byte_1st[0]" for "time_high" changed from the last time
                    time_high = ( ( byte_2nd[0] & config.MASK_VALUE_0F[0] )*256 + byte_1st[0] - time_high_1st + timestamp_offset ) * 4096

                    if time_high < last_time_high:
                        timestamp_offset += 4096
                        time_high = ( ( byte_2nd[0] & config.MASK_VALUE_0F[0] )*256 + byte_1st[0] - time_high_1st + timestamp_offset ) * 4096

                    last_time_high = time_high
                    last_time_high_byte_1st = byte_1st[0]

            elif cur_event_type == config.EVENT_TYPE_X_BASE[0]:
                #cnter_x_base += 1

                # check Contrast Detection Polarity
                polarity =  bool(byte_2nd[0] & config.MASK_VALUE_08[0]) # bool(0x0800 or 0x0000)

                # concatenate 3 + 8 bits
                if config.HORIZONTAL_OPTICAL_FLIP:
                    x_base = config.WIDTH-1 - ( ( byte_2nd[0] & config.MASK_VALUE_07[0] )*256 + byte_1st[0] )
                else:
                    x_base = ( byte_2nd[0] & config.MASK_VALUE_07[0] )*256 + byte_1st[0]

            elif cur_event_type == config.EVENT_TYPE_VECT_12[0]:
                bit_mask = 1
                for j in range(8):
                    if byte_1st[0] & bit_mask:
                        add_new_pxl( x_base, tmp_y, polarity, flg_subcluster )
                    bit_mask = bit_mask << 1
                    x_base = x_base + x_base_shift

                bit_mask = 1
                for j in range(4):
                    if byte_2nd[0] & bit_mask:
                        add_new_pxl( x_base, tmp_y, polarity, flg_subcluster )
                    bit_mask = bit_mask << 1
                    x_base = x_base + x_base_shift

            elif cur_event_type == config.EVENT_TYPE_VECT_8[0]:
                bit_mask = 1
                for j in range(8):
                    if byte_1st[0] & bit_mask:
                        add_new_pxl( x_base, tmp_y, polarity, flg_subcluster )
                    bit_mask = bit_mask << 1
                    x_base = x_base + x_base_shift

    #         elif cur_event_type == config.EVENT_TYPE_CONTINUED_4[0]:
    #             pass

    #         elif cur_event_type == config.EVENT_TYPE_EXT_TRIGGER[0]:
    #             pass

    #         elif cur_event_type == config.EVENT_TYPE_OTHERS[0]:
    #             pass

    #         elif cur_event_type == config.EVENT_TYPE_CONTINUED_12[0]:
    #             pass

    #         else :
    #             pass

            # read the next 2 bytes
            byte_1st = common.f_input.read(1)
            byte_2nd = common.f_input.read(1)

        common.f_input.close()
    return 0

    # print( "cnter_time_low: " + str(cnter_time_low) )
    # print( "cnter_time_high: " + str(cnter_time_high) )
    # print( "cnter_cd_y: " + str(cnter_cd_y) )
    # print( "cnter_x_pos: " + str(cnter_x_pos) )
    # print( "cnter_x_base: " + str(cnter_x_base) )


#@jit("void(int64[:,:], int64[:,:])")
def merge_clusters(merged_cluster, sub_merged_cluster):
    # merge "sub_merged_cluster" to "merged_cluster"

    # merge "xy_limit"
    merged_cluster.xy_limit = [ min( merged_cluster.xy_limit[0], sub_merged_cluster.xy_limit[0] ) ,
                                max( merged_cluster.xy_limit[1], sub_merged_cluster.xy_limit[1] ) ,
                                min( merged_cluster.xy_limit[2], sub_merged_cluster.xy_limit[2] ) ,
                                max( merged_cluster.xy_limit[3], sub_merged_cluster.xy_limit[3] ) ]

    # merge the event history
    idx_toStart = 0 # offset for "merged_cluster.event_history"
    for j,sub_event_entry in enumerate(sub_merged_cluster.event_history):
        found_xflg = 1 # when found yes:0, when no:1
        for i,event_entry in enumerate( merged_cluster.event_history[idx_toStart:] ):
            if sub_event_entry[0] == event_entry[0]:
                # when sub_event_entry's timestamp is same as event_entry's, merge to "event_entry"
                event_entry[1] += sub_event_entry[1]
                event_entry[2] += sub_event_entry[2]
                found_xflg = 0 # same timestamp is found and merged
                idx_toStart += i+1 # update "idx_toStart"
                break # jump to the next "sub_event_entry"
            elif sub_event_entry[0] < event_entry[0]:
                # when the timestamp of "sub_event_entry" is smaller than that of "event_entry", insert ahead of "event_entry"
                merged_cluster.event_history = np.insert(merged_cluster.event_history, idx_toStart+i, sub_event_entry, axis=0)
                found_xflg = 0 # insertion is done
                idx_toStart += i+1 # update "idx_toStart"
                break # jump to the next sub_event_entry

        if found_xflg: # if not found, it means the timestamp of "sub_event_entry" is larger than any of remaining "merged_cluster", so let's append to the end
            merged_cluster.event_history = np.concatenate( (merged_cluster.event_history, sub_merged_cluster.event_history[j:]), axis=0 )
            break

    # merge the track history
    idxB_offset = (sub_merged_cluster.first_track_t - merged_cluster.first_track_t) // config.track_step_ms
    idxB_range = (sub_merged_cluster.last_track_t - sub_merged_cluster.first_track_t) // config.track_step_ms + 1 # =len( sub_merged_cluster.track_r_history)
    idxA_ceiling = (merged_cluster.last_track_t - merged_cluster.first_track_t) // config.track_step_ms + 1 - idxB_offset
    for idxB in range(idxB_range):
        if idxB < idxA_ceiling:
            # add to an existing track record of "merged_cluster"
            (merged_cluster.track_r_history[ idxB + idxB_offset ]).extend( copy.deepcopy( sub_merged_cluster.track_r_history[idxB] ) )
        else:
            # append new track records at the end of "merged_cluster"
            (merged_cluster.track_r_history).extend( copy.deepcopy( sub_merged_cluster.track_r_history[idxB:] ) )
            break

    # update "last_track_t"
    merged_cluster.last_track_t = max(merged_cluster.last_track_t, sub_merged_cluster.last_track_t)


@jit("int16(int16[:,:], int16[:,:], int64)", nopython=True, cache=True)
def judge_track_overlap_at_a_time(rectanglesA_at_a_time, rectanglesB_at_a_time, range_margin):
    # return 1 when there is no gap larger than "range_margin" between the two tracks at a certain time
    for pieceA in rectanglesA_at_a_time:
        xminA,xmaxA,yminA,ymaxA,lenA,lenPosA = pieceA
        for pieceB in rectanglesB_at_a_time:
            if (xmaxA + range_margin) < pieceB[0]: # pieceB[0]: xminB
                break
            if (pieceB[1] + range_margin) < xminA: # pieceB[1]: xmaxB
                break
            if (ymaxA + range_margin) < pieceB[2]: # pieceB[2]: yminB
                break
            if (pieceB[3] + range_margin) < yminA: # pieceB[3]: ymaxB
                break
            return 1
    return 0


def judge_insignificance(cluster):
    if cluster.last_track_t - cluster.first_track_t < 2000:
        # when "cluster" stayed active less than 2 sec

        # calc the first centroid
        head_total_weight = 0
        head_center_x_x2 = 0
        head_center_y_x2 = 0

        for rectangle in cluster.track_r_history[0]:
            weight = rectangle[4].item() # change to native python list (64bits) since numpy 16bits is not enough
            head_total_weight += weight
            head_center_x_x2 += (rectangle[0] + rectangle[1]).item() * weight
            head_center_y_x2 += (rectangle[2] + rectangle[3]).item() * weight

        # calc the last centroid
        tail_total_weight = 0
        tail_center_x_x2 = 0
        tail_center_y_x2 = 0

        for rectangle in cluster.track_r_history[-1]:
            weight = rectangle[4].item() # change to native python list (64bits) since numpy 16bits is not enough
            tail_total_weight += weight
            tail_center_x_x2 += (rectangle[0] + rectangle[1]).item() * weight
            tail_center_y_x2 += (rectangle[2] + rectangle[3]).item() * weight

        # return 1 indicating "cluster" is insignificant if the first and last centroids are close
        if ( (head_center_x_x2//head_total_weight) - (tail_center_x_x2//tail_total_weight) ) < 200:
            if ( (head_center_y_x2//head_total_weight) - (tail_center_y_x2//tail_total_weight) ) < 200:
                return 1
    return 0