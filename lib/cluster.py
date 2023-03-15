# coding: utf-8

# Copyright 2023 Sony Group Corporation - All Rights Reserved.
# Subject to the terms and conditions contained in LICENSE.txt accompanying this file, you may use this file.

import os
import numpy as np
from numba import jit
import copy
import pickle
from enum import Enum, unique
import datetime
import time
import glob
from lib import utils
import config_by_gui
import config
from lib import common



all_subclusters_decoded = [] # list of "Subcluster" class decoded from binary file: .sub
merged_clusters = []
unmerged_clusters = []
unmerged_cluster_ids = np.array( [], dtype=np.int32 )



#@jit("int32(int32[:,:], int32[:,:])", nopython=True)
def final_judge_to_merge(subclusterA, idxA, subclusterB, idxB):
    # check if the two subclusters have (almost) always been overlapped over each other while they coexisted

    overlap_time_head = subclusterB.first_track_t
    overlap_time_tail = min( subclusterA.last_track_t , subclusterB.last_track_t )
    idx_shift_range = (overlap_time_tail - overlap_time_head) // config.track_step_ms + 1
    track_idxA_head = (overlap_time_head - subclusterA.first_track_t) // config.track_step_ms

    for idx_shift in range(idx_shift_range):
        rectangleA = subclusterA.track_r_history[ track_idxA_head + idx_shift ][0]
        rectangleB = subclusterB.track_r_history[idx_shift][0]

        # return 0 when there is a gap between the two subclusters
        if (rectangleB[1] + config.MARGIN_TO_JUDGE) < rectangleA[0]:
            return 0
        if (rectangleA[1] + config.MARGIN_TO_JUDGE) < rectangleB[0]:
            return 0
        if (rectangleB[3] + config.MARGIN_TO_JUDGE) < rectangleA[2]:
            return 0
        if (rectangleA[3] + config.MARGIN_TO_JUDGE) < rectangleB[2]:
            return 0

    # record brother relationship
    subclusterA.list_brother_idx[1] = np.append(subclusterA.list_brother_idx[1], idxB)
    subclusterB.list_brother_idx[0] = np.append(subclusterB.list_brother_idx[0], idxA)
    return 1 # return 1 when there is no gap found between the two subclusters



@jit("int16[:,:](int16[:,:], int16[:,:])", nopython=True, cache=True)
def calc_coverage(a, _new):
    b = a.ravel()
    #new = np.array( [[_new]], dtype='int16' )
    new = _new[0]

    idx0 = np.searchsorted( b , new[0] )
    idx1 = np.searchsorted( b , new[1] )
    #print( "idx: " + str(idx0) + ", " + str(idx1) )

    if idx0%2: # odd -> idx0 is inside
        if idx1%2: # odd -> idx1 is inside
            #print("case0")
            #a[ (idx0-1)//2 ][1] = np.amax( [ new[1], b[idx1] ] )
            a[ (idx0-1)//2 ][1] = max( [ new[1], b[idx1] ] )
            #a = np.delete( a, range( (idx0+1)//2, (idx1+1)//2 ), axis=0 )
            #new_head = a[ : (idx0+1)//2 ]
            #new_tail = a[ (idx1+1)//2 : ]
            a = np.concatenate( (a[ : (idx0+1)//2 ], a[ (idx1+1)//2 : ]), axis=0 )
        else: # even -> idx1 is outside
            if ( idx1 != len(b) ) and ( ( b[idx1] == new[1] ) or ( (b[idx1]-1) == new[1] ) ):
                #print("case1")
                a[ (idx0-1)//2 ][1] = max( [ new[1], b[idx1+1] ] )
                #a = np.delete( a, range( (idx0+1)//2, (idx1+2)//2 ), axis=0 )
                a = np.concatenate( (a[ : (idx0+1)//2 ], a[ (idx1+2)//2 : ]), axis=0 )
            else:
                #print("case2")
                a[ (idx0-1)//2 ][1] = new[1]
                #a = np.delete( a, range( (idx0+1)//2, idx1//2 ), axis=0 )
                a = np.concatenate( (a[ : (idx0+1)//2 ], a[ idx1//2 : ]), axis=0 )
    else: # even -> idx0 is outside
        if idx1%2: # odd -> idx1 is inside
            if (idx0 != 0) and ( (b[idx0-1]+1) == new[0] ):
                #print("case3")
                a[ (idx0-1)//2 ][1] = b[idx1]
                #a = np.delete( a, range( idx0//2, (idx1+1)//2 ), axis=0 )
                a = np.concatenate( (a[ : idx0//2 ], a[ (idx1+1)//2 : ]), axis=0 )
            else:
                #print("case4")
                a[ idx0//2 ] = [ new[0], b[idx1] ]
                #a = np.delete( a, range( (idx0+2)//2, (idx1+1)//2 ), axis=0 )
                a = np.concatenate( (a[ : (idx0+2)//2 ], a[ (idx1+1)//2 : ]), axis=0 )
        else: # even -> idx1 is outside
            if idx0 == len(b):
                if idx0 == 0:
                    # when "a" is empty, fill it with "_new"
                    #print("case empty")
                    a = _new
                elif (b[idx0-1]+1) == new[0]:
                    #print("case5")
                    a[ (idx0-1)//2 ][1] = new[1]
                else:
                    #print("case6")
                    a = np.append( a, _new, axis=0 )
            elif (idx0 != 0) and ( (b[idx0-1]+1) == new[0] ):
                if ( idx1 != len(b) ) and ( ( b[idx1] == new[1] ) or ( (b[idx1]-1) == new[1] ) ):
                    #print("case7")
                    #a[ (idx0-1)//2 ][1] = np.amax( [ new[1], b[idx1+1] ] )
                    a[ (idx0-1)//2 ][1] = max( [ new[1], b[idx1+1] ] )
                    #a = np.delete( a, range( idx0//2, (idx1+2)//2 ), axis=0 )
                    a = np.concatenate( (a[ : idx0//2 ], a[ (idx1+2)//2 : ]), axis=0 )
                else:
                    #print("case8")
                    a[ (idx0-1)//2 ][1] = new[1]
                    #a = np.delete( a, range( idx0//2, idx1//2 ), axis=0 )
                    a = np.concatenate( (a[ : idx0//2 ], a[ idx1//2 : ]), axis=0 )
            else:
                if ( idx1 != len(b) ) and ( ( b[idx1] == new[1] ) or ( (b[idx1]-1) == new[1] ) ):
                    #print("case9")
                    a[ idx0//2 ][0] = new[0]
                    #a[ idx0//2 ][1] = np.amax( [ new[1], b[idx1+1] ] )
                    a[ idx0//2 ][1] = max( [ new[1], b[idx1+1] ] )
                    #a = np.delete( a, range( (idx0+2)//2, (idx1+2)//2 ), axis=0 )
                    a = np.concatenate( (a[ : (idx0+2)//2 ], a[ (idx1+2)//2 : ]), axis=0 )
                elif idx0 == idx1:
                    #print("case10")
                    #a = np.insert( a, idx0//2, _new, axis=0 )
                    a = np.concatenate( (a[ : idx0//2 ], _new, a[ idx0//2 : ]), axis=0 )
                else:
                    #print("case11")
                    a[ idx0//2 ] = new
                    #a = np.delete( a, range( (idx0+2)//2, idx1//2 ), axis=0 )
                    a = np.concatenate( (a[ : (idx0+2)//2 ], a[ idx1//2 : ]), axis=0 )
    return a



def judge_overlap(subclusterA, subclusterB, margin):
    # return 1 when there is no gap larger than "range_margin" between the two subclusters at a certain time

    overlap_time_head = subclusterB.first_track_t
    overlap_time_tail = min( subclusterA.last_track_t , subclusterB.last_track_t )
    idx_shift_range = (overlap_time_tail - overlap_time_head) // config.track_step_ms + 1
    track_idxA_head = (overlap_time_head - subclusterA.first_track_t) // config.track_step_ms

    for idx_shift in range(idx_shift_range):
        rectangleA = subclusterA.track_r_history[ track_idxA_head + idx_shift ][0]
        rectangleB = subclusterB.track_r_history[idx_shift][0]

        # break when there is a gap between the two subclusters
        if (rectangleB[1] + margin) < rectangleA[0]:
            break
        if (rectangleA[1] + margin) < rectangleB[0]:
            break
        if (rectangleB[3] + margin) < rectangleA[2]:
            break
        if (rectangleA[3] + margin) < rectangleB[2]:
            break
        return 1 # return 1 when there is an overlap found between the two subclusters

    return 0 # return 0 when there is no overlap found between the two subclusters



#@jit("int64(int64[:,:], int64[:,:])")
def judge_to_assign(cluster, subcluster, margin):

    # If the timestamp ranges of "cluster" and "subcluster" are not overlapped at all, return 0
    if cluster.last_track_t < subcluster.first_track_t:
        return 0
    if subcluster.last_track_t < cluster.first_track_t:
        return 0

    # compare xy limits between "subcluster" and "cluster". If they are distant away, return 0
    # xy_limit = [xmin, xmax, ymin, ymax]
    if cluster.xy_limit[0] > (subcluster.xy_limit[1] + config.XY_LIMIT_MARGIN):
        return 0
    if (cluster.xy_limit[1] + config.XY_LIMIT_MARGIN) < subcluster.xy_limit[0]:
        return 0
    if cluster.xy_limit[2] > (subcluster.xy_limit[3] + config.XY_LIMIT_MARGIN):
        return 0
    if (cluster.xy_limit[3] + config.XY_LIMIT_MARGIN) < subcluster.xy_limit[2]:
        return 0

    # check if any subcluster merged in "cluster" is close enough in space/time to the given "subcluster"
    for idx in cluster.list_subcluster_idx:
        tmp_subcluster = all_subclusters_decoded[idx]

        # check which subcluster was born earlier
        if tmp_subcluster.ID < subcluster.ID:
            subcluster0, subcluster1 = tmp_subcluster, subcluster
        else:
            subcluster0, subcluster1 = subcluster, tmp_subcluster

        if subcluster0.last_track_t < subcluster1.first_track_t:
            # when the two subclusters' timestamp are not overlapped at all
            continue

        # compare xy limits between subcluster0 and 1. If they are distant away, return 0
        # xy_limit = [xmin, xmax, ymin, ymax]
        if subcluster1.xy_limit[0] > (subcluster0.xy_limit[1] + config.XY_LIMIT_MARGIN):
            continue
        if (subcluster1.xy_limit[1] + config.XY_LIMIT_MARGIN) < subcluster0.xy_limit[0]:
            continue
        if subcluster1.xy_limit[2] > (subcluster0.xy_limit[3] + config.XY_LIMIT_MARGIN):
            continue
        if (subcluster1.xy_limit[3] + config.XY_LIMIT_MARGIN) < subcluster0.xy_limit[2]:
            continue

        if judge_overlap(subcluster0, subcluster1, margin):
            if cluster.ID < subcluster.ID:
                # when the first subcluster in "cluster" is born earlier than "subcluster"
                return 1
            else:
                return 2
    return 0 # not to be assigned



def merge_subclusters_with_list_subcluster_idx(list_subcluster_idx):
    list_merged_cluster = [] # list to be returned

    for subcluster_idx in list_subcluster_idx:
        subcluster = all_subclusters_decoded[subcluster_idx]

        if list_merged_cluster == []:
            list_merged_cluster.append( utils.Cluster(subcluster, subcluster_idx) ) # add the first "Cluster"
            merged_cluster = list_merged_cluster[0] # "merged_cluster" points to the first "Cluster"
        elif subcluster.first_track_t <= merged_cluster.last_track_t:
            # when there is timestamp overlap between "merged_cluster" and "subcluster", merge them
            utils.merge_clusters(merged_cluster, subcluster)
            merged_cluster.list_subcluster_idx = np.append( merged_cluster.list_subcluster_idx, subcluster_idx ) # register as one of the merged subclusters
        else:
            # when there is no timestamp overlap between "merged_cluster" and "subcluster", create a separate "merged_cluster"
            list_merged_cluster.append( utils.Cluster(subcluster, subcluster_idx) ) # add the next "Cluster". Notice later "subcluster" is always born later
            merged_cluster = list_merged_cluster[-1] # "merged_cluster" points to the latest "Cluster"
            # "subcluster.merged_idx" do not need to be updated any more since "merged_clusters" themselves are going to be deleted

    return list_merged_cluster



def separate_a_merged_cluster(merged_cluster, h_or_v, list_range, timestamp_idx):
    merged_cluster0 = None
    merged_cluster1 = None

    # calc the border between "merged_cluster0" and "merged_cluster1"
    list_range_flatten = list_range.ravel() # flatten list_x/y_range
    list_odd = list_range_flatten[2::2]    # min coordinates of filled ranges
    list_even = list_range_flatten[1:-1:2] # max coordinates of filled ranges
    list_range_gap = list_odd - list_even # vacant gaps between filled ranges
    max_gap_idx = list_range_gap.argmax() # find the idx for the largest gap. This is the gap to unmerge "merged_cluster" into two
    merged_cluster0_max = list_range[max_gap_idx  ][1] # max x/y coordinate of "merged_cluster0". This is the border for "merged_cluster0"
    merged_cluster1_min = list_range[max_gap_idx+1][0] # max x/y coordinate of "merged_cluster1". This is the border for "merged_cluster1"

    timestamp_ms = merged_cluster.first_track_t + config.track_step_ms*timestamp_idx

    new_list_subcluster_idx = [] # list of subclusters which did not exist at the given "timestamp_ms"

    # first, assign all subclusters which existed at the given "timestamp_ms"
    for subcluster_idx in merged_cluster.list_subcluster_idx:
        subcluster = all_subclusters_decoded[subcluster_idx]
        if (timestamp_ms < subcluster.first_track_t) or (subcluster.last_track_t < timestamp_ms):
            # when "subcluster" did not exist at the given "timestamp_ms", continue after updating "new_list_subcluster_idx"
            new_list_subcluster_idx.append(subcluster_idx)
            continue

        track_idx = (timestamp_ms - subcluster.first_track_t) // config.track_step_ms # track idx at the given "timestamp_ms"
        if subcluster.track_r_history[track_idx][0][h_or_v] <= merged_cluster0_max:
            # when "subcluster" is inside the border for "merged_cluster0"
            if merged_cluster0 == None:
                merged_cluster0 = utils.Cluster(subcluster, subcluster_idx) # initialize "merged_cluster0"
            else:
                utils.merge_clusters(merged_cluster0, subcluster)
                merged_cluster0.list_subcluster_idx = np.append( merged_cluster0.list_subcluster_idx, subcluster_idx ) # register as one of the merged subclusters
        #elif merged_cluster1_min <= subcluster.track_r_history[track_idx][0][h_or_v]: # not needed
        else:
            if merged_cluster1 == None:
                merged_cluster1 = utils.Cluster(subcluster, subcluster_idx) # initialize "merged_cluster1"
            else:
                utils.merge_clusters(merged_cluster1, subcluster)
                merged_cluster1.list_subcluster_idx = np.append( merged_cluster1.list_subcluster_idx, subcluster_idx ) # register as one of the merged subclusters
                # "subcluster.merged_idx" do not need to be updated any more since "merged_clusters" themselves are going to be deleted

    merged_cluster.list_subcluster_idx = np.array(new_list_subcluster_idx, dtype=np.int32) # update "list_subcluster_idx" with remaining subcluster idxes

    # the merged_cluster with larger num of sublusters has the priority
    if len(merged_cluster0.list_subcluster_idx) >= len(merged_cluster1.list_subcluster_idx):
        merged_clusterA, merged_clusterB = merged_cluster0, merged_cluster1
    else:
        merged_clusterA, merged_clusterB = merged_cluster1, merged_cluster0


    # arrange older/younger borthers of "merged_clusterA/B"
    list_brother_idxAB = [ [np.array([], dtype=np.int32), np.array([], dtype=np.int32)], [np.array([], dtype=np.int32), np.array([], dtype=np.int32)] ] # [[A's older, A's younger], [B's older, B's younger]]
    for i,merged_clusterAorB in enumerate([merged_clusterA, merged_clusterB]):
        for j in range(2): # older, younger
            cur_list_brother_idx = merged_clusterAorB.list_subcluster_idx # initialize

            while 1:
                new_list_brother_idx = np.array([], dtype=np.int32) # initialize
                for subcluster_idx in cur_list_brother_idx:
                    subcluster = all_subclusters_decoded[subcluster_idx]
                    for brother_idx in subcluster.list_brother_idx[j]: # "brother_idx" is the idx of an older/younger brother of "subcluster"
                        search_idx = np.searchsorted(merged_cluster.list_subcluster_idx, brother_idx)
                        if search_idx < len(merged_cluster.list_subcluster_idx):
                            if merged_cluster.list_subcluster_idx[ search_idx ] == brother_idx:
                                # when "brother_idx" was included in the "merged_cluster.list_subcluster_idx"
                                merged_cluster.list_subcluster_idx = np.delete( merged_cluster.list_subcluster_idx, search_idx )
                                new_list_brother_idx = np.append(new_list_brother_idx, brother_idx)

                if len(new_list_brother_idx) == 0:
                    break

                new_list_brother_idx = np.unique(new_list_brother_idx)
                list_brother_idxAB[i][j] = np.concatenate( (list_brother_idxAB[i][j], new_list_brother_idx) ) # register as new brother idxes
                cur_list_brother_idx = new_list_brother_idx # update "cur_list_brother_idx" for the next while loop

            list_brother_idxAB[i][j] = np.unique(list_brother_idxAB[i][j])

    list_older_brother_idxA = np.setdiff1d(list_brother_idxAB[0][0], list_brother_idxAB[1][0], assume_unique=True) # older brother idxes belonging to "merged_clusterA" only
    list_older_brother_idxB = np.setdiff1d(list_brother_idxAB[1][0], list_brother_idxAB[0][0], assume_unique=True) # older brother idxes belonging to "merged_clusterB" only
    list_older_brother_idxAandB = np.setdiff1d(list_brother_idxAB[0][0], list_older_brother_idxA, assume_unique=True) # older brother idxes common between "merged_clusterA/B"
    list_younger_brother_idxA = np.setdiff1d(list_brother_idxAB[0][1], list_brother_idxAB[1][1], assume_unique=True) # younger brother idxes belonging to "merged_clusterA" only
    list_younger_brother_idxB = np.setdiff1d(list_brother_idxAB[1][1], list_brother_idxAB[0][1], assume_unique=True) # younger brother idxes belonging to "merged_clusterB" only
    list_younger_brother_idxAandB = np.setdiff1d(list_brother_idxAB[0][1], list_younger_brother_idxA, assume_unique=True) # younger brother idxes common between "merged_clusterA/B"

    if len(list_older_brother_idxAandB):
        # when there are older brother idxes common between "merged_clusterA/B", let "merged_clusterA" to merge them and older brother idxes belonging to "merged_clusterB" as well
        list_older_brother_idxA = np.sort( np.concatenate( (list_brother_idxAB[0][0], list_older_brother_idxB) ) )
        list_older_brother_idxB = np.array([], dtype=np.int32) # delete
        list_older_brother_idxAandB = np.array([], dtype=np.int32) # delete
    if len(list_younger_brother_idxAandB):
        # when there are younger brother idxes common between "merged_clusterA/B", let "merged_clusterA" to merge them and younger brother idxes belonging to "merged_clusterB" as well
        list_younger_brother_idxA = np.sort( np.concatenate( (list_brother_idxAB[0][1], list_younger_brother_idxB) ) )
        list_younger_brother_idxB = np.array([], dtype=np.int32) # delete
        list_younger_brother_idxAandB = np.array([], dtype=np.int32) # delete


    # merge older/younger borthers of "merged_clusterA/B"
    if len(list_older_brother_idxA):
        if merged_clusterA.list_subcluster_idx[0] < list_older_brother_idxA[0]:
            # in this case, no need to merge here, let "list_younger_brother_idxA" absorb "list_older_brother_idxA"
            list_younger_brother_idxA = np.sort( np.concatenate( (list_older_brother_idxA, list_younger_brother_idxA) ) )
        else:
            list_merged_cluster = merge_subclusters_with_list_subcluster_idx( list_older_brother_idxA )
            tmp_merged_cluster = list_merged_cluster[0]
            utils.merge_clusters(tmp_merged_cluster, merged_clusterA)
            tmp_merged_cluster.list_subcluster_idx = np.concatenate( (tmp_merged_cluster.list_subcluster_idx , merged_clusterA.list_subcluster_idx) )
            for tmp_sub_merged_cluster in list_merged_cluster[1:]: # concatenat the rest of list if "list_merged_cluster" has more than 1
                utils.merge_clusters(tmp_merged_cluster, tmp_sub_merged_cluster)
                tmp_merged_cluster.list_subcluster_idx = np.concatenate( (tmp_merged_cluster.list_subcluster_idx , tmp_sub_merged_cluster.list_subcluster_idx) )
            tmp_merged_cluster.list_subcluster_idx = np.sort( tmp_merged_cluster.list_subcluster_idx )
            merged_clusterA = tmp_merged_cluster

    if len(list_older_brother_idxB):
        if merged_clusterB.list_subcluster_idx[0] < list_older_brother_idxB[0]:
            # in this case, no need to merge here, let "list_younger_brother_idxB" absorb "list_older_brother_idxB"
            list_younger_brother_idxB = np.sort( np.concatenate( (list_older_brother_idxB, list_younger_brother_idxB) ) )
        else:
            list_merged_cluster = merge_subclusters_with_list_subcluster_idx( list_older_brother_idxB )
            tmp_merged_cluster = list_merged_cluster[0]
            utils.merge_clusters(tmp_merged_cluster, merged_clusterB)
            tmp_merged_cluster.list_subcluster_idx = np.concatenate( (tmp_merged_cluster.list_subcluster_idx , merged_clusterB.list_subcluster_idx) )
            for tmp_sub_merged_cluster in list_merged_cluster[1:]: # concatenat the rest of list if "list_merged_cluster" has more than 1
                utils.merge_clusters(tmp_merged_cluster, tmp_sub_merged_cluster)
                tmp_merged_cluster.list_subcluster_idx = np.concatenate( (tmp_merged_cluster.list_subcluster_idx , tmp_sub_merged_cluster.list_subcluster_idx) )
            tmp_merged_cluster.list_subcluster_idx = np.sort( tmp_merged_cluster.list_subcluster_idx )
            merged_clusterB = tmp_merged_cluster

    if len(list_younger_brother_idxA):
        list_merged_cluster = merge_subclusters_with_list_subcluster_idx( list_younger_brother_idxA )
        for tmp_sub_merged_cluster in list_merged_cluster:
            utils.merge_clusters(merged_clusterA, tmp_sub_merged_cluster)
            merged_clusterA.list_subcluster_idx = np.concatenate( (merged_clusterA.list_subcluster_idx , tmp_sub_merged_cluster.list_subcluster_idx) )
        merged_clusterA.list_subcluster_idx = np.sort( merged_clusterA.list_subcluster_idx )

    if len(list_younger_brother_idxB):
        list_merged_cluster = merge_subclusters_with_list_subcluster_idx( list_younger_brother_idxB )
        for tmp_sub_merged_cluster in list_merged_cluster:
            utils.merge_clusters(merged_clusterB, tmp_sub_merged_cluster)
            merged_clusterB.list_subcluster_idx = np.concatenate( (merged_clusterB.list_subcluster_idx , tmp_sub_merged_cluster.list_subcluster_idx) )
        merged_clusterB.list_subcluster_idx = np.sort( merged_clusterB.list_subcluster_idx )


    # merge still remaining subcluster idxes (ex. an older brother of a younger brother and the like)
    while( len(merged_cluster.list_subcluster_idx) ):
        subcluster_idx = merged_cluster.list_subcluster_idx[0] # pick the first subcluster
        merged_cluster.list_subcluster_idx = merged_cluster.list_subcluster_idx[1:] # update "list_subcluster_idx"
        subcluster = all_subclusters_decoded[subcluster_idx]

        new_list_subcluster_idx = np.array([subcluster_idx], dtype=np.int32) # initialize
        flg_relativeAorB = 0 # 0:Neither, 1:A, 2:B

        new_list_brother_idx = np.concatenate( (subcluster.list_brother_idx[0], subcluster.list_brother_idx[1]) ) # both older and younger
        while( len(new_list_brother_idx) ):
            next_list_brother_idx = np.array([], dtype=np.int32) # initialize
            for brother_idx in new_list_brother_idx:

                # check if "brother_idx" is already among "new_list_subcluster_idx"
                search_idx = np.searchsorted(new_list_subcluster_idx, brother_idx)
                if search_idx < len(new_list_subcluster_idx):
                    if new_list_subcluster_idx[ search_idx ] == brother_idx:
                        continue # if yes, skip

                # else, check if "brother_idx" is among "merged_cluster.list_subcluster_idx"
                search_idx = np.searchsorted(merged_cluster.list_subcluster_idx, brother_idx)
                if search_idx < len(merged_cluster.list_subcluster_idx):
                    if merged_cluster.list_subcluster_idx[ search_idx ] == brother_idx:
                        # when "brother_idx" was included in the "merged_cluster.list_subcluster_idx"
                        idx = np.searchsorted( new_list_subcluster_idx, brother_idx ) # look for where to insert in "new_list_subcluster_idx"
                        new_list_subcluster_idx = np.insert( new_list_subcluster_idx, idx, brother_idx ) # update "new_list_subcluster_idx"

                        merged_cluster.list_subcluster_idx = np.delete( merged_cluster.list_subcluster_idx, search_idx ) # update "merged_cluster.list_subcluster_idx"

                        brother = all_subclusters_decoded[brother_idx]
                        next_list_brother_idx = np.concatenate( (next_list_brother_idx, brother.list_brother_idx[0], brother.list_brother_idx[1]) ) # update "next_list_younger_brother_idx
                        continue

                # else, check if "brother_idx" is among "merged_clusterA.list_subcluster_idx"
                # if "brother_idx" does not belong to "new_list_subcluster_idx" or "merged_cluster.list_subcluster_idx", then it belongs to either A or B without doubt
                search_idx = np.searchsorted(merged_clusterA.list_subcluster_idx, brother_idx)
                if search_idx < len(merged_clusterA.list_subcluster_idx):
                    if merged_clusterA.list_subcluster_idx[ search_idx ] == brother_idx:
                        #if flg_relativeAorB == 0:
                        flg_relativeAorB = 1 # "merged_clusterA" has priority
                        continue

                # else, check if "brother_idx" is among "merged_clusterB.list_subcluster_idx"
                search_idx = np.searchsorted(merged_clusterB.list_subcluster_idx, brother_idx)
                if search_idx < len(merged_clusterB.list_subcluster_idx):
                    if merged_clusterB.list_subcluster_idx[ search_idx ] == brother_idx:
                        if flg_relativeAorB == 0: # "merged_clusterB" has no priority. So cannot overwrite when "flg_relativeAorB" is already 1
                            flg_relativeAorB = 2
                        continue


            new_list_brother_idx = np.unique(next_list_brother_idx)

        list_merged_cluster = merge_subclusters_with_list_subcluster_idx( new_list_subcluster_idx )
        tmp_merged_cluster = list_merged_cluster[0] # len(list_merged_cluster) is always 1
        if flg_relativeAorB == 0:
            # try merging "tmp_merged_cluster" to "merged_clusterA", which has priority

            if (merged_clusterA.last_track_t < tmp_merged_cluster.first_track_t) or (tmp_merged_cluster.last_track_t < merged_clusterA.first_track_t):
                # when the two clusters' timestamp are not overlapped at all, give up merging to "merged_clusterA" and append to "merged_clusters" as a new "merged_cluster"
                merged_clusters.append( tmp_merged_cluster )
                #print(f"branched out with len(list): {len(new_list_subcluster_idx)}")
                continue

            # when the two clusters' timestamp are overlapped, merge them
            if merged_clusterA.ID < subcluster.ID:
                utils.merge_clusters(merged_clusterA, tmp_merged_cluster)
                merged_clusterA.list_subcluster_idx = np.sort( np.concatenate( (merged_clusterA.list_subcluster_idx , tmp_merged_cluster.list_subcluster_idx) ) )
            else:
                utils.merge_clusters(tmp_merged_cluster, merged_clusterA)
                tmp_merged_cluster.list_subcluster_idx = np.sort( np.concatenate( (tmp_merged_cluster.list_subcluster_idx , merged_clusterA.list_subcluster_idx) ) )
                merged_clusterA = tmp_merged_cluster

        elif flg_relativeAorB == 1:
            #merged_cluster_selected = merged_clusterB
            if merged_clusterA.ID < subcluster.ID:
                utils.merge_clusters(merged_clusterA, tmp_merged_cluster)
                merged_clusterA.list_subcluster_idx = np.sort( np.concatenate( (merged_clusterA.list_subcluster_idx , tmp_merged_cluster.list_subcluster_idx) ) )
            else:
                utils.merge_clusters(tmp_merged_cluster, merged_clusterA)
                tmp_merged_cluster.list_subcluster_idx = np.sort( np.concatenate( (tmp_merged_cluster.list_subcluster_idx , merged_clusterA.list_subcluster_idx) ) )
                merged_clusterA = tmp_merged_cluster
        #elif flg_relativeAorB == 2:
        else:
            if merged_clusterB.ID < subcluster.ID:
                utils.merge_clusters(merged_clusterB, tmp_merged_cluster)
                merged_clusterB.list_subcluster_idx = np.sort( np.concatenate( (merged_clusterB.list_subcluster_idx , tmp_merged_cluster.list_subcluster_idx) ) )
            else:
                utils.merge_clusters(tmp_merged_cluster, merged_clusterB)
                tmp_merged_cluster.list_subcluster_idx = np.sort( np.concatenate( (tmp_merged_cluster.list_subcluster_idx , merged_clusterB.list_subcluster_idx) ) )
                merged_clusterB = tmp_merged_cluster

    # update "last_packed_track_idx"
    if merged_clusterA.ID < merged_clusterB.ID:
        merged_clusterA.last_packed_track_idx = (merged_clusterB.first_track_t - merged_clusterA.first_track_t) // config.track_step_ms
    else:
        merged_clusterB.last_packed_track_idx = (merged_clusterA.first_track_t - merged_clusterB.first_track_t) // config.track_step_ms

    # finally, "add merged_clusterA/B" to "merged_clusters"    
    merged_clusters.append( merged_clusterA )
    merged_clusters.append( merged_clusterB )



def check_track_sparsity_without_skip(merged_cluster, h_or_v, track_idx, bool_separate):
    last_idx = (merged_cluster.last_track_t - merged_cluster.first_track_t) // config.track_step_ms + 1 # =len(merged_cluster.track_r_history)
    head_track_idx = max(track_idx-49, 0)
    tail_track_idx = max(track_idx+50, last_idx)

    unpacked_track_cnter = 0
    for i,rectangles_at_a_time in enumerate( merged_cluster.track_r_history[ head_track_idx : tail_track_idx ] ):
        if head_track_idx+i == track_idx:
            unpacked_track_cnter += 1 # increment because this idx is already known to be sparse
            continue

        list_range = np.array( [[]], dtype=np.int16 ) # initialize
        for rectangle in rectangles_at_a_time:
            list_range = calc_coverage( list_range , np.array( [ rectangle[ h_or_v*2 : (h_or_v+1)*2 ] ], dtype=np.int16 ) ) # rectangle[0:2] => [xmin, xmax] . rectangle[2:4] => [ymin, ymax]

        list_range_len = list_range[:,1] - list_range[:,0] # preparation to sum up
        if ( list_range_len.sum() + len(list_range_len) )*1.2 < ( list_range[-1][1] - list_range[0][0] ): # if (filled x/y range)*1.2 < (edge to edge of x/y range):
            unpacked_track_cnter += 1 # increment when the track is sparse
            if unpacked_track_cnter >= 50:
                # when the track has been sparse enough continuously for >=0.2 sec
                if bool_separate:
                    # separate the "merged_cluster" into two or more clusters
                    separate_a_merged_cluster(merged_cluster, h_or_v+1, list_range, head_track_idx+i) # second arg: 1 results in horizontal separation . second arg: 2 results in vertical separation
                return 1
        else:
            if head_track_idx+i > track_idx:
                return 0
            unpacked_track_cnter = 0 # when packed enough, reset the cnter to 0
    return 0



def check_track_sparsity_with_skip(merged_cluster, first_idx, last_idx, bool_separate):
    for i,rectangles_at_a_time in enumerate( merged_cluster.track_r_history[ first_idx : last_idx ] ):
        if i%50 == 0: # process only one out of fifty "rectangles_at_a_time"
            list_x_range = np.array( [[]], dtype=np.int16 ) # initialize
            list_y_range = np.array( [[]], dtype=np.int16 ) # initialize
            for rectangle in rectangles_at_a_time:
                list_x_range = calc_coverage( list_x_range , np.array( [ rectangle[0:2] ], dtype=np.int16 ) ) # rectangle[0:2] => [xmin, xmax]
                list_y_range = calc_coverage( list_y_range , np.array( [ rectangle[2:4] ], dtype=np.int16 ) ) # rectangle[2:4] => [ymin, ymax]

            list_x_range_len = list_x_range[:,1] - list_x_range[:,0] # preparation to sum up
            if ( list_x_range_len.sum() + len(list_x_range_len) )*1.2 < ( list_x_range[-1][1] - list_x_range[0][0] ): # if (filled x range)*1.2 < (edge to edge of x range):
                if check_track_sparsity_without_skip( merged_cluster, 0, first_idx+i, bool_separate ): # second arg 0: potential to be horizontally sparse
                    return 1

            list_y_range_len = list_y_range[:,1] - list_y_range[:,0] # preparation to sum up
            if ( list_y_range_len.sum() + len(list_y_range_len) )*1.2 < ( list_y_range[-1][1] - list_y_range[0][0] ): # if (filled y range)*1.2 < (edge to edge of y range):
                if check_track_sparsity_without_skip( merged_cluster, 1, first_idx+i, bool_separate ): # second arg 1: potential to be vertically sparse
                    return 1

    return 0



def unmerge_clusters(merged_clusters):
    global unmerged_cluster_ids

    tmp_cnter = 0 # needed to delete an inactive "merged_cluster" avoiding index shifting
    for merged_cluster in merged_clusters[:]:
        if merged_cluster.inactive_flg:
            del merged_clusters[ tmp_cnter ]
            continue

        last_track_idx = (merged_cluster.last_track_t - merged_cluster.first_track_t) // config.track_step_ms

        # check if the "merged_cluster" track is packed enough to be recoginized as a single cluster or not
        if check_track_sparsity_with_skip( merged_cluster, merged_cluster.last_packed_track_idx, last_track_idx, bool_separate=1 ):
            # when not packed enough, "merged_cluster" has been already separated and stored at the end of "merged_clusters". So current "merged_cluster" is no more needed
            del merged_clusters[ tmp_cnter ]
            continue

        # when packed enough, add "merged_cluster" to "unmerged_clusters"
        idx = np.searchsorted( unmerged_cluster_ids, merged_cluster.ID ) # look for where to insert in "unmerged_clusters"
        unmerged_cluster_ids = np.insert( unmerged_cluster_ids, idx, merged_cluster.ID ) # update "unmerged_cluster_ids"
        unmerged_clusters.insert( idx, copy.deepcopy(merged_cluster) ) # insert to "unmerged_clusters"
        merged_cluster.inactive_flg = 1 # "merged_cluster" will be deleted later

        tmp_cnter += 1



def main(target_nm):
    global unmerged_cluster_ids

    utils.set_analysis_path(target_nm, flg_subcluster=0)

    timestamp_milestone = config.CLUSTER_TIMESTAMP_MILESTONE_STEP

    #print(f"\nClustering the contents of {common.analysis_path}")
    list_f_sub_path = sorted( glob.glob(common.analysis_path + "*s.sub") )
    for f_sub_path in list_f_sub_path:



        # Read .sub (while decoding)

        f_bin_input = open(f_sub_path, "rb")  # open .sub file

        # read the binary file: .sub containing subcluster info
        while 1:
            # read the subcluster IDs
            new_subcluster_id = int.from_bytes(f_bin_input.read(4), "little")
            if new_subcluster_id == 0: # this means EOF
                break

            # look for where to insert the new subcluster. Note a subcluster which died earlier is listed ahead in the file
            insert_pos = len(all_subclusters_decoded) # look from the tail
            while insert_pos > 0:
                if all_subclusters_decoded[ insert_pos-1 ].ID > new_subcluster_id:
                    insert_pos -= 1
                else:
                    break

            new_xy_limit = [5000, 0, 5000, 0] # [xmin, xmax, ymin, ymax ] with initialization values

            # read the first timestamp of event history
            tmp_event_timestamp = int.from_bytes(f_bin_input.read(4), "little")
            event_timestamp_diff = int.from_bytes(f_bin_input.read(2), "little")

            # read the subcluster event history
            new_event_history = []
            while 1:
                tmp_event_timestamp += event_timestamp_diff # update "tmp_event_timestamp"
                num_new_pxls = int.from_bytes(f_bin_input.read(2), "little")
                num_new_pos_pxls = int.from_bytes(f_bin_input.read(2), "little")
                new_event_history.append( [tmp_event_timestamp, num_new_pxls, num_new_pos_pxls] )

                # next event entry
                event_timestamp_diff = int.from_bytes(f_bin_input.read(2), "little")
                if event_timestamp_diff == 0: # this means the end of the event history
                    break

            # read the subcluster track history 
            new_track_r_history = [] # rectangles
            first_track_timestamp = ( (new_event_history[0][0] + (config.track_step_ms-1))//config.track_step_ms ) * config.track_step_ms # the first track timestamp
            last_track_timestamp = first_track_timestamp - config.track_step_ms # initialize
            while 1:
                xmin = int.from_bytes(f_bin_input.read(2), "little")
                if xmin == 0xFFFF: # this means the end of the track history
                    break
                xmax = int.from_bytes(f_bin_input.read(2), "little")
                ymin = int.from_bytes(f_bin_input.read(2), "little")
                ymax = int.from_bytes(f_bin_input.read(2), "little")
                len_pxls = int.from_bytes(f_bin_input.read(2), "little")
                len_pos_pxls = int.from_bytes(f_bin_input.read(2), "little")
                iteration = int.from_bytes(f_bin_input.read(1), "little")

                for i in range( iteration ): # iterate for each track piece
                    new_track_r_history.append( [ np.array([xmin, xmax, ymin, ymax, len_pxls, len_pos_pxls], dtype=np.int16) ] ) # "iteration" is not needed anymore
                    last_track_timestamp += config.track_step_ms # increment timestamp by "track_step_ms" [ms]. This value depends on "UPDATING_SUBCLUSTERS_DUR"

                # new_xy_limit
                if xmin < new_xy_limit[0]:
                    new_xy_limit[0] = xmin
                if xmax > new_xy_limit[1]:
                    new_xy_limit[1] = xmax
                if ymin < new_xy_limit[2]:
                    new_xy_limit[2] = ymin
                if ymax > new_xy_limit[3]:
                    new_xy_limit[3] = ymax

            merged_idx = -1 # initialize. -1 means this subcluster has not been merged yet

            # check if the subcluster has always been confined in a banned region
            for banned_region in config.LIST_BANNED_REGION:
                if (banned_region[0] <= new_xy_limit[0]) and (new_xy_limit[1] <= banned_region[1]):
                    if (banned_region[2] <= new_xy_limit[2]) and (new_xy_limit[3] <= banned_region[3]):
                        merged_idx = -2 # overwrite with -2. -2 means this subcluster has always been confined in a banned region and will be ignored for the rest of process

            # assemble all subcluster info and store into new Subcluster_decoded instance
            all_subclusters_decoded.insert( insert_pos, utils.Subcluster_decoded( new_subcluster_id, merged_idx, first_track_timestamp, last_track_timestamp, new_xy_limit, new_event_history, new_track_r_history ) )
            # now a subcluster born earlier has younger idx in "all_subclusters_decoded"

        f_bin_input.close()
        print(f"Reading {f_sub_path} finished.")





        # Merge subclusters

        time_start = time.time()

        timestamp_ms = 0

        # merge subclusters which are close enough in space/time
        for idxA,subclusterA in enumerate(all_subclusters_decoded):
            #if subclusterA.first_track_t > 5000:
                #break
            if subclusterA.merged_idx == -2:
                # when confined in a banned region, continue
                continue

            # to monitor progress
            timestamp_ms = subclusterA.first_track_t
            if timestamp_ms > timestamp_milestone:
                print( str(timestamp_milestone//1000) + " sec passed." )
                timestamp_milestone += config.CLUSTER_TIMESTAMP_MILESTONE_STEP

            for idxB,subclusterB in enumerate( all_subclusters_decoded[idxA+1:] ):
                if subclusterB.merged_idx == -2: # confined in a banned region
                    continue

                #if subclusterB.first_track_t - subclusterA.last_track_t > 200: #2000 # when the two subclusters' timestamp is not overlapped with a large gap
                if subclusterA.last_track_t < subclusterB.first_track_t:
                    # when the two subclusters' timestamp ranges are not overlapped at all
                    break # since there is no hope with the rest of subclusterBs, skip to the next subclusterA

                if (subclusterA.merged_idx >= 0) and (subclusterA.merged_idx == subclusterB.merged_idx):
                    # when subclusterA and B are already assigned to the same merged cluster, skip to the next subclusterB
                    continue

                # compare xy limits between subclusterA and B. If they are distant away, skip to the next subclusterB. xy_limit = [xmin, xmax, ymin, ymax]
                if subclusterB.xy_limit[0] > (subclusterA.xy_limit[1] + config.XY_LIMIT_MARGIN):
                    continue
                if (subclusterB.xy_limit[1] + config.XY_LIMIT_MARGIN) < subclusterA.xy_limit[0]:
                    continue
                if subclusterB.xy_limit[2] > (subclusterA.xy_limit[3] + config.XY_LIMIT_MARGIN):
                    continue
                if (subclusterB.xy_limit[3] + config.XY_LIMIT_MARGIN) < subclusterA.xy_limit[2]:
                    continue

                if final_judge_to_merge(subclusterA, idxA, subclusterB, idxA+idxB+1):
                    # two subclusters have been close enough to each other throughout the overlap time. Let's merge them
                    if subclusterA.merged_idx >= 0:
                        # when "subclusterA" is already merged
                        if subclusterB.merged_idx >= 0:
                            # when "subclusterB" is already merged too, decide which one absorbs the other
                            if subclusterA.merged_idx < subclusterB.merged_idx:
                                merged_cluster_idx = subclusterA.merged_idx         # the one to absorb
                                merged_idx_toBeInactivated = subclusterB.merged_idx # the one to be absorbed
                            elif subclusterA.merged_idx > subclusterB.merged_idx:
                                merged_cluster_idx = subclusterB.merged_idx         # the one to absorb
                                merged_idx_toBeInactivated = subclusterA.merged_idx # the one to be absorbed

                            merged_cluster = merged_clusters[ merged_cluster_idx ]             # the one to absorb
                            sub_merged_cluster = merged_clusters[ merged_idx_toBeInactivated ] # the one to be absorbed
                            sub_merged_cluster.inactive_flg = 1 # "sub_merged_cluster" will be deleted later

                            # update "merged_idx" of all the subclusters in "sub_merged_cluster"
                            for sub_cluster_idx in sub_merged_cluster.list_subcluster_idx:
                                all_subclusters_decoded[sub_cluster_idx].merged_idx = merged_cluster_idx

                            # update "list_subcluster_idx" of "merged_cluster"
                            #(merged_cluster.list_subcluster_idx).extend( sub_merged_cluster.list_subcluster_idx )
                            #(merged_cluster.list_subcluster_idx).sort()
                            merged_cluster.list_subcluster_idx = np.sort( np.concatenate( (merged_cluster.list_subcluster_idx , sub_merged_cluster.list_subcluster_idx) ) )
                        else:
                            # when "subclusterB" is not merged yet
                            merged_cluster = merged_clusters[ subclusterA.merged_idx ] # the one to absorb
                            sub_merged_cluster = subclusterB                           # the one to be absorbed
                            idx = np.searchsorted( merged_cluster.list_subcluster_idx, idxA+idxB+1 )
                            merged_cluster.list_subcluster_idx = np.insert( merged_cluster.list_subcluster_idx, idx, idxA+idxB+1 ) # register as one of the merged subclusters
                            subclusterB.merged_idx = subclusterA.merged_idx
                    else:
                        # when "subclusterA" is not merged yet
                        if subclusterB.merged_idx >= 0:
                            # when "subclusterB" is already merged
                            merged_cluster = merged_clusters[ subclusterB.merged_idx ] # the one to absorb
                            sub_merged_cluster = subclusterA                           # the one to be absorbed
                            idx = np.searchsorted( merged_cluster.list_subcluster_idx, idxA )
                            merged_cluster.list_subcluster_idx = np.insert( merged_cluster.list_subcluster_idx, idx, idxA ) # register as one of the merged subclusters
                            subclusterA.merged_idx = subclusterB.merged_idx
                        else:
                            # when "subclusterB" is not merged neither, create a new merged cluster
                            merged_cluster_idx = len(merged_clusters) # calc "merged_cluster_idx" before appending
                            merged_clusters.append( utils.Cluster(subclusterA, idxA) ) # add to "merged_clusters"
                            merged_cluster = merged_clusters[merged_cluster_idx]
                            merged_cluster.list_subcluster_idx = np.append( merged_cluster.list_subcluster_idx, idxA+idxB+1 ) # register as one of the merged subclusters
                            sub_merged_cluster = subclusterB # the one to be absorbed
                            subclusterA.merged_idx = merged_cluster_idx
                            subclusterB.merged_idx = merged_cluster_idx

                    utils.merge_clusters(merged_cluster, sub_merged_cluster)

                    #if merged_idx_toBeInactivated >= 0:
                        #del sub_merged_cluster[2:] # delete needless data

        print(f"Elapsed time for initial merging: {time.time()-time_start:.2f} sec.\n")





        # Unmerge clusters

        time_start = time.time()

        while 1:
            if len(merged_clusters):
                unmerge_clusters(merged_clusters)
            else:
                break

            print( f"Remaining merged_cluster to be unmerged: { len(merged_clusters) }" )

        merged_clusters.clear() # all necessary info has been moved to "unmerged_clusters". So "merged_clusters" is unnecessary any more
        print(f"Elapsed time for unmerging: {time.time()-time_start:.2f} sec.\n" )





        # Remerge clusters

        time_start = time.time()

        #remerged_clusters = unmerged_clusters

        # first, try to remerge isolated subclusters to an "unmerged_cluster"
        for range_margin in range(5,15,5):
            while_cnter = 0
            remerge_isolated_cnter = 1
            while remerge_isolated_cnter:
                remerge_isolated_cnter = 0

                for i,isolated_subcluster in enumerate(all_subclusters_decoded):
                    if isolated_subcluster.merged_idx != -1:
                        # when "subcluster" is already merged or confined in a banned region, skip it
                        continue

                    # earlier "unmerged_cluster" has more chance to absorb isolated subclusters!!! (temporarily ignore this inequality)
                    for unmerged_idx, unmerged_cluster in enumerate(unmerged_clusters):
                        if while_cnter:
                            if unmerged_cluster.remerged_flg != while_cnter:
                                # after the first while loop, skip a "unmerged_cluster" which is not remerged in the last loop
                                continue

                        if unmerged_cluster.inactive_flg:
                            continue

                        if isolated_subcluster.last_track_t < unmerged_cluster.first_track_t:
                            # when the two timestamp range of "unmerged_cluster" is entirely over that of "isolated_subcluster"
                            break
                        if unmerged_cluster.last_track_t < isolated_subcluster.first_track_t:
                            # when the two timestamp range of "isolated_subcluster" is entirely over that of "unmerged_cluster"
                            continue

                        judge_result = judge_to_assign(unmerged_cluster, isolated_subcluster, margin=range_margin)
                        if judge_result:
                            if judge_result == 1:
                                utils.merge_clusters(unmerged_cluster, isolated_subcluster)
                                idx = np.searchsorted( unmerged_cluster.list_subcluster_idx, i )
                                unmerged_cluster.list_subcluster_idx = np.insert( unmerged_cluster.list_subcluster_idx, idx, i ) # register as one of the merged subclusters

                                unmerged_cluster.remerged_flg = while_cnter + 1
                                isolated_subcluster.merged_idx = unmerged_idx # actually any value except for -1 works fine because "merged_clusters" is already cleared

                            elif judge_result == 2:
                                # create a new unmerged cluster and make "unmerged_cluster" absorbed in it
                                tmp_merged_cluster = utils.Cluster(isolated_subcluster, i) # the one to absorb
                                tmp_merged_cluster.remerged_flg = while_cnter + 1
                                utils.merge_clusters(tmp_merged_cluster, unmerged_cluster)
                                #(tmp_merged_cluster.list_subcluster_idx).extend( unmerged_cluster.list_subcluster_idx ) # update "list_subcluster_idx". No need for sort()
                                tmp_merged_cluster.list_subcluster_idx = np.concatenate( (tmp_merged_cluster.list_subcluster_idx , unmerged_cluster.list_subcluster_idx) ) # update "list_subcluster_idx". No need for sort()

                                # add to "unmerged_clusters"
                                idx = np.searchsorted( unmerged_cluster_ids, isolated_subcluster.ID ) # look for where to insert in "unmerged_clusters"
                                unmerged_cluster_ids = np.insert( unmerged_cluster_ids, idx, isolated_subcluster.ID ) # update "unmerged_cluster_ids"
                                unmerged_clusters.insert( idx, tmp_merged_cluster ) # insert to "unmerged_clusters"
                                unmerged_cluster.inactive_flg = 1 # "unmerged_cluster" will be deleted later
                                isolated_subcluster.merged_idx = idx # actually any value except for -1 works fine because "merged_clusters" is already cleared

                            remerge_isolated_cnter += 1
                            break

                print( f"range_margin: {range_margin}  remerge_isolated_cnter: {remerge_isolated_cnter}" )
                while_cnter += 1

            # delete inactive "unmerged_cluster"
            tmp_cnter = 0 # needed to delete an "unmerged_cluster" avoiding index shifting
            for unmerged_cluster in unmerged_clusters[:]:
                if unmerged_cluster.inactive_flg:
                    # when already remerged to another "unmerged_clusters" starting from an isolated subcluster, delete it
                    del unmerged_clusters[ tmp_cnter ]
                    unmerged_cluster_ids = np.delete(unmerged_cluster_ids, tmp_cnter) # update "unmerged_cluster_ids"
                    continue
                unmerged_cluster.remerged_flg = 0
                tmp_cnter += 1


        # second, try to remerge an "unmerged_cluster" to another "unmerged_cluster"
        for range_margin in range(0,35,5): #0,25,5
            while_cnter = 0
            remerge_cnter = 1
            while remerge_cnter:
                remerge_cnter = 0

                for idxA,unmerged_clusterA in enumerate( unmerged_clusters ):
                    # check if "unmerged_clusterA" has already been remerged as a "unmerged_clusterB" in the past
                    if unmerged_clusterA.inactive_flg:
                        # when already remerged, continue
                        continue
                    list_remerge_candidates = [] # list of candidates chosen among "unmerged_clusterB" to remerge with "unmerged_clusterA"

                    for idxB,unmerged_clusterB in enumerate( unmerged_clusters[idxA+1:] ):
                        if unmerged_clusterB.inactive_flg:
                            # when already remerged, continue
                            continue

                        if while_cnter:
                            if (unmerged_clusterA.remerged_flg != while_cnter) and (unmerged_clusterB.remerged_flg != while_cnter):
                                # after the first while loop, skip if both unmerged_clusters were not remerged in the last loop
                                continue

                        if unmerged_clusterA.last_track_t < unmerged_clusterB.first_track_t:
                            # when the two unmerged_clusters' timestamp are not overlapped at all, break
                            break

                        # focus on the beginning of the overlap time
                        first_track_idxA = (unmerged_clusterB.first_track_t - unmerged_clusterA.first_track_t) // config.track_step_ms
                        rectanglesA_at_a_time = np.array( unmerged_clusterA.track_r_history[first_track_idxA], dtype=np.int16 )
                        rectanglesB_at_a_time = np.array( unmerged_clusterB.track_r_history[0], dtype=np.int16 )

                        if utils.judge_track_overlap_at_a_time(rectanglesA_at_a_time, rectanglesB_at_a_time, range_margin):
                            # when there was track overlap at the beginning, check if there is track overlap at the end of the overlap time as well
                            last_overlap_time = min(unmerged_clusterA.last_track_t, unmerged_clusterB.last_track_t)
                            last_track_idxA = (last_overlap_time - unmerged_clusterA.first_track_t) // config.track_step_ms
                            last_track_idxB = (last_overlap_time - unmerged_clusterB.first_track_t) // config.track_step_ms
                            rectanglesA_at_a_time = np.array( unmerged_clusterA.track_r_history[last_track_idxA], dtype=np.int16 )
                            rectanglesB_at_a_time = np.array( unmerged_clusterB.track_r_history[last_track_idxB], dtype=np.int16 )
                            if utils.judge_track_overlap_at_a_time(rectanglesA_at_a_time, rectanglesB_at_a_time, range_margin) == 0:
                                # when there is no track overlap at the end of the overlap time
                                if utils.judge_insignificance(unmerged_clusterB) == 0:
                                    # when "unmerged_clusterB" is significant enough, add as a candidate to remerge and skip
                                    list_remerge_candidates.append( idxA+idxB+1 )
                                    continue

                            # merge "unmerged_clusterB" to "unmerged_clusterA" if the result is packed enough
                            tmp_merged_cluster = copy.deepcopy( unmerged_clusterA ) # avoid altering "unmerged_clusterA" itself
                            utils.merge_clusters( tmp_merged_cluster, unmerged_clusterB )
                            if check_track_sparsity_with_skip( tmp_merged_cluster, first_track_idxA, last_track_idxA, bool_separate=0 ) == 0:
                                # when resultant "tmp_merged_cluster" is packed enough
                                tmp_merged_cluster.list_subcluster_idx = np.sort( np.concatenate( (tmp_merged_cluster.list_subcluster_idx , unmerged_clusterB.list_subcluster_idx) ) )
                                tmp_merged_cluster.remerged_flg = while_cnter + 1
                                unmerged_clusters[idxA] = tmp_merged_cluster
                                unmerged_clusterB.inactive_flg = 1 # mark it to be deleted later
                                remerge_cnter += 1
                                list_remerge_candidates = []
                                break # jump to next "unmerged_clusterA". Maybe this "break" is not needed? -> No it is better to have this here

                    if len(list_remerge_candidates) == 1:
                        # when there is only one candidate to remerge
                        sub_merged_cluster = unmerged_clusters[ list_remerge_candidates[0] ] # the only candidate "unmerged_cluster"

                        first_track_idxA = (sub_merged_cluster.first_track_t - unmerged_clusterA.first_track_t) // config.track_step_ms
                        last_overlap_time = min(unmerged_clusterA.last_track_t, sub_merged_cluster.last_track_t)
                        last_track_idxA = (last_overlap_time - unmerged_clusterA.first_track_t) // config.track_step_ms

                        tmp_merged_cluster = copy.deepcopy( unmerged_clusterA ) # avoid altering "unmerged_clusterA" itself
                        utils.merge_clusters( tmp_merged_cluster, sub_merged_cluster )
                        if check_track_sparsity_with_skip( tmp_merged_cluster, first_track_idxA, last_track_idxA, bool_separate=0 ) == 0:
                            # when resultant "tmp_merged_cluster" is packed enough
                            tmp_merged_cluster.list_subcluster_idx = np.sort( np.concatenate( (tmp_merged_cluster.list_subcluster_idx , sub_merged_cluster.list_subcluster_idx) ) )
                            tmp_merged_cluster.remerged_flg = while_cnter + 1
                            unmerged_clusters[idxA] = tmp_merged_cluster
                            sub_merged_cluster.inactive_flg = 1 # mark it to be deleted later
                            remerge_cnter += 1

                print( f"range_margin: {range_margin}  remerge_cnter: {remerge_cnter}" )
                while_cnter += 1

            # delete inactive "unmerged_cluster"
            tmp_cnter = 0 # needed to delete an "unmerged_cluster" avoiding index shifting
            for unmerged_cluster in unmerged_clusters[:]:
                if unmerged_cluster.inactive_flg:
                    # when already remerged to another "unmerged_clusters", delete it
                    del unmerged_clusters[ tmp_cnter ]
                    continue
                unmerged_cluster.remerged_flg = 0 # refresh "remerged_flg"
                tmp_cnter += 1


        for unmerged_cluster in unmerged_clusters:
            if utils.judge_insignificance(unmerged_cluster):
                unmerged_cluster.insignificance = 1
            else:
                unmerged_cluster.insignificance = 0


        # third, try to remerge an insignificant "unmerged_cluster" to another "unmerged_cluster"
        for range_margin in range(0,35,5): #0,25,5
            while_cnter = 0
            remerge_cnter = 1
            while remerge_cnter:
                remerge_cnter = 0

                for idx0,unmerged_cluster0 in enumerate( unmerged_clusters ):
                    # check if "unmerged_cluster0" has already been remerged as a "unmerged_cluster1" in the past
                    if unmerged_cluster0.inactive_flg:
                        # when already remerged, continue
                        continue
                    list_remerge_candidates = []

                    if unmerged_cluster0.insignificance == 0:
                        # when "unmerged_cluster0" is significant, skip
                        continue

                    for idx1,unmerged_cluster1 in enumerate( unmerged_clusters ):
                        if idx0 == idx1: # when looking at the same "unmerged_clusters", skip
                            continue

                        if unmerged_cluster1.inactive_flg:
                            # when already remerged, continue
                            continue

                        if while_cnter:
                            if (unmerged_cluster0.remerged_flg != while_cnter) and (unmerged_cluster1.remerged_flg != while_cnter):
                                # after the first while loop, skip if both unmerged_clusters were not remerged in the last loop
                                continue

                        if unmerged_cluster1.last_track_t < unmerged_cluster0.first_track_t:
                            continue
                        if unmerged_cluster0.last_track_t < unmerged_cluster1.first_track_t:
                            break

                        # focus on the beginning of the overlap time
                        overlap_head_t = max(unmerged_cluster0.first_track_t, unmerged_cluster1.first_track_t)
                        head_track_idx0 = (overlap_head_t - unmerged_cluster0.first_track_t) // config.track_step_ms
                        head_track_idx1 = (overlap_head_t - unmerged_cluster1.first_track_t) // config.track_step_ms
                        rectangles0_at_head = np.array( unmerged_cluster0.track_r_history[head_track_idx0], dtype=np.int16 )
                        rectangles1_at_head = np.array( unmerged_cluster1.track_r_history[head_track_idx1], dtype=np.int16 )

                        #overlap_tail_t = min(unmerged_cluster0.last_track_t, unmerged_cluster1.last_track_t)
                        #tail_track_idx0 = (overlap_tail_t - unmerged_cluster0.first_track_t) // config.track_step_ms
                        #tail_track_idx1 = (overlap_tail_t - unmerged_cluster1.first_track_t) // config.track_step_ms
                        #rectangles0_at_tail = np.array( unmerged_cluster0.track_r_history[tail_track_idx0], dtype=np.int16 )
                        #rectangles1_at_tail = np.array( unmerged_cluster1.track_r_history[tail_track_idx1], dtype=np.int16 )

                        if utils.judge_track_overlap_at_a_time(rectangles0_at_head, rectangles1_at_head, range_margin) == 0:
                            #if utils.judge_track_overlap_at_a_time(rectangles0_at_tail, rectangles1_at_tail, range_margin) == 0:
                            # when there is no track overlap at the beginning, skip
                            continue

                        if unmerged_cluster1.insignificance:
                            # when both unmerged_clusters are insignificant, choose which one to be merged to the other
                            if unmerged_cluster0.ID < unmerged_cluster1.ID:
                                unmerged_clusterA, unmerged_clusterB = unmerged_cluster0, unmerged_cluster1
                            else:
                                unmerged_clusterA, unmerged_clusterB = unmerged_cluster1, unmerged_cluster0

                            utils.merge_clusters( unmerged_clusterA, unmerged_clusterB )
                            unmerged_clusterA.list_subcluster_idx = np.sort( np.concatenate( (unmerged_clusterA.list_subcluster_idx , unmerged_clusterB.list_subcluster_idx) ) )
                            unmerged_clusterA.remerged_flg = while_cnter + 1
                            unmerged_clusterB.inactive_flg = 1 # mark it to be deleted later
                            remerge_cnter += 1
                            list_remerge_candidates = []
                            break
                        else:
                            # when "unmerged_cluster1" is significant, add it as a candidate to remerge
                            list_remerge_candidates.append( idx1 )

                    #if len(list_remerge_candidates):
                        #max_len = 0
                        #max_candidate_idx = 0
                        #for candidate_idx in list_remerge_candidates:
                            #tmp_len = len( unmerged_clusters[ candidate_idx ].list_subcluster_idx )
                            #if tmp_len > max_len:
                                #max_candidate_idx = candidate_idx
                        #unmerged_cluster1 = unmerged_clusters[ max_candidate_idx ] # candidate "unmerged_cluster"

                    if len(list_remerge_candidates) == 1:
                        # when there is only one candidate to remerge
                        unmerged_cluster1 = unmerged_clusters[ list_remerge_candidates[0] ] # the only candidate "unmerged_cluster"

                        # choose which one to be merged to the other
                        if unmerged_cluster0.ID < unmerged_cluster1.ID:
                            unmerged_clusterA, unmerged_clusterB = unmerged_cluster0, unmerged_cluster1
                        else:
                            unmerged_clusterA, unmerged_clusterB = unmerged_cluster1, unmerged_cluster0

                        utils.merge_clusters( unmerged_clusterA, unmerged_clusterB )
                        unmerged_clusterA.list_subcluster_idx = np.sort( np.concatenate( (unmerged_clusterA.list_subcluster_idx , unmerged_clusterB.list_subcluster_idx) ) )
                        unmerged_clusterA.remerged_flg = while_cnter + 1
                        unmerged_clusterA.insignificance = 0
                        unmerged_clusterB.inactive_flg = 1 # mark it to be deleted later
                        remerge_cnter += 1

                print( f"range_margin: {range_margin}  remerge_insignificant_cnter: {remerge_cnter}" )
                while_cnter += 1

            # delete inactive "unmerged_cluster"
            tmp_cnter = 0 # needed to delete an "unmerged_cluster" avoiding index shifting
            for unmerged_cluster in unmerged_clusters[:]:
                if unmerged_cluster.inactive_flg:
                    # when already remerged to another "unmerged_clusters", delete it
                    del unmerged_clusters[ tmp_cnter ]
                    continue
                unmerged_cluster.remerged_flg = 0 # refrest "remerged_flg"
                tmp_cnter += 1

        print(f"Elapsed time for remerging: {time.time()-time_start:.2f} sec.\n\n" )


        # write output "*s.pkl"
        f_remerged_clusters_path = f_sub_path[:-4] + ".pkl"
        with open(f_remerged_clusters_path, 'wb') as f_pkl_nm:
            pickle.dump(unmerged_clusters, f_pkl_nm)
        unmerged_clusters.clear()
        unmerged_cluster_ids = np.array( [], dtype=np.int32 )


        # delete merged subclusters before writing output "_isolated_subclusters.pkl"
        tmp_cnter = 0 # needed to delete a subcluster avoiding index shifting
        for subcluster in all_subclusters_decoded[:]:
            if subcluster.merged_idx >= 0:
                # when already merged to a "merged_cluster", delete it
                del all_subclusters_decoded[ tmp_cnter ]
                continue
            tmp_cnter += 1

        f_isolated_subclusters_path = f_sub_path[:-4] + "_isolated.pkl"
        with open(f_isolated_subclusters_path, 'wb') as f_pkl_nm:
            pickle.dump(all_subclusters_decoded, f_pkl_nm)
        all_subclusters_decoded.clear()


    print(f"\nClustering finished at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")



if __name__ == "__main__":
    main()
