# coding: utf-8

# Copyright 2023 Sony Group Corporation - All Rights Reserved.
# Subject to the terms and conditions contained in LICENSE.txt accompanying this file, you may use this file.

import numpy as np
from enum import Enum, unique

# User-defined variables
VERTICAL_OPTICAL_FLIP = 0 # optically flipped or not
HORIZONTAL_OPTICAL_FLIP = 0 # optically flipped or not
WIDTH = 1280
HEIGHT = 720 #720 #736
TIMESTAMP_JUMP_TH = 500 # [us]

NUM_BINARY_IMG_BUF = 4
KERNEL_SIZE = 5 # filter evs data with a kernel of "KERNEL_SIZE"*"KERNEL_SIZE" pixel size. Supposed to be an odd number. default: 5

CLUSTERING_NEW_PXLS_DUR = 1000 # [us]. Supposed to be a multiple of 1000
cluster_step_ms = CLUSTERING_NEW_PXLS_DUR // 1000
event_fft_fs = (1000*1000) // CLUSTERING_NEW_PXLS_DUR # sampling freq for event FFT (default: 1000 Hz (1ms period))

UPDATING_SUBCLUSTERS_DUR = 4000 # [us]. Supposed to be a multiple of 1000
track_step_ms = UPDATING_SUBCLUSTERS_DUR // 1000
coor_fft_fs = (1000*1000) // UPDATING_SUBCLUSTERS_DUR # sampling freq for coordinate FFT (default: 250 Hz (4ms period))

SUBCLUSTER_TIMESTAMP_MILESTONE_STEP = 1000*1000*60 # default: 1 min (1000*1000*60us)
SUBCLUSTER_DUR = 100 #6000 #1000000 # each pixel in a subcluster remains for "SUBCLUSTER_DUR" [ms]
SUBCLUSTER_SIZE_LIMIT = 19 # allow for each subcluster to grow up to 20(=19+1)) pxl length

LIST_BANNED_REGION = []
#LIST_BANNED_REGION = [ [187,247,155,215], [1062,1122,151,211], [185,245,515,575], [1059,1119,518,578] ] # [x_min, x_max, y_min, y_max ] #ROV day1
# LIST_BANNED_REGION = [ [22,102,70,150], [1218,1280,65,145], [24,104,555,635], [1203,1283,558,638],
#                        [345,375,214,244], [927,957,208,238], [342,372,449,479], [921,951,453,483],
#                        [513,543,42,72], [752,782,45,75], [511,541,619,649], [744,774,617,647] ] # [x_min, x_max, y_min, y_max ] #ROV day2
#LIST_BANNED_REGION = [ [174,234,155,215], [1065,1125,146,206], [170,230,515,575], [1055,1115,520,580] ] # [x_min, x_max, y_min, y_max ] #ROV day3

FLG_CALC_FEATURES = 1 # if FLG_CALC_FEATURES = 0, then ENABLE_FFT is automatically considered to be 0 and cluster.width is not going to be calculated
ENABLE_FFT = 1
FLG_INFERENCE = 1

# params for drawing
CD_ON = [255,255,255]
CD_OFF = [127,127,127] #[191,191,191] [63,63,63]
COLORS = [[255,0,0], [0,255,255], [255,0,255], [127,0,0], [0,127,0],  [0,0,127], [127,127,0], [0,127,127], [127,0,127],  [127,63,0], [0,127,63], [63,0,127], [63,127,0]]
FPS = 50 #250 #50
frame_dur = (1000//FPS) # 20 [ms] step at 50 fps
DUR_TO_SHOW = 1000
dur_to_show_div = DUR_TO_SHOW // track_step_ms

MARGIN_TO_JUDGE = 10
XY_LIMIT_MARGIN = 10
CLUSTER_TIMESTAMP_MILESTONE_STEP = 1000*60 # just for monitoring progress. [ms] 

NUM_TRACK_TO_AVERAGE = 25 #10
GRAPH_LINE_WIDTH = 0.5

TRAVEL_DISTANCE_TH = 200.

# threshold to judge activeness
ACTIVE_VEL_TH = 0.025 # [pxl/ms]
ACTIVE_VAR_TH = 0.14
VERY_ACTIVE_VAR_TH = 1.0

ANTENNA_WEIGHT_BIAS = 10
SQ_JUMP_CNTER_TH = 5.0*5.0 # large acceleration peaks with values over five times larger than the average acceleration across all the trajectories were counted. Actually, squared value of 5.0 is used in the program for the ease of calculation
JUMP_CNTER_OFFSET = 250 # to avoid counting one jump as multiple jumps or counting vanishment as jumping
EVENT_FFT_TH = 1000 # in order to be processed by event FFT, a cluster needs to have an event history length equal to or more than "EVENT_FFT_TH" (default: 1000 [ms])
LIST_FQ_BORDER = [1., 2.15443469003188, 4.64158883361278, 10., 21.5443469003188] # [Hz]

# when inference by neural network is enabled
@unique
class Plankton(Enum):
    Passive = 0
    Pcra_C = 1 # Copepod adult
    Pcra_N = 2 # Copepod larva
    Nfus = 3 # Kusairo aogai
    Ppec = 4 # Itomaki hitode
    Artemia = 5
    Osedax = 6 # Honekui hanamushi
    Septifer = 7 # Murasaki inko
    Unknown = 96
    Collision = 97
    Shadow = 98
    Other = 99
#LIST_PLANKTON_NM = [Passive", "Copepod", "Nipponacmea", "Patiria", "Artemia", "Osedax", "Septifer"]

NP_NORM_CEIL = np.array([0.5, 5., 11., 400., 400., 0.5, 0.1, 0.1, 0.1, 0.1, 1.01, 1.01, 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.])

W_I_H__PATH = "nn_param/w_i_h__mix.csv"
W_H_O__PATH = "nn_param/w_h_o__mix.csv"
B_I_H__PATH = "nn_param/b_i_h__mix.csv"
B_H_O__PATH = "nn_param/b_h_o__mix.csv"

NUM_FEATURES = 22
COLUMN_IDX_FEATURE_OFFSET = 3
COLUMN_IDX_ANNOTATION = 31
NUM_CLASSIFICATION = 5
NUM_H_NORD = 20
LEARNING_RATE = 0.1
NUM_EPOCH = 10000

ENABLE_STDOUT_REDIRECTOR = 1
FOCUS_MARGIN = 5

RADAR_WIDTH = 1472*2 # 1472 is almost eaqual to the value of sqrt(1280*1280+720*720)
RADAR_HEIGHT = 1472*2
SHRINK_SCALE = 4

# global variables for CD_Y, X_POS, X_BASE, VECT_12, VECT_8, TIME_LOW, CONTINUED_4, TIME_HIGH, EXT_TRIGGER, OTHERS, CONTINUED_12
EVENT_TYPE_CD_Y         = b'\x00'
EVENT_TYPE_X_POS        = b'\x20'
EVENT_TYPE_X_BASE       = b'\x30'
EVENT_TYPE_VECT_12      = b'\x40'
EVENT_TYPE_VECT_8       = b'\x50'
EVENT_TYPE_TIME_LOW     = b'\x60'
EVENT_TYPE_CONTINUED_4  = b'\x70'
EVENT_TYPE_TIME_HIGH    = b'\x80'
EVENT_TYPE_EXT_TRIGGER  = b'\xA0'
EVENT_TYPE_OTHERS       = b'\xE0'
EVENT_TYPE_CONTINUED_12 = b'\xF0'

MASK_EVENT_TYPE = b'\xF0'
MASK_VALUE_0F = b'\x0F'
MASK_VALUE_08 = b'\x08'
MASK_VALUE_07 = b'\x07'
