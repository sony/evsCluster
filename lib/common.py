# coding: utf-8

# Copyright 2023 Sony Group Corporation - All Rights Reserved.
# Subject to the terms and conditions contained in LICENSE.txt accompanying this file, you may use this file.

# Common variables
raw_file_mode = 1
target_basename = ""
analysis_path = ""
list_raw_or_bin_path = []
parent_dir_nm = ""
border_time_idx = 1
blank_binary_img = []
binary_img_buf = []
binary_img_idx = 0
img_subcluster_idxes = []
list_new_pxls = [] # [ [x, y, polarity, timestamp(TIME_HIGH + TIME_LOW) ] , [] ,,, [] ] -> [ [x, y, polarity] , [] ,,, [] ]
timestamp_ms = 0
subclusters = []
subcluster_cnter = 0 # for indexing of the "clusters" list
subcluster_id = 0
f_input = None
f_sub_path = ""
f_sub = None
f_bin_border = None
img = []
blank_img = []
blank_img_idx = []
video = None

last_time_updating_subclusters = 0 # [us]
timestamp_step = 0
last_time = 0

clusters = None
isolated_subclusters = None

selected_phase = 0
f_clusters_path = ""
cluster_idx_to_merge = -1
cluster_idx_to_be_merged = -1
list_new_and_old_clusters = [] # [new cluster's idx, new cluster, old cluster, absorbed cluster]
valid_new_cluster_cnter = 0
tb_fc_iv = None
tb_sc_iv = None
tb_msg_sv = None
flg_annotation = 0