# coding: utf-8

# Copyright 2023 Sony Group Corporation - All Rights Reserved.
# Subject to the terms and conditions contained in LICENSE.txt accompanying this file, you may use this file.

import os
import sys
import cv2
from scipy.spatial import ConvexHull
import numpy as np
import copy
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import scrolledtext
#from tkinter import font
import glob
import threading
import pickle
import time
from lib import utils
from lib import subcluster
from lib import cluster
from lib import analysis
from lib import video
import config_by_gui
import config
from lib import common
from lib import gui_utils



def draw_convex_hull(obj, rectangles, list_bgr):
    points = [] # points of cluster's "rectangle" corners at "timestamp_ms". Used for Convex-Hull
    for rectangle in rectangles:
        # if the rectangle does not have length or width, skip
        if rectangle[0] == rectangle[1]:
            continue
        if rectangle[2] == rectangle[3]:
            continue

        # register each "rectangle" corners to "points"
        points.extend( [ [rectangle[0], rectangle[2]] , [rectangle[1], rectangle[2]] , [rectangle[1], rectangle[3]] , [rectangle[0], rectangle[3]] ] )

    if len(points):
        # calc Convex-Hull
        convex_hull = ConvexHull(points)
        hull_points = convex_hull.points
        hull_points_selected = hull_points[convex_hull.vertices]
        hull_points = np.vstack( ( hull_points_selected, hull_points_selected[0] ) )
        hull_points_int = hull_points.astype(int)

        # draw Convex-Hull
        hull_points_item = hull_points_int[-1]
        for hull_point in hull_points_int:
            cv2.line( obj.frame, hull_points_item, hull_point, list_bgr, thickness=2 )
            hull_points_item = hull_point

        return 1
    else:
        return 0



class Sub_window(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        
        
        
        # frame for analysis directory
        frame_pkl_dir = ttk.Frame(parent, padding=2)
        frame_pkl_dir.pack(fill=tk.X)

        # label for analysis directory
        lbl_pkl_dir = ttk.Label(frame_pkl_dir, text="Select an analysis* folder") # padding=(5, 2)
        lbl_pkl_dir.pack(side=tk.LEFT)

        # textbox for analysis directory
        self.pkl_dir_path_sv = tk.StringVar()
        tb_pkl_dir = ttk.Entry( frame_pkl_dir, textvariable=self.pkl_dir_path_sv, validate="focusout", validatecommand=lambda: self.set_analysis_directory(dialog=0) )
        tb_pkl_dir.pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.pkl_dir_path_sv.set( config_by_gui.DIR_PATH.replace('/', '\\') )

        # button for analysis directory
        btn_pkl_dir = ttk.Button( frame_pkl_dir, text="Select", command=lambda: self.set_analysis_directory(dialog=1) )
        btn_pkl_dir.pack(side=tk.RIGHT)



        # frame for listbox for .pkl files
        frame_lb_pkl = ttk.Frame(parent, padding=2)
        frame_lb_pkl.pack(side=tk.LEFT, anchor=tk.NW, expand=True, fill=tk.BOTH)

        # label for listbox
        lbl_lb_pkl = ttk.Label(frame_lb_pkl, text="Select a .pkl file")
        lbl_lb_pkl.pack()


        # subsubframe for listbox
        subframe_lb_pkl = ttk.Frame(frame_lb_pkl, padding=0)
        subframe_lb_pkl.pack(expand=True, fill=tk.BOTH)

        # listbox
        self.lb_pkl_sv = tk.StringVar()
        lb_pkl = tk.Listbox(subframe_lb_pkl, listvariable=self.lb_pkl_sv, exportselection=0) # , height=8
        lb_pkl.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        lb_pkl.bind('<<ListboxSelect>>', self.lb_pkl_select_one)
        self.set_analysis_directory(dialog=0) # after "lb_pkl_sv" is defined

        # vertical scrollbar
        scrollbar_pkl_v = ttk.Scrollbar(subframe_lb_pkl, orient=tk.VERTICAL, command=lb_pkl.yview)
        lb_pkl['yscrollcommand'] = scrollbar_pkl_v.set
        scrollbar_pkl_v.pack(side=tk.RIGHT, fill=tk.Y)

        # horizontal scrollbar
        scrollbar_pkl_h = ttk.Scrollbar(frame_lb_pkl, orient=tk.HORIZONTAL, command=lb_pkl.xview)
        lb_pkl['xscrollcommand'] = scrollbar_pkl_h.set
        scrollbar_pkl_h.pack(side=tk.BOTTOM, fill=tk.X)



        # frame for manual merging
        frame_mm = ttk.Frame(parent)
        frame_mm.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)


        # subframe for "Focused cluster"
        frame_fc = ttk.Frame(frame_mm, padding=2)
        frame_fc.pack()

        # label for "Focused cluster"
        lbl_fc = ttk.Label(frame_fc, text="Focused cluster: ")
        lbl_fc.pack(side=tk.LEFT)

        # textbox for "Focused cluster"
        common.tb_fc_iv = tk.IntVar()
        tb_fc = ttk.Entry(frame_fc, width=12, textvariable=common.tb_fc_iv, justify=tk.RIGHT)
        tb_fc.pack(side=tk.LEFT)
        tb_fc.configure(state=tk.DISABLED)


        # subframe for "Selected cluster"
        frame_sc = ttk.Frame(frame_mm, padding=2)
        frame_sc.pack()

        # label for "Selected cluster"
        lbl_sc = ttk.Label(frame_sc, text="Selected cluster: ")
        lbl_sc.pack(side=tk.LEFT)

        # textbox for "Selected cluster"
        common.tb_sc_iv = tk.IntVar()
        tb_sc = ttk.Entry(frame_sc, width=12, textvariable=common.tb_sc_iv, justify=tk.RIGHT)
        tb_sc.pack(side=tk.LEFT)
        tb_sc.configure(state=tk.DISABLED)


        # subframe for "Cluster to be merged"
        frame_ctbm = ttk.Frame(frame_mm, padding=2)
        frame_ctbm.pack()

        # label for "Cluster to be merged"
        lbl_ctbm = ttk.Label(frame_ctbm, text="Cluster to be merged: ")
        lbl_ctbm.pack(side=tk.LEFT)

        # textbox for "Cluster to be merged"
        common.tb_ctbm_iv = tk.IntVar()
        tb_ctbm = ttk.Entry(frame_ctbm, width=12, textvariable=common.tb_ctbm_iv, justify=tk.RIGHT)
        tb_ctbm.pack(side=tk.LEFT)
        tb_ctbm.configure(state=tk.DISABLED)


        # subframe for "Message"
        frame_msg = ttk.Frame(frame_mm, padding=2)
        frame_msg.pack()

        # label for "Message"
        lbl_msg = ttk.Label(frame_msg, text="Message: ")
        lbl_msg.pack(side=tk.LEFT)

        # textbox for "Message"
        common.tb_msg_sv = tk.StringVar()
        tb_msg = ttk.Entry(frame_msg, width=35, textvariable=common.tb_msg_sv)
        tb_msg.pack(side=tk.LEFT)
        tb_msg.configure(state=tk.DISABLED)


        # subframe for find function
        frame_find = ttk.Frame(frame_mm, padding=10)
        frame_find.pack()

        lbl_find = ttk.Label(frame_find, text="Find:")
        lbl_find.pack(side=tk.LEFT)

        self.tb_find_iv = tk.IntVar()
        tb_find = ttk.Entry(frame_find, width=12, textvariable=self.tb_find_iv, justify=tk.RIGHT)
        tb_find.pack(side=tk.LEFT)

        btn_find = ttk.Button(frame_find, text="Go", command=self.find_cluster)
        btn_find.pack(side=tk.LEFT)


        # buttons to call functions
        # subframe for "Merge/Undo/Redo/Clear"
        frame_merge = ttk.Frame(frame_mm, padding=10)
        frame_merge.pack()

        btn_merge = ttk.Button(frame_merge, text="Merge", command=self.execute_merging)
        btn_merge.pack(side=tk.LEFT)

        btn_undo = ttk.Button(frame_merge, text="Undo", command=self.undo_merging)
        btn_undo.pack(side=tk.LEFT)

        btn_redo = ttk.Button(frame_merge, text="Redo", command=self.redo_merging)
        btn_redo.pack(side=tk.LEFT)

        btn_clear = ttk.Button(frame_merge, text="Clear", command=self.cancel_merging)
        btn_clear.pack(side=tk.LEFT)


        # subframe for other buttons
        frame_obtn = ttk.Frame(frame_mm, padding=10)
        frame_obtn.pack()

        btn_overwrite = ttk.Button(frame_obtn, text="Overwrite .pkl", command=self.overwrite_pkl)
        btn_overwrite.pack(side=tk.LEFT) # anchor=tk.W



        # frame for listbox for annotation
        frame_lb_anno = ttk.Frame(parent, padding=2)
        frame_lb_anno.pack(side=tk.RIGHT, anchor=tk.NE, expand=True, fill=tk.BOTH)

        # label for listbox
        lbl_lb_anno = ttk.Label(frame_lb_anno, text="Select annotation")
        lbl_lb_anno.pack()


        # subsubframe for listbox
        subframe_lb_anno = ttk.Frame(frame_lb_anno, padding=0)
        subframe_lb_anno.pack(expand=True, fill=tk.BOTH)

        # listbox
        lb_anno_sv = tk.StringVar()
        lb_anno = tk.Listbox(subframe_lb_anno, listvariable=lb_anno_sv, exportselection=0) # , height=8
        lb_anno.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        lb_anno.bind('<<ListboxSelect>>', self.lb_anno_select_one)
        lb_anno_sv.set( [member.name for member in config.Plankton] )

        # vertical scrollbar
        scrollbar_anno_v = ttk.Scrollbar(subframe_lb_anno, orient=tk.VERTICAL, command=lb_anno.yview)
        lb_anno['yscrollcommand'] = scrollbar_anno_v.set
        scrollbar_anno_v.pack(side=tk.RIGHT, fill=tk.Y)

        # horizontal scrollbar
        scrollbar_anno_h = ttk.Scrollbar(frame_lb_anno, orient=tk.HORIZONTAL, command=lb_anno.xview)
        lb_anno['xscrollcommand'] = scrollbar_anno_h.set
        scrollbar_anno_h.pack(side=tk.BOTTOM, fill=tk.X)



    def set_analysis_directory(self, dialog):
        # update "analysis_path"
        dir_given = self.pkl_dir_path_sv.get()
        dir_given = dir_given.replace('\\', '/')
        if len(dir_given) > 0:
            if dir_given[-1] == '/':
                dir_given = dir_given[:-1]

        if dialog:
            dir_returned = filedialog.askdirectory(initialdir=dir_given)
            if dir_returned == '':
                # when "filedialog.askdirectory" dialog was canceled
                return
        else:
            dir_returned = dir_given
        common.analysis_path = dir_returned
        dir_path_backslash = common.analysis_path.replace('/', '\\')
        self.pkl_dir_path_sv.set(dir_path_backslash)


        # update listbox
        pkl_path = self.pkl_dir_path_sv.get() + "/*s.pkl"
        pkl_path = pkl_path.replace('\\', '/')
        list_pkl_path = sorted( glob.glob(pkl_path) )

        len_pkl_dir_path_sv = len(self.pkl_dir_path_sv.get()) + 1
        list_pkl = [ item[len_pkl_dir_path_sv:] for item in list_pkl_path ] # store basename only
        self.lb_pkl_sv.set(list_pkl)



    def find_cluster(self):
        cluster_id_to_find = self.tb_find_iv.get()

        flg_found = 0
        for i,cluster in enumerate(common.clusters):
            if cluster.ID == cluster_id_to_find:
                flg_found = 1
                
                if cluster.inactive_flg:
                    common.tb_msg_sv.set("Found cluster is already merged.")

                #frame_idx = cluster.first_track_t // config.frame_dur
                frame_idx = (cluster.first_track_t + cluster.last_track_t - 2*self.player_obj.timestamp_ms_offset) // (2*config.frame_dur)
                cv2.setTrackbarPos("Frame", self.player_obj.main_window, frame_idx)

                timestamp_ms = self.player_obj.timestamp_ms_offset + frame_idx * config.frame_dur
                track_idx = (timestamp_ms - cluster.first_track_t) // config.track_step_ms # "track_step_ms" is 4 [ms] by default
                cur_rectangles = cluster.track_r_history[track_idx]

                # draw Convex-Hull of the focused cluster
                if draw_convex_hull(self.player_obj, cur_rectangles, [255,0,255]):
                    cv2.imshow(self.player_obj.main_window, self.player_obj.frame)


                if (common.cluster_idx_to_merge == -1) or (common.flg_annotation):
                    common.cluster_idx_to_merge = i
                    common.tb_sc_iv.set(cluster.ID)
                    common.tb_msg_sv.set("Select annotation or another cluster.")
                elif i != common.cluster_idx_to_merge:
                    common.cluster_idx_to_be_merged = i
                    common.tb_ctbm_iv.set(cluster.ID)
                    common.tb_msg_sv.set("Ready to merge.")

        if not flg_found:
            common.tb_msg_sv.set("Not found.")



    def execute_merging(self):
        if common.cluster_idx_to_merge == common.cluster_idx_to_be_merged:
            common.tb_msg_sv.set("Same clusters.")

        # if the cluster to merge is younger than the cluster to be merged, swap them
        if common.cluster_idx_to_merge > common.cluster_idx_to_be_merged:
            tmp_cluster_idx = common.cluster_idx_to_be_merged
            common.cluster_idx_to_be_merged = common.cluster_idx_to_merge
            common.cluster_idx_to_merge = tmp_cluster_idx

        cluster_to_merge = common.clusters[ common.cluster_idx_to_merge ]
        cluster_to_be_merged = common.clusters[ common.cluster_idx_to_be_merged ]
        if cluster_to_be_merged.first_track_t <= cluster_to_merge.last_track_t:
            # when there is timestamp overlap between "cluster_to_merge" and "cluster_to_be_merged"
            old_cluster_to_merge = copy.deepcopy(cluster_to_merge)
            utils.merge_clusters( cluster_to_merge, cluster_to_be_merged )
            # no need to maintain "list_subcluster_idx" anymore
            #(cluster_to_merge.list_subcluster_idx).extend( cluster.list_subcluster_idx ) # update "list_subcluster_idx"
            #(cluster_to_merge.list_subcluster_idx).sort()
            common.list_new_and_old_clusters = common.list_new_and_old_clusters[:common.valid_new_cluster_cnter] # to overwrite history after undoing remerging
            common.list_new_and_old_clusters.append( [common.cluster_idx_to_merge, cluster_to_merge, old_cluster_to_merge, cluster_to_be_merged] )
            common.valid_new_cluster_cnter += 1
            cluster_to_be_merged.inactive_flg = 1
            common.cluster_idx_to_merge = -1
            common.cluster_idx_to_be_merged = -1
            common.tb_sc_iv.set(0)
            common.tb_ctbm_iv.set(0)
            common.tb_msg_sv.set("Clusters were merged.")
        else:
            # when there is no timestamp overlap between "cluster_to_merge" and "cluster"
            common.tb_msg_sv.set("No time overlap.")



    def undo_merging(self):
        if common.valid_new_cluster_cnter > 0:
            common.valid_new_cluster_cnter -= 1
            cluster_idx_to_be_unmerged = common.list_new_and_old_clusters[ common.valid_new_cluster_cnter ][0]
            common.clusters[cluster_idx_to_be_unmerged] = common.list_new_and_old_clusters[ common.valid_new_cluster_cnter ][2]
            common.list_new_and_old_clusters[ common.valid_new_cluster_cnter ][3].inactive_flg = 0
            common.tb_msg_sv.set("Last merging was undone.")
        else:
            common.tb_msg_sv.set("Nothing more to undo.")



    def redo_merging(self):
        if common.valid_new_cluster_cnter < len(common.list_new_and_old_clusters):
            cluster_idx_to_be_remerged = common.list_new_and_old_clusters[ common.valid_new_cluster_cnter ][0]
            common.clusters[cluster_idx_to_be_remerged] = common.list_new_and_old_clusters[ common.valid_new_cluster_cnter ][1]
            common.list_new_and_old_clusters[ common.valid_new_cluster_cnter ][3].inactive_flg = 1
            common.valid_new_cluster_cnter += 1
            common.tb_msg_sv.set("Last un-merging was redone.")
        else:
            common.tb_msg_sv.set("Nothing more to redo.")



    def cancel_merging(self):
        common.cluster_idx_to_merge = -1
        common.cluster_idx_to_be_merged = -1
        common.tb_sc_iv.set(0)
        common.tb_ctbm_iv.set(0)
        common.tb_msg_sv.set("Cleared selection.")



    def thread_play(self):
        self.player_obj = Player()
        self.player_obj.run()



    def lb_pkl_select_one(self, event):
        w = event.widget
        pkl_nm = w.get( int(w.curselection()[0]) )

        # read "*s.pkl"
        common.f_clusters_path = common.analysis_path + '/' + pkl_nm
        with open(common.f_clusters_path, "rb") as f_pkl_nm:
            common.clusters = pickle.load(f_pkl_nm)
        print(f"\nReading {common.f_clusters_path} finished.")

        th1_run = threading.Thread(target=self.thread_play)
        th1_run.start()



    def lb_anno_select_one(self, event):
        w = event.widget
        common.clusters[ common.cluster_idx_to_merge ].annotation = w.get( int(w.curselection()[0]) )
        common.flg_annotation = 1
        common.tb_msg_sv.set(f"Labeled cluster { common.clusters[ common.cluster_idx_to_merge ].ID } as {w.get( w.curselection()[0])}")



    def overwrite_pkl(self):
        # delete inactive clusters
        tmp_cnter = 0 # needed to delete a cluster avoiding index shifting
        for cluster in common.clusters[:]:
            if cluster.inactive_flg:
                # when already merged to another cluster, delete it
                del common.clusters[ tmp_cnter ]
                continue
            tmp_cnter += 1

        # overwrite "*s.pkl"
        with open(common.f_clusters_path, "wb") as f_pkl_nm:
            pickle.dump(common.clusters, f_pkl_nm)
        print(f"\nOverwriting {common.f_clusters_path} finished.")
        common.tb_msg_sv.set("Overwriting .pkl finished.")



class Player():
    def __init__(self):
        # setup the player window for OpenCV
        self.main_window = 'avi player'
        cv2.namedWindow(self.main_window)

        # setup capturing .avi
        self.f_avi = common.f_clusters_path[:-4] + ".avi"
        self.cap = cv2.VideoCapture(self.f_avi)
        self.frame = None
        self.frame_ori = None
        self.total_n_frame = int( self.cap.get(cv2.CAP_PROP_FRAME_COUNT) ) # the total number of frames
        #cap.set(cv2.CAP_PROP_POS_MSEC, self.timestamp_ms)

        self.command_mode = 0
        self.frame_cnter = 0
        
        from_hour = int( self.f_avi[-24:-22] )
        from_min = int( self.f_avi[-21:-19] )
        from_sec = int( self.f_avi[-18:-16] )
        self.timestamp_ms_offset = ( ((from_hour*60) + from_min)*60 + from_sec )*1000
        self.timestamp_ms = self.timestamp_ms_offset

        cv2.setMouseCallback(self.main_window, self.mouse_callback)
        cv2.createTrackbar("Frame", self.main_window, 1, self.total_n_frame, self.trackbar_callback)

        self.x_pos = None
        self.y_pos = None
        self.font = cv2.FONT_HERSHEY_SIMPLEX
    
    def run(self):
        flg_break = 0
        
        while not flg_break:
            while self.frame_cnter <= self.total_n_frame:
                if self.frame_cnter < self.total_n_frame:
                    self.frame_cnter += 1
                    self.timestamp_ms = self.timestamp_ms_offset + self.frame_cnter * config.frame_dur # frame_dur = is 20 [ms] by default
                    ret, self.frame = self.cap.read()
                    cv2.imshow(self.main_window, self.frame)
                    cv2.setTrackbarPos("Frame", self.main_window, self.frame_cnter-1)

                code = cv2.waitKeyEx(1)
                if (code == ord(' ')) or (self.frame_cnter == self.total_n_frame):
                    # when the whitespace key is pressed
                    self.command_mode = 1
                    self.frame_ori = copy.deepcopy(self.frame) # backup original display
                    code = cv2.waitKeyEx(0)

                    while (code == 2424832) or (code == 2555904): # left or right arrow key
                        if code == 2424832: # left arrow
                            if self.frame_cnter >= 2:
                                cv2.setTrackbarPos("Frame", self.main_window, self.frame_cnter-2)
                        elif code == 2555904: # right arrow
                            if self.frame_cnter < self.total_n_frame:
                                cv2.setTrackbarPos("Frame", self.main_window, self.frame_cnter)
                        code = cv2.waitKeyEx(0)

                    self.command_mode = 0
                    if code == ord('q'):
                        flg_break = 1
                        break
                elif code == ord('q'):
                    flg_break = 1
                    break
            
#             if not flg_break:
#                 self.command_mode = 1
#                 self.frame_ori = copy.deepcopy(self.frame) # backup original display
#                 code = cv2.waitKeyEx(0)
                
#                 code = cv2.waitKeyEx(0)
#                 if code == ord(' '):
#                     # when the whitespace key is pressed
#                     self.command_mode = 1
#                     self.frame_ori = copy.deepcopy(self.frame) # backup original display
#                     code = cv2.waitKeyEx(0)
#                     self.command_mode = 0
#                     if code == ord('q'):
#                         break
#                 elif code == ord('q'):
#                     break

        self.cap.release()
        cv2.destroyAllWindows()

    def mouse_callback(self, event, x, y, flags, param):
        if self.command_mode:
            # after the whitespace key is pressed

            #if event == cv2.EVENT_MBUTTONUP and flags == cv2.EVENT_FLAG_SHIFTKEY | cv2.EVENT_FLAG_CTRLKEY:
            if (event == cv2.EVENT_MOUSEMOVE) or (flags == cv2.EVENT_FLAG_LBUTTON):
                xmax = x + config.FOCUS_MARGIN
                xmin = x - config.FOCUS_MARGIN
                ymax = y + config.FOCUS_MARGIN
                ymin = y - config.FOCUS_MARGIN

                xflg_focused_ID = 1
                for i,cluster in enumerate(common.clusters):
                    if cluster.inactive_flg:
                        continue
                    
                    if cluster.last_track_t < self.timestamp_ms:
                        # when "cluster" disappeared before the given "timestamp_ms"
                        continue
                    if self.timestamp_ms < cluster.first_track_t:
                        # when "cluster" appeared after the given "timestamp_ms"
                        break

                    # check if "cluster" has been around the neighbourhood of the mouse position
                    if cluster.xy_limit[0] > x:
                        continue
                    if cluster.xy_limit[1] < x:
                        continue
                    if cluster.xy_limit[2] > y:
                        continue
                    if cluster.xy_limit[3] < y:
                        continue

                    track_idx = (self.timestamp_ms - cluster.first_track_t) // config.track_step_ms # "track_step_ms" is 4 [ms] by default
                    cur_rectangles = cluster.track_r_history[track_idx]

                    # check if any "rectangle" of "cluster" at "timestamp_ms" has been around the neighbourhood of the mouse position
                    xflg_focused_rectangle = 1
                    for rectangle in cur_rectangles:
                        if rectangle[0] > xmax:
                            continue
                        if rectangle[1] < xmin:
                            continue
                        if rectangle[2] > ymax:
                            continue
                        if rectangle[3] < ymin:
                            continue
                        xflg_focused_rectangle = 0

                    if xflg_focused_rectangle:
                        continue

                    xflg_focused_ID = 0 # succeeded to focus on a cluster
                    common.tb_fc_iv.set(cluster.ID)

                    # draw Convex-Hull of the focused cluster
                    if draw_convex_hull(self, cur_rectangles, [255,255,0]):
                        cv2.imshow(self.main_window, self.frame)

                    if flags == cv2.EVENT_FLAG_LBUTTON:
                        # At the event of left click, register as a cluster to merge or to be merged
                        if (common.cluster_idx_to_merge == -1) or (common.flg_annotation):
                            common.cluster_idx_to_merge = i
                            common.tb_sc_iv.set(cluster.ID)
                            common.tb_msg_sv.set("Select annotation or another cluster.")
                        elif i != common.cluster_idx_to_merge:
                            common.cluster_idx_to_be_merged = i
                            common.tb_ctbm_iv.set(cluster.ID)
                            common.tb_msg_sv.set("Ready to merge.")
                
                if xflg_focused_ID:
                    common.tb_fc_iv.set(0)
                    self.frame = copy.deepcopy(self.frame_ori)
                    cv2.imshow(self.main_window, self.frame)
    
    def trackbar_callback(self, val): # setup trackbar
        if self.command_mode:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, val)
            ret, self.frame = self.cap.read()
            self.frame_ori = copy.deepcopy(self.frame) # backup original display
            cv2.imshow(self.main_window, self.frame)
            self.frame_cnter = val+1
            self.timestamp_ms = self.timestamp_ms_offset + self.frame_cnter * config.frame_dur # frame_dur = is 20 [ms] by default
        
        #for new_and_old_clusters in common.list_new_and_old_clusters:
        for i in range(common.valid_new_cluster_cnter):
            new_cluster = common.clusters[ common.list_new_and_old_clusters[i][0] ]
            
            if new_cluster.last_track_t < self.timestamp_ms:
                # when the new cluster disappeared before the given "timestamp_ms"
                continue
            if self.timestamp_ms < new_cluster.first_track_t:
                # when the new cluster appeared after the given "timestamp_ms"
                continue
            
            track_idx = (self.timestamp_ms - new_cluster.first_track_t) // config.track_step_ms # "track_step_ms" is 4 [ms] by default
            cur_rectangles = new_cluster.track_r_history[track_idx]
            
            # draw Convex-Hull of the focused new cluster
            if draw_convex_hull(self, cur_rectangles, [0,255,255]):
                cv2.imshow(self.main_window, self.frame)



class Main_window(gui_utils.Common_window):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent)
        self.sub_window = None
        
        
        
        # frame for parent directory
        frame_dir = ttk.Frame(parent, padding=2)
        frame_dir.pack(fill=tk.X)

        # label for parent directory
        lbl_dir = ttk.Label(frame_dir, text="Folder to search") # padding=(5, 2)
        lbl_dir.pack(side=tk.LEFT)

        # textbox for parent directory
        self.dir_path_sv = tk.StringVar()
        tb_dir = ttk.Entry( frame_dir, textvariable=self.dir_path_sv, validate="focusout", validatecommand=lambda: self.set_parent_directory(dialog=0) )
        tb_dir.pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.dir_path_sv.set( config_by_gui.DIR_PATH.replace('/', '\\') )

        # button for parent directory
        btn_dir = ttk.Button( frame_dir, text="Select", command=lambda: self.set_parent_directory(dialog=1) )
        btn_dir.pack(side=tk.RIGHT)



        # frame for listbox and scrolledtext
        frame_lb_st = ttk.Frame(root, padding=2)
        frame_lb_st.pack(anchor=tk.NW, expand=True, fill=tk.BOTH)


        # subframe for listbox
        frame_lb = ttk.Frame(frame_lb_st, padding=0)
        frame_lb.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        # label for listbox
        lbl_lb = ttk.Label(frame_lb, text="Select .raw files\nor Index* folders\ncontaining .bin files")
        lbl_lb.pack()


        # subsubframe for listbox
        subframe_lb = ttk.Frame(frame_lb, padding=0)
        subframe_lb.pack(expand=True, fill=tk.BOTH)

        # listbox
        self.lb_sv = tk.StringVar()
        self.lb = tk.Listbox(subframe_lb, selectmode="extended", listvariable=self.lb_sv, exportselection=0) # , height=8
        self.lb.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        self.set_parent_directory(dialog=0) # after "lb_sv" is defined

        # vertical scrollbar
        scrollbar_v = ttk.Scrollbar(subframe_lb, orient=tk.VERTICAL, command=self.lb.yview)
        self.lb['yscrollcommand'] = scrollbar_v.set
        scrollbar_v.pack(side=tk.RIGHT, fill=tk.Y)

        # horizontal scrollbar
        scrollbar_h = ttk.Scrollbar(frame_lb, orient=tk.HORIZONTAL, command=self.lb.xview)
        self.lb['xscrollcommand'] = scrollbar_h.set
        scrollbar_h.pack(side=tk.BOTTOM, fill=tk.X)


        # subframe for scrolledtext
        frame_st = ttk.Frame(frame_lb_st, padding=0)
        frame_st.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        # label for scrolledtext
        lbl_st = ttk.Label(frame_st, text="Progress monitor", padding=(5,0))
        lbl_st.pack()

        # scrolledtext
        self.st = scrolledtext.ScrolledText(frame_st) # ,width=40, height=12
        self.st.pack(expand=True, fill=tk.BOTH)
        self.st.configure(state=tk.DISABLED)
        self.redirect_print_output()


        # subsubframe for progress phase
        frame_phase = ttk.Frame(frame_st)
        frame_phase.pack()

        # label for "Current target"
        lbl_ct = ttk.Label(frame_phase, text="Current target:  ", padding=(1,0))
        lbl_ct.pack(side=tk.LEFT)

        # textbox for "Current target"
        self.cur_target_sv = tk.StringVar()
        tb_cur_target = ttk.Entry( frame_phase, textvariable=self.cur_target_sv, width=15 )
        tb_cur_target.pack(side=tk.LEFT)
        #tb_cur_target.configure(state=tk.DISABLED)

        # label for "Progress phase"
        lbl_pp = ttk.Label(frame_phase, text="   Progress phase: ", padding=(1,0))
        lbl_pp.pack(side=tk.LEFT)

        # label for "Subcluster"
        self.lbl_subcluster = ttk.Label(frame_phase, text="Subcluster", padding=(1,0))
        self.lbl_subcluster.pack(side=tk.LEFT)
        #self.default_font = tk.font.Font(self.lbl_subcluster, self.lbl_subcluster.cget("font"))
        #self.bold_font = tk.font.Font(self.lbl_subcluster, self.lbl_subcluster.cget("font"))
        #self.bold_font.configure(weight="bold")

        # label for right arrow 1
        lbl_ra1 = ttk.Label(frame_phase, text="->", padding=(1,0))
        lbl_ra1.pack(side=tk.LEFT)

        # label for "Cluster"
        self.lbl_cluster = ttk.Label(frame_phase, text="Cluster", padding=(1,0))
        self.lbl_cluster.pack(side=tk.LEFT)

        # label for right arrow 2
        lbl_ra2 = ttk.Label(frame_phase, text="->", padding=(1,0))
        lbl_ra2.pack(side=tk.LEFT)

        # label for "Analysis"
        self.lbl_analysis = ttk.Label(frame_phase, text="Analysis", padding=(1,0))
        self.lbl_analysis.pack(side=tk.LEFT)

        # label for right arrow 3
        lbl_ra3 = ttk.Label(frame_phase, text="->", padding=(1,0))
        lbl_ra3.pack(side=tk.LEFT)

        # label for "Video"
        self.lbl_video = ttk.Label(frame_phase, text="Video", padding=(1,0))
        self.lbl_video.pack(side=tk.LEFT)



        # label for "separator"
        lbl_separator = ttk.Label(root, text="--- Below are subcluster opitions ---")
        lbl_separator.pack()



        # frame for settings
        frame_set = ttk.Frame(root, padding=2)
        frame_set.pack(fill=tk.X)

        # checkbutton for "ENABLE_MONITORING"
        cb = ttk.Checkbutton(frame_set, text="Output *_sub.avi     ", variable=self.enable_monitoring_iv)
        cb.grid(row=0, column=0)
        self.enable_monitoring_iv.set(config_by_gui.ENABLE_MONITORING)

        # label for "NEIGHBOUR_TH"
        lbl_nt = ttk.Label(frame_set, text="            NEIGHBOUR_TH")
        lbl_nt.grid(row=0, column=1)

        # textbox for "NEIGHBOUR_TH"
        tb_nt = ttk.Entry(frame_set, width=6, textvariable=self.neighbour_th_iv, justify=tk.RIGHT)
        tb_nt.grid(row=0, column=2)
        self.neighbour_th_iv.set(config_by_gui.NEIGHBOUR_TH)
        
        
        
        # frame for "LIST_BORDER_TIME"
        frame_lbt = ttk.Frame(root, padding=2)
        frame_lbt.pack(fill=tk.X)
        
        # label for "LIST_BORDER_TIME"
        lbl_lbt = ttk.Label(frame_lbt, text="Border time list")
        lbl_lbt.pack(side=tk.LEFT)
        
        # textbox for "LIST_BORDER_TIME"
        tb_lbt = ttk.Entry(frame_lbt, textvariable=self.list_border_time_sv) #, width=40
        tb_lbt.pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.list_border_time_sv.set( ','.join( str(item) for item in config_by_gui.LIST_BORDER_TIME ) )
        
        # radiobuttons for "LIST_BORDER_TIME"
        rm=ttk.Radiobutton(frame_lbt, text="[min]", variable=self.min_or_sec_iv, value=1)
        rm.pack(side=tk.LEFT)
        rs=ttk.Radiobutton(frame_lbt, text="[sec]", variable=self.min_or_sec_iv, value=0)
        rs.pack(side=tk.RIGHT)
        if config_by_gui.MIN_OR_SEC == "min":
            self.min_or_sec_iv.set(1)
        else:
            self.min_or_sec_iv.set(0)
        
        
        
        # frame for buttons
        frame_btn = ttk.Frame(root, padding=2)
        frame_btn.pack()

        # button for starting process
        btn_run = ttk.Button(frame_btn, text='Run all', command=lambda: self.run_process(lb=self.lb, phase=0))
        btn_run.grid(row=0, column=0, padx=0)

        btn_run = ttk.Button(frame_btn, text='Subcluster', command=lambda: self.run_process(lb=self.lb, phase=1))
        btn_run.grid(row=0, column=1, padx=0)

        btn_run = ttk.Button(frame_btn, text='Cluster', command=lambda: self.run_process(lb=self.lb, phase=2))
        btn_run.grid(row=0, column=2, padx=0)

        btn_analysis = ttk.Button(frame_btn, text='Analysis', command=lambda: self.run_process(lb=self.lb, phase=3))
        btn_analysis.grid(row=0, column=3, padx=0)

        btn_video = ttk.Button(frame_btn, text='Video', command=lambda: self.run_process(lb=self.lb, phase=4))
        btn_video.grid(row=0, column=4, padx=0)

        # # button for killing on-going process
        # btn_stop = ttk.Button(frame_btn, text='Stop', command=show_selection)
        # btn_stop.grid(row=0, column=1, padx=5)

        # button for opening a subwindow to manually merge clusters
        btn_rc = ttk.Button(frame_btn, text='Modify clusters', command=self.call_modify_cluster)
        btn_rc.grid(row=0, column=5, padx=30)

        # button for saving settings
        btn_ss = ttk.Button(frame_btn, text='Save settings', command=lambda: self.overwrite_config_by_gui_py(self.lb))
        btn_ss.grid(row=0, column=6, padx=0)
    
    
    
    def print_redirector(self, input_str):
        self.st.configure(state=tk.NORMAL)
        self.st.insert(tk.END, input_str)
        self.st.configure(state=tk.DISABLED)
    
    
    
    def redirect_print_output(self):
        if config.ENABLE_STDOUT_REDIRECTOR:
            sys.stdout.write = self.print_redirector
            #sys.stderr.write = self.print_redirector
    
    
    
    # overwrite set_parent_directory()
    def set_parent_directory(self, dialog):
        super().set_parent_directory(dialog)
        
        # update listbox
        index_path = self.dir_path_sv.get() + "/Index*"
        index_path = index_path.replace('\\', '/')
        list_dir_path = sorted( glob.glob(index_path) )
        
        raw_path = self.dir_path_sv.get() + "/*.raw"
        raw_path = raw_path.replace('\\', '/')
        list_raw_path = sorted( glob.glob(raw_path) )
        
        list_all_target_path = list_dir_path + list_raw_path # concatenate two lists
        
        len_dir_path_sv = len(self.dir_path_sv.get()) + 1
        list_all_target = [ item[len_dir_path_sv:] for item in list_all_target_path ] # store basename only
        self.lb_sv.set(list_all_target)
        
        self.lb.selection_clear(0, tk.END)
        for i,target in enumerate(list_all_target): # highlight items in "LIST_TARGET"
            if target in config_by_gui.LIST_TARGET:
                self.lb.select_set(i)
        
        return 1 # required for the callback function to be called more than once
    
    
    
    # overwrite thread_run()
    def thread_run(self):
        for target_nm in config_by_gui.LIST_TARGET:
            self.cur_target_sv.set(target_nm) # display the current target name

            if (common.selected_phase == 0) or (common.selected_phase == 1):
                self.lbl_subcluster.configure(foreground="red")
                result = subcluster.main(target_nm, flg_evs2video=0)
                self.lbl_subcluster.configure(foreground="black")
                if result:
                    break

            if (common.selected_phase == 0) or (common.selected_phase == 2):
                self.lbl_cluster.configure(foreground="red")
                cluster.main(target_nm)
                self.lbl_cluster.configure(foreground="black")

            if (common.selected_phase == 0) or (common.selected_phase == 3):
                self.lbl_analysis.configure(foreground="red")
                analysis.main(target_nm)
                self.lbl_analysis.configure(foreground="black")

            if (common.selected_phase == 0) or (common.selected_phase == 4):
                self.lbl_video.configure(foreground="red")
                result = video.main(target_nm)
                self.lbl_video.configure(foreground="black")
                if result:
                    break

        print("\n\n")



    def call_modify_cluster(self):
        if (self.sub_window == None) or (not self.sub_window.winfo_exists()):
            # update "LIST_TARGET"
            config_by_gui.LIST_TARGET = []
            for i in self.lb.curselection():
                config_by_gui.LIST_TARGET.append( self.lb.get(i) )

            self.sub_window = tk.Toplevel()
            self.sub_window.wm_transient(root)
            self.sub_window.title("Modify clusters")
            #self.sub_window.geometry("350x180")
            Sub_window(self.sub_window).pack(side=tk.TOP, fill=tk.BOTH, expand=True)




if __name__ == "__main__":
    root = tk.Tk()
    root.title("evs2cluster")
    Main_window(root).pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    root.mainloop()
