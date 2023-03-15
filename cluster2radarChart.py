# coding: utf-8

# Copyright 2023 Sony Group Corporation - All Rights Reserved.
# Subject to the terms and conditions contained in LICENSE.txt accompanying this file, you may use this file.

import os
import sys
import cv2
import numpy as np
import pickle
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
from tkinter import messagebox
import glob
from lib import utils
import config_by_gui
import config
from lib import gui_utils



class Main_window(gui_utils.Common_window):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent)

        self.start_ms = 0
        self.end_ms = 0
        self.clusters = None
        self.list_focused_cluster_id = []



        # frame for parent directory
        frame_dir = ttk.Frame(root, padding=2)
        frame_dir.pack(anchor=tk.NW, expand=True, fill=tk.X)

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
        lbl_lb = tk.Label(frame_lb, text="Select .pkl files")
        lbl_lb.pack()


        # subsubframe for listbox
        subframe_lb = ttk.Frame(frame_lb, padding=0)
        subframe_lb.pack(expand=True, fill=tk.BOTH)

        # listbox
        self.lb_sv = tk.StringVar()
        self.lb = tk.Listbox(subframe_lb, selectmode="extended", listvariable=self.lb_sv) # , height=8
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
        lbl_st = tk.Label(frame_st, text="Progress monitor", padx=5)
        lbl_st.pack()

        # scrolledtext
        self.st= scrolledtext.ScrolledText(frame_st) # ,width=40, height=12
        self.st.pack(expand=True, fill=tk.BOTH)
        self.st.configure(state=tk.DISABLED)
        self.redirect_print_output()



        # frame for time range setup
        frame_trs = ttk.Frame(root, padding=2)
        frame_trs.pack()

        # label for "Start second"
        lbl_ss = tk.Label(frame_trs, text="Start second ", padx=1)
        lbl_ss.pack(side=tk.LEFT)

        # textbox for "Start second"
        self.start_sec_iv = tk.IntVar()
        tb_start_sec = ttk.Entry( frame_trs, textvariable=self.start_sec_iv, width=15 )
        tb_start_sec.pack(side=tk.LEFT)
        self.start_sec_iv.set(0)

        # label for "End second"
        lbl_es = tk.Label(frame_trs, text="   End second ", padx=1)
        lbl_es.pack(side=tk.LEFT)

        # textbox for "End second"
        self.end_sec_iv = tk.IntVar()
        tb_end_sec = ttk.Entry( frame_trs, textvariable=self.end_sec_iv, width=15 )
        tb_end_sec.pack(side=tk.LEFT)
        self.end_sec_iv.set(1)


        # frame for "list_focused_cluster_id"
        frame_lfci = ttk.Frame(root, padding=2)
        frame_lfci.pack(expand=True, fill=tk.X)

        # label for "list_focused_cluster_id"
        lbl_lfci = tk.Label(frame_lfci, text="Focused cluster's ID list")
        lbl_lfci.pack(side=tk.LEFT)

        # textbox for "list_focused_cluster_id"
        self.list_focused_cluster_id_sv = tk.StringVar()
        tb_lfci = tk.Entry(frame_lfci, textvariable=self.list_focused_cluster_id_sv) #, width=40
        tb_lfci.pack(side=tk.LEFT, expand=True, fill=tk.X)



        # frame for buttons
        frame_btn = ttk.Frame(root, padding=2)
        frame_btn.pack()

        # button for loading a .pkl file
        btn_load = ttk.Button(frame_btn, text='Load .pkl', command=self.load_pkl)
        btn_load.grid(row=0, column=0, padx=5)

        # button for processing a .pkl file
        btn_run = ttk.Button(frame_btn, text='Draw', command=self.draw_radar)
        btn_run.grid(row=0, column=1, padx=5)

        # button for saving results
        btn_ss = ttk.Button(frame_btn, text='Save .png', command=self.save_png)
        btn_ss.grid(row=0, column=2, padx=5)



        # draw radar chart
        
        self.center_x = config.RADAR_WIDTH//2
        self.center_y = config.RADAR_HEIGHT//2
        #COLORS = [[0,255,0], [0,0,255], [0,255,255], [255,0,255], [255,255,0], [0,0,0]] # currently not used

        self.img = np.zeros((config.RADAR_HEIGHT, config.RADAR_WIDTH, 3), dtype=np.uint8)
        self.img.fill(255) # fill with white color

        # draw circles for reference of travel distance
        # cv2.circle(self.img, (self.center_x,self.center_y), 272//config.SHRINK_SCALE, (127,127,127), thickness=1) # results in a circle of 272 pixel radius
        # cv2.circle(self.img, (self.center_x,self.center_y), 672//config.SHRINK_SCALE, (127,127,127), thickness=1)
        # cv2.circle(self.img, (self.center_x,self.center_y), 1072//config.SHRINK_SCALE, (127,127,127), thickness=1)
        # cv2.circle(self.img, (self.center_x,self.center_y), 1472//config.SHRINK_SCALE, (127,127,127), thickness=1) # results in a circle of 1472 pixel radius. 1472 is almost eaqual to the value of sqrt(1280*1280+720*720)
        cv2.circle(self.img, (self.center_x,self.center_y), 200, (127,127,127), thickness=1)
        cv2.circle(self.img, (self.center_x,self.center_y), 400, (127,127,127), thickness=1)
        cv2.circle(self.img, (self.center_x,self.center_y), 600, (127,127,127), thickness=1)
        cv2.circle(self.img, (self.center_x,self.center_y), 800, (127,127,127), thickness=1)
        cv2.circle(self.img, (self.center_x,self.center_y), 1000, (127,127,127), thickness=1)
        cv2.circle(self.img, (self.center_x,self.center_y), 1200, (127,127,127), thickness=1)
        cv2.circle(self.img, (self.center_x,self.center_y), 1400, (127,127,127), thickness=1)



    def print_redirector(self, input_str):
        self.st.configure(state=tk.NORMAL)
        self.st.insert(tk.END, input_str)
        self.st.configure(state=tk.DISABLED)



    def redirect_print_output(self):
        if config.ENABLE_STDOUT_REDIRECTOR:
            sys.stdout.write = self.print_redirector
            #sys.stderr.write = self.print_redirector



    # overwrite update_global_var()
    def update_global_var(self):
        config_by_gui.LIST_TARGET = []
        for i in self.lb.curselection():
            config_by_gui.LIST_TARGET.append( self.lb.get(i) )

        self.start_ms = self.start_sec_iv.get() * 1000 # convert second to millisecond
        self.end_ms = self.end_sec_iv.get() * 1000

        list_focused_cluster_id_txt = self.list_focused_cluster_id_sv.get()
        if len(list_focused_cluster_id_txt):
            self.list_focused_cluster_id = [ int(item) for item in self.list_focused_cluster_id_sv.get().split(',') ]


        # check the adequacy of configuration variables
        if len(config_by_gui.DIR_PATH) == 0:
            messagebox.showwarning("Warning", "Folder to search is not designated.")
            return 0

        if len(config_by_gui.LIST_TARGET) == 0:
            messagebox.showwarning("Warning", "No .pkl file is selected.")
            return 0
        elif len(config_by_gui.LIST_TARGET) != 1:
            messagebox.showwarning("Warning", "Only one .pkl file needs to be selected at a time.")
            return 0

        if self.start_ms < 0:
            messagebox.showwarning("Warning", "Start second needs to be 0 or more.")
            return 

        if self.start_ms >= self.end_ms:
            messagebox.showwarning("Warning", "Start second needs to be smaller than End second.")
            return 0

        return 1



    # overwrite set_parent_directory()
    def set_parent_directory(self, dialog):
        super().set_parent_directory(dialog)

        # update listbox
        pkl_path = self.dir_path_sv.get() + "/*s.pkl"
        pkl_path = pkl_path.replace('\\', '/')
        list_pkl_path = sorted( glob.glob(pkl_path) )
        list_all_target_path = list_pkl_path

        len_dir_path_sv = len(self.dir_path_sv.get()) + 1
        list_all_target = [ item[len_dir_path_sv:] for item in list_all_target_path ] # store basename only
        self.lb_sv.set(list_all_target)

        return 1 # required for the callback function to be called more than once



    def load_pkl(self):
        if self.update_global_var():
            f_pkl_path = config_by_gui.DIR_PATH + "/" + config_by_gui.LIST_TARGET[0]
            with open(f_pkl_path, "rb") as f_pkl_nm:
                self.clusters = pickle.load(f_pkl_nm)
            print(f"\nReading {f_pkl_path} finished.")

            # detect time range
            self.start_ms = self.clusters[0].first_track_t
            self.end_ms = self.start_ms
            for cluster in self.clusters:
                if self.end_ms < cluster.last_track_t:
                    self.end_ms = cluster.last_track_t

            self.start_sec_iv.set( self.start_ms // 1000 )
            self.end_sec_iv.set( (self.end_ms+999) // 1000 )
        else:
            messagebox.showwarning("Warning", "Fail to load a .pkl file.")



    def draw_radar(self):
        if not self.update_global_var():
            return

        f_output_txt_var_velocity = open(config_by_gui.DIR_PATH + "/var_velocity.csv", 'a')

        # plot the trajectories of all the clusters which, at least partially, existed from "start_ms" to "end_ms"
        for cluster in self.clusters:
            if self.start_ms <= cluster.last_track_t:
                if self.end_ms < cluster.first_track_t:
                    break

                if cluster.var_velocity != 0.:
                    # draw trajectory of the centroid
                    if len(cluster.track_g_all_history) >= 2:
                        # when "track_g_all_history" is long enough

                        f_output_txt_var_velocity.write( f"{cluster.ID},{cluster.var_velocity}\n" ) # save "var_velocity" to .txt

                        # extract centroid x and y histories separately
                        center_x_history = np.array( [ row[0] for row in cluster.track_g_all_history ] )
                        center_y_history = np.array( [ row[1] for row in cluster.track_g_all_history ] )

                        # subtract the initial coordinates from "center_*_history" to make "center_*_history" start at the coordinate of 0
                        center_x_history -= center_x_history[0]
                        center_y_history -= center_y_history[0]

                        # add offset for drawing
                        center_x_history += self.center_x
                        center_y_history += self.center_y

                        for i in range( len(center_x_history)-1 ):
                            cv2.line( self.img, ( int(center_x_history[i]), int(center_y_history[i]) ), ( int(center_x_history[i+1]), int(center_y_history[i+1]) ), [255,0,0], thickness=1 )

        f_output_txt_var_velocity.close()


        # plot the trajectories of the focused clusters with a different color
        color_idx = -1
        for cluster in self.clusters:
            if self.start_ms <= cluster.last_track_t:
                if self.end_ms < cluster.first_track_t:
                    break

                # overwrite old plot result above
                if cluster.ID in self.list_focused_cluster_id:
                    color_idx += 1

                    # draw trajectory of the centroid
                    if len(cluster.track_g_all_history) >= 2:
                        # when "track_g_all_history" is long enough

                        # extract centroid x and y histories separately
                        center_x_history = np.array( [ row[0] for row in cluster.track_g_all_history ] )
                        center_y_history = np.array( [ row[1] for row in cluster.track_g_all_history ] )

                        # subtract the initial coordinates from "center_*_history" to make "center_*_history" start at the coordinate of 0
                        center_x_history -= center_x_history[0]
                        center_y_history -= center_y_history[0]

                        # add offset for drawing
                        center_x_history += self.center_x
                        center_y_history += self.center_y

                        for i in range( len(center_x_history)-1 ):
                            #cv2.line( self.img, ( int(center_x_history[i]), int(center_y_history[i]) ), ( int(center_x_history[i+1]), int(center_y_history[i+1]) ), COLORS[ color_idx ], thickness=1 )
                            cv2.line( self.img, ( int(center_x_history[i]), int(center_y_history[i]) ), ( int(center_x_history[i+1]), int(center_y_history[i+1]) ), [0,0,255], thickness=1 )

        cv2.imshow( 'img', cv2.resize( self.img, ( config.RADAR_WIDTH//config.SHRINK_SCALE, config.RADAR_HEIGHT//config.SHRINK_SCALE) ) )



    def save_png(self):
        cv2.imwrite(config_by_gui.DIR_PATH + "/radar.png", self.img)
        print(f"\n.png file is saved to {config_by_gui.DIR_PATH}")

        f_output_txt_var_velocity = open(config_by_gui.DIR_PATH + "/var_velocity.csv", 'a')
        f_output_txt_var_velocity.write( "\n\n" ) # append two blank lines to "var_velocity.cs"
        f_output_txt_var_velocity.close()
        print(f"\nUpdatig var_velocity.csv completed.\n")



if __name__ == "__main__":
    root = tk.Tk()
    root.title("cluster2radarChart")
    Main_window(root).pack(side=tk.TOP, fill="both", expand=True)
    root.mainloop()