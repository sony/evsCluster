# coding: utf-8

# Copyright 2023 Sony Group Corporation - All Rights Reserved.
# Subject to the terms and conditions contained in LICENSE.txt accompanying this file, you may use this file.

import os
import sys
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
import glob
import shutil
from lib import subcluster
import config_by_gui
import config
from lib import common
from lib import gui_utils



class Main_window(gui_utils.Common_window):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent)

        self.enable_monitoring_iv.set(1)
        self.neighbour_th_iv.set(1)



        # frame for parent directory
        frame_dir = ttk.Frame(parent, padding=2)
        frame_dir.pack(anchor=tk.NW, expand=True, fill=tk.X)

        # label for parent directory
        lbl_dir = ttk.Label(frame_dir, text="Folder to search") # padding=(5, 2)
        lbl_dir.pack(side=tk.LEFT)

        # textbox for parent directory
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
        lbl_lb = tk.Label(frame_lb, text="Select .raw files\nor Index* folders\ncontaining .bin files")
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


        # subsubframe for progress phase
        frame_phase = ttk.Frame(frame_st)
        frame_phase.pack()

        # label for "Current target"
        lbl_ct = tk.Label(frame_phase, text="Current target:  ", padx=1)
        lbl_ct.pack(side=tk.LEFT)

        # textbox for "Current target"
        self.cur_target_sv = tk.StringVar()
        tb_cur_target = ttk.Entry( frame_phase, textvariable=self.cur_target_sv, width=15 )
        tb_cur_target.pack(side=tk.LEFT)
        #tb_cur_target.configure(state=tk.DISABLED)



        # frame for "LIST_BORDER_TIME"
        frame_lbt = ttk.Frame(root, padding=2)
        frame_lbt.pack(expand=True, fill=tk.X)

        # label for "LIST_BORDER_TIME"
        lbl_lbt = tk.Label(frame_lbt, text="Border time list")
        lbl_lbt.pack(side=tk.LEFT)

        # textbox for "LIST_BORDER_TIME"
        tb_lbt = tk.Entry(frame_lbt, textvariable=self.list_border_time_sv) #, width=40
        tb_lbt.pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.list_border_time_sv.set( ','.join( str(item) for item in config_by_gui.LIST_BORDER_TIME ) )

        # radiobuttons for "LIST_BORDER_TIME"
        rm=tk.Radiobutton(frame_lbt, text="[min]", variable=self.min_or_sec_iv, value=1)
        rm.pack(side=tk.LEFT)
        rs=tk.Radiobutton(frame_lbt, text="[sec]", variable=self.min_or_sec_iv, value=0)
        rs.pack(side=tk.RIGHT)
        if config_by_gui.MIN_OR_SEC == "min":
            self.min_or_sec_iv.set(1)
        else:
            self.min_or_sec_iv.set(0)



        # frame for buttons
        frame_btn = ttk.Frame(root, padding=2)
        frame_btn.pack()

        # button for starting process
        btn_run = ttk.Button(frame_btn, text='Run', command=lambda: self.run_process(lb=self.lb, phase=1))
        btn_run.grid(row=0, column=0, padx=5)

        # button for saving settings
        btn_ss = ttk.Button(frame_btn, text='Save settings', command=lambda: self.overwrite_config_by_gui_py(self.lb))
        btn_ss.grid(row=0, column=1, padx=5)



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
        avi_path = config_by_gui.DIR_PATH + "/avi/"
        os.makedirs(avi_path, exist_ok=True)

        for target_nm in config_by_gui.LIST_TARGET:
            self.cur_target_sv.set(target_nm) # display the current target name
            result = subcluster.main(target_nm, flg_evs2video=1)
            if result:
                break # when 'q' is pressed

            # arrange .avi file to /avi/ folder
            list_cur_video_path = glob.glob(common.analysis_path + "*_sub.avi")
            for cur_video_path in list_cur_video_path:
                target_basename = os.path.basename(cur_video_path)
                th_idx = target_basename.rfind("_th") # omit "_th" from .avi file name
                new_video_path = avi_path + target_basename[0:th_idx] + target_basename[th_idx+4:-8] + ".avi"

                os.replace(cur_video_path, new_video_path)
            
            try:
                shutil.rmtree(common.analysis_path)
            except Exception as e:
                print(e)

        print("\n\n")



if __name__ == "__main__":
    root = tk.Tk()
    root.title("evs2video")
    Main_window(root).pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    root.mainloop()
