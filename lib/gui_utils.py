# coding: utf-8

# Copyright 2023 Sony Group Corporation - All Rights Reserved.
# Subject to the terms and conditions contained in LICENSE.txt accompanying this file, you may use this file.

import sys
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import threading
import config_by_gui
import config
from lib import common


class Common_window(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        # change the format of "DIR_PATH"
        config_by_gui.DIR_PATH = config_by_gui.DIR_PATH.replace('\\', '/')
        if config_by_gui.DIR_PATH[-1] == '/':
            config_by_gui.DIR_PATH = config_by_gui.DIR_PATH[:-1]

        self.th_run = None
        self.dir_path_sv = tk.StringVar()
        self.enable_monitoring_iv = tk.IntVar()
        self.neighbour_th_iv = tk.IntVar()
        self.min_or_sec_iv=tk.IntVar()
        self.list_border_time_sv = tk.StringVar()



    def set_parent_directory(self, dialog):
        # update "DIR_PATH"
        dir_given = self.dir_path_sv.get()
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
        config_by_gui.DIR_PATH = dir_returned
        dir_path_backslash = config_by_gui.DIR_PATH.replace('/', '\\')
        self.dir_path_sv.set(dir_path_backslash)



    def update_global_var(self, lb):
        config_by_gui.LIST_TARGET = []
        for i in lb.curselection():
            config_by_gui.LIST_TARGET.append( lb.get(i) )

        config_by_gui.ENABLE_MONITORING = self.enable_monitoring_iv.get()

        config_by_gui.NEIGHBOUR_TH = self.neighbour_th_iv.get() # floating value automatically turns into integer

        if self.min_or_sec_iv.get():
            config_by_gui.MIN_OR_SEC = "min"
        else:
            config_by_gui.MIN_OR_SEC = "sec"

        list_border_time_txt = self.list_border_time_sv.get()
        if config_by_gui.MIN_OR_SEC == "min":
            config_by_gui.LIST_BORDER_TIME = [ int(item)*60 for item in list_border_time_txt.split(',') ]
        else:
            config_by_gui.LIST_BORDER_TIME = [ int(item) for item in list_border_time_txt.split(',') ]


        # check the adequacy of configuration variables
        if len(config_by_gui.DIR_PATH) == 0:
            messagebox.showwarning("Warning", "Folder to search is not designated.")
            return 0

        if len(config_by_gui.LIST_TARGET) == 0:
            messagebox.showwarning("Warning", "No folder or .raw file is selected.")
            return 0

        if (config_by_gui.NEIGHBOUR_TH <= 0):
            messagebox.showwarning("Warning", "NEIGHBOUR_TH needs to be a positive integer.")
            return 0

        len_list_border_time = len(config_by_gui.LIST_BORDER_TIME)
        if len_list_border_time < 2:
            messagebox.showwarning("Warning", "Border time list needs to have 2 elements or more.")
            return 0

        if config_by_gui.LIST_BORDER_TIME[0] < 0:
            messagebox.showwarning("Warning", "Border time list needs to start from 0 or a positive ineger.")
            return 0

        i = 1
        while i < len_list_border_time:
            if( config_by_gui.LIST_BORDER_TIME[i-1] >= config_by_gui.LIST_BORDER_TIME[i] ):
                messagebox.showwarning("Warning", "Border time list needs to be in an ascending order.")
                return 0
            i += 1

        return 1



    def thread_run(self): # dummy
        pass



    def run_process(self, lb, phase):
        if self.th_run != None:
            if self.th_run.is_alive():
                messagebox.showwarning("Warning", "Process is currently running.")
                return

        if self.update_global_var(lb):
            print("The selected targets are the following:")
            for target in config_by_gui.LIST_TARGET:
                print(target)
            print("")

            common.selected_phase = phase
            self.th_run = threading.Thread(target=self.thread_run)
            self.th_run.start()
        else:
            messagebox.showwarning("Warning", "Fail to start processing.")



    def overwrite_config_by_gui_py(self, lb):
        if self.update_global_var(lb):
            f_config_by_gui = open("config_by_gui.py", 'w')

            f_config_by_gui.write("# directory_path: the path to the directory where Index* subdirectories or .raw files are stored.\n")
            dir_path_txt = config_by_gui.DIR_PATH.replace('/', '\\')
            f_config_by_gui.write(f"DIR_PATH = r\"{ dir_path_txt }\"\n\n")

            f_config_by_gui.write("# target_list: the list of Index* subdirectories or .raw files to be processed\n")
            f_config_by_gui.write(f"LIST_TARGET = [\"{ config_by_gui.LIST_TARGET[0] }\"")
            for target in config_by_gui.LIST_TARGET[1:]:
                f_config_by_gui.write(f", \"{ target }\"")
            f_config_by_gui.write("]\n\n")

            f_config_by_gui.write("# enable_monitoring:0=disable monitoring during subcluster process and its avi output. 1=enable\n")
            f_config_by_gui.write(f"ENABLE_MONITORING = { config_by_gui.ENABLE_MONITORING }\n\n")

            f_config_by_gui.write("# neighbour_threshold: Default=1. Larger value results in shorter process time. If EVS data is too noisy, increase this value to filter out noise.\n")
            f_config_by_gui.write(f"NEIGHBOUR_TH = { config_by_gui.NEIGHBOUR_TH }\n\n")

            f_config_by_gui.write("# minutes_or_seconds: the unit for \"LIST_BORDER_TIME\". Options are \"min\" or \"sec\"\n")
            f_config_by_gui.write(f"MIN_OR_SEC = \"{ config_by_gui.MIN_OR_SEC }\"\n\n")

            if config_by_gui.MIN_OR_SEC == "min":
                list_border_time_to_save = [ item//60 for item in config_by_gui.LIST_BORDER_TIME ]
            else:
                list_border_time_to_save = config_by_gui.LIST_BORDER_TIME
            f_config_by_gui.write("# border_time_list: [capture_start_time, 1st_section_end, 2nd_section_end, 3rd_section_end, ,,, ,capture_end_time]. Unit is selected by \"MIN_OR_SEC\"\n")
            f_config_by_gui.write(f"LIST_BORDER_TIME = [{ list_border_time_to_save[0] }")
            for border in list_border_time_to_save[1:]:
                f_config_by_gui.write(f", { border }")
            f_config_by_gui.write("]")

            f_config_by_gui.close()

            messagebox.showinfo("Message", "Successfully overwrote config_by_gui.py.")
        else:
            messagebox.showwarning("Warning", "Fail to overwrite config_by_gui.py.")



