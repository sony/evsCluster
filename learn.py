# coding: utf-8

# Copyright 2023 Sony Group Corporation - All Rights Reserved.
# Subject to the terms and conditions contained in LICENSE.txt accompanying this file, you may use this file.

import sys
import numpy as np
import copy
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import scrolledtext
from tkinter import messagebox
import glob
import threading
import datetime
import config_by_gui
import config
from lib import gui_utils



class Main_window(gui_utils.Common_window):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent)

        self.th_run = None
        self.list_f_teacher = []
        self.list_f_test = []



        # frame for teacher data directory
        frame_teacher = ttk.Frame(root, padding=2)
        frame_teacher.pack(fill=tk.X)

        # label for teacher data directory
        lbl_teacher = ttk.Label(frame_teacher, text="Folder with teacher data")
        lbl_teacher.pack(side=tk.LEFT)


        # frame for test data directory
        frame_test = ttk.Frame(root, padding=2)
        frame_test.pack(fill=tk.X)

        # label for test data directory
        lbl_test = ttk.Label(frame_test, text="Folder with test data")
        lbl_test.pack(side=tk.LEFT)


        # frame for listboxes and scrolledtext
        frame_lb_st = ttk.Frame(root, padding=2)
        frame_lb_st.pack(anchor=tk.NW, expand=True, fill=tk.BOTH)


        # subframe for teacher data listbox
        frame_lb1 = ttk.Frame(frame_lb_st, padding=0)
        frame_lb1.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        # label for teacher data listbox
        lbl_lb1 = tk.Label(frame_lb1, text="Select teacher data")
        lbl_lb1.pack()


        # subsubframe for teacher data listbox
        subframe_lb1 = ttk.Frame(frame_lb1, padding=0)
        subframe_lb1.pack(expand=True, fill=tk.BOTH)

        # listbox
        lb1_sv = tk.StringVar()
        self.lb1 = tk.Listbox(subframe_lb1, selectmode="extended", listvariable=lb1_sv, exportselection=0) # , height=8
        self.lb1.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        # textbox for teacher data directory
        self.teacher_path_sv = tk.StringVar()
        tb_teacher = ttk.Entry( frame_teacher, textvariable=self.teacher_path_sv, validate="focusout", validatecommand=lambda: self.set_parent_directory(self.teacher_path_sv, lb1_sv, dialog=0) )
        tb_teacher.pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.teacher_path_sv.set( config_by_gui.DIR_PATH.replace('/', '\\') )
        self.set_parent_directory(self.teacher_path_sv, lb1_sv, dialog=0)

        # button for teacher data directory
        btn_teacher = ttk.Button( frame_teacher, text="Select", command=lambda: self.set_parent_directory(self.teacher_path_sv, lb1_sv, dialog=1) )
        btn_teacher.pack(side=tk.RIGHT)

        # vertical scrollbar
        scrollbar1_v = ttk.Scrollbar(subframe_lb1, orient=tk.VERTICAL, command=self.lb1.yview)
        self.lb1['yscrollcommand'] = scrollbar1_v.set
        scrollbar1_v.pack(side=tk.RIGHT, fill=tk.Y)

        # horizontal scrollbar
        scrollbar1_h = ttk.Scrollbar(frame_lb1, orient=tk.HORIZONTAL, command=self.lb1.xview)
        self.lb1['xscrollcommand'] = scrollbar1_h.set
        scrollbar1_h.pack(side=tk.BOTTOM, fill=tk.X)


        # subframe for test data listbox
        frame_lb2 = ttk.Frame(frame_lb_st, padding=0)
        frame_lb2.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        # label for test data listbox
        lbl_lb2 = tk.Label(frame_lb2, text="  Select test data ")
        lbl_lb2.pack()


        # subsubframe for test data listbox
        subframe_lb2 = ttk.Frame(frame_lb2, padding=0)
        subframe_lb2.pack(expand=True, fill=tk.BOTH)

        # listbox
        lb2_sv = tk.StringVar()
        self.lb2 = tk.Listbox(subframe_lb2, selectmode="extended", listvariable=lb2_sv, exportselection=0) # , height=8
        self.lb2.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        # textbox for test data directory
        self.test_path_sv = tk.StringVar()
        tb_test = ttk.Entry( frame_test, textvariable=self.test_path_sv, validate="focusout", validatecommand=lambda: self.set_parent_directory(self.test_path_sv, lb2_sv, dialog=0) )
        tb_test.pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.test_path_sv.set( config_by_gui.DIR_PATH.replace('/', '\\') )
        self.set_parent_directory(self.test_path_sv, lb2_sv, dialog=0)

        # button for test data directory
        btn_test = ttk.Button( frame_test, text="Select", command=lambda: self.set_parent_directory(self.test_path_sv, lb2_sv, dialog=1) )
        btn_test.pack(side=tk.RIGHT)

        # vertical scrollbar
        scrollbar2_v = ttk.Scrollbar(subframe_lb2, orient=tk.VERTICAL, command=self.lb2.yview)
        self.lb2['yscrollcommand'] = scrollbar2_v.set
        scrollbar2_v.pack(side=tk.RIGHT, fill=tk.Y)

        # horizontal scrollbar
        scrollbar2_h = ttk.Scrollbar(frame_lb2, orient=tk.HORIZONTAL, command=self.lb2.xview)
        self.lb2['xscrollcommand'] = scrollbar2_h.set
        scrollbar2_h.pack(side=tk.BOTTOM, fill=tk.X)


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


        # frame for settings
        frame_set = ttk.Frame(root, padding=2)
        frame_set.pack()

        # label for "NUM_EPOCH"
        lbl_ne = tk.Label(frame_set, text="NUM_EPOCH")
        lbl_ne.grid(row=0, column=1)

        # textbox for "NUM_EPOCH"
        self.num_epoch_iv = tk.IntVar()
        tb_ne = tk.Entry(frame_set, width=8, textvariable=self.num_epoch_iv, justify=tk.RIGHT)
        tb_ne.grid(row=0, column=2)
        self.num_epoch_iv.set(config.NUM_EPOCH)


        # frame for buttons
        frame_btn = ttk.Frame(root, padding=2)
        frame_btn.pack()

        # button for starting process
        btn_run = ttk.Button(frame_btn, text='Run', command=self.run_process)
        btn_run.grid(row=0, column=0, padx=0)



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
        self.list_f_teacher = []
        for i in self.lb1.curselection():
            self.list_f_teacher.append( self.lb1.get(i) )

        self.list_f_test = []
        for i in self.lb2.curselection():
            self.list_f_test.append( self.lb2.get(i) )

        config.NUM_EPOCH = self.num_epoch_iv.get()

        # check the adequacy of configuration variables
        if len(self.teacher_path_sv.get()) == 0:
            messagebox.showwarning("Warning", "Folder with teacher data is not designated.")
            return 0

        if len(self.test_path_sv.get()) == 0:
            messagebox.showwarning("Warning", "Folder with test data is not designated.")
            return 0

        if len(self.list_f_teacher) == 0:
            messagebox.showwarning("Warning", "No .csv file for teacher data is selected.")
            return 0

        if len(self.list_f_test) == 0:
            messagebox.showwarning("Warning", "No .csv file for test data is selected.")
            return 0

        return 1



    # overwrite set_parent_directory()
    def set_parent_directory(self, dir_path_sv, lb_sv, dialog):
        # update "DIR_PATH"
        dir_given = dir_path_sv.get()
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
        dir_path_backslash = dir_returned.replace('/', '\\')
        dir_path_sv.set(dir_path_backslash)


        # update listbox
        csv_path = dir_path_sv.get() + "/*.csv"
        csv_path = csv_path.replace('\\', '/')
        list_csv_path = sorted( glob.glob(csv_path) )

        len_dir_path_sv = len(dir_path_sv.get()) + 1
        list_csv = [ item[len_dir_path_sv:] for item in list_csv_path ] # store basename only
        lb_sv.set(list_csv)

        return 1 # required for the callback function to be called more than once



    def read_csv(self, list_f, path_sv):
        list_input = []
        list_label = []
        
        for f in list_f:
            #with open( path_sv.get() + '/' + f, 'r', encoding="utf-8_sig" ) as f_input:
            with open( path_sv.get() + '/' + f, 'r' ) as f_input:
                print(f)
                line = f_input.readline() # skip the 1st line
                #line = f_input.readline() # skip the 2nd line (line of file path)

                while 1:
                    line = f_input.readline()
                    if len(line) == 0: # this means EOF
                        break

                    list_cluster_feature_txt = line.split(",") # convert a line of .csv file to a list of string
                    if len(list_cluster_feature_txt) < 2:
                        continue # skip a line of .pkl file path
                    if list_cluster_feature_txt[1] == "":
                        continue # skip a line of .pkl file path
                    list_cluster_feature = list( map( float, list_cluster_feature_txt[ config.COLUMN_IDX_FEATURE_OFFSET:(config.COLUMN_IDX_FEATURE_OFFSET+config.NUM_FEATURES) ] ) ) # convert to a list of float

                    label_txt = list_cluster_feature_txt[ config. COLUMN_IDX_ANNOTATION ] # read label string
                    if len(label_txt) >= 1:
                        while (label_txt[-1] == '\n') or (label_txt[-1] == '\r'): # omit '\n' or '\r' at the end
                            label_txt = label_txt[:-1]
                            if len(label_txt) == 0:
                                break
                    
                    #print(config.Plankton[label_txt].name, config.Plankton[label_txt].value)
                    label = [0.] * config.NUM_CLASSIFICATION
                    if (0 <= config.Plankton[label_txt].value) & (config.Plankton[label_txt].value <= 4):
                        # when "label_txt" is one of the valid strings (annotations)
                        label[ config.Plankton[label_txt].value - 0 ] = 1. # convert to a label vector (one-hot vector)
                        list_input.append(list_cluster_feature) # append to the end of feature vectors
                        list_label.append(label) # append to the end of labal vectors

        np_array_input = np.array( list_input )
        np_array_label = np.array( list_label )

        np_array_input /= config.NP_NORM_CEIL # normalize

        #np_array_input[ np_array_input > 1. ] = 1.
        np_array_input = np.where(np_array_input > 1., 1., np_array_input) # overwrite values larger than 1. with 1.
        np_array_input = np.where(np_array_input < 0., 0., np_array_input) # overwrite values smaller than 0. with 0.

        return np_array_input, np_array_label



    def machine_learning(self):
        print("Teacher data:")
        np_array_teacher_input, np_array_teacher_label = self.read_csv(self.list_f_teacher, self.teacher_path_sv)

        print("Test data:")
        np_array_test_input, np_array_test_label = self.read_csv(self.list_f_test, self.test_path_sv)

        # machine learning
        f_learning_curve_txt = open( self.test_path_sv.get() + "/learning_curve.csv", 'w' )

        highest_test_accuracy = 0.

        b_i_h = np.zeros((config.NUM_H_NORD, 1))
        b_h_o = np.zeros((config.NUM_CLASSIFICATION, 1))
        w_i_h = np.random.uniform(-0.5, 0.5, (config.NUM_H_NORD, config.NUM_FEATURES))
        w_h_o = np.random.uniform(-0.5, 0.5, (config.NUM_CLASSIFICATION, config.NUM_H_NORD))
        #np.savetxt("nn_param/w_i_h__seed.csv", w_i_h, delimiter=',')
        #np.savetxt("nn_param/w_h_o__seed.csv", w_h_o, delimiter=',')
        #w_i_h = np.loadtxt("nn_param/w_i_h__seed.csv", delimiter=',')
        #w_h_o = np.loadtxt("nn_param/w_h_o__seed.csv", delimiter=',')

        for epoch in range(config.NUM_EPOCH):
            # forward and back propagation with teacher data
            teacher_correct_cnter = 0 # initialize
            for teacher_input, label in zip(np_array_teacher_input, np_array_teacher_label):
                teacher_input.shape += (1,)
                label.shape += (1,)

                # input to hidden layer
                h_tmp = b_i_h + w_i_h @ teacher_input
                h = 1 / (1 + np.exp(-h_tmp))
                # hidden to output layer
                o_tmp = b_h_o + w_h_o @ h
                o = 1 / (1 + np.exp(-o_tmp))

                # count correct answer
                teacher_correct_cnter += int(np.argmax(o) == np.argmax(label))

                # calc delta
                delta_o = (o - label) * (o * (1 - o))
                delta_h = np.transpose(w_h_o) @ delta_o * (h * (1 - h))

                # from output back to hidden layer
                w_h_o -= config.LEARNING_RATE * delta_o @ np.transpose(h)
                b_h_o -= config.LEARNING_RATE * delta_o
                # from hidden back to input layer
                w_i_h -= config.LEARNING_RATE * delta_h @ np.transpose(teacher_input)
                b_i_h -= config.LEARNING_RATE * delta_h

            # validation with test data at this epoch
            test_correct_cnter = 0 # initialize
            for test_input, label in zip(np_array_test_input, np_array_test_label):
                test_input.shape += (1,)
                label.shape += (1,)

                # input to hidden layer
                h_tmp = b_i_h + w_i_h @ test_input
                h = 1 / (1 + np.exp(-h_tmp))
                # hidden to output layer
                o_tmp = b_h_o + w_h_o @ h
                o = 1 / (1 + np.exp(-o_tmp))

                # count correct answer
                test_correct_cnter += int(np.argmax(o) == np.argmax(label))

            # calc accuracy
            teacher_accuracy = (teacher_correct_cnter / np_array_teacher_input.shape[0]) * 100
            test_accuracy = (test_correct_cnter / np_array_test_input.shape[0]) * 100

            # preserve the best weights
            if test_accuracy >= highest_test_accuracy:
                highest_test_accuracy = test_accuracy
                w_i_h_best = copy.deepcopy(w_i_h)
                w_h_o_best = copy.deepcopy(w_h_o)
                b_i_h_best = copy.deepcopy(b_i_h)
                b_h_o_best = copy.deepcopy(b_h_o)

            # write to "learning_curve.csv"
            f_learning_curve_txt.write(f"{epoch+1},{teacher_accuracy},{test_accuracy}\n")

            if epoch%100 == 99:
                print(f"@{epoch+1} Teacher: {round(teacher_accuracy, 2)}% , Test: {round(test_accuracy, 2)}%")

        f_learning_curve_txt.close()
        print(f"\nHighest accuracy with test data: {round(highest_test_accuracy, 2)}%")


        # save the machine learning result
        np.savetxt(config.W_I_H__PATH, w_i_h_best, delimiter=',')
        np.savetxt(config.W_H_O__PATH, w_h_o_best, delimiter=',')
        np.savetxt(config.B_I_H__PATH, b_i_h_best, delimiter=',')
        np.savetxt(config.B_H_O__PATH, b_h_o_best, delimiter=',')

        print(f"Machine learning finished at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        print("\n\n")



    # overwrite run_process()
    def run_process(self):
        if self.th_run != None:
            if self.th_run.is_alive():
                #print("Already running!")
                messagebox.showwarning("Warning", "Process is currently running.")
                return

        if self.update_global_var():
            self.th_run = threading.Thread(target=self.machine_learning)
            self.th_run.start()
        else:
            messagebox.showwarning("Warning", "Fail to start processing.")



if __name__ == "__main__":
    root = tk.Tk()
    root.title("learn")
    Main_window(root).pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    root.mainloop()
