# directory_path: the path to the directory where Index* subdirectories or .raw files are stored.
#DIR_PATH = r"C:\Users\0000140048\Videos\rec"
DIR_PATH = r"D:\NIES\isolated"

# target_list: the list of Index* subdirectories or .raw files to be processed
LIST_TARGET = ["recording_2023-04-11_19-07-06.raw"]

# enable_monitoring:0=disable monitoring during subcluster process and its avi output. 1=enable
ENABLE_MONITORING = 0

# neighbour_threshold: Default=1. Larger value results in shorter process time. If EVS data is too noisy, increase this value to filter out noise.
NEIGHBOUR_TH = 1

# minutes_or_seconds: the unit for "LIST_BORDER_TIME". Options are "min" or "sec"
MIN_OR_SEC = "sec"

# border_time_list: [capture_start_time, 1st_section_end, 2nd_section_end, 3rd_section_end, ,,, ,capture_end_time]. Unit is selected by "MIN_OR_SEC"
LIST_BORDER_TIME = [0, 1000000]