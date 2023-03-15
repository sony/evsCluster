# coding: utf-8

# Copyright 2023 Sony Group Corporation - All Rights Reserved.
# Subject to the terms and conditions contained in LICENSE.txt accompanying this file, you may use this file.

import sys
import datetime
from lib import utils


def main(target_nm):
    utils.set_analysis_path(target_nm, flg_subcluster=0)
    result = utils.process_data(flg_subcluster=0, flg_evs2video=0)
    utils.close_files(flg_subcluster=0)
    print(f"\nCreating .avi file finished at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    return result


if __name__ == "__main__":
    try:
        main(sys.argv[1])
    except IndexError:
        print(f"Usage: {sys.argv[0]} target_file_name")
