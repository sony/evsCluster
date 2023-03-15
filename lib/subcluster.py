# coding: utf-8

# Copyright 2023 Sony Group Corporation - All Rights Reserved.
# Subject to the terms and conditions contained in LICENSE.txt accompanying this file, you may use this file.

import sys
import datetime
from lib import utils


def main(target_nm, flg_evs2video):
    utils.set_analysis_path(target_nm, flg_subcluster=1)
    result = utils.process_data(flg_subcluster=1, flg_evs2video=flg_evs2video)
    utils.close_files(flg_subcluster=1)
    if not flg_evs2video:
        print(f"\nSubclustering finished at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    return result


if __name__ == "__main__":
    try:
        main(sys.argv[1], sys.argv[2])
    except IndexError:
        print(f"Usage: {sys.argv[0]} target_file_name boolean_evs2video")
