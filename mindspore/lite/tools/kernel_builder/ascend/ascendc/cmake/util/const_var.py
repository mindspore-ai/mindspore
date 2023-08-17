
#!/usr/bin/env python
# coding=utf-8
"""
Function:
The replay function entry
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2020
"""

import os
import stat


REPLAY_BATCH = 'batch'
REPLAY_ITERATE = 'iterate'
CFG_IMPL_DIR = 'impl_dir'
CFG_OUT_DIR = 'out_dir'
WFLAGS = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
WMODES = stat.S_IWUSR | stat.S_IRUSR
SOC_MAP_EXT = {'ascend310p': 'Ascend310P3', 'ascend310b': 'Ascend310B1',
               'ascend910': 'Ascend910A', 'ascend910b': 'Ascend910B1'}
BIN_CMD = 'opc $1 --main_func={fun} --input_param={param} --soc_version={soc} \
--output=$2 --impl_mode={impl} --simplified_key_mode=0 --op_mode=dynamic\n'
CHK_CMD = '''
if ! test -f $2/{res_file} ; then
  echo "$2/{res_file} not generated!"
  exit 1
fi
'''
ATTR_DEF_VAL = {'str': '', 'int': 0, 'float': 0.0, 'bool': False, 'list_bool': [],
                'list_int': [], 'list_float': [], 'list_list_int': [[]]}
