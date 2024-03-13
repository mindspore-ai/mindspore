#!/usr/bin/env python
# coding=utf-8
"""
Function:
The replay funtion entry
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2020
"""

import sys
import os
import stat
import re
import const_var


def gen_tiling(tiling_header_file: str, tiling_file_out: str):
    if not os.path.exists(tiling_header_file):
        print("warning: no userdef tiling header file: ", tiling_header_file)
        return
    print("generate tiling def header file: ", tiling_file_out)
    tmp_name = os.path.splitext(os.path.basename(tiling_header_file))[0].upper()
    tiling_source = '#ifndef __{}_H__\n'.format(tmp_name)
    tiling_source += '#define __{}_H__\n\n'.format(tmp_name)
    tiling_source += '#include <cstdint>\n'
    tiling_source += '#include <cstring>\n\n'
    tiling_source += '#include "kernel_tiling/kernel_tiling.h"\n\n'
    end_source = ""
    pattern = re.compile(r'[(](.*)[)]', re.S)
    with open(tiling_header_file, 'r') as fd:
        lines = fd.readlines()
        for line in lines:
            line = line.strip()
            if (line.startswith('BEGIN_TILING_DATA_DEF')):
                tiling_source += '#pragma pack(1)\n'
                tiling_source += 'struct '
                struct_def  = re.findall(pattern, line)[0]
                tiling_source += struct_def + ' {\n'
            elif (line.startswith('TILING_DATA_FIELD_DEF_ARR')):
                field_params = re.findall(pattern, line)[0]
                fds = field_params.split(',')
                tiling_source += '    {} {}[{}] = {{}};\n'.format(fds[0].strip(), fds[2].strip(), fds[1].strip())
            elif (line.startswith('TILING_DATA_FIELD_DEF_STRUCT')):
                field_params = re.findall(pattern, line)[0]
                fds = field_params.split(',')
                tiling_source += '    {} {};\n'.format(fds[0].strip(), fds[1].strip())
            elif (line.startswith('TILING_DATA_FIELD_DEF')):
                field_params = re.findall(pattern, line)[0]
                fds = field_params.split(',')
                tiling_source += '    {} {} = 0;\n'.format(fds[0].strip(), fds[1].strip())
            elif (line.startswith('END_TILING_DATA_DEF')):
                tiling_source += '};\n'
                tiling_source += '#pragma pack()\n\n'
                tiling_source += '#ifdef __NPU_TILING__\n'
                tiling_source += \
                    'inline [aicore] void Init{stru}(const __gm__ uint8_t* tiling, {stru}* const_data)\n'\
                        .format(stru=struct_def)
                tiling_source += '{\n'
                tiling_source += '    const __gm__ uint32_t *src = (const __gm__ uint32_t *)tiling;\n'
                tiling_source += '    uint32_t *dst = (uint32_t *)const_data;\n'
                tiling_source += '    for (auto i = 0; i < sizeof({}) / 4; i++) *(dst + i) = *(src + i);\n'\
                    .format(struct_def)
                tiling_source += '}\n'
                tiling_source += '#else\n'
                tiling_source += 'inline void Init{stru}(uint8_t* tiling, {stru}* const_data)\n'.format(stru=struct_def)
                tiling_source += '{\n'
                tiling_source += '    uint64_t *src = (uint64_t *)tiling;\n'
                tiling_source += '    uint64_t *dst = (uint64_t *)const_data;\n'
                tiling_source += '    for (auto i = 0; i < sizeof({}) / 8; i++) *(dst + i) = *(src + i);\n'\
                    .format(struct_def)
                tiling_source += '}\n'
                tiling_source += '#endif\n\n'
                end_source = '''
#define GET_TILING_DATA(tiling_data, tiling_arg) \\
{stru} tiling_data; \\
Init{stru}(tiling_arg, &tiling_data)\n
'''.format(stru=struct_def)
    tiling_source += end_source
    tiling_source += '#endif'
    with os.fdopen(os.open(tiling_file_out, const_var.WFLAGS, const_var.WMODES), 'w') as ofd:
        ofd.write(tiling_source)


if __name__ == '__main__':
    if len(sys.argv) <= 2:
        raise RuntimeError('arguments must greater than 2')
    gen_tiling(sys.argv[1], sys.argv[2])
