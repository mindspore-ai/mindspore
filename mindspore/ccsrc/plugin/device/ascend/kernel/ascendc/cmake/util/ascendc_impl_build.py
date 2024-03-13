#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Created on Feb  28 20:56:45 2020
Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
"""

import sys
import os
import re
import stat
import opdesc_parser
import const_var

PYF_PATH = os.path.dirname(os.path.realpath(__file__))

IMPL_HEAD = '''
import os, sys
import ctypes
import json
import shutil
from tbe.common.platform import get_soc_spec
from tbe.common.utils import para_check
from tbe.tikcpp import compile_op, replay_op, check_op_cap, generalize_op_params, get_code_channel, OpInfo
from tbe.common.buildcfg import get_default_build_config
from impl.util.platform_adapter import tbe_register
from tbe.common.buildcfg import get_current_build_config
PYF_PATH = os.path.dirname(os.path.realpath(__file__))

DTYPE_MAP = {"float32": ["DT_FLOAT", "float"],
    "float16": ["DT_FLOAT16", "half"],
    "int8": ["DT_INT8", "int8_t"],
    "int16": ["DT_INT16", "int16_t"],
    "int32": ["DT_INT32", "int32_t"],
    "int64": ["DT_INT64", "int64_t"],
    "uint1": ["DT_UINT1", "uint8_t"],
    "uint8": ["DT_UINT8", "uint8_t"],
    "uint16": ["DT_UINT16", "uint16_t"],
    "uint32": ["DT_UINT32", "uint32_t"],
    "uint64": ["DT_UINT64", "uint64_t"],
    "bool": ["DT_BOOL", "bool"],
    "double": ["DT_DOUBLE", "double"],
    "dual": ["DT_DUAL", "unknown"],
    "dual_sub_int8": ["DT_DUAL_SUB_INT8", "unknown"],
    "dual_sub_uint8": ["DT_DUAL_SUB_UINT8", "unknown"],
    "string": ["DT_STRING", "unknown"],
    "complex64": ["DT_COMPLEX64", "unknown"],
    "complex128": ["DT_COMPLEX128", "unknown"],
    "qint8": ["DT_QINT8", "unknown"],
    "qint16": ["DT_QINT16", "unknown"],
    "qint32": ["DT_QINT32", "unknown"],
    "quint8": ["DT_QUINT8", "unknown"],
    "quint16": ["DT_QUINT16", "unknown"],
    "resource": ["DT_RESOURCE", "unknown"],
    "string_ref": ["DT_STRING_REF", "unknown"],
    "int4": ["DT_INT4", "int4b_t"],
    "bfloat16": ["DT_BF16", "bfloat16_t"]}

def add_dtype_fmt_option_single(x, x_n, is_ref: bool = False):
    options = []
    x_fmt = x.get("format")
    x_dtype = x.get("dtype")
    x_n_in_kernel = x_n + '_REF' if is_ref else x_n
    options.append("-DDTYPE_{n}={t}".format(n=x_n_in_kernel, t=DTYPE_MAP.get(x_dtype)[1]))
    options.append("-DORIG_DTYPE_{n}={ot}".format(n=x_n_in_kernel, ot=DTYPE_MAP.get(x_dtype)[0]))
    options.append("-DFORMAT_{n}=FORMAT_{f}".format(n=x_n_in_kernel, f=x_fmt))
    return options
 
def get_dtype_fmt_options(__inputs__, __outputs__):
    options = []
    unique_param_name_set = set()
    for x in __inputs__:
        if x is None:
            continue
        x_n = x.get("param_name")[:-5].upper()
        unique_param_name_set.add(x_n)
        options += add_dtype_fmt_option_single(x, x_n)
 
    for x in __outputs__:
        if x is None:
            continue
        x_n = x.get("param_name")[:-5].upper()
        if x_n in unique_param_name_set:
            options += add_dtype_fmt_option_single(x, x_n, True)
        else:
            options += add_dtype_fmt_option_single(x, x_n)
    return options

def load_dso(so_path):
    try:
        ctypes.CDLL(so_path)
    except OSError as error :
        print(error)
        raise RuntimeError("cannot open %s" %(so_path))
    else:
        print("load so succ ", so_path)

def get_shortsoc_compile_option(compile_option_list: list, shortsoc:str):
    compile_options = []
    if shortsoc in compile_option_list:
        compile_options = compile_option_list[shortsoc]
    elif '__ALLSOC__' in compile_option_list:
        compile_options = compile_option_list['__ALLSOC__']
    return compile_options

def get_kernel_source(src_file, dir_snake, dir_ex):
    src_ex = os.path.join(PYF_PATH, "..", "ascendc", dir_ex, src_file)
    if os.path.exists(src_ex):
        return src_ex
    src = os.path.join(PYF_PATH, "..", "ascendc", dir_snake, src_file)
    if os.path.exists(src):
        return src
    src = os.path.join(PYF_PATH, src_file)
    if os.path.exists(src):
        return src
    return src_ex

'''

IMPL_API = '''
@tbe_register.register_operator("{}", trans_bool_to_s8=False)
@para_check.check_op_params({})
def {}({}, kernel_name="{}", impl_mode=""):
    if get_current_build_config("enable_op_prebuild"):
        return
    __inputs__, __outputs__, __attrs__ = _build_args({})
    options = get_dtype_fmt_options(__inputs__, __outputs__)
    options += ["-x", "cce"]
    bisheng = shutil.which("bisheng")
    if bisheng != None:
        bisheng_path = os.path.dirname(bisheng)
        tikcpp_path = os.path.realpath(os.path.join(bisheng_path, "..", "..", "tikcpp"))
    else:
        tikcpp_path = os.path.realpath("/usr/local/Ascend/latest/compiler/tikcpp")
    options.append("-I" + tikcpp_path)
    options.append("-I" + os.path.join(tikcpp_path, "tikcfw"))
    options.append("-I" + os.path.join(tikcpp_path, "tikcfw", "impl"))
    options.append("-I" + os.path.join(tikcpp_path, "tikcfw", "interface"))
    options.append("-I" + os.path.join(PYF_PATH, "..", "ascendc", "common"))
    if impl_mode == "high_performance":
        options.append("-DHIGH_PERFORMANCE=1")
    elif impl_mode == "high_precision":
        options.append("-DHIGH_PRECISION=1")
    if get_default_build_config("enable_deterministic_mode") == 1:
        options.append("-DDETEMINISTIC_MODE=1")

    custom_compile_options = {},
    custom_all_compile_options = {},
    soc_version = get_soc_spec("SOC_VERSION")
    soc_short = get_soc_spec("SHORT_SOC_VERSION").lower()
    custom_compile_options_soc = get_shortsoc_compile_option(custom_compile_options[0], soc_short)
    custom_all_compile_options_soc = get_shortsoc_compile_option(custom_all_compile_options[0], soc_short)
    options += custom_all_compile_options_soc
    options += custom_compile_options_soc

    origin_func_name = "{}"
    ascendc_src_dir_ex = "{}"
    ascendc_src_dir = "{}"
    ascendc_src_file = "{}"
    src = get_kernel_source(ascendc_src_file, ascendc_src_dir, ascendc_src_dir_ex)
'''

REPLAY_OP_API = '''
    print("start replay Acend C Operator {}, kernel name is {}")
    tikreplay_codegen_path = tikcpp_path + "/tikreplaylib/lib"
    tikreplay_stub_path = tikcpp_path + "/tikreplaylib/lib/" + soc_version
    print("start load libtikreplaylib_codegen.so and libtikreplaylib_stub.so")
    codegen_so_path = tikreplay_codegen_path + "/libtikreplaylib_codegen.so"
    replaystub_so_path = tikreplay_stub_path + "/libtikreplaylib_stub.so"
    if PYF_PATH.endswith("dynamic"):
        op_replay_path = os.path.join(PYF_PATH, "..", "..", "op_replay")
    else:
        op_replay_path = os.path.join(PYF_PATH, "..", "op_replay")
    replayapi_so_path = os.path.join(op_replay_path, "libreplay_{}_" + soc_short + ".so")
    load_dso(codegen_so_path)
    load_dso(replaystub_so_path)
    load_dso(replayapi_so_path)
    op_type = "{}"
    entry_obj = os.path.join(op_replay_path, "{}_entry_" + soc_short + ".o")
    code_channel = get_code_channel(src, kernel_name, op_type, options)
    op_info = OpInfo(kernel_name = kernel_name, op_type = op_type, inputs = __inputs__, outputs = __outputs__,\\
        attrs = __attrs__, impl_mode = impl_mode)
    res, msg = replay_op(op_info, entry_obj, code_channel, src, options)
    if not res:
        print("call replay op failed for %s and get into call compile op" %(msg))
        compile_op(src, origin_func_name, op_info, options, code_channel, '{}')
'''

COMPILE_OP_API = '''
    print("start compile Ascend C operator {}. kernel name is " + kernel_name)
    op_type = "{}"
    code_channel = get_code_channel(src, kernel_name, op_type, options)
    op_info = OpInfo(kernel_name = kernel_name, op_type = op_type, inputs = __inputs__, outputs = __outputs__,\\
        attrs = __attrs__, impl_mode = impl_mode, origin_inputs=[{}], origin_outputs = [{}])
    compile_op(src, origin_func_name, op_info, options, code_channel, '{}')
'''

SUP_API = '''
def {}({}, impl_mode=""):
    __inputs__, __outputs__, __attrs__ = _build_args({})
    ret_str = check_op_cap("{}", "{}", __inputs__, __outputs__, __attrs__)
    ret_dict = json.loads(ret_str)
    err_code = ret_dict.get("ret_code")
    sup = "Unknown"
    reason = "Unknown reason"
    if err_code is not None:
        if err_code is 0:
            sup = "True"
            reason = ""
        elif err_code is 1:
            sup = "False"
            reason = ret_dict.get("reason")
        else:
            sup = "Unknown"
            reason = ret_dict.get("reason")
    return sup, reason
'''
CAP_API = '''
def {}({}, impl_mode=""):
    __inputs__, __outputs__, __attrs__ = _build_args({})
    result = check_op_cap("{}", "{}", __inputs__, __outputs__, __attrs__)
    return result.decode("utf-8")
'''
GLZ_API = '''
@tbe_register.register_param_generalization("{}")
def {}_generalization({}, generalize_config=None):
    __inputs__, __outputs__, __attrs__ = _build_args({})
    ret_str = generalize_op_params("{}", __inputs__, __outputs__, __attrs__, generalize_config)
    return [json.loads(ret_str)]
'''

ATTR_DEFAULT = {'bool': 'False', 'int': '0', 'float': '0.0', 'list_int': '[]',
                'list_float': '[]', 'list_bool': '[]', 'list_list_int': '[[]]', 'str': ''}


def _get_snake_str(s, i):
    if s[i - 1] != '_':
        if not s[i - 1].isupper():
            return "_"
        elif s[i - 1].isupper() and (i + 1) < len(s) and s[i + 1].islower():
            return "_"
        return ""
    return ""


def optype_snake(origin_str):
    temp_str = origin_str[0].lower() + origin_str[1:]
    new_str = re.sub(r'([A-Z])', r'_\1', temp_str).lower()
    return new_str


def optype_snake_ex(s):
    snake_case = ""
    for i, c in enumerate(s):
        if i == 0:
            snake_case += c.lower()
        elif c.isupper():
            snake_case += _get_snake_str(s, i)
            snake_case += c.lower()
        else:
            snake_case += c
    return snake_case


class AdpBuilder(opdesc_parser.OpDesc):
    def __init__(self: any, op_type: str):
        self.argsname = []
        self.argsdefv = []
        self.op_compile_option:str = '{}'
        super().__init__(op_type)


    def write_adapt(self: any, impl_path, path: str, op_compile_option_all:list = None):
        self._build_paradefault()
        if impl_path != "":
            src_file = os.path.join(impl_path, self.op_file + '.cpp')
            if not os.path.exists(src_file):
                return
        out_path = os.path.abspath(path)
        if self.dynamic_shape and not out_path.endswith('dynamic'):
            out_path = os.path.join(path, 'dynamic')
            os.makedirs(out_path, mode=0o700, exist_ok=True)
        adpfile = os.path.join(out_path, self.op_file + '.py')
        self._gen_op_compile_option(op_compile_option_all)
        with os.fdopen(os.open(adpfile, const_var.WFLAGS, const_var.WMODES), 'w') as fd:
            self._write_head(fd)
            self._write_argparse(fd)
            self._write_impl(fd)
            if self.op_chk_support:
                self._write_cap('check_supported', fd)
                self._write_cap('get_op_support_info', fd)
            if self.op_fmt_sel:
                self._write_cap('op_select_format', fd)
                self._write_cap('get_op_specific_info', fd)
            if self.op_range_limit == 'limited' or self.op_range_limit == 'dynamic':
                self._write_glz(fd)


    def _gen_op_compile_option(self:any, op_compile_option_all:list =None):
        if op_compile_option_all is not None:
            if self.op_type in op_compile_option_all:
                self.op_compile_option = op_compile_option_all[self.op_type]
            elif "__all__" in op_compile_option_all:
                self.op_compile_option = op_compile_option_all["__all__"]


    def _ip_argpack(self: any, default: bool = True) -> list:
        args = []
        for i in range(len(self.input_name)):
            arg = self.input_name[i]
            if default and self.argsdefv[i] is not None:
                arg += '=' + self.argsdefv[i]
            args.append(arg)
        return args

    def _op_argpack(self: any, default: bool = True) -> list:
        args = []
        argidx = len(self.input_name)
        for i in range(len(self.output_name)):
            arg = self.output_name[i]
            if default and self.argsdefv[i + argidx] is not None:
                arg += '=' + self.argsdefv[i + argidx]
            args.append(arg)
        return args

    def _attr_argpack(self: any, default: bool = True) -> list:
        args = []
        argidx = len(self.input_name) + len(self.output_name)
        for i in range(len(self.attr_list)):
            att = self.attr_list[i]
            arg = att
            if default and self.argsdefv[i + argidx] is not None:
                if self.attr_val.get(att).get('type') == 'str':
                    arg += '="' + self.argsdefv[i + argidx] + '"'
                elif self.attr_val.get(att).get('type') == 'bool':
                    arg += '=' + self.argsdefv[i + argidx].capitalize()
                else:
                    arg += '=' + self.argsdefv[i + argidx]
            args.append(arg)
        return args

    def _build_paralist(self: any, default: bool = True) -> str:
        args = []
        args.extend(self._ip_argpack(default))
        args.extend(self._op_argpack(default))
        args.extend(self._attr_argpack(default))
        return ', '.join(args)

    def _io_parachk(self: any, types: list, type_name: str) -> list:
        chk = []
        for iot in types:
            if iot == 'optional':
                ptype = 'OPTION'
            else:
                ptype = iot.upper()
            chk.append('para_check.{}_{}'.format(ptype, type_name))
        return chk

    def _attr_parachk(self: any) -> list:
        chk = []
        for att in self.attr_list:
            if self.attr_val.get(att).get('paramType') == 'optional':
                pt = 'OPTION'
            else:
                pt = self.attr_val.get(att).get('paramType').upper()
            att_type = self.attr_val.get(att).get('type').upper()
            chk.append('para_check.{}_ATTR_{}'.format(pt, att_type))
        return chk

    def _build_parachk(self: any) -> str:
        chk = []
        chk.extend(self._io_parachk(self.input_type, 'INPUT'))
        chk.extend(self._io_parachk(self.output_type, 'OUTPUT'))
        chk.extend(self._attr_parachk())
        chk.append('para_check.KERNEL_NAME')
        return ', '.join(chk)

    def _build_paradefault(self: any):
        optional = False
        argtypes = []
        argtypes.extend(self.input_type)
        argtypes.extend(self.output_type)
        for atype in argtypes:
            if atype == 'optional':
                optional = True
            if optional:
                self.argsdefv.append('None')
            else:
                self.argsdefv.append(None)
        for attr in self.attr_list:
            atype = self.attr_val.get(attr).get('paramType')
            if atype == 'optional':
                optional = True
            attrval = self.attr_val.get(attr).get('defaultValue')
            if attrval is not None:
                optional = True
                if type == "bool":
                    attrval = attrval.capitalize()
                elif type == "str":
                    attrval = "\"" + attrval + "\""
                self.argsdefv.append(attrval)
                continue
            if optional:
                self.argsdefv.append(ATTR_DEFAULT.get(self.attr_val.get(attr).get('type')))
            else:
                self.argsdefv.append(None)

    def _write_head(self: any, fd: object):
        fd.write(IMPL_HEAD)

    def _write_argparse(self: any, fd: object):
        args = self._build_paralist(False)
        fd.write('def _build_args({}):\n'.format(args))
        fd.write('    __inputs__ = []\n')
        fd.write('    for arg in [{}]:\n'.format(', '.join(self.input_name)))
        fd.write('        if arg != None:\n')
        fd.write('            if isinstance(arg, (list, tuple)):\n')
        fd.write('                if len(arg) == 0:\n')
        fd.write('                    continue\n')
        fd.write('                __inputs__.append(arg[0])\n')
        fd.write('            else:\n')
        fd.write('                __inputs__.append(arg)\n')
        fd.write('    __outputs__ = []\n')
        fd.write('    for arg in [{}]:\n'.format(', '.join(self.output_name)))
        fd.write('        if arg != None:\n')
        fd.write('            if isinstance(arg, (list, tuple)):\n')
        fd.write('                if len(arg) == 0:\n')
        fd.write('                    continue\n')
        fd.write('                __outputs__.append(arg[0])\n')
        fd.write('            else:\n')
        fd.write('                __outputs__.append(arg)\n')
        fd.write('    __attrs__ = []\n')
        for attr in self.attr_list:
            fd.write('    if {} != None:\n'.format(attr))
            fd.write('        attr = {}\n')
            fd.write('        attr["name"] = "{}"\n'.format(attr))
            fd.write('        attr["dtype"] = "{}"\n'.format(self.attr_val.get(attr).get('type')))
            fd.write('        attr["value"] = {}\n'.format(attr))
            fd.write('        __attrs__.append(attr)\n')
        fd.write('    return __inputs__, __outputs__, __attrs__\n')

    def _write_impl(self: any, fd: object):
        argsdef = self._build_paralist()
        argsval = self._build_paralist(False)
        pchk = self._build_parachk()
        if len(self.kern_name) > 0:
            kern_name = self.kern_name
        else:
            kern_name = self.op_intf
        src = self.op_file + '.cpp'
        fd.write(IMPL_API.format(self.op_type, pchk, self.op_intf, argsdef, kern_name, argsval,\
                                 self.custom_compile_options, self.custom_all_compile_options, self.op_intf,\
                                 optype_snake_ex(self.op_type), optype_snake(self.op_type), src))
        if self.op_replay_flag:
            fd.write(REPLAY_OP_API.format(self.op_type, kern_name, self.op_file, self.op_type, self.op_file,\
                self.op_compile_option))
        else:
            fd.write(COMPILE_OP_API.format(self.op_type, self.op_type, ', '.join(self.input_name),\
                ', '.join(self.output_name), self.op_compile_option))

    def _write_cap(self: any, cap_name: str, fd: object):
        argsdef = self._build_paralist()
        argsval = self._build_paralist(False)
        if cap_name == 'check_supported':
            fd.write(SUP_API.format(cap_name, argsdef, argsval, cap_name, self.op_type))
        else:
            fd.write(CAP_API.format(cap_name, argsdef, argsval, cap_name, self.op_type))

    def _write_glz(self: any, fd: object):
        argsdef = self._build_paralist()
        argsval = self._build_paralist(False)
        fd.write(GLZ_API.format(self.op_type, self.op_intf, argsdef, argsval, self.op_type))


def write_scripts(cfgfile: str, cfgs: dict, dirs: dict, ops: list = None, op_compile_option:list = None):
    batch_lists = cfgs.get(const_var.REPLAY_BATCH).split(';')
    iterator_lists = cfgs.get(const_var.REPLAY_ITERATE).split(';')
    file_map = {}
    op_descs = opdesc_parser.get_op_desc(cfgfile, batch_lists, iterator_lists, AdpBuilder,\
                                         ops, dirs.get(const_var.AUTO_GEN_DIR))
    for op_desc in op_descs:
        op_desc.write_adapt(dirs.get(const_var.CFG_IMPL_DIR), dirs.get(const_var.CFG_OUT_DIR), op_compile_option)
        file_map[op_desc.op_type] = op_desc.op_file
    return file_map

if __name__ == '__main__':
    if len(sys.argv) <= 6:
        raise RuntimeError('arguments must greater equal than 6')
    rep_cfg = {}
    rep_cfg[const_var.REPLAY_BATCH] = sys.argv[2]
    rep_cfg[const_var.REPLAY_ITERATE] = sys.argv[3]
    cfg_dir = {}
    cfg_dir[const_var.CFG_IMPL_DIR] = sys.argv[4]
    cfg_dir[const_var.CFG_OUT_DIR] = sys.argv[5]
    cfg_dir[const_var.AUTO_GEN_DIR] = sys.argv[6]
    write_scripts(cfgfile=sys.argv[1], cfgs=rep_cfg, dirs=cfg_dir)
