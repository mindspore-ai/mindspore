#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import json
import time
import argparse
import random
import string
import threading
import multiprocessing
from multiprocessing.sharedctypes import Value
from copy import deepcopy
from subprocess import getstatusoutput
import yaml

script_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_path)

# command line parameters
parser = argparse.ArgumentParser()
parser.add_argument("--env_config_file", dest="env_config_file", required=True,
                    help='env net config file', type=str)
parser.add_argument("--case_env_config_path", dest="case_env_config_path",
                    required=True,
                    help='case and env mapping configuration file', type=str)
parser.add_argument("--case_root", dest="case_root", required=True,
                    help='case root path', type=str)
parser.add_argument("--filter_keyword", dest="filter_keyword", required=True,
                    help='filter key word, decide which cases to run', type=str)
parser.add_argument("--env_type", dest="env_type", required=True,
                    help='env type', type=str)


def opener(path, flags):
    return os.open(path, flags, 0o644)


class FileClass:
    def __init__(self, fname, flags, mode):
        self.fd = os.open(fname, flags, mode)

    def __enter__(self):
        return self

    def __exit__(self, etype, evalue, etrace):
        os.close(self.fd)

    def write(self, text):
        os.write(self.fd, str.encode(text))


def fopen(file_name, file_flags, file_mode):
    return FileClass(file_name, file_flags, file_mode)


class RhineThread(threading.Thread):
    def __init__(self, func, args_in, key_args_in):
        super(RhineThread, self).__init__()
        self.args = args_in
        self.key_args = key_args_in
        self.func = func
        self.result = None

    def run(self):
        self.result = self.func(*self.args, **self.key_args)

    def get_result(self):
        return self.result


class CommonThread:
    def __init__(self):
        self.thread_list = []

    def register(self, func, args, key_args=None, is_daemon=True):
        if key_args is None:
            key_args = {}
        t_thread = RhineThread(func, args_in=args, key_args_in=key_args)
        t_thread.daemon = is_daemon
        self.thread_list.append(t_thread)

    def start(self):
        if not self.thread_list:
            return None

        for t_thread in self.thread_list:
            t_thread.start()

        for t_thread in self.thread_list:
            t_thread.join()

        result_list = [t_thread.get_result() for t_thread in self.thread_list]

        self.thread_list.clear()
        return result_list


class CaseRunner:
    def __init__(self, env_config_file, case_env_config_path, case_root,
                 filter_keyword, env_type):
        self.env_config_file = env_config_file
        self.case_env_config_path = case_env_config_path
        self.case_root = case_root
        self.filter_keyword = filter_keyword
        self.env_type = env_type
        with open(case_env_config_path) as f:
            self.case_env_config = yaml.safe_load(f)

        self.script_path = script_path
        self.pytest_ini = os.path.join(self.script_path, "config", "pytest.ini")
        self.tmp_case_txt = "/tmp/tmp_case_info.txt"
        self.tmp_attr_case_json = "/tmp/tmp_attr_case.json"

        self.python_path = None
        self.pytest_path = None
        self.extend_envs = None
        self.run_path = None
        self.log_path = None
        self.device_ids = None
        self.overall_networks = False
        self.check_env_config()

        self.one_card_txt_info = None
        self.single_txt_info = None

        # testcase output log format
        self.stty_col = 200
        self.env_type_width = 15
        self.result_width = 15
        self.run_time_width = 15
        self.test_case_width = 90
        self.case_path_width = 50
        self.progress_width = 15

    @staticmethod
    def get_filter_file_dirs(file_list):
        """ group all testcase files by directory """
        dir_info = {}
        for file in file_list:
            dir_name = os.path.dirname(file)
            case_file = file.split("/")[-1]
            if not dir_info.get(dir_name):
                dir_info[dir_name] = case_file
            else:
                dir_info[dir_name] += " {0}".format(case_file)

        print("get_filter_file_dirs finished: {0}".format(json.dumps(dir_info, indent=2)))
        return dir_info

    @staticmethod
    def write_header_to_csv():
        with fopen("run_cmds.csv", os.O_WRONLY|os.O_CREAT|os.O_TRUNC, 0o644) as fo:
            fo.write("device_id, begin_time, end_time, num_executed, num_total, run_result, run_cmd\n")

    @staticmethod
    def write_result_to_csv(result_record):
        dev_id, beg_t, end_t, num_executed, num_total, run_result, run_cmd = result_record
        with fopen("run_cmds.csv", os.O_WRONLY|os.O_CREAT|os.O_APPEND, 0o644) as fo:
            fo.write('%d, %f, %f, %d, %d, %s, %s\n'
                     % (dev_id, beg_t, end_t, num_executed, num_total, run_result, run_cmd))

    def create_init_py(self, filepath):
        init_file = os.path.join(filepath, "__init__.py")
        # already has file `__init__.py`
        if not os.path.exists(init_file):
            with fopen(init_file, os.O_WRONLY|os.O_CREAT, 0o644) as fo:
                fo.write("")
        files = os.listdir(filepath)
        for fi in files:
            if str(fi).startswith("."):
                continue
            fi_d = os.path.join(filepath, fi)
            if os.path.isdir(fi_d):
                self.create_init_py(fi_d)

    def check_env_config(self):
        def is_executable(path):
            ret, _ = getstatusoutput("test -f {0} && test -x {0}".format(path))
            return ret == 0

        def search_executable(cmd):
            return getstatusoutput("command -v {0}".format(cmd))

        def get_cfg_executable(cmd, path):
            if is_executable(path):
                return path
            if not path.startswith("XXX"):
                raise ValueError("Path `{0}` is not executable file".format(path))
            ret, path = search_executable(cmd)
            if ret == 0:
                return path
            raise ValueError("Can not find executable cmd: {0}".format(cmd))

        def is_existed(path):
            ret, _ = getstatusoutput("test -e {0}".format(path))
            return ret == 0

        def is_directory(path):
            ret, _ = getstatusoutput("test -d {0} && test -x {0}".format(path))
            return ret == 0

        def get_cfg_directory(path):
            if is_directory(path):
                return path
            if is_existed(path):
                raise ValueError("Path `{0}` already exists, but is not a directory".format(path))
            ret, _ = getstatusoutput("mkdir -p {0}".format(path))
            if ret != 0:
                raise ValueError("Try to make path `{0}` failed".format(path))
            return path

        with open(self.env_config_file) as f:
            env_net_config = yaml.safe_load(f)
        self.python_path = get_cfg_executable('python', env_net_config['python_path'])
        self.pytest_path = get_cfg_executable('pytest', env_net_config['pytest_path'])
        self.run_path = get_cfg_directory(env_net_config['run_path'])
        self.log_path = get_cfg_directory(env_net_config['log_path'])
        self.extend_envs = env_net_config['extend_envs']
        if str(self.extend_envs).strip() == "" or self.extend_envs.startswith("XXX"):
            _, exec_path = getstatusoutput("dirname {0}".format(self.pytest_path))
            self.extend_envs = "export PATH=%s:${PATH}; export PYTHONPATH=%s:${PYTHONPATH}" \
                               % (exec_path, self.case_root)
        # check and set device_ids
        device_ids = env_net_config['virtualenv']['device_ids']
        if not device_ids or len(device_ids) > 8:
            raise ValueError("Invalid config device_ids: {0}".format(device_ids))
        dev_id_list = []
        for e in device_ids:
            # if device_ids contains duplicated elements, report error
            if e in range(0, 8) and (not e in dev_id_list):
                dev_id_list.append(e)
            else:
                raise ValueError("Invalid config device_ids: {0}".format(device_ids))
        self.device_ids = device_ids
        if len(device_ids) == 8 and env_net_config['virtualenv']['overall_networks']:
            self.overall_networks = True
        else:
            self.overall_networks = False

    def clear_last_tmp_file(self):
        clear_tmp_file_cmd = """rm -f %s; rm -f %s""" % (
            self.tmp_case_txt, self.tmp_attr_case_json)
        os.system(clear_tmp_file_cmd)

    def get_ready_for_filter(self):
        # create `__init__.py` if not exist
        self.create_init_py(self.case_root)
        # clear temp files
        self.clear_last_tmp_file()
        return True

    def get_suitable_file(self, search_case_type):
        """ filter testcase files which satisfy condition (e.g. level0 and env type)"""
        get_cases_info_cmd = """  grep -rE  "%s" %s/* --include="*.py" """ % (search_case_type, self.case_root)
        status, output = getstatusoutput(get_cases_info_cmd)
        if int(status) == 1 and not output:
            print("get_singel_env_case_filter :: no suitable cases found for search_case_type:{0}"
                  .format(search_case_type))
            # write blank to result json file
            with fopen(self.tmp_attr_case_json, os.O_WRONLY|os.O_CREAT, 0o644) as fo:
                fo.write(json.dumps([]))
            return False, "cases result is null"

        filter_case_cmd = """%s | awk -F ':' '{print $1}' | uniq""" % (get_cases_info_cmd)
        status, output = getstatusoutput(filter_case_cmd)
        if int(status) != 0 or not output:
            print("get_suitable_file failed for get case filter, status:{0},output:{1},filter_case_cmd:{2}" \
                  .format(status, output, filter_case_cmd))
            return False, None

        return True, output.split("\n")

    def filter_case_with_one_dir(self, case_path, case_file, extend_envs, pytest_path, search_case_type):
        """ filter testcases from single directory """
        tmp_file = os.path.join(os.path.dirname(self.tmp_case_txt), "{0}.txt".format(str(case_path).replace("/", "_")))
        case_type_filter = ""
        for case_type in search_case_type.split("|"):
            if not case_type_filter:
                case_type_filter += "({0} ".format(case_type)
            else:
                case_type_filter += " or {0}".format(case_type)
            case_type_filter += ")"

        filter_case_cmd = """{0}; export PYTHONPATH={1}:$PYTHONPATH; """ \
                          """cd {1} &&  {2} --collect-only -m '{3} and {4}'  -c {5} {6} 2>&1 >{7}""". \
            format(extend_envs, case_path, pytest_path, self.filter_keyword, case_type_filter,
                   self.pytest_ini, case_file, tmp_file)

        print("filter_case_with_one_dir filter_case_cmd is : {0}".format(filter_case_cmd))
        status, _ = getstatusoutput(filter_case_cmd)
        if int(status) == 5:
            print("no any gate test case to be found in {0}".format(case_path))
        elif int(status) != 0:
            return {"ret_val": False, "output_file": tmp_file, "case_path": case_path}

        return {"ret_val": True, "output_file": tmp_file, "case_path": case_path}

    def filer_case_with_files(self, file_list, search_case_type):
        extend_envs = """source /etc/profile; {0}""".format(self.extend_envs)

        # create one thread for per directory to filter testcases
        t_thread_list = CommonThread()
        dir_info = self.get_filter_file_dirs(file_list)
        for case_path, case_files in dir_info.items():
            t_thread_list.register(self.filter_case_with_one_dir, args=(
                case_path, case_files, extend_envs, self.pytest_path, search_case_type))

        result_list = t_thread_list.start()

        fail_filter_list = []
        for result in result_list:
            output_path = result["output_file"]
            if not result["ret_val"]:
                fail_filter_list.append({"file_path": result["case_path"], "output_path": output_path})
                continue

            # filter successfully, write result to temp file for later use
            fuse_cmd = """ [ -f {0} ] && cat {0} >> {1} ; rm -f {0}""".format(output_path, self.tmp_case_txt)
            os.system(fuse_cmd)

        # write detailed message for failed thread
        if fail_filter_list:
            for fail_filter in fail_filter_list:
                file_path = fail_filter["file_path"]
                output_path = fail_filter["output_path"]
                print("filter failed for {0}".format(file_path))
                print_fail_cmd = "[ -f {0} ] && cat {0} ; rm -f {0}".format(output_path)
                _, output = getstatusoutput(print_fail_cmd)
                print(output)
            return False

        return True

    def get_env_case_filter(self):
        print(f'get_env_case_filter start\ncase_type: {self.case_env_config["case_type"]}')
        search_case_type = ""
        for case_type, case_env in dict(
                self.case_env_config["case_type"]).items():
            if str(case_env).__contains__(self.env_type) and case_type != "ALL":
                if not search_case_type:
                    search_case_type = case_type
                else:
                    search_case_type += "|{0}".format(case_type)
        if not search_case_type:
            print("get_singel_env_case_filter, no suitable case_env_config element found for env : {0}".format(
                self.env_type))
            return False, None

        # filter testcase files first
        ret_val, search_file_list = self.get_suitable_file(search_case_type)
        # execute success but not found any testcase
        if not ret_val and isinstance(search_file_list, (str,)) and search_file_list == "cases result is null":
            return False, "cases result is null"
        if not ret_val:
            return False, None

        # filter testcases from files
        if not self.get_ready_for_filter():
            return False, None

        # filter testcases by file
        ret_val = self.filer_case_with_files(search_file_list, search_case_type)
        if not ret_val:
            print("filer_case_with_files fail")
            return False, None

        return True, "filter success"

    def parser_case_module(self):
        """ parse testcase module info from pytest collect-only result """
        print("parser_case_module start")
        case_content = []
        tmp_dict = {}

        file_handle = open(self.tmp_case_txt, "r")
        last_package = ""
        last_class = ""
        last_module = ""
        case_root = ""
        while True:
            lines = file_handle.readlines(2000)
            if not lines:
                if tmp_dict and tmp_dict not in case_content and (
                        tmp_dict.get("function_list") or tmp_dict.get("class")):
                    case_content.append(tmp_dict)
                break
            max_len = len(lines)
            for index, line in enumerate(lines):
                if line.startswith("rootdir: "):
                    case_root = line.split(':')[1].split(',')[0].strip().replace(' ', '')
                elif line.startswith("<Package "):
                    if tmp_dict:
                        case_content.append(tmp_dict)
                        tmp_dict = {}

                    psr_pkg = line.replace("<Package ", "").split(">")[0]
                    # pytest result absolute path
                    if psr_pkg.startswith(os.sep):
                        last_package = psr_pkg
                    # pytest result relative path
                    else:
                        last_package = case_root

                    tmp_dict["package"] = last_package
                elif line.startswith("  <Module "):
                    module = line.replace("  <Module ", "").split(">")[0]
                    if last_module and module != last_module:
                        if tmp_dict.get("function_list") or tmp_dict.get("class"):
                            case_content.append(tmp_dict)
                        tmp_dict = {"package": last_package, "module": module}
                    tmp_dict["module"] = module
                    last_module = module
                elif line.startswith("<Module "):
                    module = line.replace("<Module ", "").split(">")[0]
                    if last_module and module != last_module:
                        if tmp_dict.get("function_list") or tmp_dict.get("class"):
                            case_content.append(tmp_dict)
                        tmp_dict = {"package": last_package, "module": module}

                    tmp_dict["module"] = module
                    last_module = module
                elif line.startswith("        <Function "):
                    if not tmp_dict.get("function_list", None):
                        tmp_dict["function_list"] = []
                    tmp_dict.get("function_list").append(line.replace("        <Function ", "").split(">")[0])
                    if index < max_len - 1 and str(lines[index + 1]).__contains__("<Module"):
                        case_content.append(tmp_dict)
                        tmp_dict = {"package": last_package, "module": last_module, "class": last_class}
                        continue
                    if index < max_len - 1 and not lines[index + 1].startswith("        <Function "):
                        case_content.append(tmp_dict)
                        tmp_dict = {}

                elif line.startswith("    <Function "):
                    if not tmp_dict.get("function_list", None):
                        tmp_dict["function_list"] = []
                    tmp_dict.get("function_list").append(line.replace("    <Function ", "").split(">")[0])
                    if index < max_len - 1 and str(lines[index + 1]).__contains__("<Module"):
                        case_content.append(tmp_dict)
                        tmp_dict = {"package": last_package, "module": last_module}
                        continue

                    if index < max_len - 1 and not lines[index + 1].startswith("    <Function "):
                        case_content.append(tmp_dict)
                        tmp_dict = {}
                elif line.startswith("    <Class "):
                    class_info = line.replace("    <Class ", "").split(">")[0]
                    if last_class and class_info != last_class:
                        if tmp_dict.get("function_list") or tmp_dict.get("class"):
                            case_content.append(tmp_dict)
                        tmp_dict = {"package": last_package, "module": last_module, "class": class_info}
                    tmp_dict["class"] = class_info
                    last_class = class_info
                else:
                    continue
        file_handle.close()
        return case_content

    def parser_module_attr(self, module_list):
        script_root = self.script_path
        random_str = "".join(random.sample(string.ascii_letters + string.digits, 8))
        file_name = str(time.time()) + "_" + random_str + ".json"
        result_file = os.path.join(os.path.dirname(__file__), file_name)
        json_ret = list()

        get_cmd = "cd {0}; {1} psr_module_attr.py --case_root {2} --module_list '{3}' --result_file {4}" \
                  "".format(script_root, sys.executable, self.case_root, json.dumps(module_list), result_file)
        status, output = getstatusoutput(get_cmd)
        print("parser_module_attr get_cmd : {0}, output : {1}".format(get_cmd, output))
        if int(status) != 0 or not output:
            print("parser_module_attr failed for get_cmd : {0}".format(get_cmd))
            return json_ret

        if os.path.exists(result_file):
            with open(result_file) as f_h:
                json_ret = json.load(f_h)
            getstatusoutput("rm -f {}".format(result_file))
        else:
            print("parser_module_attr result_file not found")

        return json_ret

    def mul_parser_case_attr(self, module_content):
        """
        parse attr info of testcase from module info
        """
        print("mul_parser_case_attr start")
        core_nums = 16

        split_list = []
        for index in range(0, len(module_content), core_nums):
            split_list.append(module_content[index:index + core_nums])

        # use process pool to parse mark attr info from module info
        process_pool = multiprocessing.Pool(16)
        pool_map_result = process_pool.map_async(self.parser_module_attr,
                                                 split_list)
        pool_map_result.wait()
        if pool_map_result.ready() and pool_map_result.successful():
            result_list = pool_map_result.get()
            cases_attr_info = []
            for result in result_list:
                cases_attr_info += result
            process_pool.close()
        else:
            message = "mul_parser_case_attr failed for process_pool run"
            print(message)
            print(message)
            return None

        print("mul_parser_case_attr ret_val_list is : {0}".format(
            cases_attr_info))

        if None in cases_attr_info:
            message = "mul_parser_case_attr :: some case module parser failed"
            print(message)
            return None

        return cases_attr_info

    def filter_main(self):
        filer_ret, msg = self.get_env_case_filter()
        if not filer_ret:
            if not msg:
                print("get_env_case_filter failed")
                return False
            if msg == "cases result is null":
                print("get_env_case_filter ::cases result is null")
                return True

        module_content = self.parser_case_module()
        if not module_content:
            print("filter_main parser_case_module failed, no available case for {0}".format(self.filter_keyword))
            return True

        # get attribute info of testcases
        cases_attr_info = self.mul_parser_case_attr(module_content)
        if not cases_attr_info:
            print("filter_main parser_case_attr failed, no available case attr for {0}".format(self.filter_keyword))
            return False

        # write attribute info to file
        with fopen(self.tmp_attr_case_json, os.O_WRONLY|os.O_CREAT, 0o644) as fo:
            fo.write(json.dumps(cases_attr_info, indent=2))

        return True

    def group_cases_by_run_mode(self):
        """ group testcases by run mode: env_onecard/env_single """
        self.one_card_txt_info = []
        self.single_txt_info = []
        with open(self.tmp_attr_case_json, 'r') as f:
            cases_attr_info = json.load(f)

        def group_one_obj(pkg_name, case_name, obj):
            new_obj = {}
            new_obj["package"] = pkg_name
            new_obj["case_name"] = case_name
            new_obj["obj_name"] = obj["class_name"] if "class_name" in obj else obj["function"]
            run_case_mod = obj["run_case_mod"]
            new_obj["run_case_mod"] = run_case_mod
            new_obj["env_type"] = deepcopy(obj["env_type"])
            if run_case_mod in ["env_card", "env_onecard"]:
                self.one_card_txt_info.append(new_obj)
            else:
                self.single_txt_info.append(new_obj)

        for case_attr in cases_attr_info:
            package = case_attr["package"]
            case_name = case_attr["case_name"]
            function_list = case_attr["function_list"]
            class_info = case_attr["class"]
            # test object is based function
            if not class_info:
                for func in function_list:
                    group_one_obj(package, case_name, func)
            # test object is based class
            else:
                tmp_class_info = deepcopy(class_info)
                if isinstance(tmp_class_info, (dict,)):
                    group_one_obj(package, case_name, tmp_class_info)

        print(f"\nSummary of testcases 1P: {len(self.one_card_txt_info)}, 8P: {len(self.single_txt_info)}\n" + \
              f"Devices used: {self.device_ids}")

    def get_one_card_testcases(self):
        """ get one card testcases """
        testcases = mp.Queue()
        if self.one_card_txt_info:
            for _, v in dict(self.one_card_txt_info).items():
                for testcase in v["param"]["testcase"]:
                    idx = testcases.qsize()
                    case_file = testcase['number']
                    case_path = testcase['package']
                    case_func = testcase['function']
                    test_case = "{0}/{1}.py::{2}".format(case_path, case_file, case_func)
                    testcases.put((idx, test_case, case_path))
        return testcases

    def get_rel_path(self, path):
        root_len = len(self.case_root) if self.case_root[-1] == '/' else len(self.case_root) + 1
        return path[root_len:]

    def run_multi_card_testcases(self, num_all_st):
        num_pass = 0
        num_fail = 0
        time_beg = time.time()
        for case_idx, testcase in enumerate(self.single_txt_info):
            case_file = testcase['case_name']
            case_path = testcase['package']
            case_func = testcase['obj_name']
            test_case = "{0}/{1}.py::{2}".format(case_path, case_file, case_func)

            run_cmd = f'cd {case_path}; {self.extend_envs}; ' + \
                        f'python -u -m pytest -s -v {test_case} &> {self.log_path}/{case_idx}.log'
            beg_t = time.time()
            retval, _ = getstatusoutput(run_cmd)
            end_t = time.time()

            case_type = "8P"
            run_time = float(end_t - beg_t)
            run_time = "%.3f" % (run_time)
            result = '\033[32mPASS\033[0m' if retval == 0 else '\033[31mFAIL\033[0m'
            if retval == 0:
                num_pass += 1
            else:
                num_fail += 1
            progress = "[%4d / %4d]" % (case_idx + 1, num_all_st)
            case_name = "{0}.py::{1}".format(case_file, case_func)
            test_result = (progress, case_name, case_type, result, run_time, self.get_rel_path(case_path))
            self.write_result_to_csv((-1, beg_t, end_t, case_idx + 1, num_all_st, result, run_cmd))
            self.out_summary_result(test_result)
        time_end = time.time()
        return (num_pass, num_fail, time_end - time_beg)

    def run_one_card_testcases(self, num_all_st, num_8p=None):
        time_beg = time.time()
        # one card  work function
        def one_card_task(print_lock, device_id, testcases, num_total, num_executed, num_failed):
            while not testcases.empty():
                case_idx, testcase = testcases.get()
                case_file = testcase['case_name']
                case_path = testcase['package']
                case_func = testcase['obj_name']
                test_case = "{0}/{1}.py::{2}".format(case_path, case_file, case_func)

                run_cmd = f'cd {self.run_path}; export DEVICE_ID={device_id}; {self.extend_envs}; ' + \
                          f'python -u -m pytest -s -v {test_case} &> {self.log_path}/{case_idx}.log'
                beg_t = time.time()
                retval, _ = getstatusoutput(run_cmd)
                end_t = time.time()

                num_executed.value += 1
                if retval != 0:
                    num_failed.value += 1

                case_type = "1P"
                run_time = float(end_t - beg_t)
                run_time = "%.3f" % (run_time)
                result = '\033[32mPASS\033[0m' if retval == 0 else '\033[31mFAIL\033[0m'
                progress = "[%4d / %4d]" % (num_executed.value, num_all_st)
                case_name = "{0}.py::{1}".format(case_file, case_func)
                test_result = (progress, case_name, case_type, result, run_time, self.get_rel_path(case_path))

                # acquire print lock
                with print_lock:
                    self.write_result_to_csv((device_id, beg_t, end_t, num_executed.value, num_all_st,
                                              result, run_cmd))

                    if retval == 0:
                        os.system(f'rm -f logs/{case_idx}.log')

                    # output summary of testcase
                    self.out_summary_result(test_result)


        testcases = multiprocessing.Queue()
        for idx, testcase in enumerate(self.one_card_txt_info):
            testcases.put((idx, testcase))
        lock = multiprocessing.Lock()
        jobs = []
        num_executed = Value('i', 0 if num_8p is None else num_8p)
        num_failed = Value('i', 0)
        num_total = testcases.qsize()
        for dev_id in self.device_ids:
            p = multiprocessing.Process(target=one_card_task, args=(lock, dev_id, testcases, num_total, num_executed,
                                                                    num_failed))
            jobs.append(p)
            p.start()

        for job in jobs:
            job.join()

        time_end = time.time()
        return (num_total - num_failed.value, num_failed.value, time_end - time_beg)

    def out_summary_head_info(self):
        print("Start Run Case\n")
        print("[Case Result Info]\n")
        print("-" * self.stty_col, flush=True)

        print(format("Progress", f"<{self.progress_width}"),
              format("Testcase", f"<{self.test_case_width}"),
              format("EnvType", f"<{self.env_type_width}"),
              format("Result", f"<{self.result_width}"),
              format("RunTime", f"<{self.run_time_width}"),
              format("CasePath", f"<{self.case_path_width}"),
              flush=True)
        print(format("------------", f"<{self.progress_width}"),
              format("------------", f"<{self.test_case_width}"),
              format("------------", f"<{self.env_type_width}"),
              format("------", f"<{self.result_width}"),
              format("------", f"<{self.run_time_width}"),
              format("------------", f"<{self.case_path_width}"),
              flush=True)

    def out_summary_result(self, result):
        """output result info of one testcase"""
        progress, case_name, case_type, result, run_time, case_path = result

        print(format(progress, f"<{self.progress_width}"),
              format(case_name, f"<{self.test_case_width}"),
              format(case_type, f"<{self.env_type_width}"),
              format(result, f"<{self.result_width + 9}"),
              format(run_time, f"<{self.run_time_width}"),
              format(case_path, f"<{self.case_path_width}"),
              flush=True)

    def out_summary_tail_info(self, result_1p, result_8p=None):
        print("-" * self.stty_col, flush=True)
        # result_1p --> (num_pass, num_fail, run_time)
        def print_resut_by_env(env_type, result):
            if result is None:
                return
            num_pass, num_fail, run_time = result
            print(f"\n\n[EnvType    ] {env_type}", flush=True)
            print(f"[CaseNumbers] {num_pass + num_fail}", flush=True)
            print(f"[RunTime    ] %.3f" % run_time, flush=True)


        num_pass_total, num_fail_total, _ = result_1p
        if result_8p is not None:
            num_pass_total += result_8p[0]
            num_fail_total += result_8p[1]
        print("Total Tests   : %d" % (num_pass_total + num_fail_total), flush=True)
        print("Total Failures: %d" % num_fail_total, flush=True)
        print("Total Success : %d" % num_pass_total, flush=True)
        print("\n\nEnd Run Case", flush=True)
        print_resut_by_env("8P", result_8p)
        print_resut_by_env("1P", result_1p)


if __name__ == '__main__':
    cmd_args = parser.parse_args()
    input_args = (
        cmd_args.env_config_file, cmd_args.case_env_config_path, cmd_args.case_root,
        cmd_args.filter_keyword, cmd_args.env_type)
    runner_handle = CaseRunner(*input_args)

    if not runner_handle.filter_main():
        print("case_filter failed !")
        sys.exit(1)

    runner_handle.group_cases_by_run_mode()

    runner_handle.out_summary_head_info()
    runner_handle.write_header_to_csv()

    number_all = len(runner_handle.one_card_txt_info)
    if runner_handle.overall_networks:
        number_8p = len(runner_handle.single_txt_info)
        number_all += number_8p
        result_multi_card = runner_handle.run_multi_card_testcases(number_all)
        result_one_card = runner_handle.run_one_card_testcases(number_all, number_8p)
        runner_handle.out_summary_tail_info(result_one_card, result_multi_card)
    else:
        result_one_card = runner_handle.run_one_card_testcases(number_all)
        runner_handle.out_summary_tail_info(result_one_card)

    print("\nExecute testcases finished!")
    sys.exit(0)
