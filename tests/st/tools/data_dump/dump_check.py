# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Check the dump result functionality
"""

import math
import os
import re
import numpy as np
import ast
from mindspore import log as logger

FILE_RWGR_MODE = '640'
FILE_RW_MODE = '600'

WORK_DIR = os.path.realpath('./')
DEVICE_ID = int(os.environ.get("DEVICE_ID", "0"))

STATISTIC_HEAD = "Op Type,Op Name,Task ID,Stream ID,Timestamp,IO,Slot,Data Size,Data Type,Shape"
HEAD_DICT = {"max": "Max Value", "min": "Min Value", "avg": "Avg Value", "count": "Count",
             "negative zero count": "Negative Zero Count", "positive zero count": "Positive Zero Count",
             "nan count": "NaN Count", "negative inf count": "Negative Inf Count",
             "positive inf count": "Positive Inf Count",
             "zero count": "Zero Count", "md5": "MD5", "l2norm": "L2Norm Value", "hash:md5": "MD5", "hash": "SHA1",
             "hash:sha1": "SHA1"}


IGNORE_OPTYPE_LIST = ["ge:Const", "ge:NoOp", "ge:Send", "ge:Recv", "ge:Variable", "ge:NetOutput", "ge:ConcatV2D",
                      "ge:If", "ge:RefData", "ge:StreamActive", "ge:MemcpyAsync", "ge:Assign", "ge:StreamSwitch",
                      "ge:ConcatD", "ge:SplitVD", "ge:GetNext", "ge:NPUClearFloatStatusV2", "ge:Identity", "ge:Data",
                      "ge:AssignAdd", "ge:PartitionedCall"]

IGNORE_OPNAME_LIST = ["StreamSend", "StreamRecv"]


Infinity = float("inf")
NaN = float("nan")


class FileNode:
    def __init__(self, name, dirname, level):
        '''
        记录文件/目录信息
        @name: 文件/目录名称
        @dirname: 文件/目录所在绝对路径
        @level: 文件/目录所在层级数
        '''
        self.name = name
        self.abs_path = os.path.join(dirname, name)
        self.isfile = os.path.isfile(self.abs_path)
        self.level = level
        if self.level == 0:
            self.rel_path = f"{name}"
        else:
            self.rel_path = os.path.join(*self.abs_path.split("/")[-level:], name)
        self.subfiles = []


class DirTree:
    def __init__(self, dir_path):
        '''
        记录目录树数信息
        @dir_path: 目录路径
        '''
        self.root_dir = FileNode(os.path.basename(dir_path), os.path.dirname(dir_path), 0)
        self.level_idx = [[], [], [], [], [], [], [], []]
        self.name_idx = {}
        self.init_tree(self.root_dir, 0)

    def init_tree(self, rdir, level):
        '''
        初始化目录, 记录每个文件/目录节点信息, 并按level级别分组
        '''
        self.level_idx[level].append(rdir)
        if rdir.isfile:
            self.name_idx[rdir.name] = rdir
            return
        for d in os.listdir(rdir.abs_path):
            rdir.subfiles.append(FileNode(d, rdir.abs_path, level + 1))
        for subdir in rdir.subfiles:
            self.init_tree(subdir, subdir.level)
        return


class DumpCheck:
    def __init__(self, dump_json_dict, **kwargs):
        '''
        根据dump config分析dump目录格式和数据的正确性, 当配置 iteration="all"时, 需要传入iteration_id_list= \
        '{实际iteration数(acl)/实际step数(ge_option)}', 多卡用例需要传入 expected_device_id_list=[实际device id列表]
        @dump_json_dict: MINDSPORE_DUMP_CONFIG指定的json文件内容字典
        '''
        self.dump_path = dump_json_dict["common_dump_settings"].get("path")
        if not self.dump_path and os.environ.get("MS_DIAGNOSTIC_DATA_PATH"):
            self.dump_path = os.path.join(os.environ["MS_DIAGNOSTIC_DATA_PATH"], "debug_dump")
        self.dump_mode = dump_json_dict["common_dump_settings"]["dump_mode"]
        self.kernels = dump_json_dict["common_dump_settings"]["kernels"]
        self.saved_data = dump_json_dict["common_dump_settings"]["saved_data"]
        self.input_output = dump_json_dict["common_dump_settings"]["input_output"]
        self.net_name = dump_json_dict["common_dump_settings"]["net_name"]
        self.op_debug_mode = dump_json_dict["common_dump_settings"]["op_debug_mode"]
        self.statistic_category = dump_json_dict["common_dump_settings"].get("statistic_category",
                                                                             ["max", "min", "l2norm"])
        if dump_json_dict["common_dump_settings"]["iteration"] == "all":
            iteration_id_list = kwargs.get("iteration_id_list")
            if not iteration_id_list:
                raise Exception("Iteration id list get error, please pass iteration_id_list kwargs.")
            self.iteration_id_list = [str(i) for i in range(0, int(iteration_id_list))]
        else:
            self.iteration_id_list = self.get_iteration_id_list(dump_json_dict["common_dump_settings"]["iteration"])
        # 补充参数
        self.expected_device_id_list = kwargs.get("expected_device_id_list", [str(DEVICE_ID)])
        self.graph_id_list = kwargs.get("graph_id_list", ["0"])
        self.expect_dump_op = []
        if self.dump_mode == 2 or self.op_debug_mode != 0:
            self.expect_dump_op = kwargs.get("expect_dump_op", [])
        self.check_details = kwargs.get("check_details", True)

        self.target_dir = DirTree(self.dump_path)
        self.exceptions = []
        self.net_info = {}
        self.file_map_dict = {}
        self.dump_op_list = []

    @staticmethod
    def get_iteration_id_list(iteration):
        iteration_id_list = []
        for part in iteration.split("|"):
            if "-" in part:
                start_iter, end_iter = part.split("-")[0], part.split("-")[1]
                for i in range(int(start_iter), int(end_iter) + 1):
                    iteration_id_list.append(str(i))
            else:
                iteration_id_list.append(part)
        return iteration_id_list

    @staticmethod
    def check_file_permission_and_size(file_path, expect_mode=FILE_RWGR_MODE):
        # 校验文件权限和是否为空
        if oct(os.stat(file_path).st_mode)[-3:] > expect_mode:
            logger.error(f"file: {file_path} permission check failed")
            return False
        file_size = os.path.getsize(file_path)
        if file_size <= 0:
            logger.error(f"file: {file_path} is empty")
            return False
        return True


class SyncDumpCheck(DumpCheck):
    def __init__(self, dump_json_dict, **kwargs):
        super().__init__(dump_json_dict, **kwargs)
        if not dump_json_dict.get("e2e_dump_settings"):
            raise Exception("Dump config dont have e2e_dump_settings but O0 or O1")
        if self.saved_data == "statistic":
            self.stat_calc_mode = dump_json_dict["e2e_dump_settings"].get("stat_calc_mode", "host")
        else:
            self.stat_calc_mode = "host"
        self.sample_mode = dump_json_dict["e2e_dump_settings"].get("sample_mode", 0)
        self.sample_num = dump_json_dict["e2e_dump_settings"].get("sample_num", 100)
        self.save_kernel_args = dump_json_dict["e2e_dump_settings"].get("save_kernel_args", False)
        self.expected_device_id_list = kwargs.get("expected_device_id_list", ['0'])
        self.enable = dump_json_dict["e2e_dump_settings"].get("enable", True)

    def dump_result_check(self):
        # 校验是否有空目录和空文件
        for level_list in self.target_dir.level_idx:
            for level_dir in level_list:
                if not level_dir.isfile and len(level_dir.subfiles) == 0:
                    self.exceptions.append(Exception(f"Dump dir {level_dir.abs_path} is null."))
                    continue
                if level_dir.isfile and not self.check_file_permission_and_size(level_dir.abs_path):
                    self.exceptions.append(Exception(f"Dump file {level_dir.abs_path} is null or high permission."))

        self.check_rank_dir_correct()
        for rank_dir in self.target_dir.level_idx[1]:
            self.check_net_name_correct(rank_dir)
            if self.op_debug_mode == 0:
                self.check_iteration_correct()
            if self.check_details:
                self.check_dump_mode(rank_dir)
        if len(self.exceptions) > 0:
            for expt in self.exceptions:
                print(expt)
            raise self.exceptions[0]

    def check_rank_dir_correct(self):
        # 对应config字段support_device
        expect_rank_dir = {f"rank_{i}" for i in self.expected_device_id_list}
        actual_rank_dir = {rank_dir.name for rank_dir in self.target_dir.level_idx[1]}
        if actual_rank_dir != expect_rank_dir:
            self.exceptions.append(Exception(f"Rank dir lack: {expect_rank_dir - actual_rank_dir}."))

    def check_net_name_correct(self, rank_dir):
        # 对应config字段net_name
        sub_dirs = [sub_dir.name for sub_dir in rank_dir.subfiles]
        if self.net_name not in sub_dirs:
            self.exceptions.append(Exception(f"Net_name wrong, each rank dir contain: {sub_dirs}."))

    def check_iteration_correct(self):
        actual_iteration_list = {d.name for d in self.target_dir.level_idx[4]}
        expect_iteration_list = set(self.iteration_id_list)
        if actual_iteration_list != expect_iteration_list:
            self.exceptions.append(Exception(f"Iteration lack iteration_id:" \
                                                 f" {expect_iteration_list - actual_iteration_list}"))

    def check_dump_mode(self, rank_dir):
        # 对应config字段dump_mode, kernels
        self.parse_execution_order_files(os.path.join(rank_dir.abs_path, "execution_order"))
        self.parse_graphs_files(os.path.join(rank_dir.abs_path, "graphs"))

        for iteration_dir in self.target_dir.level_idx[4]:
            if not iteration_dir.isfile:
                self.check_iteration_files_correct(iteration_dir)

    def check_iteration_files_correct(self, iteration_dir):
        current_graph_id = int(os.path.basename(os.path.dirname(iteration_dir.abs_path)))
        if self.dump_mode == 0 and self.op_debug_mode == 0:
            for k in self.net_info[current_graph_id].keys():
                op_name = k.replace(".", "_").replace("/", "_").replace("\\", "_").replace(" ", "_")
                skip_flag = False
                for skip_name in IGNORE_OPNAME_LIST:
                    if skip_name in op_name:
                        skip_flag = True
                        break
                if skip_flag:
                    continue
                if self.input_output == "0" or self.input_output == "1":
                    for input_id in range(self.net_info[current_graph_id][k][0]):
                        self.dump_op_list.append(f"{op_name}.input.{input_id}")
                if self.input_output == "0" or self.input_output == "2":
                    for output_id in range(self.net_info[current_graph_id][k][1]):
                        self.dump_op_list.append(f"{op_name}.output.{output_id}")
        elif self.dump_mode == 1 and self.op_debug_mode == 0:
            for kernel_name in self.kernels:
                for k in self.net_info[current_graph_id].keys():
                    op_name = k.replace(".", "_").replace("/", "_").replace("\\", "_").replace(" ", "_")
                    skip_flag = False
                    for skip_name in IGNORE_OPNAME_LIST:
                        if skip_name in op_name:
                            skip_flag = True
                            break
                    if skip_flag:
                        continue
                    if kernel_name.startswith("name-regex") and not re.search(kernel_name[11: -1], op_name):
                        continue
                    if kernel_name not in op_name.lower() or kernel_name != op_name:
                        continue
                    for input_id in range(self.net_info[current_graph_id][k][0]):
                        self.dump_op_list.append(f"{op_name}.input.{input_id}")
                    for output_id in range(self.net_info[current_graph_id][k][0]):
                        self.dump_op_list.append(f"{op_name}.output.{output_id}")
        else:
            self.dump_op_list = self.expect_dump_op
        mapping_path = os.path.join(iteration_dir.abs_path, "mapping.csv")
        if os.path.exists(mapping_path):
            with open(mapping_path, 'r', encoding="utf-8") as f:
                content = f.readlines()
                for line in content:
                    self.file_map_dict[line.split(",")[0]] = line.split(",")[1]
        self.check_saved_data_correct(iteration_dir)
        self.check_sample_mode(iteration_dir)
        self.check_save_kernel_args(iteration_dir)

    def check_saved_data_correct(self, iteration_dir):
        statistic_records = []
        npy_files = []
        for file in iteration_dir.subfiles:
            if file.name == 'mapping.csv':
                continue
            if file.name == "statistic.csv":
                with open(file.abs_path, 'r', encoding="utf-8") as f:
                    statistic_records = f.readlines()
            if file.name.endswith("npy"):
                if re.match(r"^[0-9]{3,}", file.name) and file.name not in self.file_map_dict:
                    self.exceptions.append(Exception("Mapping csv lack data."))
                npy_files.append(file.name)
        if self.saved_data == "statistic" and len(statistic_records) <= 1:
            self.exceptions.append(Exception("Dump config saved_data is statistic but no statistic.csv."))
        if self.saved_data == "full" and (len(npy_files) == 0 or len(statistic_records) <= 1):
            self.exceptions.append(Exception("Dump config saved_data is full but file not exist."))
        if self.saved_data == "tensor" and len(npy_files) == 0:
            self.exceptions.append(Exception("Dump config saved_data is npy but no *.npy."))
        self.check_tensor_data_correct(npy_files)
        self.check_statistic_data_correct(iteration_dir, statistic_records)

    def check_tensor_data_correct(self, npy_files):
        if self.saved_data == "tensor" or self.saved_data == "full":
            actual_dump_op_list = set()
            for file in npy_files:
                if re.match(r"^[0-9]{3,}", file):
                    file = self.file_map_dict[file]
                actual_dump_op_list.add(f'{file.split(".")[1]}.{file.split(".")[5]}.{file.split(".")[6]}')
            if len(self.dump_op_list) > 0:
                if not set(self.dump_op_list).issubset(actual_dump_op_list):
                    logger.warning(f'actual more:{actual_dump_op_list - set(self.dump_op_list)}')
                    logger.warning(f'ir graph more:{set(self.dump_op_list) - actual_dump_op_list}')
                    self.exceptions.append(
                        Exception(f"Dump op list lack:{set(self.dump_op_list) - actual_dump_op_list}"))
            if len(actual_dump_op_list) == 0:
                self.exceptions.append(
                    Exception("Dump file is null."))

    def check_statistic_data_correct(self, iteration_dir, statistic_records):
        if self.saved_data == "statistic" or self.saved_data == "full":
            actual_dump_op_list = set()
            for record in statistic_records[1:]:
                actual_dump_op_list.add(f'{record.split(",")[1]}.{record.split(",")[5]}.{record.split(",")[6]}')
            if len(self.dump_op_list) > 0:
                if not set(self.dump_op_list).issubset(actual_dump_op_list):
                    self.exceptions.append(Exception(f"Dump op list in statistic lack:" \
                                                         f"{set(self.dump_op_list) - actual_dump_op_list}"))
            if len(actual_dump_op_list) == 0:
                self.exceptions.append(
                    Exception("Dump statistic file is null."))
            actual_statistic_head = statistic_records[0].strip()
            expect_statistic_head = STATISTIC_HEAD
            if "hash:md5" in self.statistic_category:
                self.statistic_category[self.statistic_category.index("hash:md5")] = "md5"
            if "hash:sha1" in self.statistic_category:
                self.statistic_category[self.statistic_category.index("hash:sha1")] = "hash"
            self.statistic_category = list(dict.fromkeys(self.statistic_category))
            if "md5" in self.statistic_category and "hash" in self.statistic_category:
                self.statistic_category.remove("md5")
                self.statistic_category.append("md5")
            for title in self.statistic_category:
                if self.stat_calc_mode == 'device' and self.saved_data == 'statistic' and \
                        title not in ['max', 'min', 'avg', 'l2norm'] and self.op_debug_mode == 0:
                    continue
                expect_statistic_head += "," + HEAD_DICT[title]
            if actual_statistic_head != expect_statistic_head:
                self.exceptions.append(Exception(f"Statistic head wrong, actual:{actual_statistic_head}, expect:" \
                                                     f"{expect_statistic_head}"))
        if self.saved_data == "full" and self.sample_mode == 0:
            right_idx = min(6, len(statistic_records))
            for record in statistic_records[1: right_idx]:
                record = record.replace("/", "_").replace("[", "(").replace("]", ")")
                npy_file_name = ".".join(record.split(",")[0: 7]) + f".DefaultFormat.{record.split(',')[8]}.npy"
                if len(os.path.join(iteration_dir.abs_path, npy_file_name)) >= 255:
                    for k, v in self.file_map_dict.items():
                        if v.strip() == ".".join(record.split(",")[0: 7]) + \
                                f".DefaultFormat.{record.split(',')[8]}.npy":
                            npy_file_name = str(k)
                            break
                array = np.load(os.path.join(iteration_dir.abs_path, npy_file_name))
                exp_data_size = array.itemsize * array.size
                exp_data_type = str(array.dtype)
                exp_data_shape = array.shape
                exp_min_value = np.amin(array)
                exp_max_value = np.amax(array)
                # numpy转换时，溢出数据需要转成fp32计算才不会出错
                exp_avg_value = np.mean(array.astype(np.float32))
                exp_l2norm_value = np.linalg.norm(array.astype(np.float64))

                act_data_shape = re.findall(r'"(\([\w\W]*?\))"', record)[0]
                if len(exp_data_shape) == 1:
                    act_data_shape = act_data_shape.replace(")", ",)")
                if ast.literal_eval(act_data_shape) != exp_data_shape:
                    self.exceptions.append(Exception(f"{record}'s shape is error."))
                record = record.replace(act_data_shape, "")
                act_data_type = record.split(",")[8]
                if act_data_type.lower() != exp_data_type.lower() and act_data_type.lower() not in ["bfloat16", "int4"]:
                    self.exceptions.append(Exception(f"{record}'s data_type is error."))
                act_data_size = int(record.split(",")[7])
                if act_data_size != exp_data_size and \
                        act_data_type.lower() == "bfloat16" and act_data_size != exp_data_size // 2:
                    self.exceptions.append(Exception(f"{record}'s data_size is error."))

                if "min" in self.statistic_category:
                    act_min_value = float(record.split(",")[10 + self.statistic_category.index("min")])
                    if not math.isclose(act_min_value, exp_min_value, abs_tol=1e-4, rel_tol=1e-4):
                        self.exceptions.append(Exception(f"{record}'s min_value is error."))
                if "max" in self.statistic_category:
                    act_max_value = float(record.split(",")[10 + self.statistic_category.index("max")])
                    if not math.isclose(act_max_value, exp_max_value, abs_tol=1e-4, rel_tol=1e-4):
                        self.exceptions.append(Exception(f"{record}'s max_value is error."))
                if "avg" in self.statistic_category:
                    act_avg_value = float(record.split(",")[10 + self.statistic_category.index("avg")])
                    if not math.isclose(act_avg_value, exp_avg_value, abs_tol=1e-4, rel_tol=1e-4):
                        self.exceptions.append(Exception(f"{record}'s avg_value is error."))
                if "l2norm" in self.statistic_category:
                    act_l2norm_value = float(record.split(",")[10 + self.statistic_category.index("l2norm")])
                    if not math.isclose(act_l2norm_value, exp_l2norm_value,
                                        abs_tol=1e-4 * max(act_l2norm_value, exp_l2norm_value), rel_tol=1e-2):
                        self.exceptions.append(Exception(f"{record}'s l2norm_value is error. exp:{exp_l2norm_value}"))

    def check_sample_mode(self, iteration_dir):
        if self.sample_mode == 1 and (self.saved_data == "tensor" or self.saved_data == "full"):
            right_idx = min(5, len(iteration_dir.subfiles))
            for file in iteration_dir.subfiles[0: right_idx]:
                if file.name.endswith("npy"):
                    array = np.load(file.abs_path)
                    # shape 可能为空
                    if array.shape != () and self.op_debug_mode == 0:
                        if array.shape[0] > self.sample_num:
                            self.exceptions.append(Exception(f"Sample_mode is useless, error file:{file.abs_path}"))

    def check_save_kernel_args(self, iteration_dir):
        exist_json = False
        for file in os.listdir(iteration_dir.abs_path):
            if file.endswith("json"):
                exist_json = True
        if self.save_kernel_args and not exist_json:
            self.exceptions.append(Exception("Dump config saved_data is npy but no *.npy."))

    def parse_graphs_files(self, graphs_dir):
        graph_ids = set()
        for file in os.listdir(graphs_dir):
            tmp = re.findall(r"ms_output_trace_code_graph_([0-9]+?).ir", file)
            if tmp:
                graph_ids.add(int(tmp[0]))
        if len(graph_ids) != len(self.net_info.keys()):
            self.exceptions.append(Exception("Graphs dir lacks graph file."))
        for graph_id in graph_ids:
            with open(
                os.path.join(graphs_dir, f"ms_output_trace_code_graph_{graph_id}.ir"),
                'r',
                encoding="utf-8"
            ) as f:
                content = f.read()
            node_infos = re.split(r"  %[0-9]+?\(\w+?\) = .+\n", content)
            for node_info in node_infos[1:]:
                in_list, out_list = re.findall(r"      : (\(.*\)) -> (\(.*\))", node_info)[0]
                in_num = len(re.findall(r"<[\w\W]+?>", in_list))
                out_num = len(re.findall(r"<[\w\W]+?>", out_list))
                fullname = re.findall(r"      # Fullname with scope:[\w\W]*?\(([\w\W]*?)\)", node_info)[0]
                if not self.net_info[graph_id].get(fullname):
                    logger.info(f"Execution order file lack op fullname:{fullname}.")
                    continue
                self.net_info[graph_id][fullname] = [in_num, out_num]

    def parse_execution_order_files(self, execution_order_dir):
        graph_ids = set()
        for file in os.listdir(execution_order_dir):
            tmp = re.findall(r"ms_execution_order_graph_([0-9]+?).csv", file)
            if tmp:
                graph_ids.add(int(tmp[0]))
        for graph_id in graph_ids:
            self.net_info[graph_id] = {}
            with open(
                os.path.join(execution_order_dir, f"ms_execution_order_graph_{graph_id}.csv"),
                'r',
                encoding="utf-8"
            ) as f:
                content = f.readlines()
            for line in content[1:]:
                self.net_info[graph_id][line.strip()] = [0, 0]
