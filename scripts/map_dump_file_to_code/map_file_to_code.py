# Copyright 2020 Huawei Technologies Co., Ltd
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
"""map_file_to_code"""

import os
import argparse


class ParseIrInfo:
    """
    Parse and return the operation info from ir file.
    """

    def __init__(self, ir_file):
        self.no_in_file_operation = []
        self.ir_file_path = self.ir_path_parse(ir_file)
        self.operation_info_dict = self.ir_info_parse()

    def __len__(self):
        return len(self.operation_info_dict)

    def ir_path_parse(self, ir_file):
        """
        parse the map file path.
        """
        if ir_file == "":
            print("[WARNING] No graph_path parameter, use current path as graph path.")
            ir_file = os.path.abspath(os.path.dirname(__file__))

        map_ir_file = ""
        file_size = 0
        map_ir_filename = "trace_code_graph"
        for filename in os.listdir(os.path.join(ir_file)):
            if map_ir_filename not in filename:
                continue
            tmp_file = os.path.join(ir_file, filename)
            tmp_file_size = os.path.getsize(tmp_file)
            if tmp_file_size >= file_size:
                file_size = tmp_file_size
                map_ir_file = tmp_file
        if map_ir_file == "":
            exit("[ERROR] Please set \"save_graphs=True\" in context to save {} file!".format(map_ir_filename))
        return map_ir_file

    def ir_info_parse(self):
        """
        parse the ir file and save code line corresponding to the operator
        """

        all_op_info_dict = {}  # recode all operation info
        single_op_info_dict = {}  # recode single operation info
        op_start_char_flag = False  # Start operator fragment
        op_end_char_flag = False  # End of operator fragment
        op_start_info_num = 0  # Accumulate the num to recode operation
        operation_line = 0  # The line number of the operator
        op_start_line_num = 0  # The line number of starting operator information
        op_start_info_flag = False  # Start operator information

        with open(self.ir_file_path, 'r+') as file:
            txt_context_list = file.readlines()

        for line_num, txt_context in enumerate(txt_context_list):
            txt_context = txt_context.strip()
            # Start operator fragment
            if txt_context.endswith(") {"):
                op_start_char_flag = True
                op_end_char_flag = False

            # End of operator fragment
            if txt_context == "}":
                op_end_char_flag = True

            # Determine whether it is operator information
            if txt_context.startswith("%") and ") = " in txt_context and txt_context[1].isdigit():
                op_start_info_flag = True
                op_start_line_num = line_num
                op_start_info_num += 1
                single_op_info_dict = {"in_file": []}

            # Judge and start to recode operation info
            if op_start_char_flag and not op_end_char_flag and op_start_info_flag and line_num != op_start_line_num:
                if "-op" in txt_context and txt_context.split("-op")[-1].split(")")[0].isdigit():
                    single_op_info_dict["origin_op_name"] = txt_context.split("-op")[0].split("/")[-1]
                    single_op_info_dict["op_name"] = txt_context.split("-op")[0].split("/")[-1].lower()
                    single_op_info_dict["op_num"] = "op" + txt_context.split("-op")[-1].split(")")[0]
                    operation_line = line_num
                if "In file" in txt_context:
                    in_file_info = txt_context.split("#")[-1].strip().rstrip("/")
                    single_op_info_dict["in_file"].append(in_file_info)
                if line_num - operation_line == 1 and "In file" not in txt_context and "op_num" in single_op_info_dict:
                    self.no_in_file_operation.append(single_op_info_dict["op_num"])
                    op_start_info_flag = False
                all_op_info_dict[op_start_info_num] = single_op_info_dict

        return all_op_info_dict


class MapOperationToLine:
    """
    to show operation info
    """
    def __init__(self, dump_op, ir_info_dict):
        self.dump_op = dump_op
        self.ir_info_dict = ir_info_dict

    def show_operator_info(self):
        """
        find operator
        """
        origin_dump_op_name = self.dump_op.split("-")[0]
        dump_op_name = origin_dump_op_name.lower()
        dump_op_num = self.dump_op.split("-")[-1]
        for _, op_info in self.ir_info_dict.items():
            if op_info["op_num"] == dump_op_num and op_info["in_file"] is not None:
                if dump_op_name in (dump_op_num, op_info["op_name"]):
                    if not op_info["in_file"]:
                        print("[WARNING] Cannot find {}'s source code in ir file.".format(op_info["origin_op_name"]))
                        return False
                    print("[INFO] Find operation '{}'.".format(op_info["origin_op_name"]))
                    for line in op_info["in_file"]:
                        print("       {}".format(line.split("  ")[0]))
                        print("               {}".format(line.split("  ")[-1]))
                    return True
        print("[WARNING] Cannot find operation {}'s in ir file.".format(origin_dump_op_name))
        return False


def start_find(dump_op, map_code_file):
    """
    start find error operation in code.
    """

    print("[INFO] Start to map the dump file to source code.")
    ir_op_info_dict = ParseIrInfo(map_code_file).operation_info_dict
    MapOperationToLine(dump_op, ir_op_info_dict).show_operator_info()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find the dump operator in the user code')
    parser.add_argument('--graph_path', '-p', type=str, default="", help='Save graph files path (option)')
    parser.add_argument('--dump_op', '-o', type=str, default="", required=True,
                        help="Dump operator id, case insensitive, such as 'op3352'.")
    args_opt = parser.parse_args()
    start_find(args_opt.dump_op, args_opt.graph_path)
