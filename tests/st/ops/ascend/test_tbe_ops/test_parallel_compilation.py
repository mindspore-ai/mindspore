# Copyright 2021 Huawei Technologies Co., Ltd
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
import json
import time
import os
from mindspore._extends.parallel_compile.tbe_compiler.tbe_job_manager import TbeJobManager

MAX_COMPILE_SECONDS = 400
QUERY_INTERVAL = 10


class Compiler:

    def __init__(self):
        self.tbe_compiler = TbeJobManager()

    def initialize(self):
        with open("Initialize.info", 'r') as init_json_file:
            init_job_json = json.load(init_json_file)
            res = self.tbe_compiler.job_handler(json.dumps(init_job_json))
            print("Initialize result:" + res)
            res_json = json.loads(res)
            for item in res_json["process_info"]:
                print("### LogLevel:" + str(item["level"]) + " " + item["message"])

            if res_json["status"] == "FAILED":
                print("Initialize Failed")
                return False

            kernel_meta_temp_dir = init_job_json["job_content"]["SocInfo"]["kernel_meta_temp_dir"]
            if not os.path.exists(kernel_meta_temp_dir):
                os.mkdir(kernel_meta_temp_dir)

            print("\n################# Initialize Success #################\n")
            return True

    @staticmethod
    def process_finish_job(compile_result_json):
        print("Final Compile Result:{}".format(json.dumps(compile_result_json["result"])))
        print("Process Logs:")
        for item in compile_result_json["process_info"]:
            print("### LogLevel:" + str(item["level"]) + " " + item["message"])
        if compile_result_json["status"] == "FAILED":
            print("Compile Failed")
            return False

        print("\n################# Compile Success #################\n")
        return True

    def get_query_json(self, compile_result_json):
        job_id = self.job_id + 1
        query_job_json = {"source_id": self.source_id,
                          "job_id": job_id,
                          "job_type": "Query",
                          "job_content": {
                              "source_id": self.source_id,
                              "job_id": compile_result_json["job_id"]}
                          }
        return query_job_json

    def process_running_job(self, compile_result_json):
        print("Query Running job with max compile seconds {}".format(MAX_COMPILE_SECONDS))
        query_job_json = self.get_query_json(compile_result_json)
        repeat_time = 0
        while True:
            print("Dispatch a Query Job")
            res = self.tbe_compiler.job_handler(json.dumps(query_job_json))
            res_json = json.loads(res)
            print("Query result:{}".format(res))
            if res_json["status"] == "SUCCESS":
                print("Target Job info :{}".format(res_json["result"]))
                target_job = json.loads(res_json["result"])
                if target_job["status"] == "RUNNING":
                    self.job_id = self.job_id + 1
                    query_job_json["job_id"] = query_job_json["job_id"] + 1
                    for item in res_json["process_info"]:
                        print("### LogLevel:" + str(item["level"]) + " " + item["message"])
                    repeat_time = repeat_time + 1
                    if repeat_time > MAX_COMPILE_SECONDS / QUERY_INTERVAL:
                        print("Query TimeOut")
                        print("\n################# Compile Failed #################\n")
                        return False
                    print("Sleep {} seconds".format(QUERY_INTERVAL))
                    time.sleep(QUERY_INTERVAL)
                else:
                    print("\n $$$Final Compile Result:{}\n".format(json.dumps(target_job["result"])))
                    print("Process Logs:")
                    for item in res_json["process_info"]:
                        print("### LogLevel:" + str(item["level"]) + " " + item["message"])
                    print("Target Job Process Logs:")
                    for item in target_job["process_info"]:
                        print("### LogLevel:" + str(item["level"]) + " " + item["message"])
                    if target_job["status"] == "SUCCESS":
                        print("\n################# Compile Success #################\n")
                        return True

                    print("\n################# Compile Failed #################\n")
                    return False

            else:
                print("Final Compile Failed:{}".format(res))
                print("Process Logs:")
                for item in res_json["process_info"]:
                    print("### LogLevel:" + str(item["level"]) + " " + item["message"])
                print("\n################# Compile Failed #################\n")
                return False

    def compile(self):
        with open("op.info", "r") as op_json_file:
            op_json = json.load(op_json_file)
            res = self.tbe_compiler.job_handler(json.dumps(op_json))
            print("Compile result:" + res)
            compile_result_json = json.loads(res)
            self.source_id = compile_result_json["source_id"]
            self.job_id = compile_result_json["job_id"]
            if compile_result_json["status"] != "RUNNING":
                return Compiler.process_finish_job(compile_result_json)

            return self.process_running_job(compile_result_json)

    def finilize(self):
        job_id = self.job_id + 1
        finalize_job_json = {"source_id": self.source_id,
                             "job_id": job_id,
                             "job_type": "Finalize",
                             "job_content":
                                 {"SocInfo":
                                      {"op_debug_level": "3",
                                       "op_debug_dir": "./rank_0"
                                       }
                                  }
                             }
        res = self.tbe_compiler.job_handler(json.dumps(finalize_job_json))
        print("Finalize result:{}".format(res))
        res_json = json.loads(res)
        if res_json["status"] == "FAILED":
            print("\n################# Finalize Failed #################\n")
            return False

        print("\n################# Finalize Success #################\n")
        return True


def test_parallel_compilation():
    """
    Feature: Test TBE Python Parallel compiler
    Description: python debug script for tbe python compiler
    Expectation: success with correct Initialize.info and op.info
    """
    compiler = Compiler()
    if not compiler.initialize():
        return False

    if not compiler.compile():
        return False

    if not compiler.finilize():
        return False

    print("Test Pass")
    return True
