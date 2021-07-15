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

from mindspore._extends.parallel_compile.tbe_compiler.tbe_job_manager import TbeJobManager

MAX_COMPILE_SECONDS = 400
QUERY_INTERVAL = 10


def test_parallel_compilation(compile_job_json_str):
    with open("Initialize.info", 'r') as init_json_file:
        # Initialize
        init_job_json = json.load(init_json_file)
        tbe_compiler = TbeJobManager()
        res = tbe_compiler.job_handler(json.dumps(init_job_json))
        print("Initialize result:" + res)
        res_json = json.loads(res)
        for item in res_json["process_info"]:
            print("### LogLevel:" + str(item["level"]) + " " + item["message"])
        if res_json["status"] == "FAILED":
            print("Initialize Failed")
            return False

        print("\n################# Initialize Success #################\n")
        # Dispatch Compile Job
        res = tbe_compiler.job_handler(compile_job_json_str)
        print("Compile result:" + res)
        compile_result_json = json.loads(res)
        source_id = compile_result_json["source_id"]
        job_id = compile_result_json["job_id"]
        if compile_result_json["status"] != "RUNNING":
            # Process Finish Job
            print("Final Compile Result:{}".format(json.dumps(compile_result_json["result"])))
            print("Process Logs:")
            for item in compile_result_json["process_info"]:
                print("### LogLevel:" + str(item["level"]) + " " + item["message"])
            res_json = json.loads(res)
            if res_json["status"] == "FAILED":
                print("Compile Failed")
                return False
        else:
            # Process Running Job
            print("Query Running job with max compile seconds {}".format(MAX_COMPILE_SECONDS))
            job_id = job_id + 1
            query_job_json = dict()
            query_job_json["source_id"] = source_id
            query_job_json["job_id"] = job_id
            query_job_json["job_type"] = "Query"
            target_job = dict()
            target_job["source_id"] = source_id
            target_job["job_id"] = compile_result_json["job_id"]
            query_job_json["job_content"] = target_job
            repeat_time = 0
            while True:
                print("Dispatch a Query Job")
                res = tbe_compiler.job_handler(json.dumps(query_job_json))
                res_json = json.loads(res)
                print("Query result:{}".format(res))
                if res_json["status"] == "SUCCESS":
                    print("Target Job info :{}".format(res_json["result"]))
                    target_job = json.loads(res_json["result"])
                    if target_job["status"] == "RUNNING":
                        job_id = job_id + 1
                        query_job_json["job_id"] = query_job_json["job_id"] + 1
                        for item in res_json["process_info"]:
                            print("### LogLevel:" + str(item["level"]) + " " + item["message"])
                        repeat_time = repeat_time + 1
                        if repeat_time > MAX_COMPILE_SECONDS / QUERY_INTERVAL:
                            print("Query TimeOut")
                            print("\n################# Compile Failed #################\n")
                            break
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
                        else:
                            print("\n################# Compile Failed #################\n")
                        break
                else:
                    print("Final Compile Failed:{}".format(res))
                    print("Process Logs:")
                    for item in res_json["process_info"]:
                        print("### LogLevel:" + str(item["level"]) + " " + item["message"])
                    print("\n################# Compile Failed #################\n")
                    break

        # Finalize Job
        job_id = job_id + 1
        finalize_job_json = dict()
        finalize_job_json["source_id"] = source_id
        finalize_job_json["job_id"] = job_id
        finalize_job_json["job_type"] = "Finalize"
        finalize_job_json["job_content"] = dict()
        res = tbe_compiler.job_handler(json.dumps(finalize_job_json))
        print("Finalize result:{}".format(res))
        res_json = json.loads(res)
        if res_json["status"] == "Failed":
            print("\n################# Finalize Failed #################\n")
            return False
        print("\n################# Finalize Success #################\n")
        return True


if __name__ == "__main__":
    with open("op.info", "r") as op_json_file:
        op_json = json.load(op_json_file)
        test_parallel_compilation(json.dumps(op_json))
