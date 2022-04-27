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
"""
Function:
    Use to control the federated learning cluster
Usage:
    python fl_restful_tool.py [http_type] [ip] [port] [request_name] [server_num] [instance_param] [metrics_file_path]
"""
import argparse
import json
import os
import warnings
from enum import Enum
import requests


class Status(Enum):
    """
    Response Status
    """
    SUCCESS = "0"
    FAILED = "1"


class Restful(Enum):
    """
    Define restful interface constant
    """
    SCALE = "scale"
    SCALE_OUT = "scaleout"
    SCALE_IN = "scalein"
    NODES = "nodes"
    GET_INSTANCE_DETAIL = "getInstanceDetail"
    NEW_INSTANCE = "newInstance"
    QUERY_INSTANCE = "queryInstance"
    ENABLE_FLS = "enableFLS"
    DISABLE_FLS = "disableFLS"
    STATE = "state"
    SCALE_OUT_ROLLBACK = "scaleoutRollback"


warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument("--http_type", type=str, default="http", help="http or https")
parser.add_argument("--ip", type=str, default="127.0.0.1")
parser.add_argument("--port", type=int, default=6666)
parser.add_argument("--request_name", type=str, default="")

parser.add_argument("--server_num", type=int, default=0)
parser.add_argument("--instance_param", type=str, default="")
parser.add_argument("--metrics_file_path", type=str, default="/opt/huawei/mindspore/hybrid_albert/metrics.json")

args, _ = parser.parse_known_args()
http_type = args.http_type
ip = args.ip
port = args.port
request_name = args.request_name
server_num = args.server_num
instance_param = args.instance_param
metrics_file_path = args.metrics_file_path

headers = {'Content-Type': 'application/json'}
session = requests.Session()
base_url = http_type + "://" + ip + ":" + str(port) + "/"


def call_scale():
    """
    call cluster scale out or scale in
    """
    if server_num == 0:
        return process_self_define_json(Status.FAILED.value, "error. server_num is 0")

    node_ids = json.loads(call_nodes())["result"]
    cluster_abstract_node_num = len(node_ids)
    if cluster_abstract_node_num == 0:
        return process_self_define_json(Status.FAILED.value, "error. cluster abstract node num is 0")

    cluster_server_node_num = 0
    cluster_worker_node_num = 0
    cluster_server_node_base_name = ''
    for i in range(0, cluster_abstract_node_num):
        if node_ids[i]['role'] == 'WORKER':
            cluster_worker_node_num = cluster_worker_node_num + 1
        elif node_ids[i]['role'] == 'SERVER':
            cluster_server_node_num = cluster_server_node_num + 1
            cluster_server_node_name = str(node_ids[i]['nodeId'])
            index = cluster_server_node_name.rindex('-')
            cluster_server_node_base_name = cluster_server_node_name[0:index]
        else:
            pass
    if cluster_server_node_num == server_num:
        return process_self_define_json(Status.FAILED.value, "error. cluster server num is same with server_num.")
    if cluster_server_node_num > server_num:
        scale_in_len = cluster_server_node_num - server_num
        scale_in_node_ids = []
        for index in range(cluster_server_node_num - scale_in_len, cluster_server_node_num):
            scale_in_node_name = cluster_server_node_base_name + "-" + str(index)
            scale_in_node_ids.append(scale_in_node_name)
        return call_scalein(scale_in_node_ids)
    return call_scaleout(server_num - cluster_server_node_num)


def call_scaleout(scale_out_server_num, scale_out_worker_num=0):
    """
    call scaleout
    """
    url = base_url + Restful.SCALE_OUT.value
    data = {"server_num": scale_out_server_num, "worker_num": scale_out_worker_num}
    res = session.post(url, headers=headers, verify=False, data=json.dumps(data))
    res_json = json.loads(res.text)
    if res_json["code"] == Status.FAILED.value:
        return process_self_define_json(Status.FAILED.value, res_json["error_message"])

    result = "scale out server num is " + str(scale_out_server_num)
    return process_result_json(Status.SUCCESS.value, res_json["message"], result)


def call_scaleout_rollback():
    """
    call scaleout rollback
    """
    url = base_url + Restful.SCALE_OUT_ROLLBACK.value
    res = session.get(url, verify=False)
    res_json = json.loads(res.text)
    if res_json["code"] == Status.FAILED.value:
        return process_self_define_json(Status.FAILED.value, res_json["error_message"])
    return process_self_define_json(Status.SUCCESS.value, res_json["message"])


def call_scalein(scale_in_node_ids):
    """
    call cluster to scale in
    """
    if not scale_in_node_ids:
        return process_self_define_json(Status.FAILED.value, "error. node ids is empty.")

    url = base_url + Restful.SCALE_IN.value
    data = {"node_ids": scale_in_node_ids}
    res = session.post(url, headers=headers, verify=False, data=json.dumps(data))
    res_json = json.loads(res.text)
    if res_json["code"] == Status.FAILED.value:
        return process_self_define_json(Status.FAILED.value, res_json["error_message"])
    result = "scale in node ids is " + str(scale_in_node_ids)
    return process_result_json(Status.SUCCESS.value, res_json["message"], result)


def call_nodes():
    """
    get nodes info
    """
    url = base_url + Restful.NODES.value
    res = session.get(url, verify=False)
    res_json = json.loads(res.text)
    if res_json["code"] == Status.FAILED.value:
        return process_self_define_json(Status.FAILED.value, res_json["error_message"])
    return process_result_json(Status.SUCCESS.value, res_json["message"], res_json["nodeIds"])


def call_get_instance_detail():
    """
    get cluster instance detail
    """
    if not os.path.exists(metrics_file_path):
        return process_self_define_json(Status.FAILED.value, "error. metrics file is not existed.")

    ans_json_obj = {}
    metrics_auc_list = []
    metrics_loss_list = []
    iteration_execution_time_list = []
    client_visited_info_list = []

    with open(metrics_file_path, 'r') as f:
        metrics_list = f.readlines()

    if not metrics_list:
        return process_self_define_json(Status.FAILED.value, "error. metrics file has no content")

    for metrics in metrics_list:
        json_obj = json.loads(metrics)
        iteration_execution_time_list.append(json_obj['iterationExecutionTime'])
        client_visited_info_list.append(json_obj['clientVisitedInfo'])
        metrics_auc_list.append(json_obj['metricsAuc'])
        metrics_loss_list.append(json_obj['metricsLoss'])

    last_metrics = metrics_list[len(metrics_list) - 1]
    last_metrics_obj = json.loads(last_metrics)

    ans_json_obj["code"] = Status.SUCCESS.value
    ans_json_obj["describe"] = "get instance metrics detail successful."
    ans_json_obj["result"] = {}
    ans_json_result = ans_json_obj.get("result")
    ans_json_result['currentIteration'] = last_metrics_obj['currentIteration']
    ans_json_result['flIterationNum'] = last_metrics_obj['flIterationNum']
    ans_json_result['flName'] = last_metrics_obj['flName']
    ans_json_result['instanceStatus'] = last_metrics_obj['instanceStatus']
    ans_json_result['iterationExecutionTime'] = iteration_execution_time_list
    ans_json_result['clientVisitedInfo'] = client_visited_info_list
    ans_json_result['metricsAuc'] = metrics_auc_list
    ans_json_result['metricsLoss'] = metrics_loss_list

    return json.dumps(ans_json_obj)


def call_new_instance():
    """
    call cluster new instance
    """
    if instance_param == "":
        return process_self_define_json(Status.FAILED.value, "error. instance_param is empty.")
    instance_param_list = instance_param.split(sep=",")
    instance_param_json_obj = {}

    url = base_url + Restful.NEW_INSTANCE.value
    for cur in instance_param_list:
        pair = cur.split(sep="=")
        instance_param_json_obj[pair[0]] = float(pair[1])

    data = json.dumps(instance_param_json_obj)
    res = session.post(url, verify=False, data=data)
    res_json = json.loads(res.text)
    if res_json["code"] == Status.FAILED.value:
        return process_self_define_json(Status.FAILED.value, res_json["error_message"])
    return process_self_define_json(Status.SUCCESS.value, res_json["message"])


def call_query_instance():
    """
    query cluster instance
    """
    url = base_url + Restful.QUERY_INSTANCE.value
    res = session.post(url, verify=False)
    res_json = json.loads(res.text)
    if res_json["code"] == Status.FAILED.value:
        return process_self_define_json(Status.FAILED.value, res_json["error_message"])
    return process_result_json(Status.SUCCESS.value, res_json["message"], res_json["result"])


def call_enable_fls():
    """
    enable cluster fls
    """
    url = base_url + Restful.ENABLE_FLS.value
    res = session.post(url, verify=False)
    res_json = json.loads(res.text)
    if res_json["code"] == Status.FAILED.value:
        return process_self_define_json(Status.FAILED.value, res_json["error_message"])
    return process_self_define_json(Status.SUCCESS.value, res_json["message"])


def call_disable_fls():
    """
    disable cluster fls
    """
    url = base_url + Restful.DISABLE_FLS.value
    res = session.post(url, verify=False)
    res_json = json.loads(res.text)
    if res_json["code"] == Status.FAILED.value:
        return process_self_define_json(Status.FAILED.value, res_json["error_message"])
    return process_self_define_json(Status.SUCCESS.value, res_json["message"])


def call_state():
    """
    get cluster state
    """
    url = base_url + Restful.STATE.value
    res = session.get(url, verify=False)
    res_json = json.loads(res.text)
    if res_json["code"] == Status.FAILED.value:
        return process_self_define_json(Status.FAILED.value, res_json["error_message"])
    result = res_json['cluster_state']
    return process_result_json(Status.SUCCESS.value, res_json["message"], result)


def process_result_json(code, describe, result):
    """
    process result json
    """
    result_dict = {"code": code, "describe": describe, "result": result}
    return json.dumps(result_dict)


def process_self_define_json(code, describe):
    """
    process self define json
    """
    result_dict = {"code": code, "describe": describe}
    return json.dumps(result_dict)


if __name__ == '__main__':
    if request_name == Restful.SCALE.value:
        print(call_scale())

    elif request_name == Restful.NODES.value:
        print(call_nodes())

    elif request_name == Restful.GET_INSTANCE_DETAIL.value:
        print(call_get_instance_detail())

    elif request_name == Restful.NEW_INSTANCE.value:
        print(call_new_instance())

    elif request_name == Restful.QUERY_INSTANCE.value:
        print(call_query_instance())

    elif request_name == Restful.ENABLE_FLS.value:
        print(call_enable_fls())

    elif request_name == Restful.DISABLE_FLS.value:
        print(call_disable_fls())

    elif request_name == Restful.STATE.value:
        print(call_state())

    elif request_name == Restful.SCALE_OUT_ROLLBACK.value:
        print(call_scaleout_rollback())

    else:
        print(process_self_define_json(1, "error. request_name is not found!"))
