import os
import argparse
import subprocess

parser = argparse.ArgumentParser(description="Run TestClient.java case")
parser.add_argument("--jarPath", type=str, default="mindspore-lite-java-flclient.jar")  # must be absolute path
parser.add_argument("--case_jarPath", type=str, default="case_jar/flclient_models.jar")  # must be absolute path
parser.add_argument("--train_dataset", type=str, default="client/train.txt")  # must be absolute path
parser.add_argument("--test_dataset", type=str, default="client/eval.txt")  # must be absolute path
parser.add_argument("--vocal_file", type=str, default="client/vocab.txt")  # must be absolute path
parser.add_argument("--ids_file", type=str, default="client/vocab_map_ids.txt")  # must be absolute path
parser.add_argument("--path_regex", type=str, default=",")

parser.add_argument("--flName", type=str, default="com.mindspore.flclient.demo.adbert.AdBertClient")

parser.add_argument("--train_model_path", type=str,
                    default="client/train/albert_ad_train.mindir.ms")  # must be absolute path of .ms files
parser.add_argument("--infer_model_path", type=str,
                    default="client/train/albert_ad_infer.mindir.ms")  # must be absolute path of .ms files

parser.add_argument("--ssl_protocol", type=str, default="TLSv1.2")
parser.add_argument("--deploy_env", type=str, default="x86")
parser.add_argument("--domain_name", type=str, default="https://10.113.216.106:6668")
parser.add_argument("--cert_path", type=str, default="certs/https_signature_certificate/client/CARoot.pem")
parser.add_argument("--use_elb", type=str, default="false")
parser.add_argument("--server_num", type=int, default=1)
parser.add_argument("--task", type=str, default="train")
parser.add_argument("--thread_num", type=int, default=1)
parser.add_argument("--cpu_bind_mode", type=str, default="NOT_BINDING_CORE")

parser.add_argument("--train_weight_name", type=str, default="null")
parser.add_argument("--infer_weight_name", type=str, default="null")
parser.add_argument("--name_regex", type=str, default=",")
parser.add_argument("--server_mode", type=str, default="FEDERATED_LEARNING")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--input_shape", type=str, default="null")

parser.add_argument("--client_num", type=int, default=0)

args, _ = parser.parse_known_args()

jarPath = args.jarPath
case_jarPath = args.case_jarPath

train_dataset = args.train_dataset
test_dataset = args.test_dataset
vocal_file = args.vocal_file
ids_file = args.ids_file
path_regex = args.path_regex

flName = args.flName

train_model_path = args.train_model_path
infer_model_path = args.infer_model_path

ssl_protocol = args.ssl_protocol
deploy_env = args.deploy_env
domain_name = args.domain_name
cert_path = args.cert_path
use_elb = args.use_elb
server_num = args.server_num
task = args.task
thread_num = args.thread_num
cpu_bind_mode = args.cpu_bind_mode

train_weight_name = args.train_weight_name
infer_weight_name = args.infer_weight_name
name_regex = args.name_regex
server_mode = args.server_mode
batch_size = args.batch_size
input_shape = args.input_shape

client_num = args.client_num


def get_client_data_path(user_path):
    bin_file_paths = os.listdir(user_path)
    train_data_path = ""
    train_label_path = ""

    test_data_path = ""
    test_label_path = ""
    for file in bin_file_paths:
        info = file.split(".")[0].split("_")
        if info[4] == "train" and info[5] == "data":
            train_data_path = os.path.join(user_path, file)
        elif info[4] == "train" and info[5] == "label":
            train_label_path = os.path.join(user_path, file)
        elif info[4] == "test" and info[5] == "data":
            test_data_path = os.path.join(user_path, file)
        elif info[4] == "test" and info[5] == "label":
            test_label_path = os.path.join(user_path, file)
    train_data_label = train_data_path + "," + train_label_path
    test_path = test_data_path + "," + test_label_path

    return train_data_label, test_path, test_path


for i in range(client_num):
    flId = "f" + str(i)
    train_path, eval_path, infer_path = "", "", ""
    if "AlbertClient" in flName:
        print("AlBertClient")
        train_path = train_dataset + "," + vocal_file + "," + ids_file
        eval_path = test_dataset + "," + vocal_file + "," + ids_file
        infer_path = test_dataset + "," + vocal_file + "," + ids_file
    elif "LenetClient" in flName:
        print("LenetClient")
        train_path, eval_path, infer_path = get_client_data_path(train_dataset)
    elif "AdBertClient" in flName:
        print("AdBertClient")
        train_path = train_dataset + "," + vocal_file + "," + ids_file
        eval_path = test_dataset + "," + vocal_file + "," + ids_file
        infer_path = test_dataset + "," + vocal_file + "," + ids_file
    elif "VaeClient" in flName:
        print("VaeClient")
        train_path = train_dataset
        eval_path = train_dataset
        infer_path = train_dataset
    elif "TagClient" in flName:
        print("TagClient")
        train_path = os.path.join(train_dataset, "sample_input_outputs_2022_02_01.csv")  # 注意确认此处csv为所需使用的csv文件名
        eval_path = os.path.join(train_dataset, "sample_input_outputs_2022_02_01.csv")  # 注意确认此处csv为所需使用的csv文件名
        infer_path = os.path.join(train_dataset, "sample_input_outputs_2022_02_01.csv")  # 注意确认此处csv为所需使用的csv文件名
    else:
        print("the flname is error")
    print("===========================")
    print("fl id: ", flId)
    print("train path: ", train_path)
    print("eval path: ", eval_path)
    print("infer path: ", infer_path)
    cmd_client = "execute_path=$(pwd) && self_path=$(dirname \"${script_self}\") && "
    cmd_client += "rm -rf ${execute_path}/client_" + task + str(i) + "/ &&"
    cmd_client += "mkdir ${execute_path}/client_" + task + str(i) + "/ &&"
    cmd_client += "cd ${execute_path}/client_" + task + str(i) + "/ || exit &&"

    jar_dir_path = os.path.abspath(os.path.dirname(jarPath))
    case_dir_path = os.path.abspath(os.path.dirname(case_jarPath))
    model_path = "--module-path=" + jar_dir_path + ":" + case_dir_path
    cmd_client += "java " + model_path + " -jar "
    cmd_client += jarPath + " "
    cmd_client += train_path + " "
    cmd_client += eval_path + " "
    cmd_client += infer_path + " "
    cmd_client += path_regex + " "
    cmd_client += flName + " "

    cmd_client += train_model_path + " "
    print("train model path: ", train_model_path)
    cmd_client += infer_model_path + " "
    print("infer model path: ", infer_model_path)

    cmd_client += ssl_protocol + " "
    cmd_client += deploy_env + " "
    cmd_client += domain_name + " "
    cmd_client += cert_path + " "
    cmd_client += use_elb + " "
    cmd_client += str(server_num) + " "
    cmd_client += task + " "
    cmd_client += str(thread_num) + " "
    cmd_client += cpu_bind_mode + " "
    cmd_client += train_weight_name + " "
    cmd_client += infer_weight_name + " "
    cmd_client += name_regex + " "
    cmd_client += server_mode + " "
    cmd_client += str(batch_size) + " "
    cmd_client += input_shape + " "
    cmd_client += " > client-" + task + ".log 2>&1 &"
    print(cmd_client)
    subprocess.call(['bash', '-c', cmd_client])
