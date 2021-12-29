import os
import argparse
import subprocess


def get_parser():
    """
    The script is used for simulating Multi-client Participation in Federated Learning in x86 environment.
    The description of the following parameters are as follows：
      - jarPath
        Specifies the path of the JAR package of the federated learning framework.
        Note, please make sure that only the JAR package is included in the path.
        For example, '--jarPath' is set to 'jarX86/mindspore-lite-java-flclient.jar', you need to make sure that
        the 'jarX86' folder contains only one JAR package 'mindspore-lite- java-flclient.jar'.
      - case_jarPath
        Specifies the path of the JAR package 'quick_start_flclient.jar' corresponding to the model script.
        Note, please make sure that only the JAR package is included in the path.
        For example, in the above reference script, '--case_jarPath' is set to 'case_jar/quick_start_flclient.jar',
        you need to make sure that the 'case_jar' folder contains only one JAR package 'quick_start_flclient.jar'.
      - train_dataset
        Specifies the root path of the training dataset.The sentiment classification task stores the
        training data (in .txt format) of each client. The LeNet image classification task stores the
        training files data.bin and label.bin of each client, for example, 'leaf-master/data/femnist/3500_clients_bin/'.
      - test_dataset
        Specifies the test dataset path. For LeNet image classification tasks, this parameter does not need to be set
        and the default value is null. For sentiment classification tasks, if this parameter is not set, validation is
        not performed during training.
      - vocal_file
        Specifies the path of the dictionary file for data preprocessing. Set the value to null for LeNet, or set the
        value to the actual absolute path for the sentiment classification task.
      - ids_file
        Specifies the path of the dictionary mapping ID. Set the value to null for LeNet, or set the value to the actual
        absolute path for the sentiment classification task.
      - path_regex
        It is used to set the data set path splicing method. The character ',' has been used in the 'run.py' script to
        splice the data set path, so there is no need to set this parameter.
      - flName
        Specifies the package path of model script used by federated learning.
      - train_model_path
        Specifies the training model path used for federated learning. The path is the directory where multiple .ms
        files copied in the preceding tutorial are stored, for example, 'ms/lenet'. The path must be an absolute path.
      - infer_model_path
        Specifies the path of the inference model used by federated learning. It is the absolute path of the model file
        in .ms format. For normal federated learning mode (training and inference use the same model, such as LeNet
        image classification task, supervised sentiment classification task), this parameter needs to be set to the
        same as trainModelPath; for mixed learning mode (training and inference use different models, and The cloud
        side also includes the training process), this parameter is set to the actual inference model path.
      - train_ms_name
        Set the same part of the multi-client training model file name. The model file name must be in the format
        '{train_ms_name}1.ms', '{train_ms_name}2.ms', '{train_ms_name}3.ms', etc.
      - infer_ms_name
        Set the same part of the multi-client inference model file name. The model file name needs to be in the format
        '{infer_ms_name}1.ms', '{infer_ms_name}2.ms', '{infer_ms_name}3.ms', etc., in the normal federated learning
        moden (such as LeNet image classification task ), it should be consistent with the 'train_ms_name'.
      - ssl_protocol
        Used to set the TLS protocol version used by the device-cloud HTTPS communication, a whitelist is set, and
        currently only 'TLSv1.3' or 'TLSv1.2' is supported. Only need to set it up in the HTTPS communication scenario.
      - deploy_env
        Used to set the deployment environment for federated learning, a whitelist is set, currently only 'x86',
        'android' are supported.
      - domain_name
        Used to set the url for device-cloud communication. Currently, https and http communication are supported, the
        corresponding formats are like as: https://......, http://......, and when 'if_use_elb' is set to true,
        the format must be: https://127.0.0.0 : 6666 or http://127.0.0.0 : 6666 , where '127.0.0.0' corresponds to
        the ip of the machine providing cloud-side services (corresponding to the cloud-side parameter
        '--scheduler_ip'), and '6666' corresponds to the cloud-side parameter '--fl_server_port'.
      - cert_path
        Specifies the absolute path of the self-signed root certificate path used for device-cloud HTTPS communication.
        When the deployment environment is 'x86' and the device-cloud uses a self-signed certificate for
        HTTPS communication authentication, this parameter needs to be set. The certificate must be consistent with
        the CA root certificate used to generate the cloud-side self-signed certificate to pass the verification.
        This parameter is used for non-Android scenarios.
      - if_use_elb
        Used for multi-server scenarios to set whether to randomly send client requests to different servers within a
        certain range. Setting to true means that the client will randomly send requests to a certain range of server
        addresses, and false means that the client's requests will be sent to a fixed server address. This parameter is
        used in non-Android scenarios, and the default value is false.
      - server_num
        Specifies the number of servers that the client can choose to connect to. When 'ifUseElb' is set to true, it
        can be set to be consistent with the 'server_num' parameter when the server is started on the cloud side.
        It is used to randomly select different servers to send information. This parameter is used in non-Android
        scenarios. The default value is 1.
      - task
        Specifies the type of the task to be started. 'train' indicates that a training task is started.
        'inference' indicates that multiple data inference tasks are started. 'getModel' indicates that the task for
        obtaining the cloud model is started. Other character strings indicate that the inference task of a single data
        record is started. The default value is 'train'. The initial model file (.ms file) is not trained.
        Therefore, you are advised to start the training task first. After the training is complete, start the inference
        task. (Note that the values of client_num in the two startups must be the same to ensure that the model file
        used by 'inference' is the same as that used by 'train'.)
      - thread_num
        Specifies the number of threads used in federated learning training and inference. The default value is 1.
      - cpu_bind_mode
        Specifies the cpu core that threads need to bind during federated learning training and inference. When this
        parameter is set to 'NOT_BINDING_CORE', it means that the core is not bound and will be automatically allocated
        by the system. 'BIND_LARGE_CORE' means that the big core is bound, and 'BIND_MIDDLE_CORE' means that the middle
        core is bound.
      - train_weight_name
        In the hybrid learning mode, set the weight name of the training model. Connect each name with '--name_regex'.
        In normal federated learning mode (such as LeNet image classification task), this parameter does not need to
        be set.
      - infer_weight_name
        In the hybrid learning mode, set the weight name of the inference model. Connect each name with '--name_regex'.
        In normal federated learning mode (such as LeNet image classification task), this parameter does not need to
        be set.
      - name_regex
        Concatenation string used when connecting '--train_weight_name' and '--infer_weight_name' .
      - server_mode
        Specifies the training mode for federated learning., set to 'FEDERATED_LEARNING' to represent normal federated
        learning mode (training and inference use the same model), set to 'HYBRID_TRAINING' to represent hybrid learning
        mode (training and inference use different models, and the cloud side also includes the training process).
        For example, the LeNet image classification task needs to be set to 'FEDERATED_LEARNING'.
      - batch_size
        Specifies the number of single-step training samples used in federated learning training and inference, that is,
        batch size. It needs to be consistent with the batch size of the input data of the model.
      - client_num
        Specifies the number of clients. The value must be the same as that of 'start_fl_job_cnt' when the server is
        started. This parameter is not required in actual scenarios.
    """

    parser = argparse.ArgumentParser(description="Run SyncFLJob.java case")
    parser.add_argument("--jarPath", type=str, default="jarX86/mindspore-lite-java-flclient.jar")
    parser.add_argument("--case_jarPath", type=str, default="case_jar/quick_start_flclient.jar")

    parser.add_argument("--train_dataset", type=str, default="leaf/data/femnist/3500_clients_bin/")
    parser.add_argument("--test_dataset", type=str, default="null")
    parser.add_argument("--vocal_file", type=str, default="null")
    parser.add_argument("--ids_file", type=str, default="null")
    parser.add_argument("--path_regex", type=str, default=",")

    parser.add_argument("--flName", type=str, default="com.mindspore.flclient.demo.lenet.LenetClient")

    parser.add_argument("--train_model_path", type=str, default="ms/lenet/")
    parser.add_argument("--infer_model_path", type=str, default="ms/lenet/")
    parser.add_argument("--train_ms_name", type=str, default="lenet_train")
    parser.add_argument("--infer_ms_name", type=str, default="lenet_train")

    parser.add_argument("--ssl_protocol", type=str, default="TLSv1.2")
    parser.add_argument("--deploy_env", type=str, default="x86")
    parser.add_argument("--domain_name", type=str, default="http://127.0.0.0:6668")
    parser.add_argument("--cert_path", type=str, default="cert/CARoot.pem")
    parser.add_argument("--use_elb", type=str, default="false")
    parser.add_argument("--server_num", type=int, default=1)
    parser.add_argument("--task", type=str, default="train")
    parser.add_argument("--thread_num", type=int, default=1)
    parser.add_argument("--cpu_bind_mode", type=str, default="NOT_BINDING_CORE")

    parser.add_argument("--train_weight_name", type=str, default="null")
    parser.add_argument("--infer_weight_name", type=str, default="null")
    parser.add_argument("--name_regex", type=str, default=",")
    parser.add_argument("--server_mode", type=str, default="FEDERATED_LEARNING")
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--client_num", type=int, default=0)
    return parser


args, _ = get_parser().parse_known_args()

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
train_ms_name = args.train_ms_name
infer_ms_name = args.infer_ms_name

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

client_num = args.client_num

users = os.listdir(train_dataset)


def get_client_data_path(data_root_path, user):
    use_path = os.path.join(data_root_path, user)
    bin_file_paths = os.listdir(use_path)

    train_data_path = ""
    train_label_path = ""
    train_batch_num = ""

    test_data_path = ""
    test_label_path = ""
    test_batch_num = ""

    for file in bin_file_paths:
        info = file.split(".")[0].split("_")
        if info[4] == "train" and info[5] == "data":
            train_data_path = os.path.join(use_path, file)
            train_batch_num = info[3]
        elif info[4] == "train" and info[5] == "label":
            train_label_path = os.path.join(use_path, file)
        elif info[4] == "test" and info[5] == "data":
            test_data_path = os.path.join(use_path, file)
            test_batch_num = info[3]
        elif info[4] == "test" and info[5] == "label":
            test_label_path = os.path.join(use_path, file)
    train_paths = train_data_path + "," + train_label_path
    test_paths = test_data_path + "," + test_label_path

    return train_paths, test_paths, test_paths, train_batch_num, test_batch_num


for i in range(client_num):
    fl_id = "f" + str(i)
    train_path, eval_path, infer_path = "", "", ""
    if "AlbertClient" in flName:
        train_path = os.path.join(train_dataset, str(i) + ".txt") + "," + vocal_file + "," + ids_file
        eval_path = test_dataset + "," + vocal_file + "," + ids_file
        infer_path = test_dataset + "," + vocal_file + "," + ids_file
    elif "LenetClient" in flName:
        train_path, eval_path, infer_path, _, _ = get_client_data_path(train_dataset, users[i])
        infer_ms_name = train_ms_name
    print("===========================")
    print("fl id: ", fl_id)
    print("train path: ", train_path)
    print("eval path: ", eval_path)
    print("infer path: ", infer_path)
    cmd_client = "execute_path=$(pwd) && self_path=$(dirname \"${script_self}\") && "
    cmd_client += "rm -rf ${execute_path}/client_" + str(i) + "/ &&"
    cmd_client += "mkdir ${execute_path}/client_" + str(i) + "/ &&"
    cmd_client += "cd ${execute_path}/client_" + str(i) + "/ || exit &&"

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

    cmd_client += train_model_path + train_ms_name + str(i) + ".ms" + " "
    print("train model path: ", train_model_path + train_ms_name + str(i) + ".ms" + " ")
    cmd_client += infer_model_path + infer_ms_name + str(i) + ".ms" + " "
    print("infer model path: ", infer_model_path + infer_ms_name + str(i) + ".ms" + " ")

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
    cmd_client += " > client" + ".log 2>&1 &"
    print(cmd_client)
    subprocess.call(['bash', '-c', cmd_client])
