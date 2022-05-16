import os
import socket
import inspect
import time
from abc import abstractmethod

FRAME_PKG_INIT_FLG = False


class BaseCase:
    # Common ENV
    fl_resource_path = os.getenv("FL_RESOURCE_PATH")
    x86_pkg_path = os.getenv("X86_PKG_PATH")
    fl_jdk_path = os.getenv("FL_JDK_PATH")
    fl_models_path = os.getenv("FL_MODELS_PATH")

    # ENV for Server
    script_path, _ = os.path.split(os.path.realpath(__file__))
    server_path = ""
    config_file_path = ""
    temp_path = os.path.join(script_path, "temp")
    scheduler_mgr_port = 6000
    scheduler_port = 60001
    fl_server_port = 60002
    scheduler_ip = socket.gethostbyname(socket.gethostname())
    server_num = 1
    worker_num = 0
    enable_ssl = "False"
    start_fl_job_threshold = 1
    client_batch_size = 32
    client_epoch_num = 1
    fl_iteration_num = 1
    start_fl_job_time_window = 30000
    update_model_time_window = 30000
    encrypt_type = "NOT_ENCRYPT"

    # ENV for Client
    lite_lib_path = os.path.join(script_path, "libs")
    ld_library_path = ""
    frame_jar_path = os.path.join(script_path, "frame_jar", "mindspore-lite-java-flclient.jar")
    case_jar_path = ""
    ssl_protocol = "TLSv1.2"
    deploy_env = "x86"
    domain_name = "http://{}:{}".format(scheduler_ip, fl_server_port)
    cert_path = os.path.join(fl_resource_path, "client/cert/CARoot.pem")
    client_num = 1
    use_elb = "false"
    thread_num = 1
    server_mode = "FEDERATED_LEARNING"

    def init_frame_pkg(self, case_jar):
        """
        unpack mindspore-lite-version-linux-x64.tar.gz get libs for fl client
        :return:
        """
        print("Class:{}, function:{}".format(self.__class__.__name__, inspect.stack()[1][3]), flush=True)
        global FRAME_PKG_INIT_FLG
        if FRAME_PKG_INIT_FLG is True:
            os.system('rm -rf {}/case_jar; mkdir -p {}/case_jar'.format(self.script_path, self.script_path))
            cp_case_jar = "cp {}/ci_jar/{} {}/case_jar".format(self.fl_resource_path, case_jar, self.script_path)
            os.system(cp_case_jar)
            return
        FRAME_PKG_INIT_FLG = True
        os.system('rm -rf {}'.format(self.case_jar_path))
        os.system('rm -rf {}'.format(self.frame_jar_path))

        if os.path.exists(self.temp_path):
            os.system('rm -rf {}/mindspore-lite*'.format(self.temp_path))
        os.system('mkdir -p {}'.format(self.temp_path))
        if os.path.exists(self.lite_lib_path):
            os.system('rm -rf {}'.format(self.lite_lib_path))
        os.system('mkdir -p {}'.format(self.lite_lib_path))
        unpack_cmd = "tar -zxf {}/*.tar.gz -C {}".format(self.x86_pkg_path, self.temp_path)
        os.system(unpack_cmd)
        lib_cp_cmd = "cp {}/mindspore-lite*/runtime/lib/* {}".format(self.temp_path, self.lite_lib_path)
        os.system(lib_cp_cmd)
        lib_cp_cmd = "cp {}/mindspore-lite*/runtime/third_party/libjpeg-turbo/lib/* {}" \
            .format(self.temp_path, self.lite_lib_path)
        os.system(lib_cp_cmd)

        os.system('rm -rf {}/frame_jar; mkdir -p {}/frame_jar'.format(self.script_path, self.script_path))
        cp_frame_jar = "cp {}/mindspore-lite-java-flclient.jar {}/frame_jar" \
            .format(self.lite_lib_path, self.script_path)
        os.system(cp_frame_jar)
        os.system('rm -rf {}/case_jar; mkdir -p {}/case_jar'.format(self.script_path, self.script_path))
        cp_case_jar = "cp {}/ci_jar/{} {}/case_jar".format(self.fl_resource_path, case_jar, self.script_path)
        os.system(cp_case_jar)

    def init_env(self, relative_model_path, case_jar):
        """
        :param relative_model_path: relative directory of model path
        :param case_jar: jar of case
        :return:
        """
        print("Class:{}, function:{}".format(self.__class__.__name__, inspect.stack()[1][3]), flush=True)

        # ENV for Server
        self.server_path = os.path.realpath(os.path.join(self.script_path, relative_model_path))
        self.config_file_path = os.path.join(self.server_path, "config.json")

        # ENV for Client
        env_library_path = os.getenv("LD_LIBRARY_PATH", default="")
        self.ld_library_path = env_library_path + ':' + self.lite_lib_path
        self.case_jar_path = os.path.join(self.script_path, "case_jar", case_jar)

        # prepare pkg
        self.init_frame_pkg(case_jar)

    def wait_client_exit(self, out_time):
        # wait client exit
        query_state_cmd = "ps -ef|grep mindspore-lite-java-flclient |grep -v grep | wc -l"
        finish_flg = False
        loop_times = 0
        while loop_times < out_time:
            result = os.popen(query_state_cmd)
            info = result.read()
            result.close()
            if int(info) == 0:
                finish_flg = True
                break
            time.sleep(1)
            loop_times = loop_times + 1
        # print logs while exception
        if not finish_flg:
            os.system("cat {}/../client_script/client_train0/*".format(self.server_path))
            assert finish_flg is True

    def check_client_result(self, out_time):
        self.wait_client_exit(out_time)
        self.check_client_log()

    @abstractmethod
    def check_client_log(self):
        print("The subclass must impl check_client_log")
        assert False

    def wait_cluster_ready(self, out_time=30):
        print("Class:{}, function:{}".format(self.__class__.__name__, inspect.stack()[1][3]), flush=True)
        # wait server status to ready
        query_state_cmd = "curl -k http://127.0.0.1:{}/state".format(self.scheduler_mgr_port)

        ready_flg = False
        loop_times = 0
        while loop_times < out_time:
            result = os.popen(query_state_cmd)
            info = result.read()
            result.close()
            if info.find('CLUSTER_READY') != -1:
                ready_flg = True
                break
            time.sleep(1)
            loop_times = loop_times + 1
        # print logs while exception
        if not ready_flg:
            os.system("cat {}/scheduler/scheduler.log".format(self.server_path))
            os.system("cat {}/server_0/server.log".format(self.server_path))
            assert ready_flg

    def clear_logs(self):
        clear_server_log = "rm -rf {}/server_0".format(self.server_path)
        clear_scheduler_log = "rm -rf {}/scheduler".format(self.server_path)
        clear_train_log = "rm -rf {}/../client_script/client_train0".format(self.server_path)
        clear_infer_log = "rm -rf {}/../client_script/client_inference0".format(self.server_path)
        os.system(clear_server_log)
        os.system(clear_scheduler_log)
        os.system(clear_train_log)
        os.system(clear_infer_log)
