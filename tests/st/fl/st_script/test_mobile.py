import inspect
import os
import pytest
from base_case import BaseCase

FLNAME = "com.mindspore.flclient.demo.lenet.LenetClient"


@pytest.mark.fl_cluster
class TestMobileTrain(BaseCase):
    train_model_path = os.path.join(BaseCase.fl_resource_path, "client/ms/lenet_train.mindir0.ms")
    infer_model_path = os.path.join(BaseCase.fl_resource_path, "client/ms/lenet_train.mindir0.ms")
    train_dataset = os.path.join(BaseCase.fl_resource_path, "client/data/f0049_32")
    test_dataset = os.path.join(BaseCase.fl_resource_path, "client/data/f0049_32")

    def setup_method(self):
        """
        Run before every test case
        :return:
        """
        self.init_env("../mobile", "quick_start_flclient.jar")
        finish_cluster_cmd = "cd {}; python finish_mobile.py --scheduler_port={}" \
            .format(self.server_path, self.scheduler_port)
        finish_client_cmd = "cd {}/../client_script; " \
                            "python fl_client_finish.py --kill_tag=mindspore-lite-java-flclient" \
            .format(self.server_path)
        os.system(finish_client_cmd)
        os.system(finish_cluster_cmd)

    def teardown_method(self):
        """
        Run after every test case
        :return:
        """
        finish_client_cmd = "cd {}/../client_script; " \
                            "python fl_client_finish.py --kill_tag=mindspore-lite-java-flclient" \
            .format(self.server_path)
        finish_cluster_cmd = "cd {}; python finish_mobile.py --scheduler_port={}" \
            .format(self.server_path, self.scheduler_port)
        os.system(finish_cluster_cmd)
        os.system(finish_client_cmd)
        self.clear_logs()

    def start_scheduler(self):
        """
        Start scheduler
        :return:
        """
        start_scheduler_cmd = "cd {}; python run_mobile_sched.py --scheduler_ip={} --scheduler_port={} " \
                              "--server_num={} --worker_num={} --scheduler_manage_port={} " \
                              "--enable_ssl={} --config_file_path={} " \
            .format(self.server_path, self.scheduler_ip, self.scheduler_port, self.server_num,
                    self.worker_num, self.scheduler_mgr_port, self.enable_ssl, self.config_file_path)
        print("exec:{}".format(start_scheduler_cmd), flush=True)
        os.system(start_scheduler_cmd)

    def start_server(self):
        start_server_cmd = "cd {}; python run_mobile_server.py --scheduler_ip={} " \
                           "--scheduler_port={} --fl_server_port={} --server_num={} " \
                           "--worker_num={} --start_fl_job_threshold={} " \
                           "--client_batch_size={} --client_epoch_num={} " \
                           "--fl_iteration_num={} --start_fl_job_time_window={} " \
                           "--update_model_time_window={} --encrypt_type={} " \
                           "--enable_ssl={} --config_file_path={}" \
            .format(self.server_path, self.scheduler_ip, self.scheduler_port, self.fl_server_port,
                    self.server_num, self.worker_num, self.start_fl_job_threshold, self.client_batch_size,
                    self.client_epoch_num, self.fl_iteration_num, self.start_fl_job_time_window,
                    self.update_model_time_window, self.encrypt_type, self.enable_ssl,
                    self.config_file_path)

        print("exec:{}".format(start_server_cmd), flush=True)
        os.system(start_server_cmd)

    def start_client(self):
        start_client_cmd = "cd {}/../client_script ;LD_LIBRARY_PATH={} python fl_client_run.py --jarPath={}  " \
                           "--case_jarPath={} --train_dataset={} --test_dataset={} --vocal_file={} " \
                           "--ids_file={} --flName={} --train_model_path={} --infer_model_path={} " \
                           "--ssl_protocol={}  --deploy_env={} --domain_name={} --cert_path={} " \
                           "--server_num={} --client_num={} --use_elb={} --thread_num={} --server_mode={} " \
                           "--batch_size={} --task={}" \
            .format(self.server_path, self.ld_library_path, self.frame_jar_path, self.case_jar_path,
                    self.train_dataset, "null", "null", "null", FLNAME, self.train_model_path,
                    self.infer_model_path, self.ssl_protocol, self.deploy_env, self.domain_name,
                    self.cert_path, self.server_num, self.client_num, self.use_elb, self.thread_num,
                    self.server_mode, self.client_batch_size, "train")
        print("exec:{}".format(start_client_cmd), flush=True)
        os.system(start_client_cmd)

    def check_client_log(self):
        # check client result
        query_success_cmd = "grep 'the total response of 1: SUCCESS' {}/../client_script/client_train0/* |wc -l" \
            .format(self.server_path)
        print("query_success_cmd:" + query_success_cmd)
        result = os.popen(query_success_cmd)
        info = result.read()
        result.close()
        assert int(info) == 1

        # check acc not none
        query_acc_cmd = "grep 'evaluate acc' {}/../client_script/client_train0/* |wc -l".format(self.server_path)
        print("query_acc_cmd:" + query_acc_cmd)
        result = os.popen(query_acc_cmd)
        info = result.read()
        result.close()
        assert info.find('none') == -1
        return True

    # @pytest.mark.skipif(2 > 1, reason="only support test in fl ST frame")
    def test_train_lenet(self):
        """
        fist case
        :return:
        """
        print("Class:{}, function:{}".format(self.__class__.__name__, inspect.stack()[1][3]), flush=True)
        self.start_scheduler()
        self.start_server()
        self.wait_cluster_ready(out_time=30)
        self.start_client()
        self.check_client_result(out_time=30)


@pytest.mark.fl_cluster
class TestMobileInference(BaseCase):
    train_model_path = os.path.join(BaseCase.fl_resource_path, "client/ms/lenet_train.mindir0.ms")
    infer_model_path = os.path.join(BaseCase.fl_resource_path, "client/ms/lenet_train.mindir0.ms")
    train_dataset = os.path.join(BaseCase.fl_resource_path, "client/data/f0049_32")
    test_dataset = os.path.join(BaseCase.fl_resource_path, "client/data/f0049_32")

    def setup_method(self):
        """
        Run before every test case
        :return:
        """
        self.init_env("../mobile", "quick_start_flclient.jar")
        finish_client_cmd = "cd {}/../client_script; " \
                            "python fl_client_finish.py --kill_tag=mindspore-lite-java-flclient" \
            .format(self.server_path)
        os.system(finish_client_cmd)

    def teardown_method(self):
        """
        Run after every test case
        :return:
        """
        finish_client_cmd = "cd {}/../client_script; " \
                            "python fl_client_finish.py --kill_tag=mindspore-lite-java-flclient" \
            .format(self.server_path)
        os.system(finish_client_cmd)
        self.clear_logs()

    def start_client(self):
        start_client_cmd = "cd {}/../client_script ;LD_LIBRARY_PATH={} python fl_client_run.py --jarPath={}  " \
                           "--case_jarPath={} --train_dataset={} --test_dataset={} --vocal_file={} " \
                           "--ids_file={} --flName={} --train_model_path={} --infer_model_path={} " \
                           "--ssl_protocol={}  --deploy_env={} --domain_name={} --cert_path={} " \
                           "--server_num={} --client_num={} --use_elb={} --thread_num={} --server_mode={} " \
                           "--batch_size={} --task={}" \
            .format(self.server_path, self.ld_library_path, self.frame_jar_path, self.case_jar_path,
                    self.train_dataset, self.test_dataset, "null", "null",
                    FLNAME, self.train_model_path, self.infer_model_path, self.ssl_protocol, self.deploy_env,
                    self.domain_name, self.cert_path, self.server_num, self.client_num, self.use_elb, self.thread_num,
                    self.server_mode, self.client_batch_size, "inference")
        print("exec:{}".format(start_client_cmd), flush=True)
        os.system(start_client_cmd)

    def check_client_log(self, out_time=30):
        # check client result
        query_success_cmd = "grep 'inference finish' {}/../client_script/client_inference0/* |wc -l".format(
            self.server_path)
        print("query_success_cmd:" + query_success_cmd)
        result = os.popen(query_success_cmd)
        info = result.read()
        result.close()
        assert int(info) == 1

    def test_infer_lenet(self):
        """
        fist case
        :return:
        """
        print("Class:{}, function:{}".format(self.__class__.__name__, inspect.stack()[1][3]), flush=True)
        self.start_client()
        self.check_client_result(out_time=30)
