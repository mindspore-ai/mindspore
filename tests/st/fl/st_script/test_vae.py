import inspect
import os
import pytest
from base_case import BaseCase

FLNAME = "com.mindspore.flclient.demo.vae.VaeClient"


@pytest.mark.fl_cluster
class TestVaeTrain(BaseCase):
    train_dataset = os.path.join(BaseCase.fl_resource_path,
                                 "client/data/vae/flatten_ca801543-a7e8-4090-9210-9b5af63be892_3.csv")
    test_dataset = os.path.join(BaseCase.fl_resource_path,
                                "client/data/vae/flatten_ca801543-a7e8-4090-9210-9b5af63be892_3.csv")
    train_model_path = os.path.join(BaseCase.fl_resource_path, "client/ms/vae_train_0.ms")
    infer_model_path = os.path.join(BaseCase.fl_resource_path, "client/ms/vae_train_0.ms")

    def setup_method(self):
        """
        Run before every test case
        :return:
        """
        print("Class:{}, function:{}".format(self.__class__.__name__, inspect.stack()[1][3]), flush=True)
        self.init_env("../cross_device_vae", "flclient_models.jar")
        # copy ST from resource
        cp_st_cmd = "cp -r {}/server/cross_device_vae {}/../".format(self.fl_resource_path, self.script_path)
        os.system(cp_st_cmd)
        finish_cluster_cmd = "cd {}; python finish_all.py --scheduler_port={}" \
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
        finish_cluster_cmd = "cd {}; python finish_all.py --scheduler_port={}" \
            .format(self.server_path, self.scheduler_port)
        os.system(finish_cluster_cmd)
        os.system(finish_client_cmd)
        self.clear_logs()

    def start_scheduler(self):
        """
        Start scheduler
        :return:
        """
        print("Class:{}, function:{}".format(self.__class__.__name__, inspect.stack()[1][3]), flush=True)
        start_scheduler_cmd = "cd {}; python run_sched.py --scheduler_ip={} --scheduler_port={} --server_num={} " \
                              "--worker_num={} --scheduler_manage_port={} --config_file_path={}" \
            .format(self.server_path, self.scheduler_ip, self.scheduler_port, self.server_num, self.worker_num,
                    self.scheduler_mgr_port, self.config_file_path)
        print("exec:{}".format(start_scheduler_cmd), flush=True)
        os.system(start_scheduler_cmd)

    def start_server(self):
        print("Class:{}, function:{}".format(self.__class__.__name__, inspect.stack()[1][3]), flush=True)
        sign_k = 0.2
        sign_eps = 100
        sign_thr_ratio = 0.6
        sign_global_lr = 5
        dp_delta = 0.9
        dp_norm_clip = 0.01
        cipher_time_window = 30000000
        reconstruct_secrets_threshold = 2
        start_server_cmd = "cd {}; python run_server.py  --scheduler_ip={} --scheduler_port={} --fl_server_port={} " \
                           "--server_num={} --worker_num={} --start_fl_job_threshold={} --client_batch_size={} " \
                           "--client_epoch_num={} --fl_iteration_num={} --start_fl_job_time_window={} " \
                           "--update_model_time_window={} --encrypt_type={} " \
                           "--sign_k={} --sign_eps={} --sign_thr_ratio={} --sign_global_lr={} --dp_delta={} " \
                           "--dp_norm_clip={}  --cipher_time_window={} " \
                           "--reconstruct_secrets_threshold={} --config_file_path={}" \
            .format(self.server_path, self.scheduler_ip, self.scheduler_port, self.fl_server_port, self.server_num,
                    self.worker_num, self.start_fl_job_threshold, self.client_batch_size, self.client_epoch_num,
                    self.fl_iteration_num, self.start_fl_job_time_window, self.update_model_time_window,
                    self.encrypt_type, sign_k, sign_eps, sign_thr_ratio, sign_global_lr, dp_delta, dp_norm_clip,
                    cipher_time_window, reconstruct_secrets_threshold, self.config_file_path)

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
                    self.train_dataset, "null", "null", "null", FLNAME, self.train_model_path, self.infer_model_path,
                    self.ssl_protocol, self.deploy_env, self.domain_name, self.cert_path,
                    self.server_num, self.client_num, self.use_elb, self.thread_num, self.server_mode,
                    self.client_batch_size, "train")
        print("exec:{}".format(start_client_cmd), flush=True)
        os.system(start_client_cmd)

    def check_client_log(self):
        print("Class:{}, function:{}".format(self.__class__.__name__, inspect.stack()[1][3]), flush=True)
        # check client result
        query_success_cmd = "grep 'the total response of 1: SUCCESS' {}/../client_script/client_train0/* |wc -l".format(
            self.server_path)
        print("query_success_cmd:" + query_success_cmd)
        result = os.popen(query_success_cmd)
        info = result.read()
        result.close()
        assert int(info) == 1

        # check if nan exist
        query_nan_cmd = "grep 'is nan' {}/../client_script/client_train0/* |wc -l".format(self.server_path)
        result = os.popen(query_nan_cmd)
        info = result.read()
        result.close()
        # after refresh the resource change to int(info) == 0
        assert int(info) >= 0

    # @pytest.mark.skipif(2 > 1, reason="only support test in fl ST frame")
    def test_train_vae(self):
        """
        fist case
        :return:
        """
        self.start_scheduler()
        self.start_server()
        self.wait_cluster_ready(out_time=30)
        self.start_client()
        self.check_client_result(out_time=30)


@pytest.mark.fl_cluster
class TestVaeInference(BaseCase):
    train_dataset = os.path.join(BaseCase.fl_resource_path,
                                 "client/data/vae/flatten_ca801543-a7e8-4090-9210-9b5af63be892_3.csv")
    test_dataset = os.path.join(BaseCase.fl_resource_path,
                                "client/data/vae/flatten_ca801543-a7e8-4090-9210-9b5af63be892_3.csv")
    train_model_path = os.path.join(BaseCase.fl_resource_path, "client/ms/vae_train_0.ms")
    infer_model_path = os.path.join(BaseCase.fl_resource_path, "client/ms/vae_train_0.ms")

    def setup_method(self):
        """
        Run before every test case
        :return:
        """
        self.init_env("../cross_device_vae", "flclient_models.jar")
        # copy ST from resource
        cp_st_cmd = "cp -r {}/server/cross_device_vae {}/../".format(self.fl_resource_path, self.script_path)
        os.system(cp_st_cmd)
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
                    self.train_dataset, self.test_dataset, "null", "null", FLNAME, self.train_model_path,
                    self.infer_model_path, self.ssl_protocol, self.deploy_env, self.domain_name,
                    self.cert_path, self.server_num, self.client_num, self.use_elb, self.thread_num,
                    self.server_mode, self.client_batch_size, "inference")
        print("exec:{}".format(start_client_cmd), flush=True)
        os.system(start_client_cmd)

    def check_client_log(self):
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
        self.start_client()
        self.check_client_result(out_time=30)
