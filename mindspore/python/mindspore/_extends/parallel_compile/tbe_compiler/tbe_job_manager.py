# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
"""tbe job manager"""

from __future__ import absolute_import
import json
import traceback
from enum import Enum

from .tbe_adapter import tbe_initialize, get_auto_tune_support_op_list, tbe_finalize, check_support, select_op_format, \
    parallel_pre_compile_op, do_fuzz_build_tbe_op, before_build_process, build_single_pre_op, \
    parallel_compile_fusion_op, rl_tune_single_op, rl_tune_fusion_op, ga_tune, get_finish_tasks, get_prebuild_output
from .tbe_helper import check_job_json, get_compute_op_list, get_func_names
from .tbe_job import TbeJob, JobStatus, JobType


class TbeJobManager:
    """ TBE compiler job manager """

    def __init__(self):
        self.job_handlers = {
            JobType.INITIALIZE_JOB: self.initialize_handler,
            JobType.FINALIZE_JOB: self.finalize_handler,
            JobType.CHECK_JOB: self.check_support_handler,
            JobType.SELECT_JOB: self.select_format_handler,
            JobType.PRECOMPILE_JOB: self.pre_compile_handler,
            JobType.COMPILE_JOB: self.compile_handler,
            JobType.FUSION_COMPILE_JOB: self.compile_handler,
            JobType.TUNE_JOB: self.tune_handler,
            JobType.QUERY_JOB: self.query_handler
        }

        self._all_jobs = {}
        self._finished_jobs = {}
        self._running_jobs = {}
        self._raw_finish_jobs = {}
        self.tbe_initialize = False
        self.init_cache = None
        self.para_debug_path = ""
        self.auto_tiling_mode = ""
        self.offline_tune = False
        self.tune_op_list = []
        self.tune_dump_path = ""
        self.tune_bank_path = ""
        self.auto_tune_op_list = []
        self.pre_build_ops = {}
        self.fusion_need_sync = 0
        self.imported_module = {}
        self.soc_version = ""
        self.core_num = 0
        self.op_bank_path = ""
        # license info
        self.rl_tune_switch = ""
        self.rl_tune_list = ""
        self.op_tune_switch = ""
        self.op_tune_list = ""
        self.pass_list = ""

    def __del__(self):
        self.reset()

    def reset(self):
        """
        Reset the job manager
        :return: None
        """
        self._all_jobs = {}
        self._finished_jobs = {}
        self._running_jobs = {}
        self._raw_finish_jobs = {}
        self.para_debug_path = ""
        self.auto_tiling_mode = ""
        self.offline_tune = False
        self.tune_op_list = []
        self.tune_dump_path = ""
        self.tune_bank_path = ""
        self.auto_tune_op_list = []
        self.pre_build_ops = []
        self.fusion_need_sync = 0
        self.imported_module = {}
        if self.tbe_initialize:
            tbe_finalize(self.auto_tiling_mode, self.offline_tune, self.init_cache)
            self.tbe_initialize = False
            self.init_cache = None
            self.soc_version = ""
            self.core_num = 0
            self.op_bank_path = ""

    def job_handler(self, job_str):
        """
        Tbe job handler
        :param job_str: tbe compile job string
        :return: job process result json string
        """
        job = None
        try:
            job_json = json.loads(job_str)
            check_job_json(job_json)
            job_id = job_json["job_id"]
            source_id = job_json["source_id"]
            job_type = job_json["job_type"]
            sys_info = self._get_job_sys_info()
            fusion_op_name = "NA"
            if "fusion_op_name" in job_json["job_content"]:
                fusion_op_name = job_json["job_content"]["fusion_op_name"]
            job = TbeJob(source_id, job_id, job_type, job_json["job_content"], fusion_op_name, job_str, sys_info)
            post_job(self._all_jobs, job)
            if not self.tbe_initialize and job.type != JobType.INITIALIZE_JOB:
                job.error(
                    "Initialize Job should be processed before job {}, job json string:{}".format(job.type,
                                                                                                  job.json_string))
                return self.add_to_finished_jobs(job, JobStatus.JOB_FAILED)
            func = self.job_handlers.get(job.type)
            res = func(job)
            return res
        # pylint: disable=broad-except
        except Exception as e:
            # pylint: disable=no-value-for-parameter
            sys_info = self._get_job_sys_info()
            job = TbeJob(-1, -1, "", None, job_str, sys_info) if job is None else job
            job.status = JobStatus.JOB_FAILED
            job.result = "Exception during job process"
            job.error("Process Job Failed")
            job.error("Job json string:\n{}\n".format(job_str))
            job.error("Error message:{}".format(traceback.format_exc()))
            job.error_manager(e)
            return self.add_to_finished_jobs(job, JobStatus.JOB_FAILED)
        finally:
            pass

    def initialize_handler(self, job: TbeJob):
        """ Initialize job handler """
        self._init_sys_info(job)
        res = tbe_initialize(job)
        if not res:
            job.error("Process Initialize Job failed, job json string:{}".format(job.json_string))
            return self.add_to_finished_jobs(job, JobStatus.JOB_FAILED)
        if "GA" in self.auto_tiling_mode:
            self.auto_tune_op_list = get_auto_tune_support_op_list(job)
        self.tbe_initialize = True
        self.init_cache = job
        return self.add_to_finished_jobs(job, JobStatus.JOB_SUCCESS)

    def finalize_handler(self, job: TbeJob):
        """ Finalize job handler """
        if not self.tbe_initialize:
            return self.add_to_finished_jobs(job, JobStatus.JOB_SUCCESS)
        res = tbe_finalize(self.auto_tiling_mode, self.offline_tune, job)
        if not res:
            job.error("Process Finalize Job failed, job json string:{}".format(job.json_string))
            return self.add_to_finished_jobs(job, JobStatus.JOB_FAILED)
        return self.add_to_finished_jobs(job, JobStatus.JOB_SUCCESS)

    def check_support_handler(self, job: TbeJob):
        """ Check Support job handler """
        res = check_support(job)
        if not res:
            job.error("Process CheckSupport Job failed, job json string:{}".format(job.json_string))
            return self.add_to_finished_jobs(job, JobStatus.JOB_FAILED)
        return self.add_to_finished_jobs(job, JobStatus.JOB_SUCCESS)

    def select_format_handler(self, job: TbeJob):
        """ Select Format job handler """
        res = select_op_format(job)
        if not res:
            job.error("Process SelectFormat Job failed, job json string:{}".format(job.json_string))
            return self.add_to_finished_jobs(job, JobStatus.JOB_FAILED)
        return self.add_to_finished_jobs(job, JobStatus.JOB_SUCCESS)

    def pre_compile_handler(self, job: TbeJob):
        """ Pre Compile job handler """
        res = parallel_pre_compile_op(job)
        if not res:
            job.error("Process PreCompile Job failed, job json string:{}".format(job.json_string))
            return self.add_to_finished_jobs(job, JobStatus.JOB_FAILED)
        self.pre_build_ops[job.content["fusion_op_name"]] = job
        return self.add_to_running_jobs(job)

    def compile_handler(self, job: TbeJob):
        """ Compile job handler """
        compute_op_list = get_compute_op_list(job.content)
        if len(compute_op_list) == 1:
            return self.single_op_compile(job)

        before_build_process(job)
        res = parallel_compile_fusion_op(job)
        if not res:
            job.error("Parallel_compile_fusion_op Job failed, job json string:{}".format(job.json_string))
            return self.add_to_finished_jobs(job, JobStatus.JOB_FAILED)
        return self.add_to_running_jobs(job)

    def single_op_compile(self, job: TbeJob):
        """Single operator compile"""
        res = do_fuzz_build_tbe_op(job)
        if not res:
            job.error("Process do fuzz build tbe op failed, job json string:{}".format(job.json_string))
            return self.add_to_finished_jobs(job, JobStatus.JOB_FAILED)
        if job.result == "NOT_CHANGED":
            job.result = ""
            before_build_process(job)
            res = build_single_pre_op(job)
            if not res:
                job.error("Process build single pre op failed, job json string:{}".format(job.json_string))
                return self.add_to_finished_jobs(job, JobStatus.JOB_FAILED)
            return self.add_to_running_jobs(job)
        if job.result == "SUCCESS":
            return self.add_to_finished_jobs(job, JobStatus.JOB_SUCCESS)
        job.error("Process do fuzz build tbe op failed, job json string:{}".format(job.json_string))
        return self.add_to_finished_jobs(job, JobStatus.JOB_FAILED)

    def tune_handler(self, job: TbeJob):
        """ Tune job handler """
        before_build_process(job)
        tune_mode = self._select_tune_mode(job)
        if tune_mode == TuneMode.NO_TUNE:
            return self.compile_handler(job)
        compute_op_list = get_compute_op_list(job.content)
        if len(compute_op_list) == 1:
            return self.single_op_tune(job)
        return self.fusion_op_tune(job)

    def single_op_tune(self, job: TbeJob):
        """Single operator tune"""
        tune_mode = self._select_tune_mode(job)
        if tune_mode == TuneMode.RL_TUNE:
            res = rl_tune_single_op(job)
            if not res:
                job.error(
                    "Tune Job failed, tune type {}, job json string:{}".format(tune_mode, job.json_string))
                return self.add_to_finished_jobs(job, JobStatus.JOB_FAILED)
        else:
            res = ga_tune(job)
            if not res:
                job.error("ga tune Job failed, job json string:{}".format(job.json_string))
                return self.compile_handler(job)
        if job.status == JobStatus.JOB_RUNNING:
            return self.add_to_running_jobs(job)
        return self.add_to_finished_jobs(job, JobStatus.JOB_SUCCESS)

    def fusion_op_tune(self, job: TbeJob):
        """Fusion operator tune"""
        tune_mode = self._select_tune_mode(job)
        if tune_mode == TuneMode.RL_TUNE:
            res = rl_tune_fusion_op(job)
        else:
            res = ga_tune(job)
        if not res:
            job.error(
                "Tune Job failed, tune type {}, job json string:{}".format(tune_mode, job.json_string))
            return self.add_to_finished_jobs(job, JobStatus.JOB_FAILED)
        if job.status == JobStatus.JOB_RUNNING:
            return self.add_to_running_jobs(job)
        return self.add_to_finished_jobs(job, JobStatus.JOB_SUCCESS)

    def query_handler(self, query_job: TbeJob):
        """ Query job handler """
        target_source_id = query_job.content["source_id"]
        target_job_id = query_job.content["job_id"]
        target_job = get_job(self._finished_jobs, target_source_id, target_job_id)
        if target_job:
            query_job.warning("Query a finished job: {}".format(query_job.content))
            query_job.result = target_job.get_result()
            return self.add_to_finished_jobs(query_job, JobStatus.JOB_SUCCESS)
        target_job = get_job(self._raw_finish_jobs, target_source_id, target_job_id)
        if not target_job:
            self.update_raw_finished_jobs(query_job)
            target_job = get_job(self._raw_finish_jobs, target_source_id, target_job_id)
        if target_job:
            query_job.debug("Found job in raw finished jobs, source_id:{}, job_id:{}".format(target_source_id,
                                                                                             target_job_id))
            query_job.result = target_job.get_result()
            del_job(self._raw_finish_jobs, target_job.source_id, target_job.id)
            self.add_to_finished_jobs(target_job, target_job.status)
            return self.add_to_finished_jobs(query_job, JobStatus.JOB_SUCCESS)
        target_job = get_job(self._running_jobs, target_source_id, target_job_id)
        if target_job:
            query_job.result = target_job.get_result()
            return self.add_to_finished_jobs(query_job, JobStatus.JOB_SUCCESS)
        target_job = get_job(self._all_jobs, target_source_id, target_job_id)
        if target_job:
            query_job.debug("Found job in all jobs, source_id:{}, job_id:{}".format(target_source_id,
                                                                                    target_job_id))
            target_job.debug("Be Queried")
            query_job.result = target_job.get_result()
            return self.add_to_finished_jobs(query_job, JobStatus.JOB_SUCCESS)
        query_job.error("Can't find job in finished/raw_finished/running jobs, source_id: {}".format(target_source_id))
        query_job.result = ""
        return self.add_to_finished_jobs(query_job, JobStatus.JOB_FAILED)

    def update_raw_finished_jobs(self, query_job: TbeJob):
        """
        Get new finished jobs from tbe parallel compilation and add them to raw_finished_jobs
        :param query_job: query job
        :return: Node
        """
        new_finished_jobs = get_finish_tasks(query_job.source_id)
        for new_job in new_finished_jobs:
            source_id = new_job["graph_id"]
            job_id = new_job["task_id"]
            target_job = get_job(self._running_jobs, source_id, job_id)
            if not target_job:
                query_job.error("Can't get job, source id:{}, job id:{}".format(source_id, job_id))
                continue
            target_job.result = new_job["op_res"] if "op_res" in new_job else new_job["result"]
            if target_job.type == JobType.PRECOMPILE_JOB:
                op_name = target_job.content["fusion_op_name"]
                op_params = get_prebuild_output(op_name)
                pre_compile_result = dict()
                pre_compile_result["op_pattern"] = target_job.result
                pre_compile_result["op_params"] = op_params
                pre_compile_result["core_type"] = new_job["core_type"] if "core_type" in new_job else ""
                target_job.result = json.dumps(pre_compile_result)
            target_job.info("Query result:{}".format(new_job["result"]))
            if new_job["status_code"] == 0:
                target_job.status = JobStatus.JOB_SUCCESS
                target_job.info("Query info_msg:{}".format(new_job["info_msg"]))
            else:
                target_job.status = JobStatus.JOB_FAILED
                target_job.error("Query info_msg:{}".format(new_job["info_msg"]))
                if "err_args" in new_job:
                    target_job.error("Query err_args:{}".format(new_job["err_args"]))
                if "except_msg" in new_job:
                    target_job.error("Query except_msg:{}".format(new_job["except_msg"]))
                if "except_tuple_msg" in new_job:
                    target_job.error_manager(new_job["except_tuple_msg"])
                target_job.error("\nOriginal compile json: \n {}\n".format(target_job.json_string))
            post_job(self._raw_finish_jobs, target_job)
            del_job(self._running_jobs, target_job.source_id, target_job.id)

    def add_to_finished_jobs(self, job, status):
        """
        add job to finished jobs with job process status
        :param job:
        :param status:
        :return: job process result string
        """
        job.status = status
        post_job(self._finished_jobs, job)
        return job.get_result()

    def add_to_running_jobs(self, job):
        """
        add job to running jobs
        :param job:
        :return: job process result string
        """
        job.status = JobStatus.JOB_RUNNING
        post_job(self._running_jobs, job)
        return job.get_result()

    def _get_job_sys_info(self):
        """
        Get job manager system info
        :return: system info
        """
        sys_info = dict()
        sys_info["logger"] = DummyLogger
        sys_info["para_debug_path"] = self.para_debug_path
        sys_info["tune_dump_path"] = self.tune_dump_path
        sys_info["offline_tune"] = self.offline_tune
        # license info
        sys_info["rl_tune_switch"] = self.rl_tune_switch
        sys_info["rl_tune_list"] = self.rl_tune_list
        sys_info["op_tune_switch"] = self.op_tune_switch
        sys_info["op_tune_list"] = self.op_tune_list
        sys_info["pass_list"] = self.pass_list
        # soc
        sys_info["socVersion"] = self.soc_version
        sys_info["coreNum"] = self.core_num
        sys_info["op_bank_path"] = self.op_bank_path
        return sys_info

    def _init_sys_info(self, initialize_job):
        """
        Initialize job manager system info from INITIALIZE JOB
        :param initialize_job: initialize job
        :return: None
        """
        # auto tune info
        self.auto_tiling_mode = initialize_job.content["SocInfo"]["autoTilingMode"]
        self.offline_tune = initialize_job.content["SocInfo"]["offlineTune"]
        self.tune_op_list = initialize_job.content["TuneInfo"]["tune_op_list"]
        self.tune_dump_path = initialize_job.content["TuneInfo"]["tune_dump_path"]
        self.tune_bank_path = initialize_job.content["TuneInfo"]["tune_bank_path"]
        self.para_debug_path = initialize_job.content["para_debug_path"]
        # license info
        self.rl_tune_switch = initialize_job.content["LicInfo"]["rl_tune_switch"]
        self.rl_tune_list = initialize_job.content["LicInfo"]["rl_tune_list"]
        self.op_tune_switch = initialize_job.content["LicInfo"]["op_tune_switch"]
        self.op_tune_list = initialize_job.content["LicInfo"]["op_tune_list"]
        self.pass_list = initialize_job.content["LicInfo"]["pass_list"]
        # soc
        self.soc_version = initialize_job.content["SocInfo"]["socVersion"]
        if initialize_job.content["SocInfo"]["coreNum"].isdigit():
            self.core_num = int(initialize_job.content["SocInfo"]["coreNum"])
        self.op_bank_path = initialize_job.content["SocInfo"]["op_bank_path"]

    def _select_tune_mode(self, job):
        """
        Select the corresponding tune mode according to op job content and job manager system info
        :param job: tbe tune job
        :return: NO_TUNE RL_TUNE or GA_TUNE
        """
        auto_tiling_mode = job.content["SocInfo"]["autoTilingMode"]
        offline_tune = job.content["SocInfo"]["offlineTune"]
        full_name = job.content["full_name"]
        func_names = get_func_names(job.content)
        if self.tune_op_list and full_name not in self.tune_op_list:
            return TuneMode.NO_TUNE
        if offline_tune:
            return TuneMode.RL_TUNE
        if TuneMode.GA_TUNE.value in auto_tiling_mode:
            for func_name in func_names:
                if func_name.lower() in self.auto_tune_op_list:
                    return TuneMode.GA_TUNE
        if TuneMode.RL_TUNE.value in auto_tiling_mode:
            return TuneMode.RL_TUNE
        return TuneMode.NO_TUNE


class TuneMode(Enum):
    """Class of tune mode: NO_TUNE, GA, RL"""
    NO_TUNE = "NO_TUNE"
    GA_TUNE = "GA"
    RL_TUNE = "RL"


class DummyLogger:
    """DummyLogger"""

    def __init__(self):
        pass

    @staticmethod
    def debug(msg, *args, **kwargs):
        """debug method."""
        return True

    @staticmethod
    def info(msg, *args, **kwargs):
        """info method."""
        return True

    @staticmethod
    def warning(msg, *args, **kwargs):
        """warning method."""
        return True

    @staticmethod
    def error(msg, *args, **kwargs):
        """error method."""
        return True

    @staticmethod
    def exception(msg, *args, **kwargs):
        """exception method."""
        return False


def get_job(jobs, source_id, job_id):
    """
    get the job from job list according to source_id and job_id
    :param jobs: job list
    :param source_id : target job's source_id
    :param job_id: target job's job_id
    :return: job instance if found in job list
             None if not found in job list
    """
    if source_id not in jobs.keys():
        return None
    if job_id not in jobs[source_id].keys():
        return None
    return jobs[source_id][job_id]


def post_job(jobs, new_job):
    """
    add the new job into jobs list
    :param jobs: job list
    :param new_job : new job
    :return: None
    """
    if new_job.source_id not in jobs.keys():
        jobs[new_job.source_id] = dict()
        jobs[new_job.source_id][new_job.id] = new_job
    else:
        jobs[new_job.source_id][new_job.id] = new_job


def del_job(jobs, source_id, job_id):
    """
    delete the job from job list according to source_id and job_id
    :param jobs: job list
    :param source_id : target job's source_id
    :param job_id: target job's job_id
    :return: bool True or False
    """
    if source_id not in jobs.keys():
        return False
    if job_id not in jobs[source_id].keys():
        return False
    del jobs[source_id][job_id]
    return True
