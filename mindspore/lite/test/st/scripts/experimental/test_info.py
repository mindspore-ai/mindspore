import subprocess
import enum

from typing import Dict

from model_info import ModelInfo, Fmk
from context import Context


class ExecRet(enum.Enum):
    idle = 0
    success = 1
    failed = 2

    def __str__(self):
        if self.value == ExecRet.idle.value:
            return "idle"
        if self.value == ExecRet.success.value:
            return "pass"
        if self.value == ExecRet.failed.value:
            return "failed"
        return ""

    def __repr__(self):
        return self.__str__()


class TestInfo:
    def __init__(self):
        self.model: ModelInfo = None

        self.convert_shapes = ""
        self.convert_output_name = ""

        self.need_performance = False
        self.benchmark_shapes = ""
        self.acc_threshold = 0.5
        self.warmup_loop = 3
        self.bench_loop = 10
        self.num_threads = 2

        self.cmd_envs = Context.instance().export_lib_paths
        self.cmd_envs["ENABLE_MULTI_BACKEND_RUNTIME"] = "on"

        self.convert_ret: ExecRet = ExecRet.idle
        self.bench_acc_ret: ExecRet = ExecRet.idle
        self.bench_perf_ret: ExecRet = ExecRet.idle

    def __str__(self):
        return f"{{model: {self.model}, convert_shapes: {self.convert_shapes}, " \
               f"convert_output_name: {self.convert_output_name}, need_performance: {self.need_performance}, " \
               f"benchmark_shapes: {self.benchmark_shapes}, acc_threshold: {self.acc_threshold}, " \
               f"warmup_loop: {self.warmup_loop}, bench_loop: {self.bench_loop}, num_threads: {self.num_threads}}}"

    def __repr__(self):
        return self.__str__()

    @classmethod
    def create(cls, model_fmk: Fmk, model_name, info: dict):
        config = cls()
        fmk = model_fmk
        network_suffix = info.get("network_suffix", "")
        weight_suffix = info.get("weight_suffix", "")
        config.model = ModelInfo(model_name, fmk, network_suffix, weight_suffix)
        input_number = info.get("input_number", 1)
        input_suffix = info.get("input_suffix", "")
        output_suffix = info.get("output_suffix", "")
        config.model.init(input_number, input_suffix, output_suffix)
        config.convert_shapes = info.get("convert_shapes", "")
        config.convert_output_name = info.get("convert_output_name", "")

        config.need_performance = info.get("need_performance", False)
        config.benchmark_shapes = info.get("benchmark_shapes", "")
        config.acc_threshold = info.get("acc_threshold", 0.5)
        config.warmup_loop = info.get("warmup_loop", 3)
        config.bench_loop = info.get("bench_loop", 10)
        config.num_threads = info.get("num_threads", 2)
        return config

    @staticmethod
    def exec_cmd(envs: Dict[str, str], cmd: str, args: [str]):
        cmds = [cmd] + args
        return_cmd = subprocess.run(cmds, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8', shell=False,
                                    env=envs)
        if return_cmd.returncode == 0:
            return True
        print(f"Failed to exec {cmds} \r\n {return_cmd.stderr}")
        return False

    def convert(self):
        if TestInfo.exec_cmd(self.cmd_envs, *self.model.convert_cmd(self.convert_shapes, self.convert_output_name)):
            self.convert_ret = ExecRet.success
            return True
        self.convert_ret = ExecRet.failed
        return False

    def benchmark_accuracy(self):
        if TestInfo.exec_cmd(self.cmd_envs,
                             *self.model.benchmark_accuracy_cmd(self.benchmark_shapes, self.acc_threshold)):
            self.bench_acc_ret = ExecRet.success
            return True
        self.bench_acc_ret = ExecRet.failed
        return False

    def benchmark_performance(self):
        if TestInfo.exec_cmd(self.cmd_envs,
                             *self.model.benchmark_performance_cmd(self.benchmark_shapes, self.warmup_loop,
                                                                   self.bench_loop, self.num_threads)):
            self.bench_perf_ret = ExecRet.success
            return True
        self.bench_perf_ret = ExecRet.failed
        return False
