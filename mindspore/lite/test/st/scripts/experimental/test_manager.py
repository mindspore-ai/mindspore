import os
import yaml

from test_info import TestInfo
from context import Context


class TestManager:
    summary_line_length = 140
    model_name_length = 100
    convert_res_length = 10
    bench_acc_res_length = 15
    bench_perf_res_length = 15
    progress_length = 4

    def __init__(self):
        self.infos: [TestInfo] = []
        self.exit_on_failed = Context.instance().exit_on_failed

    @classmethod
    def parse(cls, yaml_file):
        test_manager = cls()
        with open(os.path.join(yaml_file), 'r') as yf:
            try:
                cfgs = yaml.load_all(yf.read(), Loader=yaml.FullLoader)
                for cfg in cfgs:
                    if not isinstance(cfg, dict):
                        raise ValueError("cfg not a dict: ", cfg)
                    for key, value in cfg.items():
                        test_manager.append(key, value)
            except:
                raise ValueError("Failed to parse yaml")
        print(f"Find {len(test_manager.infos)} test models.")
        return test_manager

    @staticmethod
    def _print_summary_double_line():
        print("=" * TestManager.summary_line_length)

    @staticmethod
    def _print_summary_line():
        print("-" * TestManager.summary_line_length)

    @staticmethod
    def _print_summary_info(mode_name, convert, bench_accuracy, bench_performance):
        print(mode_name[:TestManager.model_name_length].ljust(TestManager.model_name_length),
              convert[:TestManager.convert_res_length].ljust(TestManager.convert_res_length),
              bench_accuracy[:TestManager.bench_acc_res_length].ljust(TestManager.bench_acc_res_length),
              bench_performance[:TestManager.bench_perf_res_length].ljust(TestManager.bench_perf_res_length)
              )

    def append(self, model_name, info: dict):
        self.infos.append(TestInfo.create(model_name, info))

    def test_by_network(self):
        for test_info in self.infos:
            if not isinstance(test_info, TestInfo):
                raise ValueError("Not a TestInfo.")
            if test_info.convert():
                print(f"convert {test_info.model.model_name} success.")
                if test_info.benchmark_performance():
                    print(f"benchmark {test_info.model.model_name} success.")
                else:
                    print(f"benchmark {test_info.model.model_name} failed.")
            else:
                print(f"convert {test_info.model.model_name} failed.")

    def test_convert(self):
        print("=" * 20, "Start converting model:", "=" * 20)
        total = str(len(self.infos)).rjust(TestManager.progress_length)
        has_failed = False
        for index, test_info in enumerate(self.infos):
            progress = f"[{str(index + 1).rjust(TestManager.progress_length)}/{total}] "
            if not isinstance(test_info, TestInfo):
                raise ValueError("Not a TestInfo.")
            if test_info.convert():
                print(f"{progress} convert {test_info.model.model_name} success.")
            else:
                has_failed = True
                print(f"{progress} convert {test_info.model.model_name} failed.")
                if self.exit_on_failed:
                    return False
        return not has_failed

    def test_bench_acc(self):
        print("=" * 20, "Start benchmark accuracy of model:", "=" * 20)
        total = str(len(self.infos)).rjust(TestManager.progress_length)
        has_failed = False
        for index, test_info in enumerate(self.infos):
            progress = f"[{str(index + 1).rjust(TestManager.progress_length)}/{total}] "
            if not isinstance(test_info, TestInfo):
                raise ValueError("Not a TestInfo.")
            if test_info.benchmark_accuracy():
                print(f"{progress} benchmark accuracy of {test_info.model.model_name} success.")
            else:
                has_failed = True
                print(f"{progress} benchmark accuracy of {test_info.model.model_name} failed.")
                if self.exit_on_failed:
                    return False
        return not has_failed

    def test_bench_performance(self):
        print("=" * 20, "Start benchmark performance of model:", "=" * 20)
        total = str(len(self.infos)).rjust(TestManager.progress_length)
        has_failed = False
        for index, test_info in enumerate(self.infos):
            progress = f"[{str(index + 1).rjust(TestManager.progress_length)}/{total}] "
            if not isinstance(test_info, TestInfo):
                raise ValueError("Not a TestInfo.")
            if test_info.benchmark_performance():
                print(f"{progress} benchmark performance of {test_info.model.model_name} success.")
            else:
                has_failed = True
                print(f"{progress} benchmark performance of {test_info.model.model_name} failed.")
                if self.exit_on_failed:
                    return False
        return not has_failed

    def summary(self):
        print("=" * TestManager.summary_line_length)
        TestManager._print_summary_info("model name", "convert", "accuracy", "performance")
        print("-" * TestManager.summary_line_length)
        for test_info in self.infos:
            if not isinstance(test_info, TestInfo):
                raise ValueError("Not a TestInfo.")
            TestManager._print_summary_info(test_info.model.model_name, str(test_info.convert_ret),
                                            str(test_info.bench_acc_ret), str(test_info.bench_perf_ret))
