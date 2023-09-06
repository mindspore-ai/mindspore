import os
import yaml

from test_info import TestInfo, ExecRet
from context import Context, Fmk


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
                cfgs = yaml.safe_load_all(yf)
                for cfg in cfgs:
                    if not isinstance(cfg, dict):
                        raise ValueError("cfg not a dict: ", cfg)
                    for fmk_str, models in cfg.items():
                        fmk = Fmk.from_str(fmk_str)
                        if fmk not in Context.instance().fmk_filter:
                            print(f"Ignoring {fmk_str} model info.")
                            continue
                        if not isinstance(models, dict):
                            raise ValueError(f"{fmk_str} segment in cfg not a dict: ", models)
                        for key, value in models.items():
                            test_manager.append(fmk, key, value)
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

    def append(self, fmk: Fmk, model_name, info: dict):
        self.infos.append(TestInfo.create(fmk, model_name, info))

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
        convert_success_count = 0
        perf_success_count = 0
        acc_success_count = 0
        total_count = len(self.infos)
        for test_info in self.infos:
            if not isinstance(test_info, TestInfo):
                raise ValueError("Not a TestInfo.")
            if test_info.convert_ret == ExecRet.success:
                convert_success_count += 1
            if test_info.bench_acc_ret == ExecRet.success:
                acc_success_count += 1
            if test_info.bench_perf_ret == ExecRet.success:
                perf_success_count += 1
            TestManager._print_summary_info(test_info.model.model_name, str(test_info.convert_ret),
                                            str(test_info.bench_acc_ret), str(test_info.bench_perf_ret))
        print("-" * TestManager.summary_line_length)
        if Context.instance().mode.convert:
            print(f"Convert {convert_success_count} succeed/{total_count}")
        if Context.instance().mode.performance:
            print(f"Performance {perf_success_count} succeed/{total_count}")
        if Context.instance().mode.accuracy:
            print(f"Accuracy {acc_success_count} succeed/{total_count}")
