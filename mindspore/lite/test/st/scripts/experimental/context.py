import os
import enum
import argparse


class LogLevel(enum.Enum):
    summary = 2
    error = 1
    detail = 0


class Fmk(enum.Enum):
    mindir = 0
    caffe = 1
    onnx = 2
    tf = 3
    tflite = 4

    def __str__(self):
        if self == Fmk.mindir:
            return "mindir"
        if self == Fmk.caffe:
            return "caffe"
        if self == Fmk.onnx:
            return "onnx"
        if self == Fmk.tf:
            return "tf"
        if self == Fmk.tflite:
            return "tflite"
        raise ValueError("Unsupported fmk: ", self)

    def __repr__(self):
        return self.__str__()

    @classmethod
    def from_str(cls, fmk: str):
        fmk = fmk.lower()
        if fmk == "mindir":
            return Fmk.mindir
        if fmk == "caffe":
            return Fmk.caffe
        if fmk == "onnx":
            return Fmk.onnx
        if fmk == "tf":
            return Fmk.tf
        if fmk == "tflite":
            return Fmk.tflite
        raise ValueError("Unsupported fmk: ", fmk)


class TestMode:
    def __init__(self):
        self.convert = False
        self.performance = False
        self.accuracy = False

    def __str__(self):
        result = "["
        first = True
        if self.convert:
            if not first:
                result += " ,"
                first = False
            result += "convert"
        if self.performance:
            if not first:
                result += " ,"
                first = False
            result += " performance"
        if self.accuracy:
            if not first:
                result += " ,"
            result += " accuracy"
        result += "]"
        return result

    @classmethod
    def from_str(cls, mode_str: str):
        obj = cls()
        if mode_str.find('c') != -1:
            obj.test_convert()
        if mode_str.find('p') != -1:
            obj.test_performance()
        if mode_str.find('a') != -1:
            obj.test_accuracy()
        return obj

    def test_convert(self):
        self.convert = True

    def test_performance(self):
        self.performance = True

    def test_accuracy(self):
        self.accuracy = True


def str2bool(v):
    if isinstance(v, bool):
        return v
    if not isinstance(v, str):
        raise argparse.ArgumentTypeError("Bool value invalid: ", v)
    if v.lower() in ('yes', 'true', 'on', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'off', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError("Bool value invalid: ", v)


class Context:
    def __init__(self):
        self.data_dir = ""
        self.work_dir = ""
        self.model_subdir = "model"
        self.model_dir_func = None
        self.input_subdir = "input"
        self.input_dir_func = None
        self.output_subdir = "output"
        self.output_dir_func = None
        self.package_path = ""
        self.config_file = ""
        self.mode: TestMode = TestMode()
        self.converter_name = "converter_lite"
        self.benchmark_name = "benchmark"
        self.converter_file = ""
        self.benchmark_file = ""
        self.export_lib_paths = {}

        self.convert_log_file = ""
        self.bench_acc_log_file = ""
        self.bench_perf_log_file = ""

        self.log_level: LogLevel = LogLevel.summary
        self.exit_on_failed = True

        self.fmk_filter = []

        self.arg_parser = argparse.ArgumentParser(description='Test config.')
        self.arg_parser.add_argument('-f', "--fmk", type=str, default="all", help="framework of model",
                                     choices=['mindir', 'tf', 'tflite', 'onnx', 'caffe', 'all'])
        self.arg_parser.add_argument('-d', "--data_dir", type=str, default="",
                                     help="path to model, input and output, for example: data_dir/onnx/model, "
                                          "data_dir/caffe/input")
        self.arg_parser.add_argument('-m', "--model_dir", type=str, default="", help="path to model")
        self.arg_parser.add_argument('-i', "--input_dir", type=str, default="", help="path to input data")
        self.arg_parser.add_argument('-o', "--output_dir", type=str, default="", help="path to output data")
        self.arg_parser.add_argument('-p', "--pkg_dir", type=str, required=True,
                                     help="path to mindspore lite package, "
                                          "for example: /root/mindspore-lite-2.1.0-linux-x64/")
        self.arg_parser.add_argument('-w', "--work_dir", type=str, default="",
                                     help="path of workspace, "
                                          "for example: /root/mindspore-lite-2.1.0-linux-x64/, "
                                          "default: {data_dir}/test_workspace")
        self.arg_parser.add_argument('-c', "--config_file", type=str, required=True,
                                     help="path to config file, for example: /root/model_mindir.yaml")
        self.arg_parser.add_argument('-t', "--test_mode", type=str, default="c",
                                     help="test mode, 'c' for convert, 'p' for performance, 'a' for accuracy, "
                                          "for example: a, ca, pc")
        self.arg_parser.add_argument('-e', "--exit_on_failed", type=str2bool, default=True,
                                     help="Exit test if any model failed. for example: yes, true, False, No, 0")

    def __str__(self):
        return "*" * 10 + "MindSpore Lite Test Context:" + "*" * 10 + "\r\n" + \
               f"fmk: {str(self.fmk_filter)}\r\n" + \
               f"mode: {str(self.mode)}\r\n" + \
               f"exit on failed: {str(self.exit_on_failed)}\r\n" + \
               f"model dir: {self.model_dir_func('xxfmk')}\r\n" + \
               f"input dir: {self.model_dir_func('xxfmk')}\r\n" + \
               f"calib file dir: {self.output_dir_func('xxfmk')}\r\n" + \
               f"work dir: {self.work_dir}\r\n" + \
               f"config file: {self.config_file}\r\n" + \
               f"library path: {self.export_lib_paths.get('LD_LIBRARY_PATH')}\r\n" + \
               "*" * 48

    @classmethod
    def instance(cls):
        if not hasattr(Context, "_instance"):
            Context._instance = Context()
        return Context._instance

    def init(self):
        args = self.arg_parser.parse_args()
        fmk_str = args.fmk
        if fmk_str == 'all':
            self.fmk_filter = [Fmk.mindir, Fmk.tf, Fmk.caffe, Fmk.onnx, Fmk.tflite]
        else:
            self.fmk_filter = [Fmk.from_str(fmk_str)]
        self.data_dir = os.path.join(args.data_dir)
        self.work_dir = os.path.join(args.work_dir)
        if not self.data_dir:
            if not self.model_dir_func or not self.input_dir_func or not self.output_dir_func or not self.work_dir:
                raise ValueError("Please set data_dir, if not, please set model_dir, input_dir, output_dir and "
                                 "work_dir")
        if not args.model_dir:
            self.model_dir_func = lambda fmk: os.path.join(self.data_dir, str(fmk), self.model_subdir)
        else:
            self.model_dir_func = lambda fmk: os.path.join(args.model_dir)
        if not args.input_dir:
            self.input_dir_func = lambda fmk: os.path.join(self.data_dir, str(fmk), self.input_subdir)
        else:
            self.input_dir_func = lambda fmk: os.path.join(args.input_dir)
        if not args.output_dir:
            self.output_dir_func = lambda fmk: os.path.join(self.data_dir, str(fmk), self.output_subdir)
        else:
            self.output_dir_func = lambda fmk: os.path.join(args.output_dir)
        if not self.work_dir:
            self.work_dir = os.path.join(self.data_dir, "test_workspace")
        self.package_path = os.path.join(args.pkg_dir)
        self.config_file = os.path.join(args.config_file)
        self.mode = TestMode.from_str(args.test_mode)
        self.exit_on_failed = args.exit_on_failed
        self.converter_file = os.path.join(self.package_path, "tools", "converter", "converter", self.converter_name)
        self.benchmark_file = os.path.join(self.package_path, "tools", "benchmark", self.benchmark_name)
        library_paths = os.path.join(self.package_path, "runtime", "lib") + ":" + \
                        os.path.join(self.package_path, "runtime", "third_party", "dnnl") + ":" + \
                        os.path.join(self.package_path, "runtime", "third_party", "glog") + ":" + \
                        os.path.join(self.package_path, "runtime", "third_party", "libjpeg-turbo", "lib") + ":" + \
                        os.path.join(self.package_path, "runtime", "third_party", "securec") + ":" + \
                        os.path.join(self.package_path, "tools", "converter", "lib") + ":${LD_LIBRARY_PATH}"
        self.export_lib_paths["LD_LIBRARY_PATH"] = library_paths

        self.convert_log_file = os.path.join(self.work_dir, "convert.log")
        self.bench_acc_log_file = os.path.join(self.work_dir, "bench_acc.log")
        self.bench_perf_log_file = os.path.join(self.work_dir, "bench_perf.log")

        self.log_level = LogLevel.summary

        print(str(self))
