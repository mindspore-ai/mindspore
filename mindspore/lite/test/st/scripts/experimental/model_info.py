import os

from context import Context, Fmk


class ModelInfo:
    def __init__(self, model_name, fmk: Fmk, network_suffix="", weight_suffix=""):
        self.model_name = model_name
        self.fmk: Fmk = fmk
        self.fmk_name = ""

        self.network_suffix = network_suffix
        self.weight_suffix = weight_suffix
        self.model_file = ""
        self.weight_file = ""
        # input output
        self.input_suffix = ""
        self.output_suffix = ""
        self.input_num = 0
        self.input_file = ""
        self.output_file = ""

        self.converted_model_file = ""

    def __str__(self):
        return f"{self.model_name}: {{fmk: {self.fmk}, fmk_name: {self.fmk_name}, " \
               f"network_suffix: {self.network_suffix}, weight_suffix: {self.weight_suffix}, " \
               f"model_file: {self.model_file}, weight_file: {self.weight_file}, input_suffix: {self.input_suffix}, " \
               f"output_suffix: {self.output_suffix}, input_num: {self.input_num}, input_file: {self.input_file}, " \
               f"output_file: {self.output_file}}}"

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def _input_data_str(input_file_prefix, input_num):
        if input_num == 1:
            return input_file_prefix
        first = True
        input_bin_str = ""
        for i in range(input_num):
            if first:
                first = False
                input_bin_str += f"{input_file_prefix}_{i}"
            else:
                input_bin_str += f",{input_file_prefix}_{i}"
        return input_bin_str

    def init(self, input_num=1, input_suffix="", output_suffix=""):
        self.input_num = input_num
        self.input_suffix = input_suffix
        self.output_suffix = output_suffix
        if self.fmk == Fmk.mindir:
            self.fmk_name = "mindir"
            if not self.network_suffix:
                self.network_suffix = ".mindir"
            if not self.input_suffix:
                self.input_suffix = ".mindir.bin"
            if not self.output_suffix:
                self.output_suffix = ".mindir.out"
        elif self.fmk == Fmk.caffe:
            self.fmk_name = "caffe"
            if not self.network_suffix:
                self.network_suffix = ".prototxt"
            if not self.weight_suffix:
                self.weight_suffix = ".caffemodel"
            if not self.input_suffix:
                self.input_suffix = ".ms.bin"
            if not self.output_suffix:
                self.output_suffix = ".ms.out"
        elif self.fmk == Fmk.onnx:
            self.fmk_name = "onnx"
            if not self.network_suffix:
                self.network_suffix = ".onnx"
            if not self.input_suffix:
                self.input_suffix = ".onnx.ms.bin"
            if not self.output_suffix:
                self.output_suffix = ".onnx.ms.out"
        elif self.fmk == Fmk.tf:
            self.fmk_name = "tf"
            if not self.network_suffix:
                self.network_suffix = ".pb"
            if not self.input_suffix:
                self.input_suffix = ".pb.ms.bin"
            if not self.output_suffix:
                self.output_suffix = ".pb.ms.out"
        elif self.fmk == Fmk.tflite:
            self.fmk_name = "tflite"
            if not self.network_suffix:
                self.network_suffix = ".tflite"
            if not self.input_suffix:
                self.input_suffix = ".tflite.ms.bin"
            if not self.output_suffix:
                self.output_suffix = ".tflite.ms.out"
        else:
            raise ValueError(f"model({self.model_name}) has unsupported fmk: {self.fmk}")
        context = Context.instance()
        self.model_file = os.path.join(context.model_dir_func(self.fmk), self.model_name + self.network_suffix)
        if self.fmk == Fmk.caffe:
            self.weight_file = os.path.join(context.model_dir_func(self.fmk), self.model_name + self.weight_suffix)
        else:
            self.weight_file = ""
        input_name = self.model_name + self.input_suffix
        output_name = self.model_name + self.output_suffix
        self.input_file = os.path.join(context.input_dir_func(self.fmk), input_name)
        self.output_file = os.path.join(context.output_dir_func(self.fmk), output_name)

    def convert_cmd(self, input_shapes="", output_name=""):
        context = Context.instance()
        if not output_name:
            self.converted_model_file = os.path.join(context.work_dir, self.model_name)
        else:
            self.converted_model_file = os.path.join(context.work_dir, output_name)
        args = [f"--fmk={self.fmk_name.upper()}", f"--modelFile={self.model_file}",
                f"--outputFile={self.converted_model_file}"]
        if self.weight_file:
            args.append(f"--weightFile={self.weight_file}")
        if input_shapes:
            args.append(f"--inputShape={input_shapes}")
        return context.converter_file, args

    def benchmark_accuracy_cmd(self, input_shapes="", acc_threshold=0.5):
        context = Context.instance()
        input_bin_str = ModelInfo._input_data_str(self.input_file, self.input_num)
        if not self.converted_model_file:
            self.converted_model_file = os.path.join(context.work_dir, self.model_name)
        if not os.path.exists(f"{self.converted_model_file}.mindir"):
            self.converted_model_file += "_graph"  # when FuncGraph split-export
        args = [f"--enableParallelPredict=false", f"--modelFile={self.converted_model_file}.mindir",
                f"--inDataFile={input_bin_str}", f"--benchmarkDataFile={self.output_file}",
                f"--inputShapes={input_shapes}", f"--accuracyThreshold={acc_threshold}", "--device=CPU"]
        return context.benchmark_file, args

    def benchmark_performance_cmd(self, input_shapes="", warmup_loop=3, loop=10, num_threads=2):
        context = Context.instance()
        if not self.converted_model_file:
            self.converted_model_file = os.path.join(context.work_dir, self.model_name)
        if not os.path.exists(f"{self.converted_model_file}.mindir"):
            self.converted_model_file += "_graph"  # when FuncGraph split-export
        args = [f"--enableParallelPredict=false", f"--modelFile={self.converted_model_file}.mindir",
                f"--inputShapes={input_shapes}", f"--warmUpLoopCount={warmup_loop}", f"--loopCount={loop}",
                f"--numThreads={num_threads}", f"--device=CPU"]
        return context.benchmark_file, args
