"""
input argument parser functions
"""
import argparse


class ArgParser:
    """
    input argument parser functions
    """
    @classmethod
    def parse_arguments(cls):
        """parse input arguments for mslite bench"""
        parser = argparse.ArgumentParser(description='Easy Infer for model benchmark')
        cls.base_arg_parse(parser)
        cls.model_arg_parse(parser)
        cls.task_arg_parse(parser)
        args = parser.parse_args()
        return args

    @classmethod
    def task_arg_parse(cls, parser):
        """parse task related arguments"""
        # for task related
        parser.add_argument('--task_type',
                            type=int,
                            default=0,
                            help='benchmark task type:'
                                 '0 for framework accuracy compare,'
                                 '1 for mslite dynamic input infer')
        # for inference related infos
        parser.add_argument('--model_file',
                            type=str,
                            default=None,
                            help='path to model file')
        parser.add_argument('--params_file',
                            type=str,
                            default=None,
                            help='path to params file')
        parser.add_argument('--cmp_model_file',
                            type=str,
                            default=None,
                            help="the model path for model to be compared")
        parser.add_argument('--test_data',
                            type=str,
                            default=None,
                            help='path to data to do inference')
        parser.add_argument('--test_label',
                            type=str,
                            default=None,
                            help='path to test labels to calculate accuracy of model infer')

        # for benchmark and random accuracy test
        parser.add_argument('--input_tensor_shapes',
                            type=str,
                            default=None,
                            help="input tensor infos contain input tensor name and tensor shape"
                                 "format 'tensor_name: tensor_shape;")
        parser.add_argument('--input_data_file',
                            type=str,
                            default=None,
                            help="path to files contain input data, with key is input tensor name"
                                 "value is input numpy data")

        parser.add_argument('--save_file_type',
                            type=int,
                            default=0,
                            choices=[0, 1],
                            help="file type to save output tensor info, "
                                 "0 for npy and 1 for bin file")

        parser.add_argument('--input_tensor_dtypes',
                            type=str,
                            default=None,
                            help="tensor dtype for each model input tensor, "
                                 "choices=[INT8, INT32, INT64"
                                 "FLOAT16, FLOAT, FLOAT64, UINT8]")

        parser.add_argument('--random_input_flag',
                            type=bool,
                            default=False,
                            help="flag indicate whether using random input to do inference")

        parser.add_argument('--loop_infer_times',
                            type=int,
                            default=1,
                            help="infer times for loop infer")

        parser.add_argument('--warmup_times',
                            type=int,
                            default=0,
                            help="warm times for model infer")

    @classmethod
    def model_arg_parse(cls, parser):
        """parse model and framework related arguments"""
        # for mslite config
        parser.add_argument('--thread_affinity_mode',
                            type=int,
                            default=2,
                            help='thread affinity number for mslite inference')

        parser.add_argument('--thread_num',
                            type=int,
                            default=1,
                            help='thread number for mslite inference')

        parser.add_argument('--mslite_model_type',
                            type=int,
                            default=0,
                            choices=[0, 4],
                            help='input model type for mslite inference, '
                                 '0 for MINDIR, 4 for MINDIR_LITE')

        parser.add_argument('--ascend_provider',
                            type=str,
                            default='',
                            choices=['', 'ge'],
                            help="Ascend infer method: '' for acl, 'ge' for GE")

        # for tensorrt infer
        parser.add_argument('--tensorrt_optim_input_shape',
                            type=str,
                            default=None,
                            help='optim input shape for tensorrt'
                                 'with key tensor name (str) '
                                 'and value shape info(List[int])')

        parser.add_argument('--tensorrt_min_input_shape',
                            type=str,
                            default=None,
                            help='optim input shape for tensorrt'
                                 'with key tensor name (str) '
                                 'and value shape info(List[int])')

        parser.add_argument('--tensorrt_max_input_shape',
                            type=str,
                            default=None,
                            help='optim input shape for tensorrt'
                                 'with key tensor name (str) '
                                 'and value shape info(List[int])')

        parser.add_argument('--gpu_memory_size',
                            type=int,
                            default=100,
                            help='gpu init memory size(M)')

        parser.add_argument('--is_enable_tensorrt',
                            type=bool,
                            default=False,
                            help="flag indicate whether use tensorrt engine")

        parser.add_argument('--is_fp16',
                            type=bool,
                            default=False,
                            help="flag indicate whether apply fp16 infer")

        parser.add_argument('--is_int8',
                            type=bool,
                            default=False,
                            help="flag indicate whether apply int8 infer")

    @staticmethod
    def base_arg_parse(parser):
        """parse base related arguments"""
        parser.add_argument('--device',
                            type=str,
                            default='cpu',
                            choices=['cpu', 'gpu', 'ascend'],
                            help='device type for model inference')
        parser.add_argument('--cmp_device',
                            type=str,
                            default='cpu',
                            choices=['cpu', 'gpu', 'ascend'],
                            help='device type for cmp model inference')
        parser.add_argument('--device_id',
                            type=int,
                            default=0,
                            help='device index for model inference')
        parser.add_argument('--frameworkType',
                            type=str,
                            default='MSLITE',
                            choices=['MSLITE', 'PB', 'ONNX', 'PADDLE'],
                            help='device type for model inference')
        parser.add_argument('--log_path',
                            type=str,
                            default=None,
                            help='path to save model inference log')
        parser.add_argument('--batch_size',
                            type=int,
                            default=1,
                            help='model inference batch size')
        parser.add_argument('--input_tensor_names',
                            nargs='+',
                            default=None,
                            help='model input tensor name list')
        parser.add_argument('--output_tensor_names',
                            nargs='+',
                            default=None,
                            help='model output tensor name list')
