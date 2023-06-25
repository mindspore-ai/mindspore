"""
functions for cross framework model infer result accuracy compare and summary
"""

import numpy as np

from mslite_bench.infer_base.infer_session_factory import InferSessionFactory
from mslite_bench.utils.infer_log import InferLogger
from mslite_bench.common.task_common_func import CommonFunc
from mslite_bench.common.enum_class import (
    NumpyDtype
)

_logger = InferLogger()


class CrossFrameworkAccSummary:
    """
    functions for cross framework model infer result accuracy compare and summary
    """
    @classmethod
    def acc_infos_between_features(cls,
                                   standard_feature,
                                   compare_feature):
        """
        get accuracy info between features, including mean error ratio
        and cosine similarity.
        params:
        standard_feature(Dict[str, nunmpy.ndarray): standard features to be compared
        compare_feature(Dict[str, nunmpy.ndarray): compare features to compare
        return:
        A dict, including mean_error_ratio and cosine similarity.
        """
        mean_error_ratio = cls.get_mean_error_between_features(compare_feature,
                                                               standard_feature)

        cosine_similarity = cls.get_cosine_distance_between_features(compare_feature,
                                                                     standard_feature)
        return {
            'mean_error_ratio': mean_error_ratio,
            'cosine_similarity': cosine_similarity
        }

    @classmethod
    def accuracy_compare_func(cls,
                              args,
                              logger=None):
        """
        get outputs accuracy compare info between two different framework using same model.
        params:
        args: input arguments
        logger: logger to recorder logs
        return:
        A dict, including mean_error_ratio and cosine similarity.
        """
        cmp_result = None

        src_file_path = args.model_file
        if not src_file_path.endswith('ms') and \
                not src_file_path.endswith('mindir'):
            logger.error(f'[Accuracy Compare] {src_file_path} is not a valid mslite model')

        dst_file_path = args.cmp_model_file
        logger.info(f'start to compare accuracy between '
                    f'{src_file_path} and {dst_file_path}')

        input_data_map = CommonFunc.create_numpy_data_map(args)
        ms_config = CommonFunc.get_framework_config(src_file_path,
                                                    args)
        try:
            ms_session = InferSessionFactory.create_infer_session(src_file_path,
                                                                  ms_config)
        except ValueError as e:
            logger.error(f'[Accuracy Compare] Create ms session failed: {e}')
            return cmp_result

        args.device = args.cmp_device
        cmp_cfg = CommonFunc.get_framework_config(dst_file_path,
                                                  args)
        try:
            cmp_session = InferSessionFactory.create_infer_session(dst_file_path,
                                                                   cmp_cfg,
                                                                   args.params_file)
        except ValueError as e:
            logger.error(f'[Accuracy Compare] Create dst session failed: {e}')
            return cmp_result

        try:
            cmp_result = cls.real_accuracy_compare(ms_session,
                                                   cmp_session,
                                                   input_data_map)
        except (NotImplementedError, ValueError) as e:
            logger.error(f'Accuracy test failed, get accuracy failed, {e}')
            raise
        logger.info(f'Accuracy ratio between {src_file_path} and '
                    f'{dst_file_path} is: ')
        for key, val in cmp_result.items():
            logger.info(f'{key}: {val}')
        return cmp_result

    @classmethod
    def real_accuracy_compare(cls,
                              src_session,
                              dst_session,
                              input_tensor_map):
        """
        get accuracy compare info between two different sessions with same input tensor map.
        params:
        src_session: session to be compared
        dst_session: session to compare
        input_tensor_map: tensor name and value dict for session input.
        return:
        A dict, including mean_error_ratio and cosine similarity.
        """
        src_output = src_session(input_tensor_map)
        dst_output = dst_session(input_tensor_map)

        result = {
            'mean_error_ratio': cls.get_mean_error_between_features(dst_output,
                                                                    src_output),
            'cosine_similarity': cls.get_cosine_distance_between_features(dst_output,
                                                                          src_output)
        }
        return result

    @classmethod
    def specific_accuracy_compare(cls,
                                  src_session,
                                  dst_session,
                                  args):
        """
        get accuracy compare info between two different sessions with
        specific input loading from files.
        params:
        src_session: session to be compared
        dst_session: session to compare
        args: input arguments.
        return:
        A dict, including mean_error_ratio and cosine similarity.
        """
        input_tensor_map = np.load(args.input_data_file,
                                   allow_pickle=True).item()
        result = cls.real_accuracy_compare(src_session,
                                           dst_session,
                                           input_tensor_map)
        return result

    @classmethod
    def random_accuracy_compare(cls,
                                src_session,
                                dst_session,
                                args):
        """
        get accuracy compare info between two different sessions with random inputs.
        params:
        src_session: session to be compared
        dst_session: session to compare
        args: input arguments.
        return:
        A dict, including mean_error_ratio and cosine similarity.
        """
        input_tensor_dtypes = CommonFunc.parse_dtype_infos(args.input_tensor_dtypes)
        input_tensor_shapes = CommonFunc.get_tensor_shapes(args.input_tensor_shapes)
        input_tensor_infos = {
            key: (shape, input_tensor_dtypes.get(key))
            for key, shape in input_tensor_shapes.items()
        }
        try:
            input_tensor_map = CommonFunc.create_numpy_data_map(input_tensor_infos)
        except ValueError as e:
            _logger.info(f'Random accuracy compare failed: {e}')
            raise
        result = cls.real_accuracy_compare(src_session,
                                           dst_session,
                                           input_tensor_map)

        return result

    @classmethod
    def get_cosine_distance_between_features(cls,
                                             calibrate_feature,
                                             cmp_feature):
        """
        calculate cosine distance between features.
        params:
        calibrate_feature: feature to be calibrated.
        cmp_feature: feature to compare.
        return:
        cosine similarity values between features.
        """
        cosine_similarity = {}
        for key, dst_feature in calibrate_feature.items():
            src_feature = cmp_feature.get(key)
            abs_eps = cls.absolute_tolerance()
            dst_sum = np.sum(dst_feature * dst_feature)
            src_sum = np.sum(src_feature * src_feature)
            dot_sum = np.sum(dst_feature * src_feature)

            if dst_sum < abs_eps and src_sum < abs_eps:
                value = 1.0
            elif dst_sum * src_sum < abs_eps:
                if dst_sum < abs_eps or src_sum < abs_eps:
                    value = 1.0
                else:
                    value = 0.0
            else:
                value = dot_sum / (np.sqrt(dst_sum) * np.sqrt(src_sum))

            cosine_similarity[key] = f'{round(value * 100, 3)}%'

        return cosine_similarity

    @classmethod
    def get_mean_error_between_features(cls,
                                        dst_feature,
                                        src_feature):
        """
        calculate mean error between features.
        params:
        dst_feature: feature to be calibrated.
        src_feature: feature to compare.
        return:
        mean error values between features.
        """
        absolute_tolerance = cls.absolute_tolerance()
        relative_tolerance = cls.relative_tolerance()
        mean_error_info = {}

        for key in dst_feature.keys():
            feat_a = dst_feature.get(key, None)
            feat_b = src_feature.get(key, None)
            if feat_b is None:
                raise ValueError(f'Model Inference feature '
                                 f'is not consistent in tensor: {key}')
            tolerance = absolute_tolerance + relative_tolerance * abs(feat_a)
            diff = abs(feat_a - feat_b)
            gt_tolerance_index = diff > tolerance
            lt_tolerance_index = np.logical_and(gt_tolerance_index, abs(feat_a) > absolute_tolerance)
            gt_tolerance_index = np.logical_and(gt_tolerance_index, abs(feat_a) < absolute_tolerance)
            gt_tolerance_index = np.logical_and(gt_tolerance_index, diff > relative_tolerance)
            gt_error = diff[gt_tolerance_index]
            lt_error = diff / (abs(feat_a) + absolute_tolerance)
            lt_error = lt_error[lt_tolerance_index]
            total_error = np.sum(gt_error) + np.sum(lt_error)
            total_size = gt_error.size + lt_error.size
            if total_size == 0:
                mean_error = 0.0
            else:
                mean_error = total_error / (gt_error.size + lt_error.size + 1)
            mean_error_info[key] = f'{round(mean_error * 100, 3)}%'

        return mean_error_info

    @staticmethod
    def check_np_dtype_with_model_input_dtype(tensor_map,
                                              session):
        """
        check input numpy data dtype with model input dtype
        params:
        tensor_map: a dict with key tensor name and value numpy data.
        session: model infer session
        return:
        a dict with key tensor name and value revised numpy data.
        """
        ret_map = tensor_map
        input_tensor_infos = session.input_infos
        dtype_class = session.dtype_class

        for key, np_data in tensor_map.items():
            np_dtype = np_data.dtype
            np_dtype_name = NumpyDtype(np_dtype).name
            session_dtype = input_tensor_infos.get(key, None)

            if session_dtype is None:
                raise ValueError('Input tensor name is not consistent with model inputs')
            session_dtype = session_dtype[1]

            session_dtype_name = dtype_class(session_dtype).name
            if session_dtype_name != np_dtype_name:
                _logger.warning(f'input tensor({key} input dtype{np_dtype_name} '
                                f'is not consistent with model dtype({session_dtype_name})')
                new_data = np_data.astype(getattr(NumpyDtype, session_dtype_name).value)
                ret_map[key] = new_data

        return ret_map

    @staticmethod
    def absolute_tolerance():
        """for const absolute tolerance"""
        return 1e-7

    @staticmethod
    def relative_tolerance():
        """for const relative tolerance"""
        return 1e-5
