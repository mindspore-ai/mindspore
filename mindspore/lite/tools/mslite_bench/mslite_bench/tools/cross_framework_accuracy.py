# Copyright 2023 Huawei Technologies Co., Ltd
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
"""
functions for cross framework model infer result accuracy compare and summary
"""
import os.path
import csv

import numpy as np

from mslite_bench.infer_base.infer_session_factory import InferSessionFactory
from mslite_bench.utils.infer_log import InferLogger
from mslite_bench.common.task_common_func import CommonFunc
from mslite_bench.common.enum_class import (
    NumpyDtype
)
from mslite_bench.common.model_info_enum import ErrorAlgType

_logger = InferLogger().logger


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

        mean_relative_error = cls.get_mean_relative_error_between_features(compare_feature,
                                                                           standard_feature)

        cosine_similarity = cls.get_cosine_distance_between_features(compare_feature,
                                                                     standard_feature)
        return {
            ErrorAlgType.MEAN_RELATIVE_ERROR.value: mean_relative_error,
            ErrorAlgType.COSINE_SIMILARITY.value: cosine_similarity
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
            raise ValueError(f'{src_file_path} is not a valid mslite model')

        dst_file_path = args.cmp_model_file

        input_data_map = CommonFunc.create_numpy_data_map(args)
        ms_config = CommonFunc.get_framework_config(src_file_path,
                                                    args)
        try:
            ms_session = InferSessionFactory.create_infer_session(src_file_path,
                                                                  ms_config)
        except ValueError as e:
            logger.error('[Accuracy Compare] Create ms session failed: %s', e)
            return cmp_result

        args.device = args.cmp_device
        cmp_cfg = CommonFunc.get_framework_config(dst_file_path,
                                                  args)
        try:
            cmp_session = InferSessionFactory.create_infer_session(dst_file_path,
                                                                   cmp_cfg,
                                                                   args.params_file)
        except ValueError as e:
            logger.error(f'Create dst session failed %s', e)
            return cmp_result

        try:
            cmp_result = cls.real_accuracy_compare(ms_session,
                                                   cmp_session,
                                                   input_data_map)
        except (NotImplementedError, ValueError) as e:
            logger.error(f'Accuracy test failed, get accuracy failed %s', e)
            raise
        cmp_result = cls.is_acc_ok(cmp_result)
        for key, val in cmp_result.items():
            logger.debug(f'{key}: {val}')

        if not args.cmp_result_file:
            csv_path = os.path.join(os.path.dirname(src_file_path), 'accuracy_infos.csv')
        else:
            csv_path = f'{args.cmp_result_file}.csv'
        logger.info(f'Accuracy compare done, save accuracy info in %s', csv_path)
        cls.write_csv(cmp_result, csv_path)
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
            ErrorAlgType.MEAN_RELATIVE_ERROR.value: cls.get_mean_relative_error_between_features(dst_output,
                                                                                                 src_output),
            ErrorAlgType.COSINE_SIMILARITY.value: cls.get_cosine_distance_between_features(dst_output,
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
            _logger.error('Random accuracy compare failed: %s', e)
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
                value = dot_sum / (np.sqrt(dst_sum) * np.sqrt(src_sum) + abs_eps)

            cosine_similarity[key] = cls.error_format(value)

        return cosine_similarity

    @classmethod
    def get_mean_relative_error_between_features(cls,
                                                 dst_feature,
                                                 src_feature):
        """
        calculate mean relative error between features.
        params:
        dst_feature: feature to be calibrated.
        src_feature: feature to compare.
        return:
        mean relative error values between features.
        """
        mean_relative_error_info = {}
        np.seterr(divide='ignore', invalid='ignore')

        for key in dst_feature.keys():
            feat_a = dst_feature.get(key, None)
            feat_b = src_feature.get(key, None)
            if feat_b is None:
                raise ValueError(f'Model Inference feature '
                                 f'is not consistent in tensor: {key}')
            if feat_a.size == 0:
                mean_relative_error_info[key] = '0.0'
                continue
            if feat_a.dtype != feat_b.dtype:
                _logger.warning('layer %s : different dtypes between onnx out: %s '
                                'with mslite out: %s ',
                                key,
                                feat_a.dtype,
                                feat_b.dtype)
                mean_relative_error_info[key] = '0.0'
                continue
            diff = np.abs(feat_b - feat_a)
            abs_feat_a = np.abs(feat_a)
            relative_index = diff > cls.relative_tolerance()
            if relative_index.size == 0:
                mean_relative_error_info[key] = '0.0'
                continue
            diff = diff[relative_index]
            abs_feat_a = abs_feat_a[relative_index]
            abs_index = abs_feat_a > cls.absolute_tolerance()
            if abs_index.size == 0:
                mean_relative_error_info[key] = cls.error_format(np.average(diff))
                continue
            abs_feat_a = abs_feat_a[abs_index]
            relative_diff = diff[abs_index]
            abs_diff = diff[~abs_index]
            relative_error = np.divide(relative_diff, abs_feat_a)
            mean_relative_error_info[key] = (np.sum(relative_error) + np.sum(abs_diff)) \
                                            / (relative_error.size + abs_diff.size)
            if np.isnan(mean_relative_error_info.get(key, None)):
                _logger.warning('layer: %s has nan value, '
                                '%s do not work',
                                key,
                                ErrorAlgType.MEAN_RELATIVE_ERROR.value)
            mean_relative_error_info[key] = cls.error_format(mean_relative_error_info.get(key, None))

        return mean_relative_error_info

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
            diff = abs(feat_a - feat_b)
            gt_tolerance_index = diff > (absolute_tolerance + relative_tolerance * abs(feat_a))
            lt_tolerance_index = np.logical_and(gt_tolerance_index, abs(feat_a) > absolute_tolerance)
            gt_tolerance_index = np.logical_and(gt_tolerance_index, abs(feat_a) < absolute_tolerance)
            gt_tolerance_index = np.logical_and(gt_tolerance_index, diff > relative_tolerance)
            gt_error = diff[gt_tolerance_index]
            lt_error = diff / (abs(feat_a) + absolute_tolerance)
            lt_error = lt_error[lt_tolerance_index]
            if gt_error.size + lt_error.size == 0:
                mean_error = 0.0
            else:
                mean_error = (np.sum(gt_error) + np.sum(lt_error)) / \
                             (gt_error.size + lt_error.size + 1 + cls.absolute_tolerance())
            mean_error_info[key] = cls.error_format(mean_error)
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
                _logger.warning('input tensor %s input dtype %s '
                                'is not consistent with model dtype(%s)',
                                key,
                                np_dtype_name,
                                session_dtype_name)
                new_data = np_data.astype(getattr(NumpyDtype, session_dtype_name).value)
                ret_map[key] = new_data

        return ret_map

    @staticmethod
    def absolute_tolerance():
        """for const absolute tolerance"""
        return 1e-4

    @staticmethod
    def relative_tolerance():
        """for const relative tolerance"""
        return 1e-4

    @staticmethod
    def error_format(error):
        return f'{error * 100:.4f}%'

    @staticmethod
    def write_csv(contents, csv_file):
        """write csv"""
        contents_to_write = []
        error_names = []
        error_infos = []
        for key, value in contents.items():
            error_names.append(key)
            error_infos.append(value)

        for layer_name in list(error_infos[0].keys()):
            tmp_dict = {'layer_name': layer_name}
            for error_name in error_names:
                tmp_dict[error_name] = contents.get(error_name).get(layer_name)
            contents_to_write.append(tmp_dict)

        fieldnames = ['layer_name'] + error_names
        with open(csv_file, 'w', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(contents_to_write)

    @staticmethod
    def is_acc_ok(acc_info):
        """add is ok check for accuracy result"""
        mre = acc_info.get(ErrorAlgType.MEAN_RELATIVE_ERROR.value, None)
        cos = acc_info.get(ErrorAlgType.COSINE_SIMILARITY.value, None)
        is_ok = {}
        if mre is None or cos is None:
            raise ValueError('MRE or cosine similarity is None')
        mre_thred = 0.05
        cos_thred = 0.99
        cos_bad_thred = 0.9
        def error_format_to_float(num):
            return float(num.strip('%')) / 100

        nan_set = {'nan', 'nan%'}
        for key, mre_val in mre.items():
            cos_val = cos.get(key, None)
            if mre_val in nan_set or cos_val in nan_set:
                is_ok[key] = 'Invalid'
            elif error_format_to_float(cos_val) < cos_bad_thred:
                is_ok[key] = 'Bad'
            elif error_format_to_float(mre_val) > mre_thred \
                    and error_format_to_float(cos_val) < cos_thred:
                is_ok[key] = 'Bad'
            else:
                is_ok[key] = 'Good'

        acc_info['is_ok'] = is_ok
        return acc_info
