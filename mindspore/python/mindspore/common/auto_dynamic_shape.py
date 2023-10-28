# This is the Python adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
#
# Copyright 2020-2023 Huawei Technologies Co., Ltd
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
"""Providing auto dynamic shape interface methods."""

import os
from mindspore import log as logger
from mindspore._c_expression import GraphExecutor_, Tensor
from mindspore.common._utils import is_shape_unknown, is_dim_unknown
from mindspore.common.parameter import Parameter

SHAPE_DIM_ANY = -1
SHAPE_RANK_ANY = -2

auto_dynamic_shepe_dict = {}


class _AutoDynamicShapeManager:
    """
    Represents a function to manage auto identify dynamic shape.
    """
    def __init__(self):
        self.real_shape_cache = []
        self.generalize_shape_cache = []
        self.real_phase_and_compile_args_dict = {}
        self.generalize_phase_and_compile_args_dict = {}
        self._graph_executor = GraphExecutor_.get_instance()


    def __del__(self):
        self.real_shape_cache = []
        self.generalize_shape_cache = []
        self.real_phase_and_compile_args_dict = {}
        self.generalize_phase_and_compile_args_dict = {}


    @staticmethod
    def is_tensor_equal(input_elem, cache_elem):
        """check two tensor or param is equal"""
        if input_elem.shape == cache_elem.shape and input_elem.dtype == cache_elem.dtype:
            return True
        return False


    @staticmethod
    def _get_input_generalize_number(arg_list, is_shape_input):
        """check two tensor or param is equal"""
        count = 0
        if is_shape_input:
            for arg in arg_list:
                if is_shape_unknown(arg):
                    count = count + 1
        else:
            for arg in arg_list:
                if isinstance(arg, Tensor) and is_shape_unknown(arg.shape):
                    count = count + 1
        return count


    def get_real_shape_cache_number(self):
        """get real shape cache number"""
        return len(self.real_shape_cache)


    def get_real_shape_cache(self):
        """get real shape cache"""
        return self.real_shape_cache


    def get_generalize_shape_cache_number(self):
        """get generalize shape cache number"""
        return len(self.generalize_shape_cache)


    def get_generalize_shape_cache(self):
        """get generalize shape cache"""
        return self.generalize_shape_cache


    def get_cache_by_type(self, cache_type):
        """get cache by type"""
        if cache_type == "real":
            shape_cache = self.real_shape_cache
        else:
            shape_cache = self.generalize_shape_cache

        return shape_cache


    def get_compile_args_shape_without_sink(self, input_args, res_shape):
        """get compile args shape with out sink mode"""
        for arg in input_args:
            if isinstance(arg, Tensor):
                res_shape.append(arg.shape)
            elif isinstance(arg, (int, float)):
                res_shape.append([])
            elif isinstance(arg, (tuple, list)):
                tmp_shape = []
                self.get_compile_args_shape_without_sink(arg, tmp_shape)
                res_shape.append(tmp_shape)


    def get_compile_args_shape(self, input_args, is_sink_mode):
        """get compile args shape"""
        if is_sink_mode:
            return input_args

        res_shape = []
        self.get_compile_args_shape_without_sink(input_args, res_shape)
        return res_shape


    def find_compile_args_in_shape_cache(self, input_args, cache_type):
        """find compile args in real or generalize shape cache"""
        shape_cache = self.get_cache_by_type(cache_type)
        for cache_args in shape_cache:
            res = self._compare_input_args_and_cache_args(input_args, cache_args)
            if res:
                return True
        return False


    def update_phase_and_compile_args(self, compile_args, phase, save_cache_number, is_sink_mode, aux=None):
        """update compile args and phase"""

        if phase in self.real_phase_and_compile_args_dict:
            logger.debug(f'phase=%r is in real phase and compile args dict.', phase)
            return

        if phase in self.generalize_phase_and_compile_args_dict:
            logger.debug(f'phase=%r is in generalize phase and compile args dict.', phase)
            return

        if len(self.real_shape_cache) < 2:
            logger.debug(f'The real shape cache number is {len(self.real_shape_cache)}, is less than 2,'
                         f'phase=%r should be saved in real shape cache.', phase)
            self.real_phase_and_compile_args_dict[phase] = compile_args
            self.real_shape_cache.append(compile_args)
            return

        max_save_cache_number = save_cache_number - 2

        if len(self.generalize_phase_and_compile_args_dict) >= max_save_cache_number:
            # step1: find delete phase
            phase_list = list(self.generalize_phase_and_compile_args_dict.keys())
            delete_phase = phase_list[0]
            delete_compile_args = self.generalize_phase_and_compile_args_dict.get(delete_phase)

            # step2: delete phase cache
            if is_sink_mode:
                if hasattr(aux, '__network_manage__') and delete_phase in aux.__network_manage__:
                    del aux.__network_manage__[delete_phase]
                    del self.generalize_phase_and_compile_args_dict[delete_phase]
            else:
                delete_cache = set()
                delete_cache.add(delete_phase)
                self._graph_executor.del_net_res(None, delete_cache)
                del self.generalize_phase_and_compile_args_dict[delete_phase]

            # step3: delete compile args
            self.generalize_shape_cache.remove(delete_compile_args)

        # step3 save phase and compile args into cache
        logger.info(f'The generalize shape cache number is {len(self.generalize_shape_cache)}, is less than '
                    f'{max_save_cache_number}, phase=%r should be saved in generalize shape cache.', phase)
        self.generalize_phase_and_compile_args_dict[phase] = compile_args
        self.generalize_shape_cache.append(compile_args)


    def _compare_input_args_and_cache_args(self, input_args, cache_args):
        """compare input args and cache args"""
        for (arg, cache) in zip(input_args, cache_args):
            if isinstance(arg, Tensor) and isinstance(cache, Tensor):
                if not self.is_tensor_equal(arg, cache):
                    return False
            elif isinstance(arg, int) and isinstance(cache, int):
                if arg != cache:
                    return False
            elif isinstance(arg, (tuple, list)) and isinstance(cache, (tuple, list)):
                if not self._compare_input_args_and_cache_args(arg, cache):
                    return False
        return True


class _AutoIdentifyDynamicShape:
    """
    Represents a function auto identify dynamic shape.
    """
    def __init__(self):
        self.all_shape_cache = {}
        self.is_sink_mode = False
        self.is_enable_auto_dynamic_shape = True
        self.save_cache_number = 3
        self.enable_auto_identify = os.getenv('MS_AUTO_DYNAMIC_SHAPE_ENABLE')
        self.auto_dynamic_shape_manager = _AutoDynamicShapeManager()


    def __del__(self):
        self.all_shape_cache = {}
        self.is_sink_mode = False
        self.is_enable_auto_dynamic_shape = True
        self.save_cache_number = 3


    def _check_input_args_number(self, args_list):
        """check input arg number"""
        if self.auto_dynamic_shape_manager.get_real_shape_cache_number() > 0:
            first_real_cache = self.auto_dynamic_shape_manager.get_real_shape_cache()[0]
            if len(first_real_cache) != len(args_list):
                return False
        return True


    def _check_input_tensor_type(self, args_list, cache_list):
        """check input args type"""
        for (arg, cache) in zip(args_list, cache_list):
            if isinstance(arg, Tensor) and isinstance(cache, Tensor):
                if arg.dtype != cache.dtype:
                    logger.debug((f'input tensor type = {arg.dtype}, cache tensor type = {cache.dtype}, '
                                  f'tensor types are not same.'))
                    return False
            elif isinstance(arg, (tuple, list)) and isinstance(cache, (tuple, list)):
                res = self._check_input_tensor_type(arg, cache)
                if not res:
                    return False
            elif (isinstance(arg, int) and isinstance(cache, int)) or \
                 (isinstance(arg, float) and isinstance(cache, float)):
                if arg != cache:
                    return False
            elif isinstance(arg, Tensor) and not isinstance(cache, Tensor):
                return False
            elif isinstance(arg, (int, float)) and not isinstance(cache, (int, float)):
                return False
            elif isinstance(arg, (tuple, list)) and not isinstance(cache, (tuple, list)):
                return False
        return True


    def _check_input_number_and_type(self, args_list):
        """check input number and type"""
        res = self._check_input_args_number(args_list)
        if not res:
            return False

        if self.auto_dynamic_shape_manager.get_real_shape_cache_number() > 0:
            cache_list = self.auto_dynamic_shape_manager.get_real_shape_cache()[0]
            res = self._check_input_tensor_type(args_list, cache_list)
            if not res:
                return False
        return True


    def _is_enable_auto_dynamic_shape(self, args_list, is_sink_mode):
        """is enable auto identify shape"""
        if not is_sink_mode and not args_list:
            return False

        if not self.enable_auto_identify:
            self.enable_auto_identify = "0"

        if self.enable_auto_identify == "0":
            return False

        for elem in args_list:
            if elem is None:
                continue
            if not isinstance(elem, (list, tuple, Tensor, int, float)):
                return False
            if isinstance(elem, Tensor) and (is_shape_unknown(elem.shape) or (not elem.shape)):
                return False
            if not is_sink_mode and isinstance(elem, (list, tuple)):
                return self._is_enable_auto_dynamic_shape(elem, is_sink_mode)
        return True


    @staticmethod
    def _do_generalize_in_sink(input_arg, cache, input_index, cache_index, cache_type):
        """do generalize in sink, input rank must be 2"""
        if not input_arg:
            raise ValueError("In sink mode, cell input can not be scalar.")

        if input_arg == cache:
            return cache

        shape_value = []
        if len(input_arg) != len(cache):
            shape_value.append(SHAPE_RANK_ANY)
        else:
            for _ in input_arg:
                shape_value.append(SHAPE_DIM_ANY)
        logger.info((f'In the {cache_type} cache[{cache_index}], the {input_index}th input tensor shape is {input_arg},'
                     f'cache shape is {cache}, not equal, need generalize to {shape_value}.'))
        return shape_value

    def update_phase_and_compile_args(self, args, phase, is_sink_mode, aux=None):
        """save compile args and phase into dict"""
        if not self.is_enable_auto_dynamic_shape:
            return
        self.auto_dynamic_shape_manager.update_phase_and_compile_args(args, phase, self.save_cache_number,
                                                                      is_sink_mode, aux)

    def _check_real_shape_cache(self, res_shape, args_list):
        """find cache in real_shape_cache"""
        real_cache_number = self.auto_dynamic_shape_manager.get_real_shape_cache_number()
        if real_cache_number < 2:
            logger.info((f'real shape cache cap is {real_cache_number}, smaller than 2, '
                         f'compile args shape={res_shape}.'))
            return True

        is_real_shape_exist = self.auto_dynamic_shape_manager.find_compile_args_in_shape_cache(args_list, "real")
        if is_real_shape_exist:
            logger.debug((f'find compile args in real shape cache, compile args shape={res_shape}'))
            return True

        return False

    def _generate_with_generalize_shape(self, generalize_shape_args, is_sink_mode, args_list):
        """generate with generalize_shape """
        new_generalize_shape, can_generalize = self._do_generalize_shape("generalize", generalize_shape_args,
                                                                         is_sink_mode)
        if not can_generalize:
            return args_list

        res_shape = self.auto_dynamic_shape_manager.get_compile_args_shape(new_generalize_shape, is_sink_mode)
        logger.info((f'generalize with generalize shape cache, compile args shape = {res_shape}'))
        return new_generalize_shape

    def auto_dynamic_generate_compile_args(self, args_list, is_sink_mode):
        """generate compile args in auto dynamic shape"""
        if not self._check_input_number_and_type(args_list) or \
            not self._is_enable_auto_dynamic_shape(args_list, is_sink_mode):
            self.is_enable_auto_dynamic_shape = False
            return args_list
        self.is_sink_mode = is_sink_mode

        res_shape = self.auto_dynamic_shape_manager.get_compile_args_shape(args_list, is_sink_mode)
        logger.debug((f'input args list shape = {res_shape}.'))

        # step1: find cache in real_shape_cache.
        if self._check_real_shape_cache(res_shape, args_list):
            return args_list

        # step2: if can not find cache in real_shape_cache, then generate it
        generalize_shape_args, can_generalize = self._do_generalize_shape("real", args_list, is_sink_mode)
        if not can_generalize:
            return args_list

        if self.auto_dynamic_shape_manager.get_generalize_shape_cache_number() == 0:
            res_shape = self.auto_dynamic_shape_manager.get_compile_args_shape(generalize_shape_args, is_sink_mode)
            logger.info((f'generalize shape cache cap is smaller than 1, compile args shape = {res_shape}.'))
            return generalize_shape_args

        # step3: find generalize_shape in generalize_shape_cache
        is_generalize_shape_exist = \
            self.auto_dynamic_shape_manager.find_compile_args_in_shape_cache(generalize_shape_args, "generalize")

        # step 4: if can not find cache in generalize_shape_cache, then generate it again
        if not is_generalize_shape_exist:
            return self._generate_with_generalize_shape(generalize_shape_args, is_sink_mode, args_list)

        res_shape = self.auto_dynamic_shape_manager.get_compile_args_shape(generalize_shape_args, is_sink_mode)
        logger.debug((f'find compile args in generalize shape cache, compile args shape={res_shape}'))
        return generalize_shape_args


    def _cal_unknown_shape_count(self, generalize_shape_args, is_sink_mode):
        """generalize shape by compare with real shape cache, and return the least generalize input."""
        unknown_shape_count = 0
        unknown_rank_count = 0
        for elem in generalize_shape_args:
            if isinstance(elem, (list, tuple)):
                rank_count, shape_count = self._cal_unknown_shape_count(elem, is_sink_mode)
                unknown_rank_count = unknown_rank_count + rank_count
                unknown_shape_count = unknown_shape_count + shape_count
            if isinstance(elem, Tensor):
                if is_shape_unknown(elem.shape):
                    unknown_shape_count = unknown_shape_count + 1
                if is_dim_unknown(elem.shape):
                    unknown_rank_count = unknown_rank_count + 1
            if is_sink_mode and isinstance(elem, int):
                if elem == SHAPE_DIM_ANY:
                    unknown_shape_count = unknown_shape_count + 1
                if elem == SHAPE_RANK_ANY:
                    unknown_rank_count = unknown_rank_count + 1
        return unknown_rank_count, unknown_shape_count


    def _do_generalize_one_input_shape(self, input_args, cache_args, cache_type, index, is_sink_mode):
        """do generalize shape one input by cache"""
        def generalize_tensor(arg, cache, i):
            if self.auto_dynamic_shape_manager.is_tensor_equal(arg, cache):
                return arg

            shape_value = []
            if len(arg.shape) != len(cache.shape):
                shape_value.append(SHAPE_RANK_ANY)
            else:
                shape_value = [SHAPE_DIM_ANY for _ in range(len(arg.shape))]
            shape_tuple = tuple(shape_value)
            logger.info((f'In the {cache_type} cache[{index}], the {i}th input tensor shape is {arg.shape},'
                         f'cache shape is {cache.shape}, not equal, need generalize to {shape_tuple}.'))
            return Tensor(shape=shape_tuple, dtype=arg.dtype)

        def generalize_sequence(arg, cache, i):
            if is_sink_mode:
                # when is_sink_mode=True, input must be the shape of Tensor.
                res = self._do_generalize_in_sink(arg, cache, i, index, cache_type)
                return res

            res = self._do_generalize_one_input_shape(arg, cache, cache_type, index, is_sink_mode)
            return res

        generalize_one_shape = []
        for i, (arg, cache) in enumerate(zip(input_args, cache_args)):
            if isinstance(arg, Parameter) and isinstance(cache, Parameter):
                if self.auto_dynamic_shape_manager.is_tensor_equal(arg, cache):
                    generalize_one_shape.append(arg)
                    continue

                logger.info("In auto dynamic shape mode, parameter must be equal, it can not be generalize.")
                return input_args, False

            if isinstance(arg, Tensor) and isinstance(cache, Tensor):
                res = generalize_tensor(arg, cache, i)
                generalize_one_shape.append(res)
            elif isinstance(arg, (tuple, list)) and isinstance(cache, (tuple, list)):
                res = generalize_sequence(arg, cache, i)
                generalize_one_shape.append(res)
            elif isinstance(arg, int) and isinstance(cache, int):
                # when is_sink_mode=False, the input must may be scalar, or the value of list/tuple.
                # is_sink_mode can not be True
                if arg == cache:
                    generalize_one_shape.append(arg)
                else:
                    logger.info("In auto dynamic shape mode, scalar/tuple/list must be equal, it can not be " \
                                "generalize.")
                    return input_args, False
            elif arg is None and cache is None:
                generalize_one_shape.append(arg)

        return generalize_one_shape, True


    def _do_generalize_shape(self, cache_type, input_args, is_sink_mode):
        """do generalize shape by cache"""
        shape_cache = self.auto_dynamic_shape_manager.get_cache_by_type(cache_type)
        all_generalize_shape_args = []
        for index, cache_args in enumerate(shape_cache):
            generalize_shape, can_generalize = self._do_generalize_one_input_shape(input_args, cache_args, cache_type,
                                                                                   index, is_sink_mode)
            if not can_generalize:
                return generalize_shape, False
            all_generalize_shape_args.append(tuple(generalize_shape))

        unknown_shape_dict = {}
        for generalize_shape_args in all_generalize_shape_args:
            unknown_rank_count, unknown_shape_count = self._cal_unknown_shape_count(generalize_shape_args, is_sink_mode)
            unknown_count = (unknown_rank_count, unknown_shape_count)
            if unknown_count not in unknown_shape_dict:
                unknown_shape_dict[unknown_count] = generalize_shape_args

        keys = list(unknown_shape_dict.keys())
        keys.sort(key=lambda x: (x[0], x[1]))
        return unknown_shape_dict.get(keys[0]), True

_auto_dynamic_shape = _AutoIdentifyDynamicShape()


def get_auto_dynamic_shape_args(compile_args, key_id):
    """get auto dynamic shape args."""
    if key_id not in auto_dynamic_shepe_dict:
        auto_dynamic_shepe_dict[key_id] = _AutoIdentifyDynamicShape()
    compile_args = auto_dynamic_shepe_dict[key_id].auto_dynamic_generate_compile_args(compile_args, False)
    return compile_args


def update_auto_dynamic_shape_phase(compile_args, key_id, phase):
    """update auto dynamic shape phase."""
    if key_id in auto_dynamic_shepe_dict:
        auto_dynamic_shepe_dict[key_id].update_phase_and_compile_args(compile_args, phase, False)


def get_auto_dynamic_shape_args_with_check_input_signature(compile_args, key_id, input_signature):
    """get auto dynamic shape args."""
    if input_signature is None:
        return get_auto_dynamic_shape_args(compile_args, key_id)
    return compile_args


def update_auto_dynamic_shape_phase_with_check_input_signature(compile_args, key_id, phase, input_signature):
    """update auto dynamic shape phase."""
    if input_signature is None:
        if key_id in auto_dynamic_shepe_dict:
            auto_dynamic_shepe_dict[key_id].update_phase_and_compile_args(compile_args, phase, False)
