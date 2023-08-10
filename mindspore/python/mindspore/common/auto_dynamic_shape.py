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

SHAPE_DIM_ANY = -1
SHAPE_RANK_ANY = -2


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


    def get_compile_args_shape(self, input_args, is_sink_mode):
        """get compile args shape"""
        if is_sink_mode:
            return input_args

        res_shape = []
        for input in input_args:
            if isinstance(input, Tensor):
                res_shape.append(input.shape)
            elif isinstance(input, (tuple, list)):
                if self._check_tuple_of_scalar(input):
                    shape = []
                    shape.append(len(input))
                    res_shape.append(shape)
            elif isinstance(input, int):
                res_shape.append([])
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
        logger.debug(f'The generalize shape cache number is {len(self.generalize_shape_cache)}, is less than '
                     f'{max_save_cache_number}, phase=%r should be saved in generalize shape cache.', phase)
        self.generalize_phase_and_compile_args_dict[phase] = compile_args
        self.generalize_shape_cache.append(compile_args)


    def _compare_input_args_and_cache_args(self, input_args, cache_args):
        """compare input args and cache args"""
        for (input, cache) in zip(input_args, cache_args):
            if isinstance(input, Tensor) and isinstance(cache, Tensor):
                if not self.is_tensor_equal(input, cache):
                    return False
            elif isinstance(input, int) and isinstance(cache, int):
                if input != cache:
                    return False
            elif isinstance(input, (tuple, list)) and isinstance(cache, (tuple, list)):
                if not self._compare_input_args_and_cache_args(input, cache):
                    return False
        return True


    def _check_tuple_of_scalar(self, input):
        """check tuple of scalar"""
        for elem in input:
            if not isinstance(elem, int):
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
        self.auto_dynamic_shape_manager = _AutoDynamicShapeManager()
        self.enable_auto_identify = os.getenv('MS_AUTO_DYNAMIC_SHAPE_ENABLE')


    def _is_enable_auto_dynamic_shape(self, args_list):
        """is enable auto identify shape"""
        if not self.enable_auto_identify:
            self.enable_auto_identify = "0"

        if self.enable_auto_identify == "0":
            return False

        if not args_list:
            return False

        for elem in args_list:
            if elem is None:
                return False
            if isinstance(elem, Tensor) and is_shape_unknown(elem.shape):
                return False
        return True


    @staticmethod
    def _do_generalize_in_sink(input, cache, input_index, cache_index, cache_type):
        """do generalize in sink, input rank must be 2"""
        if not input:
            raise ValueError("In sink mode, cell input can not be scalar, please close auto dynamic shape, "
                             "export MS_AUTO_DYNAMIC_SHAPE_ENABLE=0, and run again.")

        if input == cache:
            return cache

        shape_value = []
        if len(input) != len(cache):
            shape_value.append(SHAPE_RANK_ANY)
        else:
            for _ in input:
                shape_value.append(SHAPE_DIM_ANY)
        logger.info((f'In the {cache_type} cache[{cache_index}], the {input_index}th input tensor shape is {input},'
                     f'cache shape is {cache}, not equal, need generalize to {shape_value}.'))
        return shape_value

    def update_phase_and_compile_args(self, args, phase, is_sink_mode, aux=None):
        """save compile args and phase into dict"""
        if not self.is_enable_auto_dynamic_shape:
            return
        self.auto_dynamic_shape_manager.update_phase_and_compile_args(args, phase, self.save_cache_number,
                                                                      is_sink_mode, aux)


    def auto_dynamic_generate_compile_args(self, args_list, is_sink_mode):
        """generate compile args in auto dynamic shape"""
        if not self._is_enable_auto_dynamic_shape(args_list):
            self.is_enable_auto_dynamic_shape = False
            return args_list
        self.is_sink_mode = is_sink_mode

        res = self.auto_dynamic_shape_manager.get_compile_args_shape(args_list, is_sink_mode)
        logger.debug((f'input args list={res}.'))

        # step1: find cache in real_shape_cache.
        if self.auto_dynamic_shape_manager.get_real_shape_cache_number() < 2:
            logger.debug((f'real shape cache cap is smaller than 2, compile args shape={res}.'))
            return args_list

        is_real_shape_exist = self.auto_dynamic_shape_manager.find_compile_args_in_shape_cache(args_list, "real")
        if is_real_shape_exist:
            logger.debug((f'find compile args in real shape cache, compile args shape={res}'))
            return args_list

        # step2: if can not find cache in real_shape_cache, then generate it
        generalize_shape_args = self._do_generalize_shape("real", args_list, is_sink_mode)
        if self.auto_dynamic_shape_manager.get_generalize_shape_cache_number() == 0:
            res = self.auto_dynamic_shape_manager.get_compile_args_shape(generalize_shape_args, is_sink_mode)
            logger.debug((f'generalize shape cache cap is smaller than 1, compile args shape={res}.'))
            return generalize_shape_args

        # step3: find generalize_shape in generalize_shape_cache
        is_generalize_shape_exist = \
            self.auto_dynamic_shape_manager.find_compile_args_in_shape_cache(generalize_shape_args, "generalize")

        # step 4: if can not find cache in generalize_shape_cache, then generate it again
        if not is_generalize_shape_exist:
            new_generalize_shape = self._do_generalize_shape("generalize", generalize_shape_args, is_sink_mode)
            res = self.auto_dynamic_shape_manager.get_compile_args_shape(new_generalize_shape, is_sink_mode)
            logger.info((f'generalize with generalize shape cache, compile args shape={res}'))
            return new_generalize_shape

        res = self.auto_dynamic_shape_manager.get_compile_args_shape(generalize_shape_args, is_sink_mode)
        logger.debug((f'find compile args in generalize shape cache, compile args shape={res}'))
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
            elif isinstance(elem, Tensor) and is_shape_unknown(elem.shape):
                unknown_shape_count = unknown_shape_count + 1
            elif isinstance(elem, Tensor) and is_dim_unknown(elem.shape):
                unknown_rank_count = unknown_rank_count + 1
            elif isinstance(elem, int) and elem == SHAPE_DIM_ANY:
                unknown_shape_count = unknown_shape_count + 1
        return unknown_rank_count, unknown_shape_count


    def _do_generalize_one_input_shape(self, input_args, cache_args, cache_type, index, is_sink_mode):
        """do generalize shape one input by cache"""
        generalize_one_shape = []
        for i, (input, cache) in enumerate(zip(input_args, cache_args)):
            if isinstance(input, Tensor) and isinstance(cache, Tensor):
                if self.auto_dynamic_shape_manager.is_tensor_equal(input, cache):
                    generalize_one_shape.append(input)
                else:
                    shape_value = []
                    if len(input.shape) != len(cache.shape):
                        shape_value.append(SHAPE_RANK_ANY)
                    else:
                        for _ in range(len(input.shape)):
                            shape_value.append(SHAPE_DIM_ANY)
                    shape_tuple = tuple(shape_value)
                    generalize_one_shape.append(Tensor(shape=shape_tuple, dtype=input.dtype))
                    logger.info((f'In the {cache_type} cache[{index}], the {i}th input tensor shape is {input.shape},'
                                 f'cache shape is {cache.shape}, not equal, need generalize to {shape_tuple}.'))

            elif isinstance(input, (tuple, list)) and isinstance(cache, (tuple, list)):
                if is_sink_mode:
                    # when is_sink_mode=True, input must be the shape of Tensor.
                    res = self._do_generalize_in_sink(input, cache, i, index, cache_type)
                    generalize_one_shape.append(res)
                else:
                    res = self._do_generalize_one_input_shape(input, cache, cache_type, index, is_sink_mode)
                    generalize_one_shape.append(res)
            elif isinstance(input, int) and isinstance(cache, int):
                # when is_sink_mode=False, the input must may be scalar, or the value of list/tuple.
                # is_sink_mode can not be True
                if input == cache:
                    generalize_one_shape.append(input)
                else:
                    raise ValueError("In auto dynamic shape mode, scalar/tuple/list must be equal, it can not be "
                                     "generalize. Please close dynamic shape mode, "
                                     "export MS_AUTO_DYNAMIC_SHAPE_ENABLE=0, and run again.")
        return generalize_one_shape


    def _do_generalize_shape(self, cache_type, input_args, is_sink_mode):
        """do generalize shape by cache"""
        shape_cache = self.auto_dynamic_shape_manager.get_cache_by_type(cache_type)
        all_generalize_shape_args = []
        for index, cache_args in enumerate(shape_cache):
            generalize_shape = self._do_generalize_one_input_shape(input_args, cache_args, cache_type, index,
                                                                   is_sink_mode)
            all_generalize_shape_args.append(tuple(generalize_shape))

        unknown_shape_dict = {}
        for generalize_shape_args in all_generalize_shape_args:
            unknown_rank_count, unknown_shape_count = self._cal_unknown_shape_count(generalize_shape_args, is_sink_mode)
            unknown_count = (unknown_rank_count, unknown_shape_count)
            if unknown_count not in unknown_shape_dict:
                unknown_shape_dict[unknown_count] = generalize_shape_args

        keys = list(unknown_shape_dict.keys())
        keys.sort(key=lambda x: (x[0], x[1]))
        index = keys[0]
        return unknown_shape_dict.get(index)
