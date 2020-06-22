# Copyright 2020 Huawei Technologies Co., Ltd
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

"""Keywords for verification config."""

import sys


class _MindSporeTestFrameworkkeyword:
    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise TypeError("can not rebind keyword (%s)" % name)
        self.__dict__[name] = value


keyword = _MindSporeTestFrameworkkeyword()

keyword.function = "function"
keyword.inputs = "inputs"
keyword.expect = "expect"
keyword.ext = "ext"

keyword.id = "id"
keyword.group = "group"

keyword.desc_inputs = "desc_inputs"
keyword.desc_bprop = "desc_bprop"
keyword.desc_expect = "desc_expect"
keyword.block = "block"
keyword.split_outputs = "split_outputs"

keyword.compare_with = "compare_with"
keyword.compare_gradient_with = "compare_gradient_with"

keyword.max_error = "max_error"
keyword.check_tolerance = "check_tolerance"
keyword.relative_tolerance = "relative_tolerance"
keyword.absolute_tolerance = "absolute_tolerance"

keyword.sampling_times = "sampling_times"
keyword.shape_type = "shape_type"
keyword.model = "model"
keyword.loss = "loss"
keyword.opt = "opt"
keyword.num_epochs = "num_epochs"
keyword.loss_upper_bound = "loss_upper_bound"
keyword.true_params = "true_params"
keyword.num_samples = "num_samples"
keyword.batch_size = "batch_size"
keyword.dtype = "dtype"
keyword.scale = "scale"
keyword.num_inputs = "num_inputs"
keyword.num_outputs = "num_outputs"
keyword.delta = "delta"
keyword.input_selector = "input_selector"
keyword.output_selector = "output_selector"
keyword.result = "result"
keyword.shape = "shape"
keyword.type = "type"
keyword.reduce_output = "reduce_output"
keyword.init_param_with = "init_param_with"
keyword.desc_const = "desc_const"
keyword.const_first = "const_first"
keyword.add_fake_input = "add_fake_input"
keyword.fake_input_type = "fake_input_type"
keyword.exception = "exception"
keyword.error_keywords = "error_keywords"

sys.modules[__name__] = keyword
