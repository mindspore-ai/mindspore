/**
 * Copyright 2025 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "kernel/ascend/aclnn/pyboost_impl/customize/func_dropout_ext.h"

#include <algorithm>

#include "mindspore/ops/ops_utils/op_constants.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "mindspore/ccsrc/include/pynative/utils/pyboost/functions/auto_generate/functions.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr FuncDropoutExtAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input,
                                                const FP32ImmPtr &p, const BoolImmPtr &training,
                                                const BoolImmPtr &inplace, const TensorPtr &seed,
                                                const TensorPtr &offset) {
  auto training_ = training->value();
  auto inplace_ = inplace->value();
  auto p_ = static_cast<double>(p->value());
  if (MS_UNLIKELY(p_ < 0 || p_ > 1)) {
    MS_EXCEPTION(ValueError) << "For FuncDropoutExt, 'p' must be in [0, 1], but got " << p_;
  }

  auto input_shape = input->shape();
  if (MS_UNLIKELY(p_ == 0 || !training_ ||
                  std::any_of(input_shape.begin(), input_shape.end(), [](const auto &dim) { return dim == 0; }))) {
    op->set_outputs({input});
    return op->output(0);
  }

  if (p_ == 1) {
    auto input_typeid = input->Dtype()->type_id();
    auto zeros_shape = MakeValue<std::vector<int64_t>>(input_shape)->cast<ValueTuplePtr>();
    auto zeros_tensor = zeros(zeros_shape, std::make_shared<Int64Imm>(static_cast<int64_t>(input_typeid)));
    if (inplace_) {
      (void)inplace_mul(input, zeros_tensor);
      op->set_outputs({input});
    } else {
      auto zero_output = mul(input, zeros_tensor);
      op->set_outputs({zero_output});
    }
    return op->output(0);
  }

  auto results = dropout_ext(input, p, seed, offset);
  auto output = std::get<0>(results);

  if (inplace_) {
    auto non_blocking = std::make_shared<BoolImm>(False);
    (void)inplace_copy(input, output, non_blocking);
    op->set_outputs({input});
  } else {
    op->set_outputs({output});
  }
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
