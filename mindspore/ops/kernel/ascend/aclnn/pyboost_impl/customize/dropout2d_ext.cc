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
#include "kernel/ascend/aclnn/pyboost_impl/customize/dropout2d_ext.h"

#include <algorithm>

#include "mindspore/ops/ops_utils/op_constants.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "mindspore/ccsrc/include/pynative/utils/pyboost/functions/auto_generate/functions.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr Dropout2dExtAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input,
                                              const FP32ImmPtr &p, const BoolImmPtr &training,
                                              const BoolImmPtr &inplace, const TensorPtr &seed,
                                              const TensorPtr &offset) {
  auto training_ = training->value();
  auto inplace_ = inplace->value();
  auto p_ = static_cast<double>(p->value());
  if (MS_UNLIKELY(p_ < 0 || p_ > 1)) {
    MS_EXCEPTION(ValueError) << "For Dropout2d, 'p' must be in [0, 1], but got " << p_;
  }

  auto input_shape = input->shape();
  if (input_shape.size() != kDim3 || input_shape.size() != kDim4) {
    MS_LOG(WARNING) << "For Dropout2d, " << input_shape.size() << "-D input which is not recommended. "
                    << "Please use Dropout instead.";
  }

  if (MS_UNLIKELY(p_ == 0 || !training_ ||
                  std::any_of(input_shape.begin(), input_shape.end(), [](const auto &dim) { return dim == 0; }))) {
    op->set_outputs({input});
    return op->output(0);
  }

  if (p_ == 1) {
    auto input_typeid = input->Dtype()->type_id();
    auto zeros_shape = MakeValue<std::vector<int64_t>>({})->cast<ValueTuplePtr>();
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

  if (MS_UNLIKELY(input_shape.size() < kDim2)) {
    MS_EXCEPTION(ValueError) << "For Dropout2d, the dimensions of input can't be less than 2, but got "
                             << input_shape.size();
  }

  std::vector<int64_t> noise_shape(input_shape.size(), 1LL);
  noise_shape[kIndex0] = input_shape[kIndex0];
  noise_shape[kIndex1] = input_shape[kIndex1];
  auto empty =
    new_empty(input, MakeValue<std::vector<int64_t>>(noise_shape)->cast<ValueTuplePtr>(), std::nullopt, std::nullopt);
  auto noise_p = std::make_shared<FP32Imm>(static_cast<float>(1 - p_));
  auto bernoulli_out = inplace_bernoulli_scalar(empty, noise_p, seed, offset);
  auto noise = inplace_divs(bernoulli_out, noise_p);

  if (inplace_) {
    (void)inplace_mul(input, noise);
    op->set_outputs({input});
  } else {
    auto output = mul(input, noise);
    op->set_outputs({output});
  }
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
