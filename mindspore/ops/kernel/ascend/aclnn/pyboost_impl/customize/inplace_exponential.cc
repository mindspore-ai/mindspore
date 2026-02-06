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

#include "kernel/ascend/aclnn/pyboost_impl/customize/inplace_exponential.h"

#include <limits>
#include <set>
#include <memory>

#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
#include "include/pynative/utils/pyboost/op_register.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/functions/auto_generate/functions.h"
#include "include/pynative/utils/pyboost/functions/auto_grad_guard.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "mindspore/core/include/base/float16.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
double GetEps(const TypeId type_id) {
  switch (type_id) {
    case kNumberTypeFloat16:
      return static_cast<double>(std::numeric_limits<float16>::epsilon() / 2);
    case kNumberTypeBFloat16:
      return static_cast<double>(std::numeric_limits<BFloat16>::epsilon() / 2);
    case kNumberTypeFloat32:
      return static_cast<double>(std::numeric_limits<float>::epsilon() / 2);
    default:
      MS_EXCEPTION(ValueError) << "unsupported type_id " << type_id;
  }
}

double GetScalarValue(const std::shared_ptr<Scalar> &scalar) {
  if (scalar->isa<Int32Imm>()) {
    return GetValue<int32_t>(scalar);
  } else if (scalar->isa<Int64Imm>()) {
    return GetValue<int64_t>(scalar);
  } else if (scalar->isa<FP32Imm>()) {
    auto fp32imm_ptr = scalar->cast<FP32ImmPtr>();
    return ops::GetDoubleValueFromScalar(fp32imm_ptr);
  } else if (scalar->isa<FP64Imm>()) {
    return GetValue<double>(scalar);
  } else if (scalar->isa<BoolImm>()) {
    return GetValue<bool>(scalar);
  } else {
    MS_EXCEPTION(TypeError) << "Unsupported type: " << scalar->type_name();
  }
}
}  // namespace

tensor::TensorPtr InplaceExponentialAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input,
                                                    const ScalarPtr lambda, const TensorPtr &seed,
                                                    const TensorPtr &offset) {
  MS_LOG(DEBUG) << "InplaceExponentialAscendCustomize Call start";
  kernel::pyboost::RequireGradGuard no_grad(false);
  double lambda_val = GetScalarValue(lambda);
  if (lambda_val <= 0.0) {
    MS_EXCEPTION(ValueError) << "For InplaceExponential, lambd should be greater than 0.0, but got " << lambda_val;
  }
  if (std::isinf(lambda_val)) {
    auto out = inplace_zero(input);
    op->set_outputs({out});
    return out;
  }

  auto input_type = input->data_type();
  if (input_type == kNumberTypeFloat64) {
    MS_EXCEPTION(TypeError) << "For InplaceExponential, the float64 input has not been supported.";
  }

  static const auto float_zero = std::make_shared<FP64Imm>(0.);
  static const auto float_one = std::make_shared<FP64Imm>(1.);
  static const auto neg_one = std::make_shared<Int64Imm>(-1);

  auto uniform_out = inplace_uniform(input, float_zero, float_one, seed, offset);
  auto neg_out = inplace_muls(uniform_out, neg_one);
  auto add_out = inplace_adds_ext(neg_out, float_one, float_one);

  TensorPtr real_out = add_out;
  std::set<TypeId> float_types{kNumberTypeFloat16, kNumberTypeBFloat16, kNumberTypeFloat32};
  if (float_types.find(input_type) != float_types.end()) {
    auto eps = GetEps(input_type);
    auto value = std::make_shared<FP64Imm>(1.0 - eps);
    auto mask = greater_equal_scalar(add_out, value);
    real_out = inplace_masked_fill_scalar(add_out, mask, value);
  }

  real_out = inplace_log(real_out);
  real_out = inplace_muls(real_out, std::make_shared<FP64Imm>(-1.0 / lambda_val));

  MS_LOG(DEBUG) << "InplaceExponentialAscendCustomize Call end";
  op->set_outputs({real_out});
  return real_out;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
