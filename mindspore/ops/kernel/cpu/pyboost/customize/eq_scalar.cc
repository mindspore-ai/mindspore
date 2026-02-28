/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#include "mindspore/ops/kernel/cpu/pyboost/customize/eq_scalar.h"

#include <memory>
#include "mindspore/ops/kernel/cpu/pyboost/auto_generate/equal.h"
#include "ops_utils/op_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void EqScalarCPUCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor, const ScalarPtr &other) {
  MS_LOG(DEBUG) << "EqScalar Call start";
  OpRunner::InferOpOutput(op, input_tensor, other);
  const auto &device_name = device::DeviceType::kCPU;

  // the EqScalar primitive does not support CPU, so use Equal instead.
  const auto equal_op = CREATE_PYBOOST_OP(Equal, device_name);

  // Align scalar dtype by Tensor-Scalar promotion rules (PromoteTensorScalarType).
  const auto input_dtype = input_tensor->Dtype();
  MS_EXCEPTION_IF_NULL(input_dtype);
  MS_EXCEPTION_IF_NULL(other);
  const auto other_dtype = other->type();
  MS_EXCEPTION_IF_NULL(other_dtype);
  const auto promote_type_id = ops::PromoteTensorScalarType(input_dtype->type_id(), other_dtype->type_id());
  if (promote_type_id == kTypeUnknown) {
    MS_LOG(EXCEPTION) << "For 'EqScalar', failed to promote types, tensor dtype: " << input_dtype->ToString()
                      << ", scalar dtype: " << other_dtype->ToString();
  }
  const auto promote_type = TypeIdToType(promote_type_id);
  const auto other_tensor = PyBoostUtils::ScalarToTensor(other, promote_type);

  const auto equal_out = equal_op->Call(input_tensor, other_tensor);
  op->set_outputs({equal_out});
  MS_LOG(DEBUG) << "EqScalar Call end";
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
