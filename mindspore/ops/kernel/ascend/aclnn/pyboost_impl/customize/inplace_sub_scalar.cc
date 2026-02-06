/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "kernel/ascend/aclnn/pyboost_impl/customize/inplace_sub_scalar.h"
#include <memory>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/op_register.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "include/pynative/utils/pyboost/device_put.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr InplaceSubScalarAscendCustomize(const TensorPtr &input_tensor, const ScalarPtr &other,
                                                  const ScalarPtr &alpha) {
  MS_LOG(DEBUG) << "Call InplaceSubScalar start";
  if ((PyBoostUtils::IsBool(alpha) && (!PyBoostUtils::IsBool(input_tensor) || !PyBoostUtils::IsBool(other)))) {
    MS_EXCEPTION(TypeError) << "For InplaceSubScalar"
                            << ", Boolean alpha only supported for Boolean results.";
  }
  if ((PyBoostUtils::IsFloat(alpha) && (!PyBoostUtils::IsFloat(input_tensor) || !PyBoostUtils::IsFloat(other)))) {
    MS_EXCEPTION(TypeError) << "For InplaceSubScalar"
                            << ", For integral input tensors, argument alpha must not be a floating point number";
  }
  TensorToDevice(input_tensor);
  DISPATCH_ACLNN(aclnnInplaceSubs, input_tensor, other, alpha);
  return input_tensor;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
