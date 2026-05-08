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

#include "kernel/ascend/aclnn/pyboost_impl/customize/inplace_fill_tensor.h"

#include <memory>

#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "include/pynative/utils/pyboost/op_register.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr InplaceFillTensorAscendCustomize(const TensorPtr &input, const TensorPtr &value) {
  // No need to convert input
  bool is_host_tensor = value->device_address() == nullptr && value->isa<Tensor>();
  if (!is_host_tensor) {
    TensorToDevice(value);
  }
  TensorToDevice(input);
  if (is_host_tensor) {
    MS_LOG(INFO) << "For InplaceFillTensor, Input [value] is a host tensor, FillScalar will be used.";
    auto value_scalar = CreateValueFromTensor(value->cast<TensorPtr>())->cast<ScalarPtr>();
    DISPATCH_ACLNN(aclnnInplaceFillScalar, input, value_scalar);
  } else {
    DISPATCH_ACLNN(aclnnInplaceFillTensor, input, value);
  }
  return input;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
