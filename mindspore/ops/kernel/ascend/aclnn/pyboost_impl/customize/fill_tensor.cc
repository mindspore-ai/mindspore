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

#include "kernel/ascend/aclnn/pyboost_impl/customize/fill_tensor.h"
#include "include/pynative/utils/pyboost/device_put.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_f.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr FillTensorAscendCustomize(const ValueTuplePtr &size, const TensorPtr &fill_value,
                                            const std::optional<Int64ImmPtr> &dtype) {
  auto output = InferOutput<1>(prim::kPrimFillTensor, size, fill_value, dtype);
  bool is_host_tensor = fill_value->device_address() == nullptr && fill_value->isa<Tensor>();
  if (!is_host_tensor) {
    TensorToDevice(fill_value);
  }
  TensorAllocate(output);
  if (is_host_tensor) {
    MS_LOG(INFO) << "For FillTensor, Input [fill_value] is a host tensor, FillScalar will be used.";
    auto value = CreateValueFromTensor(fill_value->cast<TensorPtr>())->cast<ScalarPtr>();
    MS_LOG(DEBUG) << "Call FillScalar start";
    DISPATCH_ACLNN(aclnnInplaceFillScalar, output, value);
    MS_LOG(DEBUG) << "Launch FillScalar end";
    return output;
  }
  MS_LOG(DEBUG) << "Call FillTensor start";
  DISPATCH_ACLNN(aclnnInplaceFillTensor, output, fill_value);
  MS_LOG(DEBUG) << "Launch FillTensor end";
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
