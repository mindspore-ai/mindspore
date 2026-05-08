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

#include "kernel/ascend/aclnn/pyboost_impl/customize/log_softmax_ext.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "primitive/auto_generate/gen_ops_primitive_l.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
int64_t GetDimValue(const TensorPtr &input_tensor) {
  const auto &shape = input_tensor->shape();
  size_t ndim = shape.size();
  int64_t ret;
  if (ndim == kDim0 || ndim == kDim1 || ndim == kDim3) {
    ret = 0;
  } else {
    ret = 1;
  }
  return ret;
}
}  // namespace

tensor::TensorPtr LogSoftmaxExtAscendCustomize(const TensorPtr &input_tensor, const std::optional<Int64ImmPtr> &dim,
                                               const std::optional<Int64ImmPtr> &dtype) {
  MS_LOG(DEBUG) << "LogSoftmaxExt call start";
  auto output = InferOutput<1>(prim::kPrimLogSoftmaxExt, input_tensor, dim, dtype);

  int64_t dim_value;
  if (dim.has_value()) {
    dim_value = GetValue<int64_t>(*dim);
  } else {
    dim_value = GetDimValue(input_tensor);
  }
  auto new_tensor = input_tensor;
  if (dtype.has_value()) {
    auto dtype_value = static_cast<TypeId>(GetValue<int64_t>(*dtype));
    new_tensor = PyBoostUtils::CastTensor(input_tensor, dtype_value, device::DeviceType::kAscend);
  }

  TensorToDevice(new_tensor);
  TensorMalloc(output);

  DISPATCH_ACLNN(aclnnLogSoftmax, new_tensor, dim_value, output);
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
