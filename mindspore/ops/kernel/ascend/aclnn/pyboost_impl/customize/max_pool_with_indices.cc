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

#include "kernel/ascend/aclnn/pyboost_impl/customize/max_pool_with_indices.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/device_put.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"

namespace mindspore {
namespace kernel {
namespace pyboost {

std::tuple<tensor::TensorPtr, tensor::TensorPtr> MaxPoolWithIndicesAscendCustomize(
  const TensorPtr &x_tensor, const ValueTuplePtr &kernel_size, const std::optional<ValueTuplePtr> &strides,
  const ValueTuplePtr &pads, const ValueTuplePtr &dilation, const BoolImmPtr &ceil_mode,
  const Int64ImmPtr &argmax_type) {
  auto outputs = InferOutput<2>(prim::kPrimMaxPoolWithIndices, x_tensor, kernel_size, strides, pads, dilation,
                                ceil_mode, argmax_type);
  std::vector<int64_t> strides_array;
  if (strides.has_value()) {
    strides_array = ConvertValueTupleToVector<int64_t>(strides.value());
  }
  auto kernel_size_array = ConvertValueTupleToVector<int64_t>(kernel_size);
  auto pads_array = ConvertValueTupleToVector<int64_t>(pads);
  auto dilation_array = ConvertValueTupleToVector<int64_t>(dilation);
  auto ceil_mode_scalar = GetValue<bool>(ceil_mode);
  TensorToDevice(x_tensor);
  TensorAllocate(outputs);
  auto [o0, o1] = outputs;
  DISPATCH_ACLNN(aclnnMaxPool2dWithIndices, x_tensor, kernel_size_array, strides_array, pads_array, dilation_array,
                 ceil_mode_scalar, o0, o1);
  return outputs;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
