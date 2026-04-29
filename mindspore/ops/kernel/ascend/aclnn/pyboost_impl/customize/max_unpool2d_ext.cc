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

#include "kernel/ascend/aclnn/pyboost_impl/customize/max_unpool2d_ext.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/device_put.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "kernel/ascend/aclnn/pyboost_impl/auto_generate/sum_ext.h"
#include "kernel/ascend/aclnn/pyboost_impl/auto_generate/ne_scalar.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"

namespace mindspore {
namespace kernel {
namespace pyboost {

tensor::TensorPtr MaxUnpool2DExtAscendCustomize(const TensorPtr &input_tensor, const TensorPtr &indices,
                                                const std::optional<ValueTuplePtr> &kernel_size,
                                                const std::optional<ValueTuplePtr> &stride,
                                                const std::optional<ValueTuplePtr> &padding,
                                                const std::optional<ValueTuplePtr> &ouput_size) {
  auto output =
    InferOutput<1>(prim::kPrimMaxUnpool2DExt, input_tensor, indices, kernel_size, stride, padding, ouput_size);
  std::vector<int64_t> output_size_vector{};
  std::vector<int64_t> shape{output->shape()};
  size_t size = shape.size();

  output_size_vector.push_back(shape[size - kDim2]);
  output_size_vector.push_back(shape[size - kDim1]);
  TensorToDevice(input_tensor, indices);
  TensorAllocate(output);
  DISPATCH_ACLNN(aclnnMaxUnpool2d, input_tensor, indices, output_size_vector, output);
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
