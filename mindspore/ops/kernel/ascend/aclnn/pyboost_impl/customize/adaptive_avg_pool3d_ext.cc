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

#include "kernel/ascend/aclnn/pyboost_impl/customize/adaptive_avg_pool3d_ext.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/device_put.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "include/pynative/utils/pyboost/auto_generate/mean_ext.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
constexpr int kShapeDim1d = 1;
constexpr int kShapeDim3d = 3;
}  // namespace
tensor::TensorPtr AdaptiveAvgPool3DExtAscendCustomize(const TensorPtr &input_tensor, const ValueTuplePtr &output_size) {
  auto output = InferOutput<1>(prim::kPrimAdaptiveAvgPool3DExt, input_tensor, output_size);
  std::vector<int64_t> output_size_vector = ConvertValueTupleToVector<int64_t>(output_size);
  std::vector<int64_t> axis_vector{-1, -2, -3};  // {-1, -2, -3}, fixed axis dims for aclnnMean
  const auto keep_dims = true;                   // true, fixed keep_dims for aclnnMean
  TypeId out_dtype = input_tensor->data_type();

  auto input_shape = input_tensor->shape();
  auto input_shape_size = input_shape.size();
  std::vector<int64_t> output_size_update;
  constexpr int kShapeDimNone = -1;
  for (auto i = 0; i < kShapeDim3d; i++) {
    if (output_size_vector[i] != kShapeDimNone)
      output_size_update.emplace_back(output_size_vector[i]);
    else
      output_size_update.emplace_back(input_shape[input_shape_size - kShapeDim3d + i]);
  }

  TensorToDevice(input_tensor);
  TensorAllocate(output);

  if (output_size_update[kIndex0] == kShapeDim1d && output_size_update[kIndex1] == kShapeDim1d &&
      output_size_update[kIndex2] == kShapeDim1d) {
    DISPATCH_ACLNN(aclnnMean, input_tensor, axis_vector, keep_dims, out_dtype, output);
  } else {
    DISPATCH_ACLNN(aclnnAdaptiveAvgPool3d, input_tensor, output_size_update, output);
  }

  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
