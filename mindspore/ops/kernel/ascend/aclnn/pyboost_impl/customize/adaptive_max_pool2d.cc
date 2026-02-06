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

#include "kernel/ascend/aclnn/pyboost_impl/customize/adaptive_max_pool2d.h"
#include <tuple>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/device_put.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
constexpr int kShapeDim2d = 2;
}  // namespace
std::tuple<tensor::TensorPtr, tensor::TensorPtr> AdaptiveMaxPool2DAscendCustomize(const TensorPtr &input_tensor,
                                                                                  const ValueTuplePtr &output_size) {
  auto outputs = InferOutput<2>(prim::kPrimAdaptiveMaxPool2D, input_tensor, output_size);

  std::vector<int64_t> output_size_vector = ConvertValueTupleToVector<int64_t>(output_size);
  if (output_size_vector.size() != 2) {
    MS_LOG(EXCEPTION) << "For AdaptiveMaxPool2D, the output_size size should be 2.";
  }
  std::vector<int64_t> output_size_update;

  auto input_shape = input_tensor->shape();
  auto input_shape_size = input_shape.size();
  constexpr int kShapeDimNone = -1;
  for (auto i = 0; i < kShapeDim2d; i++) {
    if (output_size_vector[i] != kShapeDimNone) {
      output_size_update.emplace_back(output_size_vector[i]);
    } else {
      output_size_update.emplace_back(input_shape[input_shape_size - kShapeDim2d + i]);
    }
  }

  TensorToDevice(input_tensor);
  TensorAllocate(outputs);

  // Async
  const auto &output = std::get<0>(outputs);
  const auto &indices = std::get<1>(outputs);
  DISPATCH_ACLNN(aclnnAdaptiveMaxPool2d, input_tensor, output_size_update, output, indices);
  return outputs;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
