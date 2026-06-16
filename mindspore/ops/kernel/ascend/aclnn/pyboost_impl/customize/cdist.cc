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

#include "kernel/ascend/aclnn/pyboost_impl/customize/cdist.h"

#include <cmath>
#include <tuple>
#include <utility>

#include "mindspore/ccsrc/include/pynative/utils/pyboost/functions/auto_generate/functions.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
constexpr float kDefaultPValue = 2.0f;
constexpr int64_t kDefaultComputeMode = 0;
constexpr int64_t kEuclideanComputeMode = 2;
}  // namespace

std::tuple<TensorPtr, TensorPtr> BroadcastCdistBatchDim(const TensorPtr &x1_tensor, const TensorPtr &x2_tensor) {
  const auto &x1_shape = x1_tensor->shape();
  const auto &x2_shape = x2_tensor->shape();
  if (x1_shape.size() != kDim3 || x2_shape.size() != kDim3 || x1_shape[kIndex0] == x2_shape[kIndex0]) {
    return std::make_tuple(x1_tensor, x2_tensor);
  }

  auto x1_tensor_broadcast = x1_tensor;
  auto x2_tensor_broadcast = x2_tensor;
  if (x1_shape[kIndex0] == 1) {
    x1_tensor_broadcast = broadcast_to(x1_tensor, ShapeVector{x2_shape[kIndex0], x1_shape[kIndex1], x1_shape[kIndex2]});
  } else if (x2_shape[kIndex0] == 1) {
    x2_tensor_broadcast = broadcast_to(x2_tensor, ShapeVector{x1_shape[kIndex0], x2_shape[kIndex1], x2_shape[kIndex2]});
  }
  return std::make_tuple(x1_tensor_broadcast, x2_tensor_broadcast);
}

tensor::TensorPtr CdistAscendCustomize(const TensorPtr &x1_tensor, const TensorPtr &x2_tensor, const FP32ImmPtr &p) {
  auto output = InferOutput<1>(prim::kPrimCdist, x1_tensor, x2_tensor, p);
  auto p_value = GetValue<float>(p);
  const auto compute_mode = (p_value == kDefaultPValue) ? kEuclideanComputeMode : kDefaultComputeMode;

  auto [x1_tensor_broadcast, x2_tensor_broadcast] = BroadcastCdistBatchDim(x1_tensor, x2_tensor);
  TensorToDevice(x1_tensor_broadcast, x2_tensor_broadcast);
  TensorMalloc(output);
  DISPATCH_ACLNN(aclnnCdist, x1_tensor_broadcast, x2_tensor_broadcast, p_value, compute_mode, output);
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
