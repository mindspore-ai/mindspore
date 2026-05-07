/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "kernel/ascend/aclnn/pyboost_impl/customize/tile.h"

#include <vector>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "utils/log_adapter.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"

namespace mindspore::kernel::pyboost {
tensor::TensorPtr TileAscendCustomize(const TensorPtr &input_x_tensor, const ValueTuplePtr &dims) {
  auto output = InferOutput<1>(prim::kPrimTile, input_x_tensor, dims);
  std::vector<int64_t> multiples_vector = ConvertValueTupleToVector<int64_t>(dims);

  // Expand dims with 1 in head when its length is less than x rank.
  MS_EXCEPTION_IF_NULL(input_x_tensor);
  auto x_dim = LongToSize(input_x_tensor->DataDim());
  if (x_dim > multiples_vector.size()) {
    multiples_vector.reserve(x_dim);
    auto expand_len = x_dim - multiples_vector.size();
    (void)multiples_vector.insert(multiples_vector.begin(), expand_len, 1);
  }

  TensorToDevice(input_x_tensor);
  TensorMalloc(output);
  DISPATCH_ACLNN(aclnnRepeat, input_x_tensor, multiples_vector, output);
  return output;
}
}  // namespace mindspore::kernel::pyboost
