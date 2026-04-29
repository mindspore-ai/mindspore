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

#include "kernel/ascend/aclnn/pyboost_impl/customize/sort_ext.h"
#include <tuple>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/device_put.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
std::tuple<tensor::TensorPtr, tensor::TensorPtr> SortExtAscendCustomize(const TensorPtr &input_tensor,
                                                                        const Int64ImmPtr &dim,
                                                                        const BoolImmPtr &descending,
                                                                        const BoolImmPtr &stable) {
  auto outputs = InferOutput<2>(prim::kPrimSortExt, input_tensor, dim, descending, stable);

  auto dim_imm = GetValue<int64_t>(dim);
  auto descending_imm = GetValue<bool>(descending);
  auto stable_imm = GetValue<bool>(stable);

  TensorToDevice(input_tensor);
  TensorAllocate(outputs);
  const auto &output = std::get<0>(outputs);
  const auto &indices = std::get<1>(outputs);
  DISPATCH_ACLNN(aclnnSort, input_tensor, stable_imm, dim_imm, descending_imm, output, indices);
  return outputs;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
