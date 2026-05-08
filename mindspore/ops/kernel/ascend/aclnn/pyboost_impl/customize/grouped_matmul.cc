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

#include "kernel/ascend/aclnn/pyboost_impl/customize/grouped_matmul.h"
#include <functional>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "primitive/auto_generate/gen_ops_primitive_g.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
std::vector<TensorPtr> ConvertOptiaonlValueTupleToVector(const std::optional<ValueTuplePtr> &tensor_list_opt);
}  // namespace
std::vector<tensor::TensorPtr> GroupedMatmulAscendCustomize(
  const ValueTuplePtr &x_tensor_list, const ValueTuplePtr &weight_tensor_list,
  const std::optional<ValueTuplePtr> &bias_tensor_list, const std::optional<ValueTuplePtr> &scale_tensor_list,
  const std::optional<ValueTuplePtr> &offset_tensor_list,
  const std::optional<ValueTuplePtr> &antiquant_scale_tensor_list,
  const std::optional<ValueTuplePtr> &antiquant_offset_tensor_list, const std::optional<TensorPtr> &group_list_tensor,
  const Int64ImmPtr &split_item, const Int64ImmPtr &group_type, const mindspore::BoolImmPtr &transpose_a,
  const mindspore::BoolImmPtr &transpose_b) {
  MS_LOG(DEBUG) << "Call GroupedMatmul start";
  // Convert ValuePtr to c++ scalar
  auto output =
    InferDynamicOutput(prim::kPrimGroupedMatmul, x_tensor_list, weight_tensor_list, bias_tensor_list, scale_tensor_list,
                       offset_tensor_list, antiquant_scale_tensor_list, antiquant_offset_tensor_list, group_list_tensor,
                       split_item, group_type, transpose_a, transpose_b);

  std::vector<TensorPtr> x_tensor_list_vector = ConvertValueTupleToVector<TensorPtr>(x_tensor_list);
  std::vector<TensorPtr> weight_tensor_list_vector = ConvertValueTupleToVector<TensorPtr>(weight_tensor_list);
  std::vector<TensorPtr> bias_tensor_list_vector = ConvertOptiaonlValueTupleToVector(bias_tensor_list);
  std::vector<TensorPtr> scale_tensor_list_vector = ConvertOptiaonlValueTupleToVector(scale_tensor_list);
  std::vector<TensorPtr> offset_tensor_list_vector = ConvertOptiaonlValueTupleToVector(offset_tensor_list);
  std::vector<TensorPtr> antiquant_scale_tensor_list_vector =
    ConvertOptiaonlValueTupleToVector(antiquant_scale_tensor_list);
  std::vector<TensorPtr> antiquant_offset_tensor_list_vector =
    ConvertOptiaonlValueTupleToVector(antiquant_offset_tensor_list);

  auto split_item_imm = GetValue<int64_t>(split_item);
  auto group_type_imm = GetValue<int64_t>(group_type);

  TensorToDevice(x_tensor_list_vector, weight_tensor_list_vector, bias_tensor_list_vector, scale_tensor_list_vector,
                 offset_tensor_list_vector, antiquant_scale_tensor_list_vector, antiquant_offset_tensor_list_vector,
                 group_list_tensor);
  TensorMalloc(output);

  DISPATCH_ACLNN(aclnnGroupedMatmulV3, x_tensor_list_vector, weight_tensor_list_vector, bias_tensor_list_vector,
                 scale_tensor_list_vector, offset_tensor_list_vector, antiquant_scale_tensor_list_vector,
                 antiquant_offset_tensor_list_vector, group_list_tensor, split_item_imm, group_type_imm, output);
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
