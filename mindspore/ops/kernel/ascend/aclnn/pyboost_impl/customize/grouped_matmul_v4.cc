/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "kernel/ascend/aclnn/pyboost_impl/customize/grouped_matmul_v4.h"

#include <memory>
#include <functional>

#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
std::vector<TensorPtr> ConvertOptiaonlValueTupleToVector(const std::optional<ValueTuplePtr> &tensor_list_opt);

void UnifyWeightShape(const std::vector<TensorPtr> &ori_weights, std::vector<TensorPtr> *new_weights) {
  for (const auto &ori_weight : ori_weights) {
    if (ori_weight->data_type() == kNumberTypeInt4) {
      const auto &storage_info = ori_weight->storage_info();
      if (storage_info != nullptr && !storage_info->is_contiguous) {
        MS_LOG(EXCEPTION) << "Currently, GroupedMatMulV4 does not support noncontiguous input tensor for int4 quant, "
                          << "but got noncontiguous input tensor: " << ori_weight->ToString()
                          << ", storage info: " << storage_info->ToString();
      }
      auto new_weight = std::make_shared<Tensor>(*ori_weight);
      auto ori_weight_shape = ori_weight->shape();
      ori_weight_shape.back() *= kSizeTwo;
      (void)new_weight->set_shape(ori_weight_shape);
      (void)new_weights->emplace_back(new_weight);
    } else {
      (void)new_weights->emplace_back(ori_weight);
    }
  }
}
}  // namespace

void GroupedMatmulV4AscendCustomize(
  const std::shared_ptr<OpRunner> &op, const ValueTuplePtr &x_tensor_list, const ValueTuplePtr &weight_tensor_list,
  const std::optional<ValueTuplePtr> &bias_tensor_list, const std::optional<ValueTuplePtr> &scale_tensor_list,
  const std::optional<ValueTuplePtr> &offset_tensor_list,
  const std::optional<ValueTuplePtr> &antiquant_scale_tensor_list,
  const std::optional<ValueTuplePtr> &antiquant_offset_tensor_list,
  const std::optional<ValueTuplePtr> &pre_token_scale_tensor_list, const std::optional<TensorPtr> &group_list,
  const std::optional<ValueTuplePtr> &activation_input_tensor_list,
  const std::optional<ValueTuplePtr> &activation_quant_scale_tensor_list,
  const std::optional<ValueTuplePtr> &activation_quant_offset_tensor_list, const Int64ImmPtr &split_item_imm,
  const Int64ImmPtr &group_type_imm, const Int64ImmPtr &group_list_type_imm, const Int64ImmPtr &act_type_imm,
  const std::optional<Int64ImmPtr> &output_dtype_imm) {
  MS_LOG(DEBUG) << "Call GroupedMatmulV4 start";
  OpRunner::InferOpOutput(op, x_tensor_list, weight_tensor_list, bias_tensor_list, scale_tensor_list,
                          offset_tensor_list, antiquant_scale_tensor_list, antiquant_offset_tensor_list,
                          pre_token_scale_tensor_list, group_list, activation_input_tensor_list,
                          activation_quant_scale_tensor_list, activation_quant_offset_tensor_list, split_item_imm,
                          group_type_imm, group_list_type_imm, act_type_imm, output_dtype_imm);

  // Convert ValuePtr to c++ scalar
  std::vector<TensorPtr> x = ConvertValueTupleToVector<TensorPtr>(x_tensor_list);
  std::vector<TensorPtr> weight = ConvertValueTupleToVector<TensorPtr>(weight_tensor_list);
  std::vector<TensorPtr> bias = ConvertOptiaonlValueTupleToVector(bias_tensor_list);
  std::vector<TensorPtr> scale = ConvertOptiaonlValueTupleToVector(scale_tensor_list);
  std::vector<TensorPtr> offset = ConvertOptiaonlValueTupleToVector(offset_tensor_list);
  std::vector<TensorPtr> antiquant_scale = ConvertOptiaonlValueTupleToVector(antiquant_scale_tensor_list);
  std::vector<TensorPtr> antiquant_offset = ConvertOptiaonlValueTupleToVector(antiquant_offset_tensor_list);
  std::vector<TensorPtr> pre_token_scale = ConvertOptiaonlValueTupleToVector(pre_token_scale_tensor_list);
  std::vector<TensorPtr> activation_input = ConvertOptiaonlValueTupleToVector(activation_input_tensor_list);
  std::vector<TensorPtr> activation_quant_scale = ConvertOptiaonlValueTupleToVector(activation_quant_scale_tensor_list);
  std::vector<TensorPtr> activation_quant_offset =
    ConvertOptiaonlValueTupleToVector(activation_quant_offset_tensor_list);

  auto split_item = GetValue<int64_t>(split_item_imm);
  auto group_type = GetValue<int64_t>(group_type_imm);
  auto group_list_type = GetValue<int64_t>(group_list_type_imm);
  auto act_type = GetValue<int64_t>(act_type_imm);

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), x, weight, bias, scale, offset, antiquant_scale,
                                antiquant_offset, pre_token_scale, group_list, activation_input, activation_quant_scale,
                                activation_quant_offset);

  std::vector<TensorPtr> activation_feature_out;
  std::vector<TensorPtr> dyn_quant_scale_out;
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  std::vector<TensorPtr> new_weights;
  UnifyWeightShape(weight, &new_weights);

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(
    [op, x, new_weights, weight, bias, scale, offset, antiquant_scale, antiquant_offset, pre_token_scale, group_list,
     activation_input, activation_quant_scale, activation_quant_offset, split_item, group_type, group_list_type,
     act_type, activation_feature_out, dyn_quant_scale_out]() {
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, x, weight, bias, scale, offset, antiquant_scale, antiquant_offset,
                                   pre_token_scale, group_list, activation_input, activation_quant_scale,
                                   activation_quant_offset);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);

      LAUNCH_ACLNN(aclnnGroupedMatmulV4, device_context, op->stream_id(), x, new_weights, bias, scale, offset,
                   antiquant_scale, antiquant_offset, pre_token_scale, group_list, activation_input,
                   activation_quant_scale, activation_quant_offset, split_item, group_type, group_list_type, act_type,
                   outputs, activation_feature_out, dyn_quant_scale_out);
      MS_LOG(DEBUG) << "Launch GroupedMatmulV4 end";
    }));
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
