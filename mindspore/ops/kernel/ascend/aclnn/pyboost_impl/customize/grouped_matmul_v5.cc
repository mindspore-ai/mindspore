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

#include "kernel/ascend/aclnn/pyboost_impl/customize/grouped_matmul_v5.h"

#include <algorithm>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

#include "include/pynative/utils/pyboost/infer_output.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_tensor_wrapper.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_g.h"

namespace mindspore::kernel::pyboost {
namespace {
constexpr int64_t kSplitMultiFromSeparate = 0;
constexpr int64_t kSplitMultiFromGrouped = 1;
constexpr int64_t kKGroup = 2;
constexpr int64_t kCumulativeGroupList = 0;
constexpr int64_t kSizeGroupList = 1;
constexpr int64_t kSparseGroupList = 2;

std::vector<TensorPtr> ConvertOptionalTensorList(const std::optional<ValueTuplePtr> &tensor_list) {
  if (!tensor_list.has_value()) {
    return {};
  }
  return ConvertValueTupleToVector<TensorPtr>(tensor_list.value());
}

std::optional<std::vector<int64_t>> ConvertOptionalIntList(const std::optional<ValueTuplePtr> &values) {
  if (!values.has_value()) {
    return std::nullopt;
  }
  return ConvertValueTupleToVector<int64_t>(values.value());
}

aclDataType ConvertTensorType(TypeId type) {
  if (type == kNumberTypeFloat8E8M0FNU) {
    return ACL_FLOAT8_E8M0;
  }
  return device::ascend::AclConverter::ConvertType(type);
}

aclDataType ResolveTensorListType(const std::vector<TensorPtr> &tensors,
                                  const std::optional<Int64ImmPtr> &dtype_override, aclDataType empty_type,
                                  const std::string &arg_name) {
  if (dtype_override.has_value()) {
    const auto type = static_cast<TypeId>(GetValue<int64_t>(dtype_override.value()));
    if (type != kNumberTypeFloat8E8M0FNU) {
      MS_EXCEPTION(TypeError) << "For 'GroupedMatmulV5', " << arg_name << " only supports mindspore.float8_e8m0fnu.";
    }
    return ACL_FLOAT8_E8M0;
  }
  return tensors.empty() ? empty_type : ConvertTensorType(tensors[kIndex0]->data_type());
}

bool IsTransposedWeight(const TensorPtr &weight) {
  const auto &shape = weight->shape();
  const auto &strides = weight->stride();
  if (shape.size() < kIndex2 || strides.size() != shape.size()) {
    return false;
  }
  const auto last = shape.size() - kIndex1;
  const auto penultimate = shape.size() - kIndex2;
  return strides[penultimate] == 1 && strides[last] == shape[penultimate];
}

int64_t GetRuntimeWeightN(const TensorPtr &weight) {
  const auto &shape = weight->shape();
  if (shape.size() < kIndex2) {
    MS_EXCEPTION(ValueError) << "For 'GroupedMatmulV5', weight must have at least two dimensions, but got " << shape;
  }
  if (weight->data_type() == kNumberTypeInt4) {
    return shape.back() * kSizeTwo;
  }
  return IsTransposedWeight(weight) ? shape[shape.size() - kIndex2] : shape.back();
}

std::vector<int64_t> ReadGroupRows(const TensorPtr &group_list, int64_t group_list_type) {
  auto group_list_cpu = group_list->cpu();
  if (group_list_cpu->data_type() != kNumberTypeInt64) {
    MS_EXCEPTION(TypeError) << "For 'GroupedMatmulV5', group_list must have int64 dtype.";
  }
  const auto &shape = group_list_cpu->shape();
  if (shape.empty()) {
    MS_EXCEPTION(ValueError) << "For 'GroupedMatmulV5', group_list must not be scalar.";
  }
  const auto group_count = shape[kIndex0];
  const auto *values = static_cast<const int64_t *>(group_list_cpu->data_c());
  MS_EXCEPTION_IF_NULL(values);
  std::vector<int64_t> rows(LongToSize(group_count));
  int64_t previous = 0;
  for (int64_t i = 0; i < group_count; ++i) {
    if (group_list_type == kCumulativeGroupList) {
      rows[LongToSize(i)] = values[i] - previous;
      previous = values[i];
    } else if (group_list_type == kSizeGroupList) {
      rows[LongToSize(i)] = values[i];
    } else if (group_list_type == kSparseGroupList) {
      rows[LongToSize(i)] = values[i * kIndex2 + kIndex1];
    }
    if (rows[LongToSize(i)] < 0) {
      MS_EXCEPTION(ValueError) << "For 'GroupedMatmulV5', group sizes must be nonnegative, but group " << i
                               << " has size " << rows[LongToSize(i)] << ".";
    }
  }
  return rows;
}

int64_t GetRuntimeWeightNAt(const std::vector<TensorPtr> &weight, bool single_weight, size_t index) {
  return GetRuntimeWeightN(weight[single_weight ? kIndex0 : index]);
}

void SetSeparateOutputShapes(const std::vector<TensorPtr> &x, const std::vector<TensorPtr> &weight, bool single_weight,
                             const std::vector<TensorPtr> &outputs) {
  for (size_t i = 0; i < outputs.size(); ++i) {
    auto shape = x[i]->shape();
    shape.back() = GetRuntimeWeightNAt(weight, single_weight, i);
    (void)outputs[i]->set_shape(shape);
  }
}

void SetGroupedOutputShapes(const std::vector<TensorPtr> &weight, const std::optional<TensorPtr> &group_list,
                            int64_t group_list_type, bool single_weight, const std::vector<TensorPtr> &outputs) {
  if (!group_list.has_value()) {
    MS_EXCEPTION(ValueError) << "For 'GroupedMatmulV5', group_list is required when split_item is 1.";
  }
  const auto rows = ReadGroupRows(group_list.value(), group_list_type);
  if (rows.size() != outputs.size()) {
    MS_EXCEPTION(ValueError) << "For 'GroupedMatmulV5', group count " << rows.size() << " does not match output count "
                             << outputs.size() << ".";
  }
  for (size_t i = 0; i < outputs.size(); ++i) {
    (void)outputs[i]->set_shape({rows[i], GetRuntimeWeightNAt(weight, single_weight, i)});
  }
}

ShapeVector GetRuntimeSingleOutputShape(const std::vector<TensorPtr> &x, const std::vector<TensorPtr> &weight,
                                        const std::optional<TensorPtr> &group_list, int64_t group_type,
                                        bool single_weight) {
  const auto total_m = std::accumulate(x.begin(), x.end(), int64_t{0}, [](int64_t total, const TensorPtr &tensor) {
    return total + tensor->shape()[kIndex0];
  });
  ShapeVector output_shape{total_m, GetRuntimeWeightNAt(weight, single_weight, kIndex0)};
  if (group_type == kKGroup) {
    if (!group_list.has_value()) {
      MS_EXCEPTION(ValueError) << "For 'GroupedMatmulV5', group_list is required when group_type is 2.";
    }
    output_shape.insert(output_shape.begin(), group_list.value()->shape()[kIndex0]);
  }
  return output_shape;
}

void SetRuntimeOutputShapes(const std::vector<TensorPtr> &x, const std::vector<TensorPtr> &weight,
                            const std::optional<TensorPtr> &group_list, int64_t split_item, int64_t group_type,
                            int64_t group_list_type, const std::vector<TensorPtr> &outputs) {
  for (const auto &tensor : weight) {
    if (tensor->data_type() == kNumberTypeInt4 && !tensor->is_contiguous()) {
      MS_EXCEPTION(ValueError) << "For 'GroupedMatmulV5', noncontiguous int4 weight is not supported.";
    }
  }
  const bool single_weight = weight.size() == kIndex1 && weight[kIndex0]->shape().size() == kIndex3;
  if (split_item == kSplitMultiFromSeparate) {
    SetSeparateOutputShapes(x, weight, single_weight, outputs);
    return;
  }
  if (split_item == kSplitMultiFromGrouped) {
    SetGroupedOutputShapes(weight, group_list, group_list_type, single_weight, outputs);
    return;
  }
  (void)outputs[kIndex0]->set_shape(GetRuntimeSingleOutputShape(x, weight, group_list, group_type, single_weight));
}

void UnifyInt4WeightShape(const std::vector<TensorPtr> &weights, std::vector<TensorPtr> *new_weights) {
  MS_EXCEPTION_IF_NULL(new_weights);
  for (const auto &weight : weights) {
    if (weight->data_type() != kNumberTypeInt4) {
      new_weights->push_back(weight);
      continue;
    }
    auto unified_weight = std::make_shared<Tensor>(*weight);
    auto shape = weight->shape();
    shape.back() *= kSizeTwo;
    (void)unified_weight->set_shape(shape);
    new_weights->push_back(std::move(unified_weight));
  }
}
}  // namespace

std::vector<tensor::TensorPtr> GroupedMatmulV5AscendCustomize(
  const ValueTuplePtr &x, const ValueTuplePtr &weight, const std::optional<ValueTuplePtr> &bias,
  const std::optional<ValueTuplePtr> &scale, const std::optional<ValueTuplePtr> &offset,
  const std::optional<ValueTuplePtr> &antiquant_scale, const std::optional<ValueTuplePtr> &antiquant_offset,
  const std::optional<ValueTuplePtr> &per_token_scale, const std::optional<TensorPtr> &group_list,
  const std::optional<ValueTuplePtr> &activation_input, const std::optional<ValueTuplePtr> &activation_quant_scale,
  const std::optional<ValueTuplePtr> &activation_quant_offset, const Int64ImmPtr &split_item,
  const Int64ImmPtr &group_type, const Int64ImmPtr &group_list_type, const Int64ImmPtr &act_type,
  const std::optional<ValueTuplePtr> &tuning_config, const std::optional<Int64ImmPtr> &output_dtype,
  const std::optional<Int64ImmPtr> &scale_dtype, const std::optional<Int64ImmPtr> &per_token_scale_dtype) {
  auto outputs = InferDynamicOutput(
    prim::kPrimGroupedMatmulV5, x, weight, bias, scale, offset, antiquant_scale, antiquant_offset, per_token_scale,
    group_list, activation_input, activation_quant_scale, activation_quant_offset, split_item, group_type,
    group_list_type, act_type, tuning_config, output_dtype, scale_dtype, per_token_scale_dtype);

  auto x_tensors = ConvertValueTupleToVector<TensorPtr>(x);
  auto weight_tensors = ConvertValueTupleToVector<TensorPtr>(weight);
  auto bias_tensors = ConvertOptionalTensorList(bias);
  auto scale_tensors = ConvertOptionalTensorList(scale);
  auto offset_tensors = ConvertOptionalTensorList(offset);
  auto antiquant_scale_tensors = ConvertOptionalTensorList(antiquant_scale);
  auto antiquant_offset_tensors = ConvertOptionalTensorList(antiquant_offset);
  auto per_token_scale_tensors = ConvertOptionalTensorList(per_token_scale);
  auto activation_input_tensors = ConvertOptionalTensorList(activation_input);
  auto activation_quant_scale_tensors = ConvertOptionalTensorList(activation_quant_scale);
  auto activation_quant_offset_tensors = ConvertOptionalTensorList(activation_quant_offset);
  const auto split_item_value = GetValue<int64_t>(split_item);
  const auto group_type_value = GetValue<int64_t>(group_type);
  const auto group_list_type_value = GetValue<int64_t>(group_list_type);
  const auto act_type_value = GetValue<int64_t>(act_type);
  const auto tuning_config_values = ConvertOptionalIntList(tuning_config);

  SetRuntimeOutputShapes(x_tensors, weight_tensors, group_list, split_item_value, group_type_value,
                         group_list_type_value, outputs);
  TensorToDevice(x_tensors, weight_tensors, bias_tensors, scale_tensors, offset_tensors, antiquant_scale_tensors,
                 antiquant_offset_tensors, per_token_scale_tensors, group_list, activation_input_tensors,
                 activation_quant_scale_tensors, activation_quant_offset_tensors);
  TensorMalloc(outputs);

  std::vector<TensorPtr> unified_weights;
  UnifyInt4WeightShape(weight_tensors, &unified_weights);
  const device::ascend::AclTensorListWrapper scale_wrapper = {
    scale_tensors, ResolveTensorListType(scale_tensors, scale_dtype, ACL_UINT64, "scale_dtype")};
  const device::ascend::AclTensorListWrapper per_token_scale_wrapper = {
    per_token_scale_tensors,
    ResolveTensorListType(per_token_scale_tensors, per_token_scale_dtype, ACL_FLOAT, "per_token_scale_dtype")};
  const device::ascend::AclTensorListWrapper antiquant_scale_wrapper = {
    antiquant_scale_tensors, antiquant_scale_tensors.empty()
                               ? ACL_FLOAT16
                               : (antiquant_scale_tensors[kIndex0]->data_type() == kNumberTypeUInt8
                                    ? ACL_FLOAT8_E8M0
                                    : ConvertTensorType(antiquant_scale_tensors[kIndex0]->data_type()))};
  std::vector<TensorPtr> activation_feature_out;
  std::vector<TensorPtr> dyn_quant_scale_out;

  DISPATCH_ACLNN(aclnnGroupedMatmulV5, x_tensors, unified_weights, bias_tensors, scale_wrapper, offset_tensors,
                 antiquant_scale_wrapper, antiquant_offset_tensors, per_token_scale_wrapper, group_list,
                 activation_input_tensors, activation_quant_scale_tensors, activation_quant_offset_tensors,
                 split_item_value, group_type_value, group_list_type_value, act_type_value, tuning_config_values,
                 outputs, activation_feature_out, dyn_quant_scale_out);
  return outputs;
}
}  // namespace mindspore::kernel::pyboost
