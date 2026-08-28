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

#include "infer/ops_func_impl/grouped_matmul_v5.h"

#include <algorithm>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "abstract/ops/primitive_infer_map.h"
#include "ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore::ops {
namespace {
constexpr size_t kXIndex = 0;
constexpr size_t kWeightIndex = 1;
constexpr size_t kGroupListIndex = 8;
constexpr size_t kSplitItemIndex = 12;
constexpr size_t kGroupTypeIndex = 13;
constexpr size_t kGroupListTypeIndex = 14;
constexpr size_t kOutputDtypeIndex = 17;
constexpr size_t kScaleDtypeIndex = 18;
constexpr size_t kPerTokenScaleDtypeIndex = 19;

constexpr int64_t kSplitMultiFromSeparate = 0;
constexpr int64_t kSplitMultiFromGrouped = 1;
constexpr int64_t kSplitSingleFromSeparate = 2;
constexpr int64_t kSplitSingleFromGrouped = 3;
constexpr int64_t kNoGroup = -1;
constexpr int64_t kKGroup = 2;
constexpr int64_t kCumulativeGroupList = 0;
constexpr int64_t kSizeGroupList = 1;
constexpr int64_t kSparseGroupList = 2;

std::vector<InferInfoPtr> GetTensorSequence(const PrimitivePtr &primitive, const InferInfoPtr &input,
                                            const std::string &arg_name) {
  if (MS_UNLIKELY(!input->IsSequence())) {
    MS_EXCEPTION(TypeError) << "For '" << primitive->name() << "', '" << arg_name
                            << "' must be a tuple or list of Tensors.";
  }
  if (MS_UNLIKELY(input->IsDynamicSequence())) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', dynamic-length '" << arg_name
                             << "' is not supported.";
  }
  auto elements = input->GetSequenceElements();
  if (MS_UNLIKELY(elements.empty())) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', '" << arg_name << "' must not be empty.";
  }
  return elements;
}

ShapeArray GetShapes(const std::vector<InferInfoPtr> &elements) {
  ShapeArray shapes;
  std::transform(elements.begin(), elements.end(), std::back_inserter(shapes),
                 [](const InferInfoPtr &info) { return info->GetShape(); });
  return shapes;
}

void CheckTensorRanks(const PrimitivePtr &primitive, const ShapeArray &shapes, const std::string &arg_name) {
  for (size_t i = 0; i < shapes.size(); ++i) {
    if (!IsDynamicRank(shapes[i]) && shapes[i].size() < kIndex2) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', " << arg_name << "[" << i
                               << "] must have at least two dimensions, but got shape " << shapes[i] << ".";
    }
  }
}

bool IsSingleThreeDimensionalWeight(const ShapeArray &weight_shapes) {
  return weight_shapes.size() == kIndex1 && !IsDynamicRank(weight_shapes[0]) && weight_shapes[0].size() == kIndex3;
}

int64_t GetWeightN(const ShapeVector &weight_shape, TypeId weight_type) {
  if (IsDynamicRank(weight_shape)) {
    return abstract::Shape::kShapeDimAny;
  }
  auto n = weight_shape.back();
  if (weight_type == kNumberTypeInt4 && n != abstract::Shape::kShapeDimAny) {
    n = n << 1;
  }
  return n;
}

int64_t GetGroupCount(const PrimitivePtr &primitive, const InferInfoPtr &group_list, int64_t group_list_type) {
  const auto &shape = group_list->GetShape();
  if (IsDynamicRank(shape)) {
    return abstract::Shape::kShapeDimAny;
  }
  const size_t expected_rank = group_list_type == kSparseGroupList ? kIndex2 : kIndex1;
  if (MS_UNLIKELY(shape.size() != expected_rank)) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', group_list must be " << expected_rank
                             << "D when group_list_type is " << group_list_type << ", but got shape " << shape << ".";
  }
  if (group_list_type == kSparseGroupList && shape[kIndex1] != abstract::Shape::kShapeDimAny &&
      shape[kIndex1] != kIndex2) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                             << "', group_list.shape[1] must be 2 when group_list_type is 2, but got " << shape[kIndex1]
                             << ".";
  }
  return shape[kIndex0];
}

std::optional<std::vector<int64_t>> GetKnownGroupRows(const PrimitivePtr &primitive, const InferInfoPtr &group_list,
                                                      int64_t group_list_type, int64_t group_count) {
  const auto values = group_list->GetArrayValue<int64_t>();
  if (!values.has_value() || values->HasUnknownValue()) {
    return std::nullopt;
  }
  const auto &flat_values = values->ToVector();
  const int64_t expected_size = group_list_type == kSparseGroupList ? group_count * SizeToLong(kIndex2) : group_count;
  if (group_count < 0 || SizeToLong(flat_values.size()) != expected_size) {
    return std::nullopt;
  }

  std::vector<int64_t> rows(LongToSize(group_count));
  int64_t previous = 0;
  for (int64_t i = 0; i < group_count; ++i) {
    if (group_list_type == kCumulativeGroupList) {
      const auto current = flat_values[LongToSize(i)];
      rows[LongToSize(i)] = current - previous;
      previous = current;
    } else if (group_list_type == kSizeGroupList) {
      rows[LongToSize(i)] = flat_values[LongToSize(i)];
    } else {
      rows[LongToSize(i)] = flat_values[LongToSize(i * kIndex2 + kIndex1)];
    }
    if (MS_UNLIKELY(rows[LongToSize(i)] < 0)) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', group sizes must be nonnegative, but group " << i
                               << " has size " << rows[LongToSize(i)] << ".";
    }
  }
  return rows;
}

int64_t SumLeadingDimensions(const ShapeArray &x_shapes) {
  int64_t total = 0;
  for (const auto &shape : x_shapes) {
    if (IsDynamicRank(shape) || shape[kIndex0] == abstract::Shape::kShapeDimAny) {
      return abstract::Shape::kShapeDimAny;
    }
    total += shape[kIndex0];
  }
  return total;
}

void CheckDtypeOverride(const PrimitivePtr &primitive, const InferInfoPtr &dtype_info, const std::string &arg_name,
                        bool *need_retry) {
  if (dtype_info->IsNone()) {
    return;
  }
  const auto dtype = dtype_info->GetScalarValue<int64_t>();
  if (!dtype.has_value()) {
    *need_retry = true;
    return;
  }
  if (MS_UNLIKELY(static_cast<TypeId>(dtype.value()) != kNumberTypeFloat8E8M0FNU)) {
    MS_EXCEPTION(TypeError) << "For '" << primitive->name() << "', '" << arg_name
                            << "' only supports mindspore.float8_e8m0fnu.";
  }
}

int64_t GetWeightNAt(const PrimitivePtr &primitive, const ShapeArray &weight_shapes,
                     const std::vector<InferInfoPtr> &weight_elements, bool single_weight, size_t index) {
  const size_t weight_index = single_weight ? kIndex0 : index;
  if (MS_UNLIKELY(weight_index >= weight_shapes.size())) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                             << "', the weight sequence does not contain enough tensors for the requested outputs.";
  }
  return GetWeightN(weight_shapes[weight_index], weight_elements[weight_index]->GetType());
}

ShapeArray InferMultiFromSeparate(const PrimitivePtr &primitive, const ShapeArray &x_shapes,
                                  const ShapeArray &weight_shapes, const std::vector<InferInfoPtr> &weight_elements,
                                  bool single_weight) {
  if (!single_weight && x_shapes.size() != weight_shapes.size()) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                             << "' with split_item=0, x and weight must have the same length unless weight contains "
                                "one 3D tensor, but got "
                             << x_shapes.size() << " and " << weight_shapes.size() << ".";
  }
  ShapeArray outputs;
  for (size_t i = 0; i < x_shapes.size(); ++i) {
    if (IsDynamicRank(x_shapes[i])) {
      outputs.push_back({abstract::Shape::kShapeRankAny});
    } else {
      auto output_shape = x_shapes[i];
      output_shape.back() = GetWeightNAt(primitive, weight_shapes, weight_elements, single_weight, i);
      outputs.push_back(std::move(output_shape));
    }
  }
  return outputs;
}

ShapeArray InferMultiFromGrouped(const PrimitivePtr &primitive, const InferInfoPtr &group_list, int64_t group_list_type,
                                 const ShapeArray &x_shapes, const ShapeArray &weight_shapes,
                                 const std::vector<InferInfoPtr> &weight_elements, bool single_weight) {
  if (MS_UNLIKELY(group_list->IsNone())) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', group_list is required when split_item is 1.";
  }
  if (MS_UNLIKELY(x_shapes.size() != kIndex1)) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "' with split_item=1, x must contain one tensor.";
  }
  const auto group_count = GetGroupCount(primitive, group_list, group_list_type);
  if (MS_UNLIKELY(group_count == abstract::Shape::kShapeDimAny)) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                             << "' with split_item=1, group_list.shape[0] must be static because it determines the "
                                "output tuple length.";
  }
  if (!single_weight && SizeToLong(weight_shapes.size()) != group_count) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                             << "' with split_item=1, weight must contain one tensor per group, but got "
                             << weight_shapes.size() << " weights and " << group_count << " groups.";
  }
  const auto rows = GetKnownGroupRows(primitive, group_list, group_list_type, group_count);
  ShapeArray outputs;
  for (int64_t i = 0; i < group_count; ++i) {
    const auto m = rows.has_value() ? rows.value()[LongToSize(i)] : abstract::Shape::kShapeDimAny;
    outputs.push_back({m, GetWeightNAt(primitive, weight_shapes, weight_elements, single_weight, LongToSize(i))});
  }
  return outputs;
}

ShapeArray InferSingleOutput(const PrimitivePtr &primitive, const InferInfoPtr &group_list, int64_t group_list_type,
                             int64_t group_type, const ShapeArray &x_shapes, const ShapeArray &weight_shapes,
                             const std::vector<InferInfoPtr> &weight_elements, bool single_weight) {
  const auto m = SumLeadingDimensions(x_shapes);
  const auto n = GetWeightNAt(primitive, weight_shapes, weight_elements, single_weight, kIndex0);
  if (group_type != kKGroup) {
    return {{m, n}};
  }
  if (MS_UNLIKELY(group_list->IsNone())) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', group_list is required when group_type is 2.";
  }
  return {{GetGroupCount(primitive, group_list, group_list_type), m, n}};
}
}  // namespace

int32_t GroupedMatmulV5FuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                                 const InferInfoPtrList &input_infos) const {
  (void)GetTensorSequence(primitive, input_infos[kXIndex], "x");
  (void)GetTensorSequence(primitive, input_infos[kWeightIndex], "weight");
  bool need_retry = false;

  const auto split_item = input_infos[kSplitItemIndex]->GetScalarValue<int64_t>();
  if (split_item.has_value()) {
    static const std::set<int64_t> valid_split_items = {kSplitMultiFromSeparate, kSplitMultiFromGrouped,
                                                        kSplitSingleFromSeparate, kSplitSingleFromGrouped};
    if (MS_UNLIKELY(valid_split_items.count(split_item.value()) == 0)) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', split_item must be 0, 1, 2, or 3, but got "
                               << split_item.value() << ".";
    }
  } else {
    need_retry = true;
  }

  const auto group_type = input_infos[kGroupTypeIndex]->GetScalarValue<int64_t>();
  if (group_type.has_value()) {
    static const std::set<int64_t> valid_group_types = {kNoGroup, 0, kKGroup};
    if (MS_UNLIKELY(valid_group_types.count(group_type.value()) == 0)) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', group_type must be -1, 0, or 2, but got "
                               << group_type.value() << ".";
    }
  } else {
    need_retry = true;
  }

  const auto group_list_type = input_infos[kGroupListTypeIndex]->GetScalarValue<int64_t>();
  if (group_list_type.has_value()) {
    static const std::set<int64_t> valid_group_list_types = {kCumulativeGroupList, kSizeGroupList, kSparseGroupList};
    if (MS_UNLIKELY(valid_group_list_types.count(group_list_type.value()) == 0)) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', group_list_type must be 0, 1, or 2, but got "
                               << group_list_type.value() << ".";
    }
  } else {
    need_retry = true;
  }

  const auto &group_list = input_infos[kGroupListIndex];
  if (!group_list->IsNone()) {
    if (MS_UNLIKELY(group_list->GetType() != kNumberTypeInt64)) {
      MS_EXCEPTION(TypeError) << "For '" << primitive->name() << "', group_list must have int64 dtype, but got "
                              << TypeIdToString(group_list->GetType()) << ".";
    }
    if (group_list_type.has_value()) {
      (void)GetGroupCount(primitive, group_list, group_list_type.value());
    }
  } else if ((group_type.has_value() && group_type.value() != kNoGroup) ||
             (split_item.has_value() && split_item.value() == kSplitMultiFromGrouped)) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                             << "', group_list is required for grouped computation and split_item=1.";
  }

  CheckDtypeOverride(primitive, input_infos[kScaleDtypeIndex], "scale_dtype", &need_retry);
  CheckDtypeOverride(primitive, input_infos[kPerTokenScaleDtypeIndex], "per_token_scale_dtype", &need_retry);
  return need_retry ? OP_CHECK_RETRY : OP_CHECK_SUCCESS;
}

ShapeArray GroupedMatmulV5FuncImpl::InferShape(const PrimitivePtr &primitive,
                                               const InferInfoPtrList &input_infos) const {
  const auto x_elements = GetTensorSequence(primitive, input_infos[kXIndex], "x");
  const auto weight_elements = GetTensorSequence(primitive, input_infos[kWeightIndex], "weight");
  const auto x_shapes = GetShapes(x_elements);
  const auto weight_shapes = GetShapes(weight_elements);
  CheckTensorRanks(primitive, x_shapes, "x");
  CheckTensorRanks(primitive, weight_shapes, "weight");

  const auto split_item = input_infos[kSplitItemIndex]->GetScalarValue<int64_t>();
  const auto group_type = input_infos[kGroupTypeIndex]->GetScalarValue<int64_t>();
  const auto group_list_type = input_infos[kGroupListTypeIndex]->GetScalarValue<int64_t>();
  if (!split_item.has_value() || !group_type.has_value() || !group_list_type.has_value()) {
    return {{abstract::Shape::kShapeRankAny}};
  }

  const bool single_weight = IsSingleThreeDimensionalWeight(weight_shapes);
  if (split_item.value() == kSplitMultiFromSeparate) {
    return InferMultiFromSeparate(primitive, x_shapes, weight_shapes, weight_elements, single_weight);
  }

  if (split_item.value() == kSplitMultiFromGrouped) {
    return InferMultiFromGrouped(primitive, input_infos[kGroupListIndex], group_list_type.value(), x_shapes,
                                 weight_shapes, weight_elements, single_weight);
  }

  return InferSingleOutput(primitive, input_infos[kGroupListIndex], group_list_type.value(), group_type.value(),
                           x_shapes, weight_shapes, weight_elements, single_weight);
}

TypeIdList GroupedMatmulV5FuncImpl::InferType(const PrimitivePtr &primitive,
                                              const InferInfoPtrList &input_infos) const {
  const auto x_elements = GetTensorSequence(primitive, input_infos[kXIndex], "x");
  TypeId output_type = x_elements[kIndex0]->GetType();
  if (!input_infos[kOutputDtypeIndex]->IsNone()) {
    const auto output_dtype = input_infos[kOutputDtypeIndex]->GetScalarValue<int64_t>();
    if (output_dtype.has_value()) {
      output_type = static_cast<TypeId>(output_dtype.value());
    }
  }
  const auto output_count = InferShape(primitive, input_infos).size();
  return TypeIdList(output_count, output_type);
}
}  // namespace mindspore::ops
