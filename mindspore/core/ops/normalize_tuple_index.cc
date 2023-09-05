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

#include "ops/normalize_tuple_index.h"

#include <vector>
#include <string>
#include <algorithm>
#include <memory>

#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "include/common/utils/utils.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/structure_ops.h"

namespace mindspore {
namespace ops {
int64_t NormalizeTupleIndex::CheckRange(int64_t index, int64_t dim_size) {
  if (index >= dim_size || index < -dim_size) {
    MS_EXCEPTION(IndexError) << "Index " << index << " is out of bounds for dimension with size " << dim_size;
  }
  return (dim_size + (index % dim_size)) % dim_size;
}

size_t NormalizeTupleIndex::NormalizeDimIndex(const ShapeVector &data_shape, size_t dim_index,
                                              const std::vector<int64_t> &tuple_index_types, size_t expand_dims_mask) {
  constexpr size_t kMaxDimNums = 8;
  std::bitset<kMaxDimNums> expand_dims_bit(expand_dims_mask);
  size_t ellipse_position = 0;
  size_t not_ellipse_occupy_dims = 0;
  bool has_ellipsis = false;
  if (tuple_index_types.empty()) {
    return 0;
  }
  for (size_t i = 0; i < kMaxDimNums; i++) {
    if (tuple_index_types[i] == kMetaTypeEllipsis) {
      has_ellipsis = true;
      ellipse_position = i;
    } else if (tuple_index_types[i] != kTypeUnknown) {
      not_ellipse_occupy_dims += 1;
    }
  }

  size_t expand_dims_count = 0;
  for (size_t i = 0; i < dim_index; i++) {
    if (expand_dims_bit[i]) {
      expand_dims_count += 1;
    }
  }
  size_t output = 0;
  size_t ellipse_occupy_dims = data_shape.size() + expand_dims_bit.count() - not_ellipse_occupy_dims;
  if (!has_ellipsis || dim_index < ellipse_position) {
    output = dim_index - expand_dims_count;
    return output;
  }
  output = ellipse_occupy_dims + dim_index - 1 - expand_dims_count;
  return output;
}

MIND_API_OPERATOR_IMPL(NormalizeTupleIndex, BaseOperator);
AbstractBasePtr NormalizeIntIndex(const ShapeVector &data_shape, const AbstractBasePtr &index_val_abs, size_t dim_index,
                                  const std::vector<int64_t> &tuple_index_types, size_t expand_dims_mask) {
  AbstractBasePtr output_index_val_abs =
    std::make_shared<abstract::AbstractTensor>(std::make_shared<abstract::AbstractScalar>(kValueAny, kInt64));
  if (IsDynamic(data_shape) || index_val_abs->BuildValue() == kValueAny) {
    return output_index_val_abs;
  }
  auto new_dim_index =
    NormalizeTupleIndex::NormalizeDimIndex(data_shape, dim_index, tuple_index_types, expand_dims_mask);
  int64_t int_index_val = GetValue<int64_t>(index_val_abs->BuildValue());
  int64_t dim = data_shape[new_dim_index];
  int_index_val = NormalizeTupleIndex::CheckRange(int_index_val, dim);
  auto tensor = std::make_shared<tensor::Tensor>(kNumberTypeInt64, ShapeVector{}, &int_index_val, sizeof(int64_t));
  output_index_val_abs = tensor->ToAbstract();
  return output_index_val_abs;
}

AbstractBasePtr NormalizeSequenceIndex(const ShapeVector &data_shape, const AbstractBasePtr &index_val_abs,
                                       size_t dim_index, const std::vector<int64_t> &tuple_index_types,
                                       size_t expand_dims_mask) {
  auto list_index_val_abs = index_val_abs->cast<abstract::AbstractSequencePtr>();
  auto output_list_elements = std::vector<int64_t>();
  if (list_index_val_abs->dynamic_len()) {
    MS_EXCEPTION(IndexError) << "The sequence element(tuple/list) in tuple index can't be dynamic len.";
  }
  const AbstractBasePtrList &list_index_val_ele = list_index_val_abs->elements();
  size_t seq_size = list_index_val_ele.size();
  AbstractBasePtr output_index_val_abs = std::make_shared<abstract::AbstractTensor>(
    kInt64, std::make_shared<abstract::Shape>(ShapeVector({SizeToLong(seq_size)})));
  if (list_index_val_abs->BuildValue() == kValueAny || IsDynamicRank(data_shape)) {
    return output_index_val_abs;
  }
  auto new_dim_index =
    NormalizeTupleIndex::NormalizeDimIndex(data_shape, dim_index, tuple_index_types, expand_dims_mask);
  if (data_shape[new_dim_index] == -1) {
    return output_index_val_abs;
  }
  int64_t dim_size = data_shape[new_dim_index];
  for (size_t i = 0; i < seq_size; i++) {
    int64_t int_index_val = GetValue<int64_t>(list_index_val_ele[i]->BuildValue());
    int_index_val = NormalizeTupleIndex::CheckRange(int_index_val, dim_size);
    output_list_elements.emplace_back(int_index_val);
  }
  auto output = std::make_shared<tensor::Tensor>(output_list_elements);
  return output->ToAbstract();
}

AbstractBasePtr NormalizeNoneIndex(const ShapeVector &data_shape, size_t dim_index,
                                   const std::vector<int64_t> &tuple_index_types) {
  auto output_list_elements = std::vector<int64_t>();
  AbstractBasePtr output_index_val_abs = std::make_shared<abstract::AbstractTensor>(
    kInt64, std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeDimAny})));
  if (IsDynamicRank(data_shape)) {
    return output_index_val_abs;
  }
  auto new_dim_index = NormalizeTupleIndex::NormalizeDimIndex(data_shape, dim_index, tuple_index_types, 0);
  if (data_shape[new_dim_index] == -1) {
    return output_index_val_abs;
  }
  int64_t dim_size = data_shape[new_dim_index];
  for (int64_t i = 0; i < dim_size; i++) {
    (void)output_list_elements.emplace_back(i);
  }
  auto output = std::make_shared<tensor::Tensor>(output_list_elements);
  return output->ToAbstract();
}

AbstractBasePtr NormalizeBoolSequenceIndex(const ShapeVector &data_shape, const AbstractBasePtr &index_val_abs,
                                           size_t dim_index, const std::vector<int64_t> &tuple_index_types,
                                           size_t expand_dims_mask) {
  auto list_index_val_abs = index_val_abs->cast<abstract::AbstractSequencePtr>();
  auto output_list_elements = std::vector<int64_t>();
  if (list_index_val_abs->dynamic_len()) {
    MS_EXCEPTION(IndexError) << "The sequence element(tuple/list) in tuple index can't be dynamic len.";
  }
  // Handle bool list index
  AbstractBasePtr output_index_val_abs = std::make_shared<abstract::AbstractTensor>(
    kInt64, std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeDimAny})));
  if (list_index_val_abs->BuildValue() == kValueAny || IsDynamicRank(data_shape)) {
    return output_index_val_abs;
  }
  const AbstractBasePtrList &list_index_val_ele = list_index_val_abs->elements();
  size_t seq_size = list_index_val_ele.size();
  auto new_dim_index =
    NormalizeTupleIndex::NormalizeDimIndex(data_shape, dim_index, tuple_index_types, expand_dims_mask);
  if (data_shape[new_dim_index] == -1) {
    return output_index_val_abs;
  }
  int64_t dim_size = data_shape[new_dim_index];
  if (SizeToLong(seq_size) != dim_size) {
    MS_EXCEPTION(IndexError) << "dimension is " << dim_size << " but corresponding boolean dimension is " << seq_size;
  }
  for (size_t i = 0; i < seq_size; i++) {
    if (GetValue<bool>(list_index_val_ele[i]->BuildValue())) {
      (void)output_list_elements.emplace_back(SizeToLong(i));
    }
  }
  if (output_list_elements.empty()) {
    MS_EXCEPTION(IndexError) << "The sequence element(tuple/list) in tuple index can't be empty.";
  }
  auto output = std::make_shared<tensor::Tensor>(output_list_elements);
  return output->ToAbstract();
}

AbstractBasePtr NormalizeEllipsisIndex(const ShapeVector &data_shape, size_t dim_index,
                                       const std::vector<int64_t> &tuple_index_types) {
  AbstractBasePtr output_index_val_abs = std::make_shared<abstract::AbstractTensor>(
    kInt64, std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeDimAny})));
  if (IsDynamicRank(data_shape)) {
    return output_index_val_abs;
  }
  size_t ellipse_position = 0;
  size_t not_ellipse_occupy_dims = 0;
  constexpr size_t kMaxDimNums = 8;
  for (size_t i = 0; i < kMaxDimNums; i++) {
    if (tuple_index_types[i] == kMetaTypeEllipsis) {
      ellipse_position = i;
    } else if (tuple_index_types[i] != kTypeUnknown) {
      not_ellipse_occupy_dims += 1;
    }
  }
  size_t ellipse_occupy_dims = data_shape.size() - not_ellipse_occupy_dims;
  if (dim_index >= ellipse_occupy_dims) {
    auto tensor = std::make_shared<tensor::Tensor>(std::vector<int64_t>{1});
    return tensor->ToAbstract();
  }
  size_t ellipse_occupy_dims_i = ellipse_position + dim_index;
  if (data_shape[ellipse_occupy_dims_i] == -1) {
    return output_index_val_abs;
  }
  int64_t ellipse_occupy_dim = data_shape[ellipse_occupy_dims_i];
  std::vector<int64_t> ellipse_to_list;
  for (int64_t i = 0; i < ellipse_occupy_dim; i++) {
    (void)ellipse_to_list.emplace_back(i);
  }
  auto tensor = std::make_shared<tensor::Tensor>(ellipse_to_list);
  return tensor->ToAbstract();
}

AbstractBasePtr NormalizeTupleIndexInferInner(const PrimitivePtr &primitive,
                                              const std::vector<AbstractBasePtr> &input_args) {
  const AbstractBasePtr &data_abs = input_args[kIndex0];
  const AbstractBasePtr &index_val_abs = input_args[kIndex1];
  auto shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(data_abs->BuildShape());
  size_t dim_index = LongToSize(GetValue<int64_t>(primitive->GetAttr(kAttrTupleIndexAxis)));
  auto data_shape = shape_map[kShape];
  string index_types = GetValue<string>(primitive->GetAttr(kAttrOriginIndexType));
  auto tuple_index_types = GetValue<std::vector<int64_t>>(primitive->GetAttr(kAttrTupleIndexTypes));
  size_t expand_dims_mask = 0;
  if (primitive->HasAttr(kAttrExpandDimsMask)) {
    expand_dims_mask = LongToSize(GetValue<int64_t>(primitive->GetAttr(kAttrExpandDimsMask)));
  }
  if (index_types == kIntIndex) {
    return NormalizeIntIndex(data_shape, index_val_abs, dim_index, tuple_index_types, expand_dims_mask);
  }
  if (index_types == kTensorIndexSequenceIndex) {
    return NormalizeSequenceIndex(data_shape, index_val_abs, dim_index, tuple_index_types, expand_dims_mask);
  }
  if (index_types == kBoolSequenceIndex) {
    return NormalizeBoolSequenceIndex(data_shape, index_val_abs, dim_index, tuple_index_types, expand_dims_mask);
  }
  if (index_types == kNoneIndex) {
    return NormalizeNoneIndex(data_shape, dim_index, tuple_index_types);
  }
  return NormalizeEllipsisIndex(data_shape, dim_index, tuple_index_types);
}

class MIND_API NormalizeTupleIndexInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    auto tuple_index_types = GetValue<std::vector<int64_t>>(primitive->GetAttr(kAttrTupleIndexTypes));
    size_t dim_index = LongToSize(GetValue<int64_t>(primitive->GetAttr(kAttrTupleIndexAxis)));
    string index_types = GetValue<string>(primitive->GetAttr(kAttrOriginIndexType));
    size_t expand_dims_mask = 0;
    if (primitive->HasAttr(kAttrExpandDimsMask)) {
      expand_dims_mask = LongToSize(GetValue<int64_t>(primitive->GetAttr(kAttrExpandDimsMask)));
    }
    auto shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
    auto data_shape = shape_map[kShape];
    if (index_types == kIntIndex) {
      return std::make_shared<abstract::Shape>(ShapeVector{});
    }
    if (index_types == kTensorIndexSequenceIndex) {
      auto tensor = GetValue<tensor::TensorPtr>(input_args[1]->BuildValue());
      return std::make_shared<abstract::Shape>(tensor->shape());
    }
    if (index_types == kNoneIndex) {
      ShapeVector shape = ShapeVector({abstract::Shape::kShapeDimAny});
      if (!IsDynamic(data_shape)) {
        auto new_dim_index = NormalizeTupleIndex::NormalizeDimIndex(data_shape, dim_index, tuple_index_types, 0);
        shape = ShapeVector({data_shape[new_dim_index]});
      }
      return std::make_shared<abstract::Shape>(ShapeVector({shape}));
    }
    if (index_types == kBoolSequenceIndex || index_types == kSliceIndex) {
      ShapeVector max_shape;
      if (!IsDynamic(data_shape)) {
        auto new_dim_index =
          NormalizeTupleIndex::NormalizeDimIndex(data_shape, dim_index, tuple_index_types, expand_dims_mask);
        max_shape = ShapeVector({data_shape[new_dim_index]});
      }
      return std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeDimAny}), max_shape);
    }
    return NormalizeEllipsisIndex(data_shape, dim_index, tuple_index_types)->BuildShape();
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    return NormalizeTupleIndexInferInner(prim, input_args)->BuildType();
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return NormalizeTupleIndexInferInner(primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(NormalizeTupleIndex, prim::kPrimNormalizeTupleIndex, NormalizeTupleIndexInfer, false);
}  // namespace ops
}  // namespace mindspore
