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

#include "ops/normalize_dim_index.h"

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
constexpr size_t kMaxDimNums = 8;
size_t NormalizeDimIndex::ConstNormalizeDimIndex(size_t data_dims, size_t dim_index,
                                                 const std::vector<int64_t> &tuple_index_types,
                                                 size_t expand_dims_mask) {
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
  size_t ellipse_occupy_dims = data_dims + expand_dims_bit.count() - not_ellipse_occupy_dims;
  if (!has_ellipsis || dim_index < ellipse_position) {
    output = dim_index - expand_dims_count;
    return output;
  }
  output = ellipse_occupy_dims + dim_index - 1 - expand_dims_count;
  return output;
}

MIND_API_OPERATOR_IMPL(NormalizeDimIndex, BaseOperator);
AbstractBasePtr NormalizeDimIndexInferInner(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const std::string op_name = primitive->name();
  const size_t inputs_size = 1;
  CheckArgsSize(op_name, input_args, inputs_size);
  const AbstractBasePtr &data_abs = input_args[kIndex0];
  ShapeVector data_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(data_abs->BuildShape())[kShape];
  if (IsDynamicRank(data_shape)) {
    return std::make_shared<abstract::AbstractScalar>(kInt64);
  }
  size_t dim_index = LongToSize(GetValue<int64_t>(primitive->GetAttr(kAttrTupleIndexAxis)));
  size_t expand_dims_cnt = LongToSize(GetValue<int64_t>(primitive->GetAttr(kAttrExpandDimsCnt)));
  auto tuple_index_types = GetValue<std::vector<int64_t>>(primitive->GetAttr(kAttrTupleIndexTypes));
  size_t normalize_dim_index = NormalizeDimIndex::ConstNormalizeDimIndex(
    data_shape.size() + expand_dims_cnt, static_cast<size_t>(dim_index), tuple_index_types, 0);
  return std::make_shared<abstract::AbstractScalar>(SizeToLong(normalize_dim_index));
}

class MIND_API NormalizeDimIndexInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return NormalizeDimIndexInferInner(primitive, input_args)->BuildShape();
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    return NormalizeDimIndexInferInner(prim, input_args)->BuildType();
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return NormalizeDimIndexInferInner(primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(NormalizeDimIndex, prim::kPrimNormalizeDimIndex, NormalizeDimIndexInfer, false);
}  // namespace ops
}  // namespace mindspore
