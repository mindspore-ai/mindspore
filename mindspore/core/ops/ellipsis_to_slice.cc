/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "ops/ellipsis_to_slice.h"
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/structure_ops.h"
#include "mindspore/core/abstract/dshape.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
namespace {

abstract::TupleShapePtr EllipsisToSliceInferShape(const PrimitivePtr &primitive,
                                                  const std::vector<AbstractBasePtr> &input_args) {
  auto data_shape = input_args[0]->GetShape()->cast<abstract::ShapePtr>()->shape();
  const size_t kSliceInfoNums = 3;

  std::vector<abstract::BaseShapePtr> inner_tuple_elements(data_shape.size(), abstract::kNoShape);
  std::vector<abstract::BaseShapePtr> output_list(kSliceInfoNums,
                                                  std::make_shared<abstract::TupleShape>(inner_tuple_elements));
  return std::make_shared<abstract::TupleShape>(output_list);
}

}  // namespace

MIND_API_OPERATOR_IMPL(EllipsisToSlice, BaseOperator);
class MIND_API EllipsisToSliceInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return EllipsisToSliceInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    std::vector<TypePtr> type_tuple{kInt64, kInt64, kInt64};
    auto data_shape = input_args[0]->GetShape()->cast<abstract::ShapePtr>()->shape();
    std::vector<TypePtr> inner_tuple_types(data_shape.size(), kInt64);
    auto inner_tuple = std::make_shared<Tuple>(inner_tuple_types);
    std::vector<TypePtr> out_tuple_types{inner_tuple, inner_tuple, inner_tuple};
    return std::make_shared<Tuple>(out_tuple_types);
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    AbstractBasePtrList elements = input_args;
    const size_t kInputNums = 4;
    MS_CHECK_VALUE(elements.size() == kInputNums,
                   CheckAndConvertUtils::FormatCheckIntegerMsg("input num", SizeToLong(elements.size()), kEqual,
                                                               kInputNums, primitive));
    auto data_shape = input_args[0]->GetShape()->cast<abstract::ShapePtr>()->shape();
    const size_t kSliceInfoNums = 3;
    auto any_scalar = std::make_shared<abstract::AbstractScalar>(kValueAny, kInt64);
    // If any shape is dynamic rank, return a dynamic rank.
    if (IsDynamicRank(data_shape)) {
      auto any_tuple = std::make_shared<abstract::AbstractTuple>(std::vector<abstract::AbstractBasePtr>({any_scalar}));
      auto dynamic_tuple = any_tuple->BroadenToDynamicLenSequence();
      std::vector<abstract::AbstractBasePtr> tuple_elements(kSliceInfoNums, dynamic_tuple);
      return std::make_shared<abstract::AbstractTuple>(tuple_elements);
    }

    std::vector<abstract::AbstractBasePtr> inner_tuple_elements(data_shape.size(), any_scalar);
    std::vector<abstract::AbstractBasePtr> output_list(kSliceInfoNums,
                                                       std::make_shared<abstract::AbstractTuple>(inner_tuple_elements));
    return std::make_shared<abstract::AbstractTuple>(output_list);
  }
};
REGISTER_PRIMITIVE_OP_INFER_IMPL(EllipsisToSlice, prim::kPrimEllipsisToSlice, EllipsisToSliceInfer, false);
}  // namespace ops
}  // namespace mindspore
