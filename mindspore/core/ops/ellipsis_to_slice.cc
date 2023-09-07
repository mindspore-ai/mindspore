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
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
namespace {

abstract::TupleShapePtr EllipsisToSliceInferShape(const PrimitivePtr &primitive,
                                                  const std::vector<AbstractBasePtr> &input_args) {
  const auto &prim_name = primitive->name();
  AbstractBasePtrList elements = input_args;
  const size_t kIntNums = 2;
  (void)CheckAndConvertUtils::CheckInteger("input num", SizeToLong(elements.size()), kEqual, kIntNums, prim_name);
  if (!input_args[1]->isa<abstract::AbstractSequence>()) {
    MS_EXCEPTION(TypeError) << "For '" << prim_name
                            << "', the input data type must be list or tuple of tensors.But got:"
                            << input_args[0]->ToString();
  }
  auto data_shape = input_args[0]->BuildShape()->cast<abstract::ShapePtr>()->shape();
  const size_t kSliceInfoNums = 3;
  // If any shape is dynamic rank, return a dynamic rank.
  if (IsDynamicRank(data_shape)) {
    std::vector<abstract::BaseShapePtr> output_list(
      kSliceInfoNums, std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeDimAny})));
    return std::make_shared<abstract::TupleShape>(output_list);
  }
  std::vector<abstract::BaseShapePtr> output_list(
    kSliceInfoNums, std::make_shared<abstract::Shape>(ShapeVector({SizeToLong(data_shape.size())})));
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
    return std::make_shared<Tuple>(type_tuple);
  }
};
REGISTER_PRIMITIVE_OP_INFER_IMPL(EllipsisToSlice, prim::kPrimEllipsisToSlice, EllipsisToSliceInfer, false);
}  // namespace ops
}  // namespace mindspore
