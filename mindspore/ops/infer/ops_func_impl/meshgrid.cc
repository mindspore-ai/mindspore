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

#include "infer/ops_func_impl/meshgrid.h"
#include <utility>
#include <memory>
#include "mindspore/ops/ops_utils/op_utils.h"
#include "ops/ops_func_impl/simple_infer.h"

#include "utils/check_convert_utils.h"
#include "include/op_enum.h"

namespace mindspore {
namespace ops {

namespace {
const size_t kMeshgridTupleInputNum = 2;
const size_t MIN_TENSOR_SIZE = 2;
}  // namespace

BaseShapePtr MeshgridFuncImpl::InferShape(const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) const {
  AbstractBasePtrList elements = input_args;
  elements.pop_back();
  if (input_args.size() == kMeshgridTupleInputNum && input_args[kIndex0]->isa<abstract::AbstractSequence>()) {
    elements = input_args[kIndex0]->cast<abstract::AbstractSequencePtr>()->elements();
  }
  (void)CheckAndConvertUtils::CheckInteger("number of input tensors", SizeToLong(elements.size()), kGreaterThan, 1,
                                           primitive->name());
  ShapeVector output_shape;
  for (size_t i = 0; i < elements.size(); ++i) {
    auto shape = elements[i]->GetShape();
    ShapeVector input_shape;
    if (shape->isa<abstract::TensorShape>()) {
      input_shape = shape->GetShapeVector();
    }
    if (IsDynamicRank(input_shape)) {
      auto shape_ptr = std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
      return std::make_shared<abstract::TupleShape>(
        std::vector<abstract::BaseShapePtr>(SizeToLong(elements.size()), shape_ptr));
    }
    (void)CheckAndConvertUtils::CheckInteger("Each input dims", SizeToLong(input_shape.size()), kEqual, 1,
                                             primitive->name());
    output_shape.push_back(input_shape[kIndex0]);
  }

  auto indexing_imm = GetScalarValue<int64_t>(input_args.back()->GetValue());
  if (!indexing_imm.has_value()) {
    return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>(
      SizeToLong(elements.size()),
      std::make_shared<abstract::TensorShape>(ShapeVector(abstract::TensorShape::kShapeDimAny))));
  }

  Indexing indexing = static_cast<Indexing>(indexing_imm.value());
  if (indexing == Indexing::XY && output_shape.size() >= MIN_TENSOR_SIZE) {
    std::swap(output_shape[kIndex0], output_shape[kIndex1]);
  }
  auto shape_ptr = std::make_shared<abstract::Shape>(output_shape);
  return std::make_shared<abstract::TupleShape>(
    std::vector<abstract::BaseShapePtr>(SizeToLong(elements.size()), shape_ptr));
}

TypePtr MeshgridFuncImpl::InferType(const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const {
  AbstractBasePtrList elements = input_args;
  elements.pop_back();
  if (input_args.size() == 2 && input_args[kIndex0]->isa<abstract::AbstractSequence>()) {
    elements = input_args[kIndex0]->cast<abstract::AbstractSequencePtr>()->elements();
  }

  for (size_t i = 0; i < elements.size() - 1; ++i) {
    auto type_ptr_x = elements[i]->GetType();
    MS_EXCEPTION_IF_NULL(type_ptr_x);
    auto type_ptr_y = elements[i + 1]->GetType();
    MS_EXCEPTION_IF_NULL(type_ptr_y);
    MS_CHECK_VALUE(type_ptr_x->type_id() == type_ptr_y->type_id(),
                   "For Primitive [Meshgrid], all tensors should have the same type.");
  }

  return std::make_shared<Tuple>(std::vector<TypePtr>(SizeToLong(elements.size()), elements[kIndex0]->GetType()));
}

}  // namespace ops
}  // namespace mindspore
