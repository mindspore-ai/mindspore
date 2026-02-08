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

#include <map>
#include <memory>
#include "ops/ops_frontend_func_impl.h"
#include "ops_utils/op_utils.h"
#include "utils/log_adapter.h"
#include "abstract/abstract_value.h"

#include "utils/check_convert_utils.h"
#include "include/op_enum.h"

namespace mindspore {
namespace ops {

class MeshgridFrontendFuncImpl : public OpFrontendFuncImpl {
 public:
  AbstractBasePtr InferAbstract(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
    auto input_abs = input_args[kIndex0];
    auto input_shape = input_abs->GetShape();
    auto input_type = input_abs->GetType();
    auto tuple_type = input_type->cast<TuplePtr>();
    auto elements = tuple_type->elements();
    (void)CheckAndConvertUtils::CheckInteger("number of input tensors", SizeToLong(elements.size()), kGreaterThan, 1,
                                             primitive->name());

    for (size_t i = 0; i < elements.size() - 1; ++i) {
      auto type_ptr_x = elements[i]->cast<TensorTypePtr>();
      MS_EXCEPTION_IF_NULL(type_ptr_x);
      auto type_ptr_y = elements[i + 1]->cast<TensorTypePtr>();
      MS_EXCEPTION_IF_NULL(type_ptr_y);
      MS_CHECK_VALUE(type_ptr_x->element()->type_id() == type_ptr_y->element()->type_id(),
                     "For Primitive [Meshgrid], all tensors should have the same type.");
    }
    auto first_element = elements[kIndex0]->cast<TensorTypePtr>();
    auto element_type = first_element->element();
    AbstractBasePtrList output_list{};

    auto is_dynamic_sequence = input_shape->isa<abstract::DynamicSequenceShape>();
    if (is_dynamic_sequence) {
      auto dynamic_shape = std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
      output_list.push_back(abstract::MakeAbstractTensor(dynamic_shape, element_type));
      auto abs_tuple = std::make_shared<abstract::AbstractTuple>(output_list);
      abs_tuple->CheckAndConvertToDynamicLenSequence();
      return abs_tuple;
    }

    auto tuple_shape = input_shape->cast<abstract::TupleShapePtr>();
    auto out_rank = tuple_shape->size();
    ShapeVector element_shape;
    auto indexing_value = input_args.back()->GetValue();
    auto indexing_opt = GetScalarValue<int64_t>(indexing_value);
    if (!indexing_opt.has_value()) {
      element_shape = ShapeVector(out_rank, abstract::TensorShape::kShapeDimAny);
    } else {
      auto indexing_res = static_cast<ops::Indexing>(indexing_opt.value());
      for (size_t i = 0; i < tuple_shape->size(); ++i) {
        auto single_shape = (*tuple_shape)[i]->GetShapeVector();
        (void)CheckAndConvertUtils::CheckInteger("Each input dims", SizeToLong(single_shape.size()), kEqual, 1,
                                                 primitive->name());
        if (IsDynamicRank(single_shape)) {
          element_shape.push_back(abstract::TensorShape::kShapeDimAny);
        } else {
          element_shape.push_back(single_shape[kIndex0]);
        }
      }
      if (indexing_res == ops::Indexing::XY) {
        std::swap(element_shape[kIndex0], element_shape[kIndex1]);
      }
    }

    output_list.reserve(out_rank);
    for (size_t i = 0; i < out_rank; ++i) {
      output_list.push_back(std::make_shared<abstract::AbstractTensor>(element_type, element_shape));
    }

    auto out_tuple = std::make_shared<abstract::AbstractTuple>(output_list);
    return out_tuple;
  }
};

REGISTER_PRIMITIVE_FUNCTION_FRONTEND_FUNC_IMPL("Meshgrid", MeshgridFrontendFuncImpl);
}  // namespace ops
}  // namespace mindspore
