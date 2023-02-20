/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "ops/getnext.h"

#include <vector>
#include <memory>
#include <algorithm>
#include <iterator>

#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/primitive.h"
#include "ir/value.h"
#include "mindapi/base/shape_vector.h"
#include "ops/core_ops.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
void GetShapeVector(const ValuePtr &shape_attr, std::vector<std::vector<int64_t>> *shape_vec) {
  if (shape_attr == nullptr) {
    return;
  }
  std::vector<ValuePtr> shape = shape_attr->isa<ValueTuple>() ? shape_attr->cast<ValueTuplePtr>()->value()
                                                              : shape_attr->cast<ValueListPtr>()->value();
  for (ValuePtr shape_elements : shape) {
    std::vector<ValuePtr> shape_elements_list = shape_elements->isa<ValueTuple>()
                                                  ? shape_elements->cast<ValueTuplePtr>()->value()
                                                  : shape_elements->cast<ValueListPtr>()->value();
    std::vector<int64_t> shape_vec_item;
    (void)std::transform(std::begin(shape_elements_list), std::end(shape_elements_list),
                         std::back_inserter(shape_vec_item),
                         [](const ValuePtr &e) -> int64_t { return GetValue<int64_t>(e); });
    shape_vec->push_back(shape_vec_item);
  }
}

abstract::AbstractBasePtr GetnextInferShape(const PrimitivePtr &primitive) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto types = GetValue<std::vector<TypePtr>>(primitive->GetAttr("types"));
  ValuePtr shape_attr = primitive->GetAttr("shapes");

  std::vector<ShapeVector> shape;
  GetShapeVector(shape_attr, &shape);

  AbstractBasePtrList output;
  for (size_t i = 0; i < shape.size(); ++i) {
    auto ret_shape = std::make_shared<abstract::Shape>(shape[i]);
    auto element = std::make_shared<abstract::AbstractScalar>(kAnyValue, types[i]);
    auto tensor = std::make_shared<abstract::AbstractTensor>(element, ret_shape);
    output.push_back(tensor);
  }
  return std::make_shared<abstract::AbstractTuple>(output);
}
}  // namespace

MIND_API_OPERATOR_IMPL(GetNext, BaseOperator);
AbstractBasePtr GetNextInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const std::vector<AbstractBasePtr> &input_args) {
  return GetnextInferShape(primitive);
}
REGISTER_PRIMITIVE_EVAL_IMPL(GetNext, prim::kPrimGetNext, GetNextInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
