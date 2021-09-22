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

#include <set>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>

#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/tensor_construct_utils.h"
#include "abstract/primitive_infer_map.h"

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

bool IsDynamic(const std::vector<ShapeVector> &shape) {
  for (auto shape_vec : shape) {
    if (std::find(shape_vec.begin(), shape_vec.end(), -1) != shape_vec.end()) {
      return true;
    }
  }
  return false;
}

abstract::AbstractBasePtr GetnextInferShape(const PrimitivePtr &primitive) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto types = GetValue<std::vector<TypePtr>>(primitive->GetAttr("types"));
  ValuePtr shape_attr = primitive->GetAttr("shapes");
  ValuePtr min_shape_attr = primitive->GetAttr("min_shapes");
  ValuePtr max_shape_attr = primitive->GetAttr("max_shapes");

  std::vector<ShapeVector> shape;
  std::vector<ShapeVector> min_shape;
  std::vector<ShapeVector> max_shape;

  GetShapeVector(shape_attr, &shape);
  GetShapeVector(min_shape_attr, &min_shape);
  GetShapeVector(max_shape_attr, &max_shape);

  bool is_dynamic = IsDynamic(shape);

  AbstractBasePtrList output;
  for (size_t i = 0; i < shape.size(); ++i) {
    auto ret_shape = !min_shape.empty() && !max_shape.empty() && is_dynamic
                       ? std::make_shared<abstract::Shape>(shape[i], min_shape[i], max_shape[i])
                       : std::make_shared<abstract::Shape>(shape[i]);
    auto element = std::make_shared<abstract::AbstractScalar>(kAnyValue, types[i]);
    auto tensor = std::make_shared<abstract::AbstractTensor>(element, ret_shape);
    output.push_back(tensor);
  }
  return std::make_shared<abstract::AbstractTuple>(output);
}
}  // namespace

AbstractBasePtr GetNextInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const std::vector<AbstractBasePtr> &input_args) {
  return GetnextInferShape(primitive);
}
REGISTER_PRIMITIVE_EVAL_IMPL(GetNext, prim::kPrimGetNext, GetNextInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
