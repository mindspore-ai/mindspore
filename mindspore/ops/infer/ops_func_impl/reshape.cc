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
#include "infer/ops_func_impl/reshape.h"
#include <algorithm>
#include <memory>
#include <functional>
#include "mindspore/ops/ops_utils/op_utils.h"

namespace mindspore {
namespace ops {
namespace {
void ReshapeStaticInferDim(const std::vector<int64_t> &input_shape_vec, std::vector<int64_t> *const shape_vec) {
  if (MS_UNLIKELY(IsDynamic(input_shape_vec))) {
    return;
  }
  // If shape has an element -1, and the input shape is known, the -1 dim can be inferred from the remaining
  // dimensions and the number of elements in the input.
  auto input_element_num = std::accumulate(input_shape_vec.begin(), input_shape_vec.end(), static_cast<int64_t>(1),
                                           std::multiplies<int64_t>());
  auto itr = std::find(shape_vec->begin(), shape_vec->end(), -1);
  auto index = LongToSize(std::distance(shape_vec->begin(), itr));
  int64_t new_size = 1;
  (void)std::for_each(shape_vec->begin(), shape_vec->end(),
                      [&new_size](int64_t val) { new_size *= (val > -1 ? val : 1); });
  if (MS_UNLIKELY(new_size == 0)) {
    MS_LOG(WARNING) << "cannot reshape tensor of " << input_element_num << " elements into proposed_shape "
                    << (*shape_vec) << ", because the unspecified dimension size -1 can be any value and is ambiguous";
    (*shape_vec)[index] = 0;
  } else {
    (*shape_vec)[index] = input_element_num / new_size;
  }
}

void ReshapeCheckSize(const PrimitivePtr &primitive, const std::vector<int64_t> &input_shape_vec,
                      const std::vector<int64_t> &shape_vec) {
  if (MS_UNLIKELY(IsDynamic(input_shape_vec))) {
    return;
  }
  auto input_element_num = std::accumulate(input_shape_vec.begin(), input_shape_vec.end(), static_cast<int64_t>(1),
                                           std::multiplies<int64_t>());
  auto shape_number =
    std::accumulate(shape_vec.begin(), shape_vec.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());
  if (input_element_num != shape_number) {
    MS_EXCEPTION(ValueError) << "For primitive[" << primitive->name()
                             << "], the accumulate of x_shape must be equal to out_shape, but got x_shape: "
                             << input_shape_vec << ", and out_shape: " << shape_vec;
  }
}
}  // namespace

BaseShapePtr ReshapeFuncImpl::InferShape(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) const {
  auto input_shape = input_args[kIndex0]->GetShape();
  auto input_shape_vec = input_shape->GetShapeVector();
  auto shape_shape = input_args[kIndex1]->GetShape();
  if (shape_shape->isa<abstract::DynamicSequenceShape>()) {
    return std::make_shared<abstract::TensorShape>(ShapeVector({abstract::Shape::kShapeRankAny}));
  }

  auto shape_array_opt = GetArrayValue<int64_t>(input_args[kIndex1]);
  if (!shape_array_opt.has_value()) {
    if (shape_shape->isa<abstract::SequenceShape>()) {
      auto seq_shape = shape_shape->cast<abstract::SequenceShapePtr>();
      MS_EXCEPTION_IF_NULL(seq_shape);
      size_t shape_size = seq_shape->size();
      return std::make_shared<abstract::TensorShape>(ShapeVector(shape_size, abstract::Shape::kShapeDimAny));
    }
    return std::make_shared<abstract::TensorShape>(ShapeVector{abstract::Shape::kShapeRankAny});
  }

  auto shape_array = shape_array_opt.value();
  if (!shape_array.HasUnknownValue()) {
    std::vector<int64_t> shape_vec = shape_array.ToVector();
    if (std::any_of(shape_vec.begin(), shape_vec.end(), [](const int &shape_i) { return shape_i < -1; })) {
      MS_EXCEPTION(ValueError) << "For primitive[" << primitive->name()
                               << "], the component of shape can't be less than -1, but got " << shape_vec;
    }
    auto self_computed_dim_count = std::count(shape_vec.begin(), shape_vec.end(), -1);
    if (self_computed_dim_count > 1) {
      MS_EXCEPTION(ValueError) << "For primitive[" << primitive->name()
                               << "], at most one component of shape can be -1, but got " << shape_vec;
    }
    if (self_computed_dim_count == 1) {
      ReshapeStaticInferDim(input_shape_vec, &shape_vec);
    }
    ReshapeCheckSize(primitive, input_shape_vec, shape_vec);
    return std::make_shared<abstract::Shape>(shape_vec);
  }

  ShapeVector output_shape;
  int self_computed_dim_count = 0;
  for (size_t i = 0; i < shape_array.size(); i++) {
    if (shape_array.IsValueUnknown(i)) {
      output_shape.push_back(abstract::Shape::kShapeDimAny);
    } else {
      if (shape_array[i] == -1) {
        self_computed_dim_count++;
        if (self_computed_dim_count > 1) {
          MS_EXCEPTION(ValueError) << "For primitive[" << primitive->name()
                                   << "], at most one component of shape can be -1.";
        }
        output_shape.push_back(abstract::Shape::kShapeDimAny);
      } else {
        output_shape.push_back(shape_array[i]);
      }
    }
  }
  return std::make_shared<abstract::Shape>(output_shape);
}

TypePtr ReshapeFuncImpl::InferType(const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) const {
  return input_args[kIndex0]->GetType();
}
}  // namespace ops
}  // namespace mindspore
