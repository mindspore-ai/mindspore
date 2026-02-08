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
#include "infer/ops_func_impl/broadcast_to.h"
#include <algorithm>
#include <memory>
#include <string>
#include <set>
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/shape_utils.h"
#include "mindspore/ops/ops_utils/op_utils.h"

namespace mindspore {
namespace ops {
namespace {
ShapeVector CheckShapeValid(const ShapeVector &x_shape, const ShapeVector &proposed_shape,
                            const std::string &prim_name) {
  auto res_shape = proposed_shape;
  if (IsDynamicRank(x_shape)) {
    return res_shape;
  }
  MS_CHECK_VALUE(
    x_shape.size() <= proposed_shape.size(),
    "For primitive [BroadcastTo]: input's rank should be less equal to the number of proposed_shape, but got " +
      std::to_string(x_shape.size()) + " and " + std::to_string(proposed_shape.size()));
  auto outer_dim_offset = res_shape.size() - x_shape.size();
  bool need_compute_shape = true;
  if (res_shape.end() == find(res_shape.begin(), res_shape.end(), -1)) {
    need_compute_shape = false;
  } else {
    need_compute_shape = true;
  }

  if (need_compute_shape) {
    for (size_t i = 0; i < res_shape.size(); i++) {
      if (res_shape[i] == -1) {
        if (i < outer_dim_offset) {
          MS_EXCEPTION(ValueError) << "For '" << prim_name
                                   << "', -1 in init shape is in an incompatible "
                                      "location with given input tensor, -1 index in init shape: "
                                   << i << " but -1 can only be in index" << x_shape.size()
                                   << "onwards for this input.";
        }
        res_shape[i] = x_shape[i - outer_dim_offset];
      }
    }
  }

  for (size_t i = 0; i < x_shape.size(); i++) {
    if (x_shape[i] == -1) {
      continue;
    }
    if (res_shape[i + outer_dim_offset] != x_shape[i] && x_shape[i] != 1) {
      MS_EXCEPTION(ValueError)
        << "For '" << prim_name
        << "', in order to broadcast, each dimension pair must be equal or input dimension is 1 or target "
           "dimension is -1. But got x_shape: "
        << x_shape << ", target shape: " << res_shape << ".";
    }
  }
  return res_shape;
}
}  // namespace

BaseShapePtr BroadcastToFuncImpl::InferShape(const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto x_shape = input_args[kIndex0]->GetShape()->GetShapeVector();
  auto shape_shape = input_args[kIndex1]->GetShape();
  if (shape_shape->isa<abstract::DynamicSequenceShape>()) {
    return std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeRankAny}));
  }

  auto shape_array_opt = GetArrayValue<int64_t>(input_args[kIndex1]);
  if (!shape_array_opt.has_value()) {
    if (shape_shape->isa<abstract::SequenceShape>()) {
      auto seq_shape = shape_shape->cast<abstract::SequenceShapePtr>();
      MS_EXCEPTION_IF_NULL(seq_shape);
      size_t shape_size = seq_shape->size();
      return std::make_shared<abstract::Shape>(ShapeVector(shape_size, abstract::Shape::kShapeDimAny));
    }
    return std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeRankAny});
  }

  auto shape_array = shape_array_opt.value();
  if (!shape_array.HasUnknownValue()) {
    std::vector<int64_t> shape_vec = shape_array.ToVector();
    auto out_shape = CheckShapeValid(x_shape, shape_vec, prim_name);
    auto x_shape_ptr = std::make_shared<abstract::Shape>(out_shape);
    return x_shape_ptr;
  }

  auto outer_dim_offset = shape_array.size() - x_shape.size();
  ShapeVector output_shape;
  for (size_t i = 0; i < shape_array.size(); i++) {
    if (shape_array.IsValueUnknown(i)) {
      output_shape.push_back(abstract::Shape::kShapeDimAny);
    } else {
      if (shape_array[i] == -1) {
        if (i < outer_dim_offset) {
          MS_EXCEPTION(ValueError) << "For '" << prim_name
                                   << "', -1 in init shape is in an incompatible "
                                      "location with given input tensor, -1 index in init shape: "
                                   << i << " but -1 can only be in index" << x_shape.size()
                                   << "onwards for this input.";
        }
        output_shape.push_back(x_shape[i - outer_dim_offset]);
      } else {
        output_shape.push_back(shape_array[i]);
      }
    }
  }
  return std::make_shared<abstract::Shape>(output_shape);
}

TypePtr BroadcastToFuncImpl::InferType(const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) const {
  auto x_dtype = input_args[kIndex0]->GetType()->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(x_dtype);
  std::set<TypePtr> template_types = {kTensorType};
  (void)CheckAndConvertUtils::CheckSubClass("x_dtype", x_dtype, template_types, primitive->name());
  return x_dtype;
}
}  // namespace ops
}  // namespace mindspore
