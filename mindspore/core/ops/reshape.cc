/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include <numeric>
#include <functional>
#include "ops/reshape.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
AbstractBasePtr ReshapeInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  CheckAndConvertUtils::CheckInteger("input number", input_args.size(), kEqual, 2, prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto x = input_args[0]->cast<abstract::AbstractTensorPtr>();
  MS_EXCEPTION_IF_NULL(x);
  auto shape = input_args[1]->cast<abstract::AbstractTuplePtr>();
  MS_EXCEPTION_IF_NULL(shape);
  auto shape_v = GetValue<std::vector<int64_t>>(shape->BuildValue());
  int64_t neg_index = -1;
  int64_t dim_prod = 1;
  for (size_t i = 0; i < shape_v.size(); ++i) {
    if (shape_v[i] == -1) {
      if (neg_index != -1) {
        MS_LOG(EXCEPTION) << "The Reshape's shape input can only has one -1 at most.";
      }
      neg_index = SizeToLong(i);
    } else {
      dim_prod *= shape_v[i];
    }
  }
  MS_EXCEPTION_IF_NULL(x->shape());
  auto x_shape = x->shape()->shape();
  int64_t arr_prod =
    std::accumulate(x_shape.begin(), x_shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());
  if (arr_prod <= 0) {
    ShapeVector x_max_shape = x->shape()->max_shape();
    ShapeVector x_min_shape = x->shape()->min_shape();
    if (x_max_shape.empty()) {
      x_max_shape = x_shape;
    }
    if (x_min_shape.empty()) {
      x_min_shape = x_shape;
    }
    int64_t max_arr_prod =
      std::accumulate(x_max_shape.begin(), x_max_shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());
    int64_t min_arr_prod =
      std::accumulate(x_min_shape.begin(), x_min_shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());
    ShapeVector max_shape = shape_v;
    ShapeVector min_shape = shape_v;
    if (neg_index != -1) {
      max_shape[neg_index] = max_arr_prod / dim_prod;
      min_shape[neg_index] = min_arr_prod / dim_prod;
    } else {
      MS_LOG(EXCEPTION) << "For dynamic shape, Reshape's shape input must have neg index";
    }
    return std::make_shared<abstract::AbstractTensor>(x->element(),
                                                      std::make_shared<abstract::Shape>(shape_v, min_shape, max_shape));
  } else {
    if (dim_prod <= 0 || arr_prod % dim_prod != 0) {
      MS_LOG(EXCEPTION) << "The product of input_x's shape should > 0, and can be divided by product of input_shape, "
                           "but product of input_x's shape is "
                        << arr_prod << ", product of input_shape is" << dim_prod;
    }
    if (neg_index != -1) {
      shape_v[neg_index] = arr_prod / dim_prod;
      dim_prod *= shape_v[neg_index];
    }
    if (arr_prod != dim_prod) {
      MS_LOG(EXCEPTION) << "The product of input_x's shape should be equal to product of input_shape, "
                           "but product of input_x's shape is "
                        << arr_prod << ", product of input_shape is" << dim_prod;
    }
    return std::make_shared<abstract::AbstractTensor>(x->element(), std::make_shared<abstract::Shape>(shape_v));
  }
}
REGISTER_PRIMITIVE_C(kNameReshape, Reshape);
}  // namespace ops
}  // namespace mindspore
