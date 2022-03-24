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

#include "ops/slice.h"
#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr SliceInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kEqual, 3, prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
  auto input_shape = shape_map[kShape];
  auto min_shape = shape_map[kMinShape];
  auto max_shape = shape_map[kMaxShape];
  std::vector<std::vector<int64_t>> input_values;
  // get begin and size value
  for (size_t i = 1; i <= 2; ++i) {
    std::vector<int64_t> tmp_input;
    auto input_value = input_args[i]->BuildValue();
    if (input_value->isa<tensor::Tensor>()) {
      tmp_input = CheckAndConvertUtils::CheckTensorIntValue("slice args value", input_value, prim_name);
    } else {
      tmp_input = CheckAndConvertUtils::CheckTupleInt("slice args value", input_value, prim_name);
    }
    (void)input_values.emplace_back(tmp_input);
  }

  auto begin_v = input_values[0];
  auto size_v = input_values[1];
  auto rank = input_shape.size();
  if (begin_v.size() != rank || size_v.size() != rank) {
    MS_LOG(EXCEPTION) << "For '" << prim_name
                      << "', the shape of input|begin|size must be equal, but got input shape: " << rank
                      << ", begin shape: " << begin_v.size() << ", size shape: " << size_v.size();
  }

  for (size_t i = 0; i < size_v.size(); ++i) {
    if (size_v[i] == -1) {
      size_v[i] = input_shape[i] - begin_v[i];
    }
  }
  if (max_shape.empty() && min_shape.empty()) {
    return std::make_shared<abstract::Shape>(size_v);
  }
  return std::make_shared<abstract::Shape>(size_v, min_shape, max_shape);
}

TypePtr SliceInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  (void)CheckAndConvertUtils::CheckInteger("slice_prim_infer", input_args.size(), kEqual, 3, prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  MS_EXCEPTION_IF_NULL(input_args[0]);
  auto x_type_map = input_args[0]->BuildType();
  MS_EXCEPTION_IF_NULL(x_type_map);
  auto x_dtype = x_type_map->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(x_dtype);
  std::set<TypePtr> template_types = {kTensorType};
  return CheckAndConvertUtils::CheckTensorTypeValid("x_dtype", x_dtype, template_types, prim_name);
}
}  // namespace

MIND_API_BASE_IMPL(Slice, PrimitiveC, BaseOperator);
AbstractBasePtr SliceInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                           const std::vector<AbstractBasePtr> &input_args) {
  return abstract::MakeAbstract(SliceInferShape(primitive, input_args), SliceInferType(primitive, input_args));
}

REGISTER_PRIMITIVE_C(kNameSlice, Slice);
}  // namespace ops
}  // namespace mindspore
