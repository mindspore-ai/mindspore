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
#include "ops/reluv2.h"
#include <string>
#include <algorithm>
#include <map>
#include <set>
#include <vector>

#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
namespace {
std::vector<int64_t> GetOutputMaskShape(const std::vector<int64_t> &input_shape, const std::shared_ptr<Type> &x_dtype) {
  std::vector<int64_t> mask_shape;
  if (input_shape.size() != 4) {
    MS_EXCEPTION(ValueError) << "The `input_x` should be a 4-D tensor,but got a " + std::to_string(input_shape.size()) +
                                  "-D tensor whose shape is " + std::to_string(input_shape.size());
  }
  for (size_t i = 0; i < input_shape.size(); i++) {
    if (i == 1) {
      if (x_dtype == kUInt8 || x_dtype == kInt8) {
        mask_shape.push_back(UlongToLong(ceil((input_shape[1] + 31) / 32)));
      } else {
        mask_shape.push_back(UlongToLong(ceil((input_shape[1] + 15) / 16)));
      }
    } else {
      mask_shape.push_back(input_shape[i]);
    }
  }
  const int64_t shape_end_4d = 4;
  const int64_t shape_end_2d = 2;
  if (x_dtype == kUInt8 || x_dtype == kInt8) {
    (void)mask_shape.insert(mask_shape.end(), shape_end_4d);
  } else {
    (void)mask_shape.insert(mask_shape.end(), shape_end_2d);
  }
  return mask_shape;
}
abstract::TupleShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("input numbers", int64_t(input_args.size()), kEqual, 1, prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
  auto input_shape = shape_map[kShape];
  auto min_shape = shape_map[kMinShape];
  auto max_shape = shape_map[kMaxShape];
  auto x_type_tmp = input_args[0]->BuildType();
  MS_EXCEPTION_IF_NULL(x_type_tmp);
  auto input_type = x_type_tmp->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(input_type);
  auto x_dtype = input_type->element();
  auto mask_shape = GetOutputMaskShape(input_shape, x_dtype);
  abstract::ShapePtr inputs_shape;
  abstract::ShapePtr masks_shape;
  if (min_shape.empty() || max_shape.empty()) {
    inputs_shape = std::make_shared<abstract::Shape>(input_shape);
    masks_shape = std::make_shared<abstract::Shape>(mask_shape);
    return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{inputs_shape, masks_shape});
  }
  auto min_mask_shape = GetOutputMaskShape(min_shape, x_dtype);
  auto max_mask_shape = GetOutputMaskShape(max_shape, x_dtype);
  inputs_shape = std::make_shared<abstract::Shape>(input_shape, min_shape, max_shape);
  masks_shape = std::make_shared<abstract::Shape>(mask_shape, min_mask_shape, max_mask_shape);
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{inputs_shape, masks_shape});
}

TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  (void)CheckAndConvertUtils::CheckInteger("ReLUV2 infer", int64_t(input_args.size()), kEqual, 1, prim_name);
  MS_EXCEPTION_IF_NULL(input_args[0]);
  auto x_type = input_args[0]->BuildType();
  MS_EXCEPTION_IF_NULL(x_type);
  if (!x_type->isa<TensorType>()) {
    MS_EXCEPTION(TypeError) << "The " << prim_name << "'s "
                            << " input must be tensor type but got " << x_type->ToString();
  }
  auto mask_dtype = kUInt8;
  return std::make_shared<Tuple>(std::vector<TypePtr>{x_type, mask_dtype});
}
}  // namespace
AbstractBasePtr ReLUV2Infer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                            const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto types = InferType(primitive, input_args);
  auto shapes = InferShape(primitive, input_args);
  return abstract::MakeAbstract(shapes, types);
}
REGISTER_PRIMITIVE_EVAL_IMPL(ReLUV2, prim::kPrimReluV2, ReLUV2Infer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
