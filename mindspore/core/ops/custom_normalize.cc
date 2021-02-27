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

#include "ops/custom_normalize.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr CustomNormalizeInferShape(const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto custom_normalize_prim = primitive->cast<PrimCustomNormalizePtr>();
  MS_EXCEPTION_IF_NULL(custom_normalize_prim);
  // auto prim_name = custom_normalize_prim->name();
  MS_EXCEPTION_IF_NULL(input_args[0]);
  MS_EXCEPTION_IF_NULL(input_args[0]->BuildShape());
  //  auto input_shape =
  //    CheckAndConvertUtils::ConvertShapePtrToShape("input_shape", input_args[0]->BuildShape(), prim_name);
  if (input_args[0]->BuildValue()->cast<tensor::TensorPtr>()->data_c() == nullptr) {
    MS_LOG(ERROR) << "Do infer shape in runtime.";
  }
  std::vector<int64_t> infer_shape;
  auto string_num = reinterpret_cast<int64_t *>(input_args[0]->BuildValue()->cast<tensor::TensorPtr>()->data_c());
  if (*string_num == 0) {
    infer_shape.push_back(1);
  } else {
    infer_shape.push_back(*string_num);
  }
  return std::make_shared<abstract::Shape>(infer_shape);
}

TypePtr CustomNormalizeInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto infer_type = input_args[0]->BuildType();
  auto tensor_type = infer_type->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(tensor_type);
  auto data_type = tensor_type->element();
  MS_EXCEPTION_IF_NULL(data_type);
  return data_type;
}
}  // namespace

AbstractBasePtr CustomNormalizeInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) {
  return std::make_shared<abstract::AbstractTensor>(CustomNormalizeInferType(primitive, input_args),
                                                    CustomNormalizeInferShape(primitive, input_args)->shape());
}
REGISTER_PRIMITIVE_C(kNameCustomNormalize, CustomNormalize);
}  // namespace ops
}  // namespace mindspore
