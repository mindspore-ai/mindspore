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
abstract::ShapePtr CustomNormalizeInferShape(const std::vector<AbstractBasePtr> &input_args) {
  auto base_value = input_args[0]->BuildValue();
  MS_EXCEPTION_IF_NULL(base_value);
  auto tensor_value = base_value->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(tensor_value);
  MS_EXCEPTION_IF_NULL(tensor_value->data_c());
  std::vector<int64_t> infer_shape;
  auto string_num = reinterpret_cast<int64_t *>(tensor_value->data_c());
  if (*string_num == 0) {
    infer_shape.push_back(1);
  } else {
    infer_shape.push_back(*string_num);
  }
  return std::make_shared<abstract::Shape>(infer_shape);
}

TypePtr CustomNormalizeInferType(const std::vector<AbstractBasePtr> &input_args) {
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
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, input_num, primitive->name());
  return std::make_shared<abstract::AbstractTensor>(CustomNormalizeInferType(input_args),
                                                    CustomNormalizeInferShape(input_args));
}
REGISTER_PRIMITIVE_C(kNameCustomNormalize, CustomNormalize);
}  // namespace ops
}  // namespace mindspore
