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

#include "ops/custom_extract_features.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
AbstractBasePtr CustomExtractFeaturesInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto extract_prim = primitive->cast<PrimCustomExtractFeaturesPtr>();
  MS_EXCEPTION_IF_NULL(extract_prim);
  auto prim_name = extract_prim->name();
  MS_EXCEPTION_IF_NULL(input_args[0]);
  // auto input = input_args[0];

  // Infer type
  auto output0_type = TypeIdToType(kNumberTypeInt32);
  auto output1_type = TypeIdToType(kNumberTypeFloat32);

  // Infer shape
  std::vector<int64_t> out_shape;
  auto input_shape =
    CheckAndConvertUtils::ConvertShapePtrToShape("input_shape", input_args[0]->BuildShape(), prim_name);
  auto string_num = input_shape[0];
  if (string_num == 0) {
    out_shape.push_back(1);
  } else {
    out_shape.push_back(string_num);
  }

  auto output0 = std::make_shared<abstract::AbstractTensor>(output0_type, out_shape);
  auto output1 = std::make_shared<abstract::AbstractTensor>(output1_type, out_shape);
  AbstractBasePtrList output = {output0, output1};
  return std::make_shared<abstract::AbstractTuple>(output);
}
REGISTER_PRIMITIVE_C(kNameCustomExtractFeatures, CustomExtractFeatures);
}  // namespace ops
}  // namespace mindspore
