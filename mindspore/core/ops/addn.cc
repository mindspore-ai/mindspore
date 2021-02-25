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

#include <set>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include "ops/addn.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
AbstractBasePtr AddNInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  CheckAndConvertUtils::CheckInteger("input number", input_args.size(), kEqual, 1, prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto input_tuple = input_args[0]->cast<abstract::AbstractTuplePtr>();
  MS_EXCEPTION_IF_NULL(input_tuple);
  auto elements = input_tuple->elements();
  CheckAndConvertUtils::CheckInteger("concat element num", elements.size(), kGreaterEqual, 1, prim_name);
  auto element0 = elements[0]->cast<abstract::AbstractTensorPtr>();
  MS_EXCEPTION_IF_NULL(element0);
  auto element0_shape =
    CheckAndConvertUtils::ConvertShapePtrToShape("element0 shape", element0->BuildShape(), prim_name);

  std::map<std::string, TypePtr> types;
  types.emplace("element0", element0->BuildType());
  for (size_t i = 1; i < elements.size(); ++i) {
    std::string elementi = "element" + std::to_string(i);
    auto elementi_shape =
      CheckAndConvertUtils::ConvertShapePtrToShape(elementi + " shape", elements[i]->BuildShape(), prim_name);
    CheckAndConvertUtils::CheckInteger(elementi + " shape rank", elementi_shape.size(), kEqual, element0_shape.size(),
                                       prim_name);
    for (size_t j = 0; j < element0_shape.size(); ++j) {
      if (elementi_shape[j] != element0_shape[j]) {
        MS_LOG(EXCEPTION) << "element " << i << " shape in input can not concat with first element.";
      }
    }
    types.emplace(elementi, elements[i]->BuildType());
  }
  std::set<TypeId> valid_types = common_valid_types;
  valid_types.insert(kNumberTypeBool);
  auto infer_type = CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim_name);

  return std::make_shared<abstract::AbstractTensor>(TypeIdToType(infer_type),
                                                    std::make_shared<abstract::Shape>(element0_shape));
}
REGISTER_PRIMITIVE_C(kNameAddN, AddN);
}  // namespace ops
}  // namespace mindspore
