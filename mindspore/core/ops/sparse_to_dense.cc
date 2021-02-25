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
#include <vector>
#include <memory>
#include "ops/sparse_to_dense.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
AbstractBasePtr SparseToDenseInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto spasetodense_prim = primitive->cast<PrimSparseToDensePtr>();
  MS_EXCEPTION_IF_NULL(spasetodense_prim);
  auto prim_name = spasetodense_prim->name();
  CheckAndConvertUtils::CheckInteger("input numbers", input_args.size(), kEqual, 3, prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  // infer shape
  auto dense_shape =
    CheckAndConvertUtils::ConvertShapePtrToShape("dense_shape", input_args[3]->BuildShape(), prim_name);
  // infer type
  auto indices_type = input_args[0]->BuildType()->cast<TensorTypePtr>()->element();
  auto values_type = input_args[1]->BuildType()->cast<TensorTypePtr>()->element();
  std::set<TypePtr> valid_type = {TypeIdToType(kObjectTypeTensorType)};
  CheckAndConvertUtils::CheckSubClass("indices_type", indices_type, valid_type, prim_name);
  CheckAndConvertUtils::CheckSubClass("values_type", values_type, valid_type, prim_name);
  return std::make_shared<abstract::AbstractTensor>(values_type, dense_shape);
}
REGISTER_PRIMITIVE_C(kNameSparseToDense, SparseToDense);
}  // namespace ops
}  // namespace mindspore
