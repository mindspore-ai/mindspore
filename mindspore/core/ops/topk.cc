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
#include "ops/topk.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
void TopK::Init(const bool sorted) { this->set_sorted(sorted); }
void TopK::set_sorted(const bool sorted) { this->AddAttr(kSorted, MakeValue(sorted)); }

bool TopK::get_sorted() const {
  auto value_ptr = this->GetAttr(kSorted);
  return GetValue<bool>(value_ptr);
}
AbstractBasePtr TopKInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto top_prim = primitive->cast<PrimTopKPtr>();
  MS_EXCEPTION_IF_NULL(top_prim);
  auto prim_name = top_prim->name();
  CheckAndConvertUtils::CheckInteger("top_k_infer", input_args.size(), kEqual, 2, prim_name);

  // Infer dtype
  auto output0_type = input_args[0]->BuildType()->cast<TensorTypePtr>()->element();
  auto output1_type = TypeIdToType(kNumberTypeInt32);
  const std::set<TypeId> valid_types = {kNumberTypeFloat16, kNumberTypeFloat32};
  CheckAndConvertUtils::CheckTensorTypeValid("input_x", input_args[0]->BuildType(), valid_types, prim_name);

  // Infer shape
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShape("x_shape", input_args[0]->BuildShape(), prim_name);
  auto k_v = GetValue<int>(input_args[1]->BuildValue());
  auto ndims = x_shape.size() - 1;
  x_shape[ndims] = k_v;

  auto output0 = std::make_shared<abstract::AbstractTensor>(output0_type, x_shape);
  auto output1 = std::make_shared<abstract::AbstractTensor>(output1_type, x_shape);
  AbstractBasePtrList output = {output0, output1};
  return std::make_shared<abstract::AbstractTuple>(output);
}
REGISTER_PRIMITIVE_C(kNameTopK, TopK);
}  // namespace ops
}  // namespace mindspore
