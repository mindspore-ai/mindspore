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

#include "ops/unsqueeze.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
void Unsqueeze::Init(const std::vector<int64_t> axis) { this->set_axis(axis); }

void Unsqueeze::set_axis(std::vector<int64_t> axis) { this->AddAttr(kAxis, MakeValue(axis)); }

std::vector<int64_t> Unsqueeze::get_axis() const {
  auto value_ptr = this->GetAttr(kAxis);
  return GetValue<std::vector<int64_t>>(value_ptr);
}
AbstractBasePtr UnsqueezeInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto unsqueeze_prim = primitive->cast<PrimUnsqueezePtr>();
  MS_EXCEPTION_IF_NULL(unsqueeze_prim);
  auto prim_name = unsqueeze_prim->name();
  CheckAndConvertUtils::CheckInteger("unsqueeze_infer", input_args.size(), kEqual, 1, prim_name);
  MS_EXCEPTION_IF_NULL(input_args[0]);
  auto input = input_args[0];

  // Infer type
  auto input_type = input->BuildType()->cast<TensorTypePtr>()->element();

  // Infer shape
  auto dims = unsqueeze_prim->get_axis();
  auto input_shape = CheckAndConvertUtils::ConvertShapePtrToShape("input", input->BuildShape(), prim_name);
  auto input_rank = input_shape.size();
  auto dim_rank = dims.size();
  std::vector<int64_t> out_shape;
  if (dim_rank == 0) {
    std::copy_if(input_shape.begin(), input_shape.end(), out_shape.begin(), [](const auto item) { return item == 1; });
  } else {
    auto sz = input_rank + dim_rank;
    size_t in_itr = 0;
    size_t ax_itr = 0;
    for (size_t i = 0; i < sz; i++) {
      if (ax_itr < dim_rank && dims[ax_itr] == (int64_t)i) {
        out_shape.emplace_back(1);
        ax_itr++;
      } else if (ax_itr < dim_rank && dims[ax_itr] + sz == i) {
        out_shape.emplace_back(1);
        ax_itr++;
      } else {
        out_shape.emplace_back(input_shape[in_itr]);
        in_itr++;
      }
    }
  }
  return std::make_shared<abstract::AbstractTensor>(input_type, out_shape);
}
REGISTER_PRIMITIVE_C(kNameUnsqueeze, Unsqueeze);
}  // namespace ops
}  // namespace mindspore
