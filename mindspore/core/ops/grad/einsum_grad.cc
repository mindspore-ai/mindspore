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

#include "ops/grad/einsum_grad.h"
#include <algorithm>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
void EinsumGrad::Init(const std::string equation) { this->set_equation(equation); }

void EinsumGrad::set_equation(const std::string equation) { (void)this->AddAttr(kEquation, MakeValue(equation)); }

std::string EinsumGrad::get_equation() const {
  auto value_ptr = this->GetAttr(kEquation);
  return GetValue<std::string>(value_ptr);
}
AbstractBasePtr EinsumGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  for (auto item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto elements = input_args[0]->isa<abstract::AbstractTuple>()
                    ? input_args[0]->cast<abstract::AbstractTuplePtr>()->elements()
                    : input_args[0]->cast<abstract::AbstractListPtr>()->elements();
  AbstractBasePtrList rets;
  std::vector<std::vector<size_t>> input_shapes;
  std::vector<size_t> cur_shape;
  for (size_t idx = 0; idx < elements.size(); ++idx) {
    auto dx = elements[idx]->Broaden();
    rets.emplace_back(dx);
    auto shape = elements[idx]->BuildShape();
    auto &shape_int = shape->cast<abstract::ShapePtr>()->shape();
    std::transform(shape_int.begin(), shape_int.end(), std::back_inserter(cur_shape), SizeToLong);
    input_shapes.emplace_back(cur_shape);
    cur_shape.clear();
  }
  (void)primitive->AddAttr("input_shape_vec", MakeValue<std::vector<std::vector<size_t>>>(input_shapes));
  return std::make_shared<abstract::AbstractTuple>(rets);
}
// REGISTER_PRIMITIVE_EVAL_IMPL(EinsumGrad, prim::kPrimEinsumGrad, EinsumGradInfer, nullptr, true);
REGISTER_PRIMITIVE_C(kNameEinsumGrad, EinsumGrad);
}  // namespace ops
}  // namespace mindspore
