/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "ops/list_inplace_extend.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "abstract/abstract_value.h"
#include "abstract/ops/primitive_infer_map.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/primitive.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/sequence_ops.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ops {
AbstractBasePtr ListInplaceExtendInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const auto &prim_name = primitive->name();
  constexpr size_t input_len = 2;
  constexpr size_t data_index = 0;
  constexpr size_t target_index = 1;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kGreaterEqual, input_len,
                                           prim_name);
  auto data_abs = dyn_cast<abstract::AbstractList>(input_args[data_index]);
  MS_EXCEPTION_IF_NULL(data_abs);
  auto target_abs = input_args[target_index];
  if (!target_abs->isa<abstract::AbstractSequence>() && !target_abs->isa<abstract::AbstractTensor>()) {
    MS_EXCEPTION(TypeError) << "For " << prim_name
                            << ", the second input should be tuple or list but got: " << target_abs->ToString();
  }

  abstract::AbstractBasePtrList new_elements;
  const auto &data_elements = data_abs->elements();
  for (auto element : data_elements) {
    (void)new_elements.emplace_back(element);
  }

  if (target_abs->isa<abstract::AbstractSequence>()) {
    auto target_abs_seq = target_abs->cast<abstract::AbstractSequencePtr>();
    if (data_abs->dynamic_len() || target_abs_seq->dynamic_len()) {
      MS_LOG(INTERNAL_EXCEPTION) << "ListInplaceExtend do not support dynamic length sequence input.";
    }
    const auto &target_elements = target_abs_seq->elements();
    for (auto element : target_elements) {
      (void)new_elements.emplace_back(element);
    }
  } else {
    auto target_base_shape = target_abs->BuildShape();
    MS_EXCEPTION_IF_NULL(target_base_shape);
    if (target_base_shape->isa<abstract::NoShape>()) {
      MS_EXCEPTION(TypeError) << "Cannot iterate over a scalar tensor.";
    }
    auto target_shape = dyn_cast<abstract::Shape>(target_base_shape);
    MS_EXCEPTION_IF_NULL(target_shape);
    const auto &target_shape_vec = target_shape->shape();
    if (target_shape_vec.size() != 0) {
      auto new_element_num = target_shape_vec[0];
      ShapeVector new_element_shape;
      if (target_shape_vec.size() == 1) {
        (void)new_element_shape.emplace_back(1);
      } else {
        (void)std::copy(target_shape_vec.begin() + 1, target_shape_vec.end(), std::back_inserter(new_element_shape));
      }
      auto new_element_abs =
        abstract::MakeAbstractTensor(std::make_shared<abstract::Shape>(new_element_shape), target_abs->BuildType());
      for (int i = 0; i < new_element_num; ++i) {
        (void)new_elements.emplace_back(new_element_abs->Clone());
      }
    }
  }
  abstract::AbstractListPtr ret = std::make_shared<abstract::AbstractList>(new_elements);
  ret = AbstractBroaden(ret)->cast<abstract::AbstractListPtr>();
  ret->set_extra_info(data_abs->extra_info());

  return ret;
}
MIND_API_OPERATOR_IMPL(ListInplaceExtend, BaseOperator);
REGISTER_PRIMITIVE_EVAL_IMPL(ListInplaceExtend, prim::kPrimListInplaceExtend, ListInplaceExtendInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
