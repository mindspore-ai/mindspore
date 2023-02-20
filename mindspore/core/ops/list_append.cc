/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "ops/list_append.h"

#include <vector>
#include <memory>
#include <string>

#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/ops/primitive_infer_map.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/primitive.h"
#include "ops/core_ops.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
AbstractBasePtr ListAppendInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  constexpr size_t input_len = 2;
  constexpr size_t data_index = 0;
  constexpr size_t target_index = 1;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_len, prim_name);
  auto data_abs = dyn_cast<abstract::AbstractList>(input_args[data_index]);
  MS_EXCEPTION_IF_NULL(data_abs);
  if (!data_abs->dynamic_len()) {
    MS_EXCEPTION(TypeError) << "The first input to ListAppend should be dynamic length sequence but got constant "
                            << "length sequence. The abstract of input sequence is: " << data_abs->ToString();
  }
  auto target_abs = input_args[target_index];
  auto data_element_abs = data_abs->dynamic_len_element_abs();
  if (data_element_abs == nullptr) {
    // The element type of the dynamic length sequence is not determined before list append.
    // Fix the element abstract as the target element
    auto ret = data_abs->Clone();
    ret->cast<abstract::AbstractListPtr>()->set_dynamic_len_element_abs(target_abs);
    return ret;
  }
  // If abstract of element is fixed, the abstract of target should have the same shape and type with the
  // abstract of element.
  CheckAndConvertUtils::CheckAbstractTypeAndShapeSame({data_element_abs, target_abs},
                                                      "For " + prim::kPrimListAppend->ToString(),
                                                      "mutable list existing item", "new added item");
  return data_abs->Clone();
}
MIND_API_OPERATOR_IMPL(ListAppend, BaseOperator);
REGISTER_PRIMITIVE_EVAL_IMPL(ListAppend, prim::kPrimListAppend, ListAppendInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
