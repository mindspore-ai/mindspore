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

#include "ops/list_inplace_insert.h"

#include <memory>
#include <string>
#include <vector>

#include "abstract/abstract_value.h"
#include "abstract/ops/primitive_infer_map.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/primitive.h"
#include "mindapi/src/helper.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ops {
AbstractBasePtr ListInplaceInsertInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const auto &prim_name = primitive->name();
  constexpr size_t input_len = 3;
  constexpr size_t data_index = 0;
  constexpr size_t index_index = 1;
  constexpr size_t target_index = 2;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kGreaterEqual, input_len,
                                           prim_name);
  auto data_abs = dyn_cast<abstract::AbstractList>(input_args[data_index]);
  MS_EXCEPTION_IF_NULL(data_abs);
  auto index_abs = input_args[index_index];
  auto target_abs = input_args[target_index];

  if (data_abs->dynamic_len()) {
    MS_EXCEPTION(TypeError) << "The first input to " << prim_name << " can not be dynamic length sequence.";
  }

  // Check index input, must satisfy:
  //   1. index input must be constant.
  //   2. index input must be int64 scalar.
  auto index_abs_value = index_abs->BuildValue();
  if (index_abs_value == kValueAny) {
    MS_EXCEPTION(ValueError) << "The second input to " << prim_name << " must be constant scalar but got variable.";
  }
  if (!utils::isa<int64_t>(index_abs_value)) {
    MS_EXCEPTION(TypeError) << "The second input to " << prim_name << " must be int scalar but got "
                            << index_abs_value->type_name();
  }
  int64_t index_value = GetValue<int64_t>(index_abs_value);

  // Generate insert position:
  int64_t insert_pos = 0;
  int64_t seq_len = SizeToInt(data_abs->size());
  if (index_value < 0) {
    insert_pos = index_value + seq_len < 0 ? 0 : index_value + seq_len;
  } else if (index_value >= seq_len) {
    insert_pos = seq_len;
  } else {
    insert_pos = index_value;
  }

  abstract::AbstractListPtr ret;
  const auto &elements = data_abs->elements();
  abstract::AbstractBasePtrList new_elements;
  for (auto i = 0; i < insert_pos; ++i) {
    const auto &element = elements[i];
    (void)new_elements.emplace_back(element);
  }
  (void)new_elements.emplace_back(target_abs);
  for (auto i = insert_pos; i < seq_len; ++i) {
    const auto &element = elements[i];
    (void)new_elements.emplace_back(element);
  }
  ret = std::make_shared<abstract::AbstractList>(new_elements);

  ret = AbstractBroaden(ret)->cast<abstract::AbstractListPtr>();
  ret->set_extra_info(data_abs->extra_info());

  return ret;
}
MIND_API_OPERATOR_IMPL(ListInplaceInsert, BaseOperator);
REGISTER_PRIMITIVE_EVAL_IMPL(ListInplaceInsert, prim::kPrimListInplaceInsert, ListInplaceInsertInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
