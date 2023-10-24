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

#include "ops/dict_inplace_setitem.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include <utility>

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
AbstractBasePtr DictInplaceSetItemInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                        const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const auto &prim_name = primitive->name();
  constexpr size_t input_len = 3;
  constexpr size_t data_index = 0;
  constexpr size_t key_index = 1;
  constexpr size_t value_index = 2;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kGreaterEqual, input_len,
                                           prim_name);
  auto data_abs = input_args[data_index];
  MS_EXCEPTION_IF_NULL(data_abs);
  auto dict_abs = dyn_cast<abstract::AbstractDictionary>(input_args[data_index]);
  MS_EXCEPTION_IF_NULL(dict_abs);

  auto key_abs = input_args[key_index];
  MS_EXCEPTION_IF_NULL(key_abs);
  auto key_value = key_abs->BuildValue();
  MS_EXCEPTION_IF_NULL(key_value);
  if (key_value == kValueAny) {
    MS_LOG(EXCEPTION) << prim_name << " only support constant key but got abstract of key: " << key_abs->ToString();
  }
  if (!(key_value->isa<StringImm>() || key_value->isa<Scalar>()) && !key_abs->isa<abstract::AbstractTensor>() &&
      !key_abs->isa<abstract::AbstractTuple>()) {
    MS_EXCEPTION(TypeError) << "Dict do not support un-hashable type key with abstract: " << key_abs->ToString();
  }

  auto dict_elems = dict_abs->elements();
  auto it = std::find_if(
    dict_elems.cbegin(), dict_elems.cend(),
    [&key_value](const abstract::AbstractElementPair &item) { return *key_value == *item.first->BuildValue(); });

  MS_EXCEPTION_IF_NULL(input_args[value_index]);
  auto new_ele = std::make_pair(input_args[key_index], input_args[value_index]);
  if (it != dict_elems.end()) {
    int64_t index = it - dict_elems.begin();
    dict_elems[LongToSize(index)] = new_ele;
  } else {
    dict_elems.push_back(new_ele);
  }

  auto ret = std::make_shared<abstract::AbstractDictionary>(dict_elems);
  ret = AbstractBroaden(ret)->cast<abstract::AbstractDictionaryPtr>();
  ret->set_extra_info(dict_abs->extra_info());
  return ret;
}
MIND_API_OPERATOR_IMPL(DictInplaceSetItem, BaseOperator);
REGISTER_PRIMITIVE_EVAL_IMPL(DictInplaceSetItem, prim::kPrimDictInplaceSetItem, DictInplaceSetItemInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
