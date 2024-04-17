/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include <map>
#include <memory>
#include "ops/ops_frontend_func_impl.h"
#include "ops/op_utils.h"
#include "utils/log_adapter.h"
#include "abstract/abstract_value.h"
#include "utils/anf_utils.h"

namespace mindspore {
namespace ops {

class ListToTupleFrontendFuncImpl : public OpFrontendFuncImpl {
 public:
  ValuePtr InferValue(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    auto input_list = input_args[kIndex0]->cast<abstract::AbstractListPtr>();
    MS_EXCEPTION_IF_NULL(input_list);
    if (input_list->dynamic_len()) {
      return nullptr;
    }
    return input_list->ElementsBuildValue<ValueTuple>();
  }

  AbstractBasePtr InferAbstract(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
    auto input_list = input_args[kIndex0]->cast<abstract::AbstractListPtr>();
    MS_EXCEPTION_IF_NULL(input_list);
    auto ret = std::make_shared<abstract::AbstractTuple>(input_list->elements(), input_list->sequence_nodes());
    ret->set_dynamic_len(input_list->dynamic_len());
    ret->set_dynamic_len_element_abs(input_list->dynamic_len_element_abs());
    if (input_list->dynamic_len()) {
      (void)primitive->AddAttr(kInputRealTuple, MakeValue(true));
      (void)primitive->AddAttr(kOutputRealTuple, MakeValue(true));
    }
    return ret;
  }
};

REGISTER_PRIMITIVE_FUNCTION_FRONTEND_FUNC_IMPL("ListToTuple", ListToTupleFrontendFuncImpl);
}  // namespace ops
}  // namespace mindspore
