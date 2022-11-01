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
#include "ops/reduce.h"
#include <string>
#include <memory>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"
#include "include/common/utils/utils.h"

namespace mindspore {
namespace ops {
void Reduce::set_keep_dims(const bool keep_dims) { (void)this->AddAttr(kKeepDims, api::MakeValue(keep_dims)); }

bool Reduce::get_keep_dims() const { return GetValue<bool>(GetAttr(kKeepDims)); }

void Reduce::Init(const bool keep_dims) { this->set_keep_dims(keep_dims); }

void Reduce::set_axis(const std::vector<int64_t> &axis) { (void)this->AddAttr(kAxis, api::MakeValue(axis)); }

std::vector<int64_t> Reduce::get_axis() {
  PrimitivePtr prim = this->GetPrim();
  MS_EXCEPTION_IF_NULL(prim);
  std::vector<int64_t> axis = {};
  if (prim->HasAttr(kAttrAxis)) {
    auto value_ptr = prim->GetAttr(kAttrAxis);
    if (value_ptr->isa<tensor::Tensor>()) {
      axis = CheckAndConvertUtils::CheckTensorIntValue(kAttrAxis, value_ptr, prim->name());
    } else {
      axis = CheckAndConvertUtils::CheckIntOrTupleInt(kAttrAxis, value_ptr, prim->name());
    }
  }
  return axis;
}

MIND_API_OPERATOR_IMPL(Reduce, BaseOperator);
REGISTER_PRIMITIVE_C(kNameReduce, Reduce);
}  // namespace ops
}  // namespace mindspore
