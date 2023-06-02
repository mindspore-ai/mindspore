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

#include "ops/base_operator.h"
#include "mindapi/src/helper.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"

namespace mindspore {
namespace ops {
MIND_API_BASE_IMPL(BaseOperator, PrimitiveC, api::Primitive);
BaseOperator::BaseOperator(const std::string &name) : api::Primitive(std::make_shared<PrimitiveC>(name)) {}

PrimitiveCPtr BaseOperator::GetPrim() {
  PrimitiveCPtr res = std::dynamic_pointer_cast<PrimitiveC>(impl_);
  return res;
}

void BaseOperator::set_batch_rank(int64_t batch_rank) { (void)this->AddAttr(kBatchRank, api::MakeValue(batch_rank)); }

int64_t BaseOperator::get_batch_rank() const {
  if (this->HasAttr(kBatchRank)) {
    auto value_ptr = this->GetAttr(kBatchRank);
    return GetValue<int64_t>(value_ptr);
  }
  return 0;
}

void BaseOperator::InitIOName(const std::vector<std::string> &inputs_name,
                              const std::vector<std::string> &outputs_name) {
  (void)AddAttr("input_names", api::MakeValue(inputs_name));
  (void)AddAttr("output_names", api::MakeValue(outputs_name));
}

OperatorRegister &OperatorRegister::GetInstance() {
  static OperatorRegister instance;
  return instance;
}

const std::map<std::string, OperatorDefineFunc> &OperatorRegister::GetOperatorMap() const { return operator_fns_; }
void OperatorRegister::SetOperatorMap(const std::string &kname, const OperatorDefineFunc &fn) {
  operator_fns_[kname] = fn;
}
}  // namespace ops
}  // namespace mindspore
