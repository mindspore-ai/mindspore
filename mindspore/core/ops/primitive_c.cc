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

#include <memory>
#include <string>

#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
namespace mindspore {
namespace ops {
void PrimitiveC::InitIOName(const std::vector<std::string> &inputs_name, const std::vector<std::string> &outputs_name) {
  this->AddAttr("input_names", MakeValue(inputs_name));
  this->AddAttr("output_names", MakeValue(outputs_name));
}

AbstractBasePtr PrimitiveC::Infer(const AbstractBasePtrList &abstract_list) {
  auto infer_map = abstract::GetPrimitiveToEvalImplMap();
  auto iter = infer_map.find(std::make_shared<Primitive>(this->name()));
  if (iter == infer_map.end()) {
    MS_EXCEPTION(NotExistsError) << "Cannot find the " << this->name() << "infer function in the infer map!";
  }
  auto infer_function = iter->second.impl_;
  return infer_function(nullptr, shared_from_base<Primitive>(), abstract_list);
}

OpPrimCRegister &OpPrimCRegister::GetInstance() {
  static OpPrimCRegister instance;
  return instance;
}

std::map<std::string, OpPrimCDefineFunc> OpPrimCRegister::GetPrimCMap() { return op_primc_fns_; }
void OpPrimCRegister::SetPrimCMap(const std::string &kname, const OpPrimCDefineFunc &fn) { op_primc_fns_[kname] = fn; }
}  // namespace ops
}  // namespace mindspore
