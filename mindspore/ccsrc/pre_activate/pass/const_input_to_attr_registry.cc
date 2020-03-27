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
#include "pre_activate/pass/const_input_to_attr_registry.h"

#include <utility>

#include "utils/log_adapter.h"

namespace mindspore {
namespace opt {
ConstInputToAttrInfoRegistry &ConstInputToAttrInfoRegistry::Instance() {
  static ConstInputToAttrInfoRegistry instance;
  return instance;
}

void ConstInputToAttrInfoRegistry::Register(const ConstInputToAttrInfoRegister &reg) {
  auto op_name = reg.GetOpName();
  if (op_input_to_attr_map_.find(op_name) == op_input_to_attr_map_.end()) {
    (void)op_input_to_attr_map_.insert(make_pair(op_name, reg));
    MS_LOG(DEBUG) << op_name << " const2attr register successfully!";
  }
}

void ConstInputToAttrInfoRegistry::Register(const std::string &op_name,
                                            const std::unordered_set<size_t> &input_attr_set) {
  if (op_input_to_attr_map_.find(op_name) == op_input_to_attr_map_.end()) {
    ConstInputToAttrInfoRegister reg(op_name);
    (void)reg.SetConstInputToAttr(input_attr_set);
    (void)op_input_to_attr_map_.insert(make_pair(op_name, reg));
    MS_LOG(DEBUG) << op_name << " const2attr register successfully!";
  }
}

bool ConstInputToAttrInfoRegistry::GetRegisterByOpName(const std::string &op_name,
                                                       ConstInputToAttrInfoRegister *reg) const {
  if (op_input_to_attr_map_.find(op_name) != op_input_to_attr_map_.end()) {
    *reg = op_input_to_attr_map_.at(op_name);
    MS_LOG(DEBUG) << op_name << " const2attr find in registery.";
    return true;
  }
  return false;
}
}  // namespace opt
}  // namespace mindspore
