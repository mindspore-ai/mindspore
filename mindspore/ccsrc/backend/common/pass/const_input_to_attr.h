/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_COMMON_CONST_INPUT_TO_ATTR_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_COMMON_CONST_INPUT_TO_ATTR_H_
#include <string>

#include "ir/anf.h"
#include "utils/hash_map.h"
#include "utils/hash_set.h"
#include "utils/ms_utils.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace opt {
class ConstInputToAttrInfoRegister {
 public:
  explicit ConstInputToAttrInfoRegister(const std::string &op_name = "") : op_name_(op_name) {}
  virtual ~ConstInputToAttrInfoRegister() = default;

  ConstInputToAttrInfoRegister &SetConstInputToAttr(size_t input_index) {
    (void)input_attr_set_.insert(input_index);
    return *this;
  }

  ConstInputToAttrInfoRegister &SetConstInputToAttr(const mindspore::HashSet<size_t> &input_attr_set) {
    input_attr_set_.insert(input_attr_set.cbegin(), input_attr_set.cend());
    return *this;
  }

  const mindspore::HashSet<size_t> &GetConstInputAttrInfo() const { return input_attr_set_; }
  const std::string &GetOpName() const { return op_name_; }

 private:
  std::string op_name_;
  mindspore::HashSet<size_t> input_attr_set_;
};

class ConstInputToAttrInfoRegistry {
 public:
  static ConstInputToAttrInfoRegistry &Instance();
  void Register(const ConstInputToAttrInfoRegister &reg);
  void Register(const std::string &op_name, const mindspore::HashSet<size_t> &input_attr_set);
  bool GetRegisterByOpName(const std::string &op_name, ConstInputToAttrInfoRegister *reg) const;

 private:
  ConstInputToAttrInfoRegistry();
  ~ConstInputToAttrInfoRegistry() = default;
  DISABLE_COPY_AND_ASSIGN(ConstInputToAttrInfoRegistry)
  mindspore::HashMap<std::string, ConstInputToAttrInfoRegister> op_input_to_attr_map_;
};

BACKEND_EXPORT CNodePtr ConstInputToAttr(const CNodePtr &cnode, const mindspore::HashSet<size_t> &input_attrs);
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_COMMON_CONST_INPUT_TO_ATTR_H_
