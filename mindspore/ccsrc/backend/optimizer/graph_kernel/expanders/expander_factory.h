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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_EXPANDERS_EXPANDER_FACTORY_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_EXPANDERS_EXPANDER_FACTORY_H_

#include <unordered_map>
#include <functional>
#include <string>
#include <memory>

#include "backend/optimizer/graph_kernel/expanders/utils.h"

namespace mindspore {
namespace opt {
namespace expanders {
class OpExpanderFactory {
 public:
  static OpExpanderFactory &Instance() {
    static OpExpanderFactory instance;
    return instance;
  }
  std::shared_ptr<OpExpander> GetExpander(const std::string &op) {
    if (auto iter = creators.find(op); iter != creators.end()) {
      auto expander_ptr = iter->second();
      expander_ptr->op_ = op;
      return expander_ptr;
    }
    return nullptr;
  }
  ~OpExpanderFactory() = default;

  using RegFunc = std::function<std::shared_ptr<OpExpander>()>;
  void Register(const std::string &op, const RegFunc &func) { creators[op] = func; }

 private:
  std::unordered_map<std::string, RegFunc> creators;
};

class OpExpanderRegister {
 public:
  OpExpanderRegister(const std::string &name, const OpExpanderFactory::RegFunc &func) : func_(func) {
    OpExpanderFactory::Instance().Register(name, func);
  }
  ~OpExpanderRegister() = default;

 private:
  // for pclint-plus
  OpExpanderFactory::RegFunc func_;
};

#define OP_EXPANDER_REGISTER(name, cls)                   \
  static const OpExpanderRegister g_##cls##_expander_reg( \
    name, []() -> std::shared_ptr<OpExpander> { return std::make_shared<cls>(); })
}  // namespace expanders
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_EXPANDERS_EXPANDER_FACTORY_H_
