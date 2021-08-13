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
#include "backend/optimizer/graph_kernel/expanders/reshape.h"
#include "backend/optimizer/graph_kernel/expanders/bias_add.h"

namespace mindspore {
namespace opt {
namespace expanders {
#define OP_EXPANDER_CREATOR(cls) []() -> std::shared_ptr<OpExpander> { return std::make_shared<cls>(); }

class OpExpanderFactory {
 public:
  static OpExpanderFactory &Instance() {
    static std::unique_ptr<OpExpanderFactory> instance = nullptr;
    if (instance == nullptr) {
      instance.reset(new OpExpanderFactory());
    }
    return *instance;
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

 private:
  using RegFunc = std::function<std::shared_ptr<OpExpander>()>;
  void Register(std::string &&op, RegFunc &&func) { creators.insert({op, func}); }
  OpExpanderFactory() {
    Register("BiasAdd", OP_EXPANDER_CREATOR(expanders::BiasAdd));
    Register("ExpandDims", OP_EXPANDER_CREATOR(expanders::ExpandDims));
  }

  std::unordered_map<std::string, RegFunc> creators;
};
}  // namespace expanders
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_EXPANDERS_EXPANDER_FACTORY_H_
