/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include <functional>
#include <string>
#include <memory>

#include "utils/hash_map.h"
#include "backend/common/graph_kernel/expanders/utils.h"
#include "include/backend/visible.h"

namespace mindspore::graphkernel::expanders {
class BACKEND_EXPORT OpDescFactory {
 public:
  static OpDescFactory &Instance() {
    static OpDescFactory instance = OpDescFactory();
    return instance;
  }
  bool HasOp(const std::string &op) const { return creators.find(op) != creators.end(); }
  std::shared_ptr<OpDesc> GetOp(const std::string &op) const {
    if (auto iter = creators.find(op); iter != creators.end()) {
      auto op_desc = iter->second();
      op_desc->name_ = op;
      return op_desc;
    }
    return nullptr;
  }
  OpDescFactory() = default;
  ~OpDescFactory() = default;

  using RegFunc = std::function<std::shared_ptr<OpDesc>()>;
  void Register(const std::string &op, const RegFunc &func) { creators[op] = func; }

 private:
  inline static mindspore::HashMap<std::string, RegFunc> creators;
};

class OpDescRegister {
 public:
  OpDescRegister(const std::string &name, const OpDescFactory::RegFunc &func) : func_(func) {
    OpDescFactory::Instance().Register(name, func);
  }
  ~OpDescRegister() = default;

 private:
  // for pclint-plus
  OpDescFactory::RegFunc func_;
};

#define JOIN(x, y) x##y
#define UNIQUE_NAME(prefix, cnt) JOIN(prefix, cnt)
#define EXPANDER_OP_DESC_REGISTER(name, cls)                         \
  const OpDescRegister UNIQUE_NAME(g_expander_opdesc_, __COUNTER__)( \
    name, []() noexcept -> std::shared_ptr<OpDesc> { return std::make_shared<cls>(); })
}  // namespace mindspore::graphkernel::expanders
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_EXPANDERS_EXPANDER_FACTORY_H_
