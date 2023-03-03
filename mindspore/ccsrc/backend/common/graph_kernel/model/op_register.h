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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_MODEL_OP_REGISTER_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_MODEL_OP_REGISTER_H_

#include <functional>
#include <string>

#include "utils/hash_map.h"
#include "backend/common/graph_kernel/model/op_node.h"
#include "include/backend/visible.h"

namespace mindspore::graphkernel::inner {
using CreatorFunc = std::function<PrimOpPtr(const std::string &)>;
class BACKEND_EXPORT OpRegistry {
 public:
  static OpRegistry &Instance() {
    static OpRegistry instance{};
    return instance;
  }
  void Register(const std::string &op_name, const CreatorFunc &func) { (void)creators.emplace(op_name, func); }

  PrimOpPtr NewOp(const std::string &op) {
    // "OpaqueOp" is registered by default.
    return creators.find(op) == creators.end() ? creators["_opaque"](op) : creators[op](op);
  }

 private:
  OpRegistry() = default;
  ~OpRegistry() = default;

  OpRegistry(const OpRegistry &) = delete;
  OpRegistry(const OpRegistry &&) = delete;
  OpRegistry &operator=(const OpRegistry &) = delete;
  OpRegistry &operator=(const OpRegistry &&) = delete;

  mindspore::HashMap<std::string, CreatorFunc> creators;
};
}  // namespace mindspore::graphkernel::inner
#endif
