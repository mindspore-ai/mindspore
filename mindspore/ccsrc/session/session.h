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
#ifndef MINDSPORE_CCSRC_SESSION_SESSION_H
#define MINDSPORE_CCSRC_SESSION_SESSION_H

#include <vector>
#include <string>
#include <unordered_map>
#include <utility>
#include <memory>
#include <map>

#include "session/session_basic.h"
#include "ir/anf.h"
#include "include/inference.h"

namespace mindspore {
namespace inference {
class Session : public MSSession {
 public:
  Session();

  uint32_t CompileGraph(std::shared_ptr<FuncGraph> funcGraphPtr) override;

  MultiTensor RunGraph(uint32_t graph_id, const std::vector<std::shared_ptr<inference::MSTensor>> &inputs) override;

  int Init(const std::string &device, uint32_t device_id);

  static void RegAllOp();

 private:
  std::shared_ptr<session::SessionBasic> session_impl_ = nullptr;
  std::vector<uint32_t> graph_id_;
};
}  // namespace inference
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_SESSION_SESSION_BASIC_H
