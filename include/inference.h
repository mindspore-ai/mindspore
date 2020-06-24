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

#ifndef MINDSPORE_INCLUDE_MS_SESSION_H
#define MINDSPORE_INCLUDE_MS_SESSION_H

#include <memory>
#include <vector>
#include <string>
#include "include/ms_tensor.h"

namespace mindspore {
class FuncGraph;
namespace inference {
class MS_API MSSession {
 public:
  MSSession() = default;

  static std::shared_ptr<MSSession> CreateSession(const std::string &device, uint32_t device_id);

  virtual uint32_t CompileGraph(std::shared_ptr<FuncGraph> funcGraphPtr) = 0;

  virtual MultiTensor RunGraph(uint32_t graph_id, const std::vector<std::shared_ptr<inference::MSTensor>> &inputs) = 0;
};

std::shared_ptr<FuncGraph> MS_API LoadModel(const char *model_buf, size_t size, const std::string &device);

void MS_API ExitInference();
}  // namespace inference
}  // namespace mindspore
#endif  // MINDSPORE_INCLUDE_MS_SESSION_H
