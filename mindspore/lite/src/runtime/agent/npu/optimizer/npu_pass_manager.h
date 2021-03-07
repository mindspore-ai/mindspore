/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_AGENT_NPU_OPTIMIZER_NPU_PASS_MANAGER_H_
#define MINDSPORE_LITE_SRC_RUNTIME_AGENT_NPU_OPTIMIZER_NPU_PASS_MANAGER_H_
#include <vector>
#include "src/runtime/agent/npu/optimizer/npu_base_pass.h"
#include "src/inner_context.h"
namespace mindspore::lite {
class NPUPassManager {
 public:
  static NPUPassManager *GetInstance() {
    static NPUPassManager pass_manager;
    return &pass_manager;
  }

  ~NPUPassManager() { Clear(); }

  void AddPass(NPUBasePass *pass);

  int Run();

  void Clear();

 private:
  std::vector<NPUBasePass *> all_pass_;
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_RUNTIME_AGENT_NPU_OPTIMIZER_NPU_PASS_MANAGER_H_
