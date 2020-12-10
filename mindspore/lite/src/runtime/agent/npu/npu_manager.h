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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_AGENT_NPU_NPU_UTILS_H_
#define MINDSPORE_LITE_SRC_RUNTIME_AGENT_NPU_NPU_UTILS_H_
#include <string>
#include <memory>
#include <vector>
#include "include/HiAiModelManagerService.h"

namespace mindspore::lite {

class NPUManager {
 public:
  static NPUManager *GetInstance() {
    static NPUManager npuManager;
    return &npuManager;
  }

  bool IsSupportNPU();

  int InitClient();

  // provide to subgraph to add model.
  int AddModel(void *model_buf, uint32_t size, const std::string &model_name, int frequency);

  // scheduler to load om model.
  int LoadOMModel();

  // provide to executor.
  std::shared_ptr<hiai::AiModelMngerClient> GetClient();

  int index();

 private:
  void CheckSupportNPU();

  bool IsKirinChip();

  bool CheckOmBuildIr(const std::string &path);

  std::string GetExecutorPath();

 private:
  int index_ = 0;

  bool is_npu_check_executor = false;

  bool is_support_npu = false;

  std::shared_ptr<hiai::AiModelMngerClient> client_ = nullptr;

  std::vector<std::shared_ptr<hiai::AiModelDescription>> model_desc_;

  std::shared_ptr<hiai::AiModelBuilder> mc_builder_ = nullptr;
};

}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_RUNTIME_AGENT_NPU_NPU_UTILS_H_
