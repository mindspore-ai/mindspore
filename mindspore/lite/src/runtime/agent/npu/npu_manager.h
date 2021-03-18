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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_AGENT_NPU_NPU_MANAGER_H_
#define MINDSPORE_LITE_SRC_RUNTIME_AGENT_NPU_NPU_MANAGER_H_
#include <string>
#include <memory>
#include <utility>
#include <vector>
#include <unordered_map>
#include <set>
#include "include/hiai_ir_build.h"
#include "schema/model_generated.h"
#include "include/HiAiModelManagerService.h"

namespace mindspore::lite {

struct SubGraphModel {
 public:
  SubGraphModel(int index, std::string model_name, std::shared_ptr<domi::ModelBufferData> model_buffer_data)
      : index_(index), model_name_(std::move(model_name)), model_buffer_data_(model_buffer_data) {}

  bool is_freed_ = false;
  bool is_loaded_ = false;
  int index_;
  std::string model_name_;
  std::shared_ptr<domi::ModelBufferData> model_buffer_data_;
  std::shared_ptr<hiai::AiModelMngerClient> client_;
  std::shared_ptr<hiai::AiModelDescription> desc_;
};
class NPUManager {
 public:
  NPUManager() = default;

  ~NPUManager() { Reset(); }

  bool IsSupportNPU();

  // provide to subgraph to add model.
  int AddModel(std::shared_ptr<domi::ModelBufferData> model_buffer_data, const std::string &model_name, int frequency);

  // scheduler to load om model.
  int LoadOMModel();

  // provide to executor.
  std::shared_ptr<hiai::AiModelMngerClient> GetClient(const std::string &model_name);

  int index() const;

  void Reset();

  int LoadModel(const std::shared_ptr<hiai::AiModelMngerClient> &client,
                std::vector<std::shared_ptr<hiai::AiModelDescription>> desc_list);

 private:
  bool IsKirinChip();

  bool CheckEMUIVersion();

  bool CheckDDKVersion();

  int CompareVersion(const std::string &version1, const std::string &version2);

  std::shared_ptr<hiai::AiModelMngerClient> CreateAiModelMngerClient();

 private:
  int index_ = 0;
  bool is_check_version_ = false;
  bool is_support_ = false;
  std::unordered_map<std::string, std::shared_ptr<SubGraphModel>> models_;
  std::vector<std::shared_ptr<hiai::AiModelMngerClient>> clients_;
};

}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_RUNTIME_AGENT_NPU_NPU_MANAGER_H_
