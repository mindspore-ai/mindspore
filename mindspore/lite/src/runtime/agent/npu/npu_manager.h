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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_AGENT_NPU_NPU_MANAGER_H_
#define MINDSPORE_LITE_SRC_RUNTIME_AGENT_NPU_NPU_MANAGER_H_
#include <string>
#include <memory>
#include <vector>
#include <unordered_map>
#include <set>
#include "schema/model_generated.h"
#include "include/HiAiModelManagerService.h"

namespace mindspore::lite {
static std::set<mindspore::schema::PrimitiveType> npu_trans_nodes = {
  schema::PrimitiveType_Conv2D,          schema::PrimitiveType_DeConv2D,
  schema::PrimitiveType_DepthwiseConv2D, schema::PrimitiveType_DeDepthwiseConv2D,
  schema::PrimitiveType_Resize,          schema::PrimitiveType_Pooling};
class NPUManager {
 public:
  static NPUManager *GetInstance() {
    static NPUManager npuManager;
    return &npuManager;
  }

  bool IsSupportNPU();

  // provide to subgraph to add model.
  int AddModel(void *model_buf, uint32_t size, const std::string &model_name, int frequency);

  // scheduler to load om model.
  int LoadOMModel();

  // provide to executor.
  std::shared_ptr<hiai::AiModelMngerClient> GetClient(const std::string &model_name);

  int index() const;

 private:
  bool IsKirinChip();

  bool CheckEMUIVersion();

  bool CheckDDKVersion();

  int CompareVersion(const std::string &version1, const std::string &version2);

 private:
  int index_ = 0;

  bool is_npu_check_executor = false;

  bool is_support_npu = false;

  std::vector<std::shared_ptr<hiai::AiModelMngerClient>> clients_;

  std::vector<std::shared_ptr<hiai::AiModelDescription>> model_desc_;

  std::shared_ptr<hiai::AiModelBuilder> mc_builder_ = nullptr;

  std::unordered_map<std::string, int> model_map_;
};

}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_RUNTIME_AGENT_NPU_NPU_MANAGER_H_
