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
#include <vector>
#include "framework/ge_runtime/model_runner.h"
#include "runtime/hccl_adapter/hccl_adapter.h"

namespace ge {
namespace model_runner {
ModelRunner &ModelRunner::Instance() {
  static ModelRunner runner;
  return runner;
}

bool ModelRunner::LoadDavinciModel(uint32_t device_id, uint64_t session_id, uint32_t model_id,
                                   std::shared_ptr<DavinciModel> ascend_model,
                                   std::shared_ptr<ge::ModelListener> listener) {
  return true;
}

bool ModelRunner::UnloadModel(uint32_t model_id) { return true; }

bool ModelRunner::LoadModelComplete(uint32_t model_id) { return true; }

bool ModelRunner::RunModel(uint32_t model_id, const ge::InputData &input_data, ge::OutputData *output_data) {
  return true;
}

void *ModelRunner::GetModelHandle(uint32_t model_id) const { return nullptr; }

bool ModelRunner::DistributeTask(uint32_t model_id) { return true; }

const std::vector<uint32_t> &ModelRunner::GetTaskIdList(uint32_t model_id) const {
  static std::vector<uint32_t> task_id_list;
  return task_id_list;
}

const std::vector<uint32_t> &ModelRunner::GetStreamIdList(uint32_t model_id) const {
  static std::vector<uint32_t> stream_id_list;
  return stream_id_list;
}

const std::map<std::string, std::shared_ptr<RuntimeInfo>> &ModelRunner::GetRuntimeInfoMap(uint32_t model_id) const {
  static std::map<std::string, std::shared_ptr<RuntimeInfo>> runtime_info_map;
  return runtime_info_map;
}
}  // namespace model_runner
}  // namespace ge

namespace mindspore {
namespace hccl {
bool InitHccl(uint32_t, std::string_view, std::string_view) { return true; }
bool FinalizeHccl() { return true; }
bool GenTask(const AnfNodePtr &, HcclDataType, std::vector<HcclTaskInfo> *) { return true; }
int64_t CalcWorkspaceSize(const AnfNodePtr &, HcclDataType) { return 0; }
void *GetHcclOpsKernelInfoStore() { return nullptr; }
std::string GetHcclType(const AnfNodePtr &) { return ""; }
}  // namespace hccl
}  // namespace mindspore
