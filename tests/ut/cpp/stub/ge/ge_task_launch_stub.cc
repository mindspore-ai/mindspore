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
#include "device/ascend/tasksink/runtime_utils.h"

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

bool ModelRunner::RunModel(uint32_t model_id, const ge::InputData &input_data, ge::OutputData *output_data) {
  return true;
}

const std::vector<uint32_t> &ModelRunner::GetTaskIdList(uint32_t model_id) const {
  static std::vector<uint32_t> task_id_list;
  return task_id_list;
}
}  // namespace model_runner
}  // namespace ge

namespace mindspore {
namespace device {
namespace ascend {
namespace tasksink {
bool RuntimeUtils::HcomBindModel(rtModel_t model, rtStream_t stream) { return true; }

bool RuntimeUtils::HcomUnbindModel(rtModel_t model) { return true; }

bool RuntimeUtils::HcomDistribute(const std::shared_ptr<HcclTaskInfo> &task_info, rtStream_t stream) { return true; }
}  // namespace tasksink
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
