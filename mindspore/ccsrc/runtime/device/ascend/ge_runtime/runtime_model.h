/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_GE_RUNTIME_RUNTIME_MODEL_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_GE_RUNTIME_RUNTIME_MODEL_H_
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <tuple>
#include "runtime/base.h"
#include "runtime/rt_model.h"
#include "runtime/device/ascend/ge_runtime/davinci_model.h"

namespace mindspore::ge::model_runner {
using RuntimeInfo = std::tuple<uint32_t, uint32_t, void *>;
class Task;
class RuntimeModel {
 public:
  RuntimeModel() = default;
  ~RuntimeModel();

  void Load(uint32_t device_id, uint64_t session_id, const std::shared_ptr<DavinciModel> &davinci_model);
  void DistributeTask();
  void LoadComplete();
  const std::vector<uint32_t> &GetTaskIdList() const;
  const std::vector<uint32_t> &GetStreamIdList() const;
  const std::map<std::string, std::shared_ptr<RuntimeInfo>> &GetRuntimeInfoMap() const { return runtime_info_map_; }
  rtModel_t GetModelHandle() const { return rt_model_handle_; }
  rtStream_t GetModelStream() const { return rt_model_stream_; }
  void Run();

 private:
  void InitResource(const std::shared_ptr<DavinciModel> &davinci_model);
  void GenerateTask(uint32_t device_id, uint64_t session_id, const std::shared_ptr<DavinciModel> &davinci_model);
  void InitStream(const std::shared_ptr<DavinciModel> &davinci_model);
  void InitEvent(uint32_t event_num);
  void InitLabel(const std::shared_ptr<DavinciModel> &davinci_model);
  void RtModelUnbindStream() noexcept;
  void RtStreamDestory() noexcept;
  void RtModelDestory() noexcept;
  void RtLabelDestory() noexcept;
  void RtEventDestory() noexcept;

  rtModel_t rt_model_handle_{};
  rtStream_t rt_model_stream_{};

  std::vector<rtStream_t> stream_list_{};
  std::vector<rtLabel_t> label_list_{};
  std::vector<rtEvent_t> event_list_{};

  std::vector<std::shared_ptr<Task>> task_list_{};

  std::vector<uint32_t> task_id_list_{};
  std::vector<uint32_t> stream_id_list_{};
  std::map<std::string, std::shared_ptr<RuntimeInfo>> runtime_info_map_;
};
}  // namespace mindspore::ge::model_runner
#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_GE_RUNTIME_RUNTIME_MODEL_H_
