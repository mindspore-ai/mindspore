/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_GE_RUNTIME_TASK_HCCL_TASK_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_GE_RUNTIME_TASK_HCCL_TASK_H_

#include <memory>
#include <set>
#include <map>
#include <vector>
#include <mutex>
#include "plugin/device/ascend/hal/device/ge_runtime/task/task.h"
#include "plugin/device/ascend/hal/device/dump/kernel_dumper.h"
#include "common/opskernel/ge_task_info.h"

namespace mindspore::ge::model_runner {
class HcclTask : public TaskRepeater<HcclTaskInfo> {
 public:
  HcclTask(const ModelContext &model_context, const std::shared_ptr<HcclTaskInfo> &task_info);

  ~HcclTask() override;

  void Distribute() override;

 private:
  class StreamGuard;
  void SetSecondaryStream();
  void CreateStream(int64_t stream_num, int64_t master_stream_id);
  void CreateStream(rtModel_t model, rtStream_t *stream) const;
  void SaveHcclSecondaryStream(int64_t master_stream_id, const std::shared_ptr<StreamGuard> &stream);
  std::shared_ptr<StreamGuard> GetSecondaryStream(std::vector<std::weak_ptr<StreamGuard>> *secondary_streams,
                                                  size_t index);

  std::shared_ptr<HcclTaskInfo> task_info_;
  void *stream_;
  void *workspace_mem_;
  rtModel_t rt_model_handle_;
  int32_t priority_;
  std::vector<std::shared_ptr<StreamGuard>> secondary_stream_list_;

  // map<key: model pointer, value: map<key: primary stream id, value: vector<secondary stream pointer>>>
  static std::map<rtModel_t, std::map<uint32_t, std::vector<std::weak_ptr<StreamGuard>>>> model_stream_mapping_;
  static std::mutex model_stream_mapping_mutex_;

  ::ge::GETaskInfo ge_task_;
  device::ascend::KernelDumper kernel_dumper_;
};

class HcclTask::StreamGuard {
 public:
  StreamGuard(rtModel_t model, rtStream_t stream) : model_(model), stream_(stream) {}
  ~StreamGuard();
  const rtStream_t GetStream() const { return stream_; }

 private:
  rtModel_t model_;
  rtStream_t stream_;
};
}  // namespace mindspore::ge::model_runner
#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_GE_RUNTIME_TASK_HCCL_TASK_H_
