/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_GE_RUNTIME_TASK_LABEL_MANAGER_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_GE_RUNTIME_TASK_LABEL_MANAGER_H_

#include <vector>
#include <memory>
#include <mutex>
#include <map>
#include <string>
#include "runtime/base.h"

namespace mindspore::ge::model_runner {
class LabelGuard {
 public:
  explicit LabelGuard(void *label_info) : label_info_(reinterpret_cast<uintptr_t>(label_info)) {}
  ~LabelGuard();
  void *GetLabelInfo() noexcept { return reinterpret_cast<void *>(label_info_); }

 private:
  uintptr_t label_info_;
};

class LabelManager {
 public:
  static std::shared_ptr<LabelManager> GetInstance();
  std::shared_ptr<LabelGuard> GetLabelInfo(rtModel_t model, const std::vector<uint32_t> &label_ids,
                                           const std::vector<void *> &all_label);

 private:
  std::mutex model_info_mapping_mutex_;
  std::map<rtModel_t, std::map<std::string, std::weak_ptr<LabelGuard>>> model_info_mapping_;

  static std::weak_ptr<LabelManager> instance_;
  static std::mutex instance_mutex_;
};
}  // namespace mindspore::ge::model_runner
#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_GE_RUNTIME_TASK_LABEL_MANAGER_H_
