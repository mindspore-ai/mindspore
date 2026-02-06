/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_HARDWARE_STRESS_DETECT_H_
#define MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_HARDWARE_STRESS_DETECT_H_
#include <thread>
#include <future>
#include <utility>
#include <memory>
#include <string>
#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "runtime/pipeline/task/task.h"

namespace mindspore {
namespace kernel {
namespace pyboost {

typedef struct AmlP2PDetectAttr {
  void *workspace;
  uint64_t workspaceSize;
  uint8_t reserve[64];
} AmlP2PDetectAttr;

typedef enum AmlDetectRunMode {
  AML_DETECT_RUN_MODE_ONLINE = 0,
  AML_DETECT_RUN_MODE_OFFLINE = 1,
  AML_DETECT_RUN_MODE_MAX,
} AmlDetectRunMode;

typedef struct AmlAicoreDetectAttr {
  AmlDetectRunMode mode;
  void *workspace;
  uint64_t workspaceSize;
  uint8_t reserve[64];
} AmlAicoreDetectAttr;

class StressDetectTask : public runtime::AsyncTask {
 public:
  StressDetectTask(std::function<int(int32_t, void *, uint64_t)> run_func, uint32_t device_id, void *workspace_addr,
                   uint64_t workspace_size, std::promise<int> &&p)
      : AsyncTask(runtime::kStressDetectTask),
        run_func_(std::move(run_func)),
        device_id_(device_id),
        workspace_addr_(workspace_addr),
        workspace_size_(workspace_size),
        p_(std::move(p)) {}
  ~StressDetectTask() override = default;
  void Run() override;

 private:
  std::function<int(int32_t, void *, uint64_t)> run_func_;
  uint32_t device_id_;
  void *workspace_addr_;
  uint64_t workspace_size_;
  std::promise<int> p_;
};

class AmlAicoreDetectTask : public runtime::AsyncTask {
 public:
  AmlAicoreDetectTask(std::function<int(int32_t, const AmlAicoreDetectAttr *)> run_func, uint32_t device_id,
                      std::shared_ptr<AmlAicoreDetectAttr> attr, std::promise<int> &&p)
      : AsyncTask(runtime::kStressDetectTask),
        run_func_(std::move(run_func)),
        device_id_(device_id),
        attr_(std::move(attr)),
        p_(std::move(p)) {}
  uint32_t device_id() const { return device_id_; }
  std::shared_ptr<AmlAicoreDetectAttr> attr() const { return attr_; }
  ~AmlAicoreDetectTask() override = default;
  void Run() override;

 private:
  std::function<int(int32_t, const AmlAicoreDetectAttr *)> run_func_;
  uint32_t device_id_;
  std::shared_ptr<AmlAicoreDetectAttr> attr_;
  std::promise<int> p_;
};

int StressDetectKernel(const std::string &detect_type);
inline std::string GetLibAscendMLName() { return "/lib64/libascend_ml.so"; }
constexpr const char *kNameAmlAicoreDetectOnline = "AmlAicoreDetectOnline";
constexpr const char *kNameAmlP2PDetectOnline = "AmlP2PDetectOnline";
constexpr int kDetectSucceeded = 0;
constexpr int kDetectFailed = 1;
constexpr int kDetectFailedWithHardwareFailure = 2;
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_HARDWARE_STRESS_DETECT_H_
