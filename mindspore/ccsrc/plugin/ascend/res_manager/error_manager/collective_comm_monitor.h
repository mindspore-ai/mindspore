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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_HARDWARE_ASCEND_WATCH_DOG_THREAD_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_HARDWARE_ASCEND_WATCH_DOG_THREAD_H_

#include <atomic>
#include <map>
#include <vector>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <thread>
#include <unordered_map>
#include <list>
#include "hccl/hccl.h"
#include "ir/anf.h"
#include "nlohmann/json.hpp"
#include "plugin/ascend/res_manager/event/ascend_event.h"
#include "plugin/ascend/res_manager/visible.h"

namespace mindspore {
namespace device {
namespace ascend {
class ASCEND_RES_MANAGER_EXPORT HcclWorkEvent {
 public:
  HcclWorkEvent() : start_event_(ACL_EVENT_CAPTURE_STREAM_PROGRESS), end_event_(ACL_EVENT_CAPTURE_STREAM_PROGRESS) {}
  ~HcclWorkEvent() = default;
  HcclWorkEvent(const CNodePtr &kernel, void *stream);
  HcclWorkEvent &operator=(const HcclWorkEvent &other);
  bool CheckAndSetEndStatus();
  bool CheckAndSetStartStatus();
  bool CheckStopRecord();
  void RecordStartEvent() { start_event_.RecordEvent(); }
  void RecordEndEvent() { end_event_.RecordEvent(); }
  void SetSeq(uint64_t seq) { seq_ = seq; }
  const std::string &group_name() const { return group_name_; }
  nlohmann::json ToJson(const std::vector<uint32_t> &comm_ids, uint32_t global_rank_size) const;

 private:
  uint64_t seq_;
  std::string op_type_;
  std::string full_name_;
  std::string group_name_;
  std::string status_;
  AscendEvent start_event_;
  AscendEvent end_event_;
  bool stop_record_;
};

class ASCEND_RES_MANAGER_EXPORT HcclWatchDogHandler {
 public:
  HcclWatchDogHandler(uint32_t global_rank_id, uint32_t device_id, const std::string &group_name, HcclComm hcom,
                      const std::vector<uint32_t> &group_ranks);
  ~HcclWatchDogHandler();
  bool Initialize();
  void Terminate();
  uint32_t rank_id() const { return rank_id_; }
  std::string group_name() const { return group_name_; }
  bool exit() const { return exit_; }
  void AddHcclWorkEvent(std::unique_ptr<HcclWorkEvent> &&event);

 private:
  void WatchDogProcess();
  void SetException(std::string *error_info, bool *disable);
  void DoProcess();
  static const int64_t GetStatusSaveInterval();
  static const std::string &SavePath();
  bool CheckHcclEvents();
  void UpdateHcclStatus();
  void RecordHcclStatus(bool is_end = false);

  uint32_t device_id_;
  uint32_t rank_size_;
  bool worker_event_updated_{false};
  std::string err_message_;
  uint32_t rank_id_;
  std::string group_name_;
  HcclComm hcom_;
  std::thread thread_;
  std::atomic<bool> exception_{false};
  std::atomic<bool> terminate_{false};
  std::atomic<bool> exit_{false};
  HcclWorkEvent current_event_;
  // Mutex for current event list
  std::mutex event_list_mutex_;
  std::list<std::unique_ptr<HcclWorkEvent>> event_list_;
  // This is a global status map, used to record all current status in all collective comms.
  // The key is the group name, value is the status json
  static std::unordered_map<std::string, nlohmann::json> status_map_;
  // Mutex for global status map
  static std::mutex status_map_mutex_;
  // The comm ids for current collective comm
  std::vector<uint32_t> comm_ids_;
  std::atomic<uint32_t> seq_{1};
};

class ASCEND_RES_MANAGER_EXPORT HcclWatchDogManager {
 public:
  static HcclWatchDogManager &GetInstance() {
    static HcclWatchDogManager instance;
    return instance;
  }

  void AddHcclWorkEvent(std::unique_ptr<HcclWorkEvent> &&event);
  static bool CheckStatusSaveEnable();
  void AddHandler(std::unique_ptr<HcclWatchDogHandler> handler) {
    auto name = handler->group_name();
    handles_[name] = std::move(handler);
  }
  bool InitHandler(const std::string &name);
  void DestroyHandlerByName(const std::string &name);
  void DestroyHandler() {
    std::unique_lock<std::mutex> lock(handle_mutex_);
    if (handles_.empty()) {
      return;
    }
    for (const auto &handle : handles_) {
      if (handle.second != nullptr) {
        handle.second->Terminate();
      }
    }
    handles_.clear();
  }

 private:
  HcclWatchDogManager() = default;
  ~HcclWatchDogManager();
  HcclWatchDogManager(const HcclWatchDogManager &) = delete;
  HcclWatchDogManager &operator=(const HcclWatchDogManager &) = delete;
  std::mutex handle_mutex_;
  std::unordered_map<std::string, std::unique_ptr<HcclWatchDogHandler>> handles_;
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_HARDWARE_ASCEND_WATCH_DOG_THREAD_H_
