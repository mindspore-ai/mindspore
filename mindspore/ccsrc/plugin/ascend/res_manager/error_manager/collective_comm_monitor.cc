/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#include "plugin/ascend/res_manager/error_manager/collective_comm_monitor.h"
#include <signal.h>
#include <unordered_map>
#include <algorithm>
#include <fstream>
#include "plugin/ascend/res_manager/hccl_adapter/hccl_adapter.h"
#include "utils/ms_utils.h"
#include "include/runtime/hardware_abstract/collective/collective_manager.h"
#include "include/utils/anfalgo.h"
#include "include/utils/callback.h"

namespace mindspore {
namespace device {
namespace ascend {
namespace {
// check exception in every 1s
constexpr int64_t kMilSec = 1000;
constexpr int64_t kInterval = 30;
constexpr int kIndent = 2;

int64_t GetCurrentTime() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
    .count();
}

class WatchDogCallback {
 public:
  static WatchDogCallback &GetInstance();

  bool OnLaunchBegin(const CNodePtr &kernel, const std::vector<KernelTensor *> &inputs,
                     const std::vector<KernelTensor *> &outputs, KernelMod *kernel_mod, void *stream,
                     const DeviceContext *device_context) {
    bool is_need_record = HcclWatchDogManager::CheckStatusSaveEnable() && common::AnfAlgo::IsCommunicationOp(kernel);
    if (is_need_record) {
      hccl_work_event_.reset(new HcclWorkEvent(kernel, stream));
    }
    if (hccl_work_event_ != nullptr) {
      hccl_work_event_->RecordStartEvent();
    }

    return false;
  }

  void OnLaunchEnd(const CNodePtr &kernel, const std::vector<KernelTensor *> &inputs,
                   const std::vector<KernelTensor *> &outputs, KernelMod *kernel_mod, void *stream,
                   const DeviceContext *device_context, bool launch_success) {
    if (hccl_work_event_ != nullptr) {
      hccl_work_event_->RecordEndEvent();
      HcclWatchDogManager::GetInstance().AddHcclWorkEvent(std::move(hccl_work_event_));
    }
  }

  void InitCallbackConfig(bool is_enable_watch_dog, bool is_enable_save_op_status, bool is_enable_status_save,
                          int64_t status_save_interval, const std::string &status_save_path) {
    is_enable_watch_dog_ = is_enable_watch_dog;
    is_enable_save_op_status_ = is_enable_save_op_status;
    is_enable_status_save_ = is_enable_status_save;
    status_save_interval_ = status_save_interval;
    status_save_path_ = status_save_path;
  }

  bool is_enable_watch_dog_{false};
  bool is_enable_save_op_status_{false};
  bool is_enable_status_save_{false};
  int64_t status_save_interval_{0};
  std::string status_save_path_{""};

 private:
  std::unique_ptr<HcclWorkEvent> hccl_work_event_ = nullptr;
};

WatchDogCallback &WatchDogCallback::GetInstance() {
  static WatchDogCallback instance;
  return instance;
}

void TftInitWatchDogCallback(bool is_enable_watch_dog, bool is_enable_save_op_status, bool is_enable_status_save,
                             int64_t status_save_interval, const std::string &status_save_path) {
  WatchDogCallback::GetInstance().InitCallbackConfig(is_enable_watch_dog, is_enable_save_op_status,
                                                     is_enable_status_save, status_save_interval, status_save_path);
}

bool TftWatchDogPreLaunchKernel(const CNodePtr &kernel, const std::vector<KernelTensor *> &inputs,
                                const std::vector<KernelTensor *> &outputs, KernelMod *kernel_mod, void *stream,
                                const DeviceContext *device_context) {
  return std::bind(&WatchDogCallback::OnLaunchBegin, &WatchDogCallback::GetInstance(), std::placeholders::_1,
                   std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5,
                   std::placeholders::_6)(kernel, inputs, outputs, kernel_mod, stream, device_context);
}

void TftWatchDogPostLaunchKernel(const CNodePtr &kernel, const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &outputs, KernelMod *kernel_mod, void *stream,
                                 const DeviceContext *device_context, bool launch_success) {
  std::bind(&WatchDogCallback::OnLaunchEnd, &WatchDogCallback::GetInstance(), std::placeholders::_1,
            std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5,
            std::placeholders::_6,
            std::placeholders::_7)(kernel, inputs, outputs, kernel_mod, stream, device_context, launch_success);
}

REGISTER_COMMON_CALLBACK(TftInitWatchDogCallback);
REGISTER_COMMON_CALLBACK(TftWatchDogPreLaunchKernel);
REGISTER_COMMON_CALLBACK(TftWatchDogPostLaunchKernel);
}  // namespace

std::mutex HcclWatchDogHandler::status_map_mutex_;
std::unordered_map<std::string, nlohmann::json> HcclWatchDogHandler::status_map_;

HcclWorkEvent::HcclWorkEvent(const CNodePtr &kernel, void *stream)
    : start_event_(ACL_EVENT_CAPTURE_STREAM_PROGRESS), end_event_(ACL_EVENT_CAPTURE_STREAM_PROGRESS) {
  op_type_ = common::AnfAlgo::GetCNodeName(kernel);
  full_name_ = kernel->fullname_with_scope();
  group_name_ = common::AnfAlgo::GetNodeAttr<std::string>(kernel, kAttrGroup);
  start_event_.set_record_stream(stream);
  end_event_.set_record_stream(stream);
  stop_record_ = false;
}

HcclWorkEvent &HcclWorkEvent::operator=(const HcclWorkEvent &other) {
  if (this != &other) {
    this->op_type_ = other.op_type_;
    this->full_name_ = other.full_name_;
    this->group_name_ = other.group_name_;
    this->seq_ = other.seq_;
    this->status_ = other.status_;
  }
  return *this;
}

bool HcclWorkEvent::CheckAndSetEndStatus() {
  try {
    if (stop_record_) {
      return false;
    }
    if (end_event_.QueryEvent()) {
      status_ = "end";
      return true;
    }
  } catch (const std::exception &e) {
    MS_LOG(WARNING) << "Query event failed, stop record hcom op status. Error message: " << e.what();
    stop_record_ = true;
  }
  return false;
}

bool HcclWorkEvent::CheckStopRecord() { return stop_record_; }
bool HcclWorkEvent::CheckAndSetStartStatus() {
  try {
    if (stop_record_) {
      return false;
    }
    if (start_event_.QueryEvent()) {
      status_ = "start";
      return true;
    }
  } catch (const std::exception &e) {
    MS_LOG(WARNING) << "Query event failed, stop record hcom op status. Error message: " << e.what();
    stop_record_ = true;
  }
  return false;
}

nlohmann::json HcclWorkEvent::ToJson(const std::vector<uint32_t> &comm_ids, uint32_t global_rank_size) const {
  nlohmann::json json_obj;
  json_obj["seq"] = seq_;
  json_obj["op_type"] = op_type_;
  json_obj["op_name"] = full_name_;
  json_obj["pg_id"] = group_name_;
  if (comm_ids.empty() || comm_ids.size() == global_rank_size) {
    json_obj["comm_ids"] = "all";
  } else {
    std::stringstream ss;
    // [1,2,3] => "1,2,3"
    for (size_t i = 0; i < comm_ids.size(); ++i) {
      if (i > 0) {
        ss << ",";
      }
      ss << comm_ids[i];
    }
    json_obj["comm_ids"] = ss.str();
  }
  json_obj["status"] = status_;
  return json_obj;
}

void HcclWatchDogManager::AddHcclWorkEvent(std::unique_ptr<HcclWorkEvent> &&event) {
  if (!CheckStatusSaveEnable()) {
    MS_LOG(INFO) << "No need save hccl op status!";
    return;
  }
  auto handle = handles_.begin();
  auto g_name = event->group_name();
  while (handle != handles_.end()) {
    if (handle->second != nullptr && handle->second->group_name() == g_name) {
      handle->second->AddHcclWorkEvent(std::move(event));
      return;
    }
    handle++;
  }
  MS_LOG(WARNING) << "No hcom  monitor handler found, group name: " << event->group_name();
}

bool HcclWatchDogManager::CheckStatusSaveEnable() { return WatchDogCallback::GetInstance().is_enable_status_save_; }

bool HcclWatchDogManager::InitHandler(const std::string &name) {
  auto it = handles_.find(name);
  if (it != handles_.end() && it->second != nullptr) {
    return it->second->Initialize();
  }
  return false;
}

void HcclWatchDogManager::DestroyHandlerByName(const std::string &name) {
  auto it = handles_.find(name);
  if (it != handles_.end() && it->second != nullptr) {
    MS_LOG(INFO) << "Destroy hcom monitor thread by group name: " << name;
    it->second->Terminate();
    while (!it->second->exit()) {
      MS_LOG(DEBUG) << "Wait exit, group name:" << name;
    }
    handles_.erase(it);
    MS_LOG(INFO) << "Destroy hcom monitor thread by group name: " << name << " success";
  }
}

HcclWatchDogManager::~HcclWatchDogManager() { handles_.clear(); }

HcclWatchDogHandler::~HcclWatchDogHandler() {
  MS_LOG(DEBUG) << "HcclWatchDogHandler destructor start";
  if (HcclWatchDogManager::CheckStatusSaveEnable()) {
    RecordHcclStatus(true);
  }
  terminate_.store(true, std::memory_order_acq_rel);
  if (thread_.joinable()) {
    thread_.join();
  }
  MS_LOG(INFO) << "HcclWatchDogHandler thread exit, rank id: " << rank_id_ << ", group name: " << group_name_;
}

HcclWatchDogHandler::HcclWatchDogHandler(uint32_t rank_id, uint32_t device_id, const std::string &group_name,
                                         HcclComm hcom, const std::vector<uint32_t> &group_ranks) {
  rank_id_ = rank_id;
  device_id_ = device_id;
  group_name_ = group_name;
  hcom_ = hcom;
  rank_size_ = distributed::collective::CollectiveManager::instance()->global_rank_size();
  comm_ids_ = group_ranks;
}

bool HcclWatchDogHandler::Initialize() {
  MS_LOG(INFO) << "Initialize hcom monitor handler. rank id: " << rank_id_ << ", group name: " << group_name_;
  thread_ = std::thread(&HcclWatchDogHandler::WatchDogProcess, this);
  return true;
}

void HcclWatchDogHandler::SetException(std::string *error_info, bool *disable) {
  MS_EXCEPTION_IF_NULL(error_info);
  MS_EXCEPTION_IF_NULL(disable);
  MS_EXCEPTION_IF_NULL(hcom_);
  if (exception_) {
    MS_LOG(WARNING) << "Already has an exception";
    return;
  }
  MS_LOG(DEBUG) << "Hcom Monitor checking for hcom: " << hcom_ << ", group name: " << group_name_
                << ", rank id: " << rank_id_;
  auto ret = hccl::HcclAdapter::GetInstance().HcclWatchdogThread(hcom_, error_info, disable);
  if (!ret) {
    exception_.store(true, std::memory_order_acq_rel);
  }
}

void HcclWatchDogHandler::Terminate() { terminate_.store(true, std::memory_order_acq_rel); }

void HcclWatchDogHandler::DoProcess() {
  std::string error_info;
  auto last_record_time = GetCurrentTime();
  while (!terminate_.load()) {
    MS_LOG(DEBUG) << "Start check hcom monitor thread in every " << kMilSec << "ms .";
    std::this_thread::sleep_for(std::chrono::milliseconds(kMilSec));
    error_info.clear();
    bool disable = false;
    if (CheckHcclEvents()) {
      auto now_time = GetCurrentTime();
      if (now_time - last_record_time > GetStatusSaveInterval()) {
        RecordHcclStatus(false);
        last_record_time = now_time;
      }
    }
    SetException(&error_info, &disable);
    if (!error_info.empty()) {
      err_message_ = error_info;
    }
    if (disable) {
      MS_LOG(WARNING) << "Call HcclGetCommAsyncError failed, close hcom monitor for group: " << group_name_;
      Terminate();
      break;
    }
    if (exception_) {
      MS_LOG(ERROR) << "Hcom Monitor thread got hccl error,rank: " << rank_id_ << ", group name:" << group_name_
                    << ",details : " << error_info;
      return;
    }
  }
}

void HcclWatchDogHandler::WatchDogProcess() {
  MS_LOG(INFO) << "WatchDogProcess start, rank id: " << rank_id_ << ", group name: " << group_name_;
  DoProcess();
  if (HcclWatchDogManager::CheckStatusSaveEnable()) {
    RecordHcclStatus(true);
  }
  if (exception_ && WatchDogCallback::GetInstance().is_enable_watch_dog_) {
    MS_LOG(ERROR) << "[HcclWatchDog] Try to kill this process by SIGTERM. Node:"
                  << common::GetEnv(distributed::kEnvWorkerIp);
    (void)killpg(getpid(), SIGTERM);
  }
  exit_.store(true, std::memory_order_acq_rel);
  MS_LOG(INFO) << "Hcom monitor thread for group:" << group_name_ << " execute end.";
}

const std::string &HcclWatchDogHandler::SavePath() { return WatchDogCallback::GetInstance().status_save_path_; }

const int64_t HcclWatchDogHandler::GetStatusSaveInterval() {
  static auto interval = ([]() -> int64_t {
    auto inter_val = WatchDogCallback::GetInstance().status_save_interval_;
    if (inter_val < 0) {
      MS_LOG(WARNING) << "HCCL_STATUS_SAVE_INTERVAL value: " << inter_val << " is invalid, using default value: 30s";
      inter_val = kInterval;
    }
    return inter_val * kMilSec;
  })();
  return interval;
}

void HcclWatchDogHandler::AddHcclWorkEvent(std::unique_ptr<HcclWorkEvent> &&event) {
  event->SetSeq(seq_.fetch_add(1, std::memory_order_relaxed));
  std::lock_guard<std::mutex> lock(event_list_mutex_);
  event_list_.push_back(std::move(event));
}

void HcclWatchDogHandler::UpdateHcclStatus() {
  if (worker_event_updated_) {
    status_map_[group_name_] = current_event_.ToJson(comm_ids_, rank_size_);
    worker_event_updated_ = false;
  }
}

void HcclWatchDogHandler::RecordHcclStatus(bool is_end) {
  if (!HcclWatchDogManager::CheckStatusSaveEnable()) {
    return;
  }
  std::lock_guard<std::mutex> lock(status_map_mutex_);
  static auto cur_record_time = GetCurrentTime();
  static auto scheduler_host = common::GetEnv("MS_SCHED_HOST", "127.0.0.1");
  static auto node_ip = common::GetEnv(distributed::kEnvWorkerIp);
  static std::string record_file = SavePath() + "/" + "ms_status_record_" + std::to_string(rank_id_) + "_" +
                                   scheduler_host + "_" + std::to_string(device_id_) + "_" +
                                   std::to_string(rank_size_) + "_" + std::to_string(getpid()) + "_" +
                                   std::to_string(cur_record_time) + ".json";
  UpdateHcclStatus();
  MS_LOG(INFO) << "Start RecordHcclStatus: status_map_ size: " << status_map_.size();
  if (status_map_.empty()) {
    MS_LOG(INFO) << "No status to record, return!";
    return;
  }
  std::ofstream ofs(record_file);
  if (!ofs.is_open()) {
    MS_LOG(ERROR) << "Open file failed, file: " << record_file;
    return;
  }
  std::vector<nlohmann::json> status_list;
  status_list.reserve(status_map_.size());
  std::transform(status_map_.begin(), status_map_.end(), std::back_inserter(status_list),
                 [](const auto &item) { return item.second; });
  nlohmann::json json_obj;
  json_obj["last_comm_op"] = status_list;
  json_obj["global_pg_end_time"] = is_end ? GetCurrentTime() : cur_record_time;
  json_obj["is_master"] = scheduler_host == node_ip;
  json_obj["node_ip"] = node_ip;
  json_obj["global_rank"] = rank_id_;
  json_obj["local_rank"] = device_id_;
  json_obj["exception_message"] = err_message_;
  ofs << json_obj.dump(kIndent) << std::endl;
  ofs.close();
  MS_LOG(INFO) << "End RecordHcclStatus: status_map_ size: " << status_map_.size();
  return;
}

bool HcclWatchDogHandler::CheckHcclEvents() {
  if (!HcclWatchDogManager::CheckStatusSaveEnable()) {
    return false;
  }
  std::lock_guard<std::mutex> lock(event_list_mutex_);
  if (event_list_.empty()) {
    return false;
  }
  auto it = event_list_.begin();
  while (it != event_list_.end()) {
    if (it->get()->CheckAndSetEndStatus()) {
      current_event_ = *(*it);
      it = event_list_.erase(it);
      worker_event_updated_ = true;
      continue;
    }
    if (it->get()->CheckAndSetStartStatus()) {
      current_event_ = *(*it);
      worker_event_updated_ = true;
    }
    if (it->get()->CheckStopRecord()) {
      return false;
    }
    it++;
  }
  if (event_list_.empty()) {
    // update op status after all event execute end
    RecordHcclStatus(false);
  }
  return true;
}

void RegisterHcclWatchDog(uint32_t global_rank, uint32_t device_id, const std::string &name, HcclComm comm,
                          const std::vector<uint32_t> &group_ranks) {
  if (common::GetEnv(kSimulationLevel).empty() && (WatchDogCallback::GetInstance().is_enable_watch_dog_ ||
                                                   WatchDogCallback::GetInstance().is_enable_save_op_status_)) {
    MS_LOG(INFO) << "Start initializing hccl watchdog on device side for group: " << name << ", rank: " << global_rank;
    HcclWatchDogManager::GetInstance().AddHandler(
      std::make_unique<HcclWatchDogHandler>(global_rank, device_id, name, comm, group_ranks));
    (void)HcclWatchDogManager::GetInstance().InitHandler(name);
    MS_LOG(INFO) << "hccl watchdog on device side is successfully initialized.";
  }
}

void DestroyWatchDogHandler(const std::string &name) { HcclWatchDogManager::GetInstance().DestroyHandlerByName(name); }

void DestroyWatchDogAllHandlers() { HcclWatchDogManager::GetInstance().DestroyHandler(); }

REGISTER_COMMON_CALLBACK(RegisterHcclWatchDog);
REGISTER_COMMON_CALLBACK(DestroyWatchDogHandler);
REGISTER_COMMON_CALLBACK(DestroyWatchDogAllHandlers);
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
