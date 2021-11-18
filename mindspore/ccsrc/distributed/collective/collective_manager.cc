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

#include "distributed/collective/collective_manager.h"
#include <string>
#include <vector>
#include <memory>

namespace mindspore {
namespace distributed {
namespace collective {
CollectiveManager::CollectiveManager()
    : inited_(false),
      finalized_(true),
      host_ctx_(nullptr),
      device_ctx_(nullptr),
      host_comm_lib_(nullptr),
      device_comm_lib_(nullptr),
      global_rank_id_(0),
      local_rank_id_(0),
      global_rank_size_(0),
      global_group_ranks_({}) {}

CollectiveManager::~CollectiveManager() {
  if (!finalized_) {
    Finalize();
  }
}

std::shared_ptr<CollectiveManager> CollectiveManager::instance() {
  static std::shared_ptr<CollectiveManager> instance = nullptr;
  if (instance == nullptr) {
    instance.reset(new (std::nothrow) CollectiveManager());
    MS_EXCEPTION_IF_NULL(instance);
  }
  return instance;
}

bool CollectiveManager::Initialize(const std::string &backend, const std::string &global_group_name) {
  if (inited_) {
    return true;
  }
  MS_LOG(INFO) << "Start initializing collective communication for backend: " << backend << "...";

  // Step 1: Initialize host side collective communication.
  if (!InitHostCommlib()) {
    MS_LOG(ERROR) << "Failed to initialize host communication library.";
    return false;
  }

  MS_EXCEPTION_IF_NULL(host_comm_lib_);
  // Step 2: Create global communication group on host side.
  if (!CreateHostGlobalCommGroup(global_group_name)) {
    MS_LOG(ERROR) << "Failed to initialize host communication library.";
    return false;
  }
  // Step 3: Assign local rank id(device id) for this process.

  MS_LOG(INFO) << "End initializing collective communication for backend: " << backend << ".";
  return true;
}

bool CollectiveManager::CreateCommunicationGroup(const std::string &group_name,
                                                 const std::vector<uint32_t> &group_ranks) {
  MS_EXCEPTION_IF_NULL(host_comm_lib_);
  MS_EXCEPTION_IF_NULL(device_comm_lib_);
  // Step 1: Create communication group on host side.
  // Step 2: Generate device information of the root node.
  // Step 3: Broadcast the device root information to all nodes.
  // Step 4: Create communication group on device side.
  return true;
}

bool CollectiveManager::DestroyCommunicationGroup(const std::string &group_name) { return true; }

uint32_t CollectiveManager::GetRankId(const std::string &group_name) { return 0; }

uint32_t CollectiveManager::GetGroupSize(const std::string &group_name) { return 0; }

bool CollectiveManager::Finalize() {
  if (finalized_) {
    return true;
  }
  return true;
}

bool CollectiveManager::InitHostCommlib() {
  device::DeviceContextKey host_key = {"CPU", 0};
  host_ctx_ = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
  MS_EXCEPTION_IF_NULL(host_ctx_);
  if (!host_ctx_->LoadCollectiveCommLib()) {
    MS_LOG(ERROR) << "Failed to load communication library on the host side.";
    return false;
  }
  return true;
}

bool CollectiveManager::CreateHostGlobalCommGroup(const std::string &global_group_name) {
  MS_EXCEPTION_IF_NULL(host_comm_lib_);
  if (global_group_ranks_.empty()) {
    MS_LOG(ERROR) << "The global group rank list is empty.";
    return false;
  }
  return true;
}

bool CollectiveManager::InitDeviceCommLib(const std::string &backend, uint32_t device_id) {
  std::string device_name;
  if (backend == "nccl") {
    device_name = "GPU";
  } else if (backend == "hccl") {
    device_name = "Ascend";
  } else {
    MS_LOG(ERROR) << "Backend " << backend << " is not supported.";
    return false;
  }

  device::DeviceContextKey device_key = {device_name, device_id};
  device_ctx_ = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(device_key);
  MS_EXCEPTION_IF_NULL(device_ctx_);
  if (!device_ctx_->LoadCollectiveCommLib()) {
    MS_LOG(ERROR) << "Failed to load communication library on the device side.";
    return false;
  }
  return true;
}
}  // namespace collective
}  // namespace distributed
}  // namespace mindspore
