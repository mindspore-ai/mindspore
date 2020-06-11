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
#include "dataset/engine/perf/profiling.h"

#include <sys/time.h>
#include <cstdlib>
#include <fstream>
#include "common/utils.h"
#include "dataset/util/path.h"
#include "dataset/engine/perf/monitor.h"
#include "dataset/engine/perf/device_queue_tracing.h"
#include "dataset/engine/perf/connector_size.h"
#include "dataset/engine/perf/dataset_iterator_tracing.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace dataset {

bool ProfilingManager::IsProfilingEnable() const {
  auto profiling = common::GetEnv("PROFILING_MODE");
  if (profiling.empty() || profiling != "true") {
    return false;
  }
  return true;
}

Status ProfilingManager::Initialize() {
  // Register nodes based on config
  std::string dir = common::GetEnv("MINDDATA_PROFILING_DIR");
  if (dir.empty()) {
    RETURN_STATUS_UNEXPECTED("Profiling dir is not set.");
  }
  char real_path[PATH_MAX] = {0};
  if (dir.size() >= PATH_MAX) {
    RETURN_STATUS_UNEXPECTED("Profiling dir is invalid.");
  }
#if defined(_WIN32) || defined(_WIN64)
  if (_fullpath(real_path, common::SafeCStr(dir), PATH_MAX) == nullptr) {
    RETURN_STATUS_UNEXPECTED("Profiling dir is invalid.");
  }
#else
  if (realpath(common::SafeCStr(dir), real_path) == nullptr) {
    RETURN_STATUS_UNEXPECTED("Profiling dir is invalid.");
  }
#endif
  dir_path_ = real_path;

  // If DEVICE_ID is not set,defult value is 0
  device_id_ = common::GetEnv("DEVICE_ID");
  if (device_id_.empty()) {
    device_id_ = "0";
  }

  // Register all profiling node.
  // device_queue node is used for graph mode
  std::shared_ptr<Tracing> device_queue_tracing = std::make_shared<DeviceQueueTracing>();
  RETURN_IF_NOT_OK(RegisterTracingNode(device_queue_tracing));
  // dataset_iterator node is used for graph mode
  std::shared_ptr<Tracing> dataset_iterator_tracing = std::make_shared<DatasetIteratorTracing>();
  RETURN_IF_NOT_OK(RegisterTracingNode(dataset_iterator_tracing));

  std::shared_ptr<Sampling> monitor_sampling = std::make_shared<ConnectorSize>(tree_);
  RETURN_IF_NOT_OK(RegisterSamplingNode(monitor_sampling));

  return Status::OK();
}

// Profiling node registration
Status ProfilingManager::RegisterTracingNode(std::shared_ptr<Tracing> node) {
  // Check if node with the same name has already been registered.
  auto exist = tracing_nodes_.find(node->Name());
  if (exist != tracing_nodes_.end()) {
    return Status(StatusCode::kProfilingError, "Profiling node already exist: " + node->Name());
  }
  // Register the node with its name as key.
  RETURN_IF_NOT_OK(node->Init(dir_path_, device_id_));
  tracing_nodes_[node->Name()] = node;
  return Status::OK();
}

// Profiling node getter
Status ProfilingManager::GetTracingNode(const std::string &name, std::shared_ptr<Tracing> *node) {
  // Check if node with the same name has already been registered.
  auto exist = tracing_nodes_.find(name);
  if (exist == tracing_nodes_.end()) {
    return Status(StatusCode::kProfilingError, "Profiling node does not exist: " + name);
  }
  // Fetch node.
  *node = tracing_nodes_[name];
  return Status::OK();
}

// Profiling node registration
Status ProfilingManager::RegisterSamplingNode(std::shared_ptr<Sampling> node) {
  // Check if node with the same name has already been registered.
  auto exist = sampling_nodes_.find(node->Name());
  if (exist != sampling_nodes_.end()) {
    return Status(StatusCode::kProfilingError, "Profiling node already exist: " + node->Name());
  }
  // Register the node with its name as key.
  RETURN_IF_NOT_OK(node->Init(dir_path_, device_id_));
  sampling_nodes_[node->Name()] = node;
  return Status::OK();
}

// Profiling node getter
Status ProfilingManager::GetSamplingNode(const std::string &name, std::shared_ptr<Sampling> *node) {
  // Check if node with the same name has already been registered.
  auto exist = sampling_nodes_.find(name);
  if (exist == sampling_nodes_.end()) {
    return Status(StatusCode::kProfilingError, "Profiling node does not exist: " + name);
  }
  // Fetch node.
  *node = sampling_nodes_[name];
  return Status::OK();
}

Status ProfilingManager::SaveProfilingData() {
  if (!IsProfilingEnable()) {
    return Status::OK();
  }
  MS_LOG(INFO) << "Start to save profiling data.";
  for (auto node : tracing_nodes_) {
    RETURN_IF_NOT_OK(node.second->SaveToFile());
  }
  for (auto node : sampling_nodes_) {
    RETURN_IF_NOT_OK(node.second->SaveToFile());
  }
  MS_LOG(INFO) << "Save profiling data end.";

  return Status::OK();
}

double ProfilingTime::GetCurMilliSecond() {
  struct timeval tv = {0, 0};
  (void)gettimeofday(&tv, nullptr);
  return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}
}  // namespace dataset
}  // namespace mindspore
