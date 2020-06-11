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
#ifndef DATASET_UTIL_PROFILE_H_
#define DATASET_UTIL_PROFILE_H_

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include "dataset/util/status.h"

namespace mindspore {
namespace dataset {

class Monitor;
class ExecutionTree;

const char kDeviceQueueTracingName[] = "Device Queue Tracing";
const char kDatasetIteratorTracingName[] = "Dataset Iterator Tracing";
const char kConnectorSizeSamplingName[] = "Connector Size Sampling";

// Profiling is a class of basic unit of profiling action
// This base class encapsulate the serialization output logic
class Profiling : std::enable_shared_from_this<Profiling> {
 public:
  // Constructor
  Profiling() = default;

  // Destructor
  virtual ~Profiling() = default;

  virtual Status Init(const std::string &dir_path, const std::string &device_id) = 0;

  // Default serialization file generator
  virtual Status SaveToFile() = 0;

  // Profiling name
  virtual std::string Name() const = 0;

 protected:
  std::string file_path_;
};

// Sampling is a class of profiling which generate samples periodically.
class Sampling : public Profiling {
 public:
  // Sampling action function. This function will be invoked by performance monitor thread.
  virtual Status Sample() = 0;
};

// Tracing is class of profiling which record samples upon request.
class Tracing : public Profiling {
  // Tracing does not define a fixed interface to provide flexible on data recording.
};

// ProfilingManager is a class manages all profiling infrastructure
// It serves the following purposes:
// 1) Fetch profiling configs from global contexts
// 2) Setup all profiling node based on config
// 3) Provide access of profiling nodes for profiling actions
// 4) Manage profiling data serialization process
class ProfilingManager {
 public:
  explicit ProfilingManager(ExecutionTree *tree) : tree_(tree) {}

  ~ProfilingManager() = default;

  Status Initialize();

  // Save profile data to file
  // @return Status - The error code return
  Status SaveProfilingData();

  // Sampling node getter
  // @param name - The name of the requested node
  // @param node - Pointer to the shared pointer for the Sampling node
  // @return Status - The error code return
  Status GetSamplingNode(const std::string &name, std::shared_ptr<Sampling> *node);

  // Tracing node getter
  // @param name - The name of the requested node
  // @param node - Pointer to the shared pointer for the Tracing node
  // @return Status - The error code return
  Status GetTracingNode(const std::string &name, std::shared_ptr<Tracing> *node);

  // If profiling is enabled.
  bool IsProfilingEnable() const;

  const std::unordered_map<std::string, std::shared_ptr<Sampling>> &GetSamplingNodes() { return sampling_nodes_; }

 private:
  std::unordered_map<std::string, std::shared_ptr<Tracing>> tracing_nodes_;

  std::unordered_map<std::string, std::shared_ptr<Sampling>> sampling_nodes_;

  // Register profile node to tree
  // @param node - Profiling node
  // @return Status - The error code return
  Status RegisterTracingNode(std::shared_ptr<Tracing> node);

  // Register profile node to tree
  // @param node - Profiling node
  // @return Status - The error code return
  Status RegisterSamplingNode(std::shared_ptr<Sampling> node);

  ExecutionTree *tree_ = nullptr;  // ExecutionTree pointer
  std::string dir_path_;           // where to create profiling file
  std::string device_id_;          // used when create profiling file,filename_deviceid.suffix
};

enum ProfilingType { TIME, CONNECTOR_DEPTH };

enum ProfilingTimeSubType {
  PIPELINE_TIME,
  TDT_PUSH_TIME,
  BATCH_TIME,
  INVALID_TIME,
};

class ProfilingTime {
 public:
  static double GetCurMilliSecond();
};

}  // namespace dataset
}  // namespace mindspore
#endif
