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
#include <memory>
#include "dataset/util/status.h"

namespace mindspore {
namespace dataset {
enum ProfilingType {
  TIME,
  CONNECTOR_DEPTH,
};

enum ProfilingTimeSubType {
  PIPELINE_TIME,
  TDT_PUSH_TIME,
  BATCH_TIME,
  INVALID_TIME,
};

class Profiling {
 public:
  // Constructor
  Profiling() = default;

  // Constructor if need save profile data to file
  Profiling(const std::string &file_name, const int32_t device_id);

  // Destructor
  ~Profiling() = default;

  Status Init();

  // Record profile data
  Status Record(const std::string &data);

  // Save profile data to file if necessary
  Status SaveToFile();

 private:
  std::vector<std::string> value_;
  std::string file_name_;
  std::string file_path_;
  int32_t device_id_;
};

class ProfilingManager {
 public:
  ProfilingManager() = default;
  ~ProfilingManager() = default;

  static ProfilingManager &GetInstance();

  // Save profile data to file
  // @return Status - The error code return
  Status SaveProfilingData();

  // Register profile node to tree
  // @param node - Profiling node
  // @return Status - The error code return
  Status RegisterProfilingNode(std::shared_ptr<Profiling> *node);

  bool IsProfilingEnable() const;

 private:
  std::vector<std::shared_ptr<Profiling>> profiling_node_;
};

class ProfilingTime {
 public:
  static double GetCurMilliSecond();
};
}  // namespace dataset
}  // namespace mindspore
#endif
