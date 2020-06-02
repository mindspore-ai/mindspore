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
#include "dataset/util/profiling.h"

#include <sys/time.h>
#include <cstdlib>
#include <fstream>
#include "dataset/util/path.h"
#include "common/utils.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace dataset {
Profiling::Profiling(const std::string &file_name, const int32_t device_id)
    : file_name_(file_name), device_id_(device_id) {}

Status Profiling::Init() {
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
  file_path_ = (Path(real_path) / Path(file_name_ + "_" + std::to_string(device_id_) + ".txt")).toString();
  return Status::OK();
}

Status Profiling::Record(const std::string &data) {
  value_.emplace_back(data);
  return Status::OK();
}

Status Profiling::SaveToFile() {
  if (file_name_.empty()) {
    RETURN_STATUS_UNEXPECTED("Profiling file name has not been set.");
  }
  std::ofstream handle(file_path_, std::ios::app);
  if (!handle.is_open()) {
    RETURN_STATUS_UNEXPECTED("Profiling file can not be opened.");
  }
  for (auto value : value_) {
    handle << value << "\n";
  }
  handle.close();

  return Status::OK();
}

ProfilingManager &ProfilingManager::GetInstance() {
  static ProfilingManager instance;
  return instance;
}

bool ProfilingManager::IsProfilingEnable() const {
  auto profiling = common::GetEnv("PROFILING_MODE");
  if (profiling.empty() || profiling != "true") {
    return false;
  }

  return true;
}

Status ProfilingManager::RegisterProfilingNode(std::shared_ptr<Profiling> *node) {
  RETURN_IF_NOT_OK((*node)->Init());
  profiling_node_.emplace_back(*node);
  return Status::OK();
}

Status ProfilingManager::SaveProfilingData() {
  if (!IsProfilingEnable()) {
    return Status::OK();
  }
  MS_LOG(INFO) << "Start to save profile data.";
  for (auto node : profiling_node_) {
    RETURN_IF_NOT_OK(node->SaveToFile());
  }
  MS_LOG(INFO) << "Save profile data end.";

  return Status::OK();
}

double ProfilingTime::GetCurMilliSecond() {
  struct timeval tv = {0, 0};
  (void)gettimeofday(&tv, nullptr);
  return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}
}  // namespace dataset
}  // namespace mindspore
