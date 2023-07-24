/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "extendrt/kernel/ascend/profiling/profiling.h"
#include <iostream>
#include <memory>
#include <map>
#include <nlohmann/json.hpp>
#include "src/common/file_utils.h"

namespace mindspore::kernel {
namespace acl {
namespace {
std::map<std::string, aclprofAicoreMetrics> kAicMetrics{{"ArithmeticUtilization", ACL_AICORE_ARITHMETIC_UTILIZATION},
                                                        {"PipeUtilization", ACL_AICORE_PIPE_UTILIZATION},
                                                        {"Memory", ACL_AICORE_MEMORY_BANDWIDTH},
                                                        {"MemoryL0", ACL_AICORE_L0B_AND_WIDTH},
                                                        {"ResourceConflictRatio", ACL_AICORE_RESOURCE_CONFLICT_RATIO},
                                                        {"MemoryUB", ACL_AICORE_MEMORY_UB},
                                                        {"None", ACL_AICORE_NONE}};
std::string RealPath(const char *path) {
  if (path == nullptr) {
    MS_LOG(ERROR) << "path is nullptr";
    return "";
  }
  if ((strlen(path)) >= PATH_MAX) {
    MS_LOG(ERROR) << "path is too long";
    return "";
  }
  auto resolved_path = std::make_unique<char[]>(PATH_MAX);
  if (resolved_path == nullptr) {
    MS_LOG(ERROR) << "new resolved_path failed";
    return "";
  }
  auto real_path = realpath(path, resolved_path.get());
  if (real_path == nullptr || strlen(real_path) == 0) {
    MS_LOG(ERROR) << "file path not exists: " << path;
    return "";
  }
  std::string res = resolved_path.get();
  return res;
}
}  // namespace

bool Profiling::Init(const std::string &profiling_file, uint32_t device_id) {
  device_id_ = device_id;
  auto real_path = RealPath(profiling_file.c_str());
  std::ifstream ifs(real_path);
  if (!ifs.good()) {
    MS_LOG(ERROR) << "file: " << real_path << " is not exist";
    return false;
  }
  if (!ifs.is_open()) {
    MS_LOG(ERROR) << "file: " << real_path << " open failed";
    return false;
  }
  try {
    profiling_json_ = nlohmann::json::parse(ifs);
  } catch (const nlohmann::json::parse_error &error) {
    MS_LOG(ERROR) << "parse json file failed, please check your file.";
    return false;
  }
  if (profiling_json_["profiler"] == nullptr) {
    MS_LOG(ERROR) << "profiler is required in profiling json file.";
    return false;
  }
  auto profiling_info = profiling_json_["profiler"];
  if (profiling_info["switch"] == "on") {
    is_profiling_open_ = true;
    profiling_mask_ = ACL_PROF_ACL_API | ACL_PROF_TASK_TIME | ACL_PROF_AICORE_METRICS | ACL_PROF_RUNTIME_API;
  }
  if (profiling_info["output"] != nullptr) {
    output_path_ = profiling_info["output"].get<std::string>();
  }
  if (profiling_info["aic_metrics"] != nullptr) {
    auto aic_metrics_key = profiling_info["aic_metrics"].get<std::string>();
    if (kAicMetrics.find(aic_metrics_key) != kAicMetrics.end()) {
      aic_metrics_ = kAicMetrics.at(aic_metrics_key);
    } else {
      MS_LOG(WARNING) << "The value of aic_metrics is invalid";
    }
  }
  if (profiling_info["training_trace"] == "on") {
    profiling_mask_ |= ACL_PROF_TRAINING_TRACE;
  }
  if (profiling_info["aicpu"] == "on") {
    profiling_mask_ |= ACL_PROF_AICPU;
  }
  if (profiling_info["hccl"] == "on") {
    profiling_mask_ |= ACL_PROF_HCCL_TRACE;
  }
  if (profiling_info["l2"] == "on") {
    profiling_mask_ |= ACL_PROF_L2CACHE;
  }
  if (profiling_info["msprofix"] == "on") {
    profiling_mask_ |= ACL_PROF_MSPROFTX;
  }
  return true;
}

bool Profiling::StartProfiling(const aclrtStream &stream) {
  MS_LOG(INFO) << "Start to profile";
  aclError ret = aclprofInit(output_path_.c_str(), output_path_.length());
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "aclprofInit failed, ret = " << ret;
    return false;
  }
  uint32_t device_list[1] = {device_id_};
  uint32_t device_num = 1;
  acl_config_ = aclprofCreateConfig(device_list, device_num, aic_metrics_, nullptr, profiling_mask_);
  ret = aclprofStart(acl_config_);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "aclprofStart start failed, ret = " << ret;
    return false;
  }
  return true;
}

bool Profiling::StopProfiling(const aclrtStream &stream) {
  MS_LOG(INFO) << "End to profile";
  aclError ret = aclprofStop(acl_config_);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "aclprofDestroyConfig failed, ret = " << ret;
    return false;
  }
  ret = aclprofDestroyConfig(acl_config_);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "aclprofDestroyConfig failed, ret = " << ret;
    return false;
  }
  ret = aclprofFinalize();
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "aclProfFinalize failed, ret = " << ret;
    return false;
  }
  return true;
}
}  // namespace acl
}  // namespace mindspore::kernel
