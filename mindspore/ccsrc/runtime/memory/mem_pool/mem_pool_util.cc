/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "include/runtime/memory/mem_pool/mem_pool_util.h"

#include "include/utils/common.h"
#include "include/utils/utils.h"
#include "utils/file_utils.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace memory {
namespace mem_pool {
const std::map<MemType, std::string> kMemTypeStr = {{MemType::kWeight, "Weight"},
                                                    {MemType::kConstantValue, "ConstantValue"},
                                                    {MemType::kKernel, "Kernel"},
                                                    {MemType::kGraphOutput, "GraphOutput"},
                                                    {MemType::kSomas, "Somas"},
                                                    {MemType::kSomasOutput, "SomasOutput"},
                                                    {MemType::kGeConst, "GeConst"},
                                                    {MemType::kGeFixed, "GeFixed"},
                                                    {MemType::kBatchMemory, "BatchMemory"},
                                                    {MemType::kContinuousMemory, "ContinuousMemory"},
                                                    {MemType::kPyNativeInput, "PyNativeInput"},
                                                    {MemType::kPyNativeOutput, "PyNativeOutput"},
                                                    {MemType::kWorkSpace, "WorkSpace"},
                                                    {MemType::kOther, "Other"}};

std::string MemTypeToStr(MemType mem_type) { return kMemTypeStr.at(mem_type); }

bool IsEnableMemTrack() {
  return IsEnableAllocConfig(kAllocMemoryTracker) || !GetAllocConfigValue(kAllocMemoryTrackerPath).empty();
}

bool IsNeedProfilieMemoryLog() {
  static bool is_need_profile_memory_log = IsDisableGeKernel() && common::IsCompileSimulation();
  return is_need_profile_memory_log;
}

bool IsMemoryPoolRecycle() {
  static bool disable_optimize_mem = IsDisableAllocConfig(kAllocMemoryRecycle);
  static bool disable_ge_kernel = IsDisableGeKernel();
  if (!disable_ge_kernel || disable_optimize_mem) {
    return false;
  }
  if (!IsJit()) {
    return false;
  }
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  auto is_ge = context_ptr->GetBackend() == kBackendGE;
  auto task_sink = context_ptr->get_param<bool>(MS_CTX_ENABLE_TASK_SINK);
  return is_ge && task_sink;
}

std::string GeneratePath(size_t rank_id, const std::string &file_name, const std::string &suffix) {
  std::string path = GetAllocConfigValue(kAllocMemoryTrackerPath);
  if (path.empty()) {
    path = "./";
  }
  if (path.back() != '/') {
    path += "/";
  }

  if (rank_id != SIZE_MAX) {
    path += "rank_" + std::to_string(rank_id) + "/";

    auto path_opt = Common::CreatePrefixPath(path);
    if (!path_opt.has_value()) {
      MS_LOG(ERROR) << "Create path : " << path << " failed.";
    }
  }

  if (!file_name.empty()) {
    path += file_name;
  }
  if (!suffix.empty()) {
    path += "." + suffix;
  }

  auto file_path_opt = Common::CreatePrefixPath(path);
  if (!file_path_opt.has_value()) {
    MS_LOG(WARNING) << "Generate path for rank id : " << rank_id << ", file_name : " << file_name
                    << ", suffix : " << suffix << "failed.";
    return "";
  }
  ChangeFileMode(path, S_IWUSR | S_IRUSR);
  return path;
}

LockGuard::LockGuard(const Lock &lock) {
  lock_ = const_cast<Lock *>(&lock);
  lock_->lock();
}

LockGuard::~LockGuard() { lock_->unlock(); }
}  // namespace mem_pool
}  // namespace memory
}  // namespace mindspore
