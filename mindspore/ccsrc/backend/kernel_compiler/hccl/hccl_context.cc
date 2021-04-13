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

#include "backend/kernel_compiler/hccl/hccl_context.h"
#include "utils/log_adapter.h"
#include "hccl/hccl.h"

constexpr auto kHcclConfigFile = "MINDSPORE_HCCL_CONFIG_PATH";
constexpr auto kHcclConfigFileOld = "RANK_TABLE_FILE";

namespace mindspore {
namespace kernel {
int GetRankId() {
  auto rank_id_env = std::getenv("RANK_ID");
  if (rank_id_env == nullptr) {
    MS_LOG(EXCEPTION) << "No RANK_ID, please export RANK_ID";
  }
  try {
    return std::stoi(rank_id_env);
  } catch (std::invalid_argument &e) {
    MS_LOG(EXCEPTION) << "Invalid rankd id env:" << rank_id_env;
  }
}

bool HcclContext::InitHccl() {
  if (hccl_comm_ != nullptr) {
    return true;
  }
  auto config_file = std::getenv(kHcclConfigFile);
  if (config_file == nullptr) {
    config_file = std::getenv(kHcclConfigFileOld);
    if (config_file == nullptr) {
      MS_LOG(ERROR) << "Get hccl rank table file failed. Please export MINDSPORE_HCCL_CONFIG_PATH or RANK_TABLE_FILE";
      return false;
    }
  }

  rank_id_ = GetRankId();
  if (rank_id_ < 0 || rank_id_ > 7) {
    MS_LOG(ERROR) << "rank_id needs to be between 0-7";
    return false;
  }

  auto hccl_result = HcclCommInitClusterInfo(config_file, rank_id_, &hccl_comm_);
  if (hccl_result != HCCL_SUCCESS) {
    MS_LOG(ERROR) << "HcclCommInitClusterInfo failed, ret:" << hccl_result;
    return false;
  }
  MS_LOG(INFO) << "HcclCommInitClusterInfo success";
  return true;
}

bool HcclContext::Finalize() {
  if (hccl_comm_ == nullptr) {
    return true;
  }
  auto hccl_result = HcclCommDestroy(hccl_comm_);
  if (hccl_result != HCCL_SUCCESS) {
    MS_LOG(ERROR) << "HcclComm destroy failed, ret:" << hccl_result;
    return false;
  }
  MS_LOG(INFO) << "HcclComm destroy success";
  hccl_comm_ = nullptr;
  return true;
}
}  // namespace kernel
}  // namespace mindspore
