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

#include "tools/silent_detect/checksum/checksum_mgr.h"
#include <string>
#include "tools/silent_detect/silent_detect_config_parser.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace checksum {
constexpr auto kMsSdcDetectEnable = "MS_SDC_DETECT_ENABLE";

int GetMsSdcDetectEnable() {
  std::string var = common::GetEnv(kMsSdcDetectEnable);
  if (var.empty()) {
    MS_VLOG(VL_ASCEND_SILENT_CHECK) << "Environment variable " << kMsSdcDetectEnable
                                    << " is not set, using default value 0.";
    return 0;
  }
  if (var.size() != 1 || var[0] < '0' || var[0] > '1') {
    MS_LOG(WARNING) << "Environment variable " << kMsSdcDetectEnable << " should be 0 or 1, but got '" << var
                    << "', using default value 0.";
    return 0;
  }
  MS_VLOG(VL_ASCEND_SILENT_CHECK) << "Environment variable " << kMsSdcDetectEnable << " is " << var;
  int ms_sdc_detect_enable = var[0] - '0';
  return ms_sdc_detect_enable;
}

bool CheckSumMgr::NeedEnableCheckSum() const {
  static int ms_sdc_detect_enable = GetMsSdcDetectEnable();
  if (ms_sdc_detect_enable == 1) {
    return true;
  }
  static bool silent_detect_with_checksum = []() {
    bool need_enable = silentdetect::SilentDetectConfigParser::GetInstance().IsEnable() &&
                       silentdetect::SilentDetectConfigParser::GetInstance().IsWithChecksum();
    if (need_enable) {
      MS_VLOG(VL_ASCEND_SILENT_CHECK) << "MS_NPU_ASD_CONFIG-with_checksum is true, need enable CheckSum.";
    }
    return need_enable;
  }();
  return silent_detect_with_checksum;
}

bool CheckSumMgr::IsCheckSumEnable() const {
  std::shared_lock<std::shared_mutex> lock(enable_mutex_);
  return enable_;
}

void CheckSumMgr::CheckSumStart() {
  if (!NeedEnableCheckSum()) {
    MS_LOG(WARNING) << "CheckSum is disable. Set environment variable " << kMsSdcDetectEnable << "=1 to enable it.";
    return;
  }
  MS_LOG(WARNING) << "CheckSum start, which will degrade performance";
  std::unique_lock<std::shared_mutex> lock(enable_mutex_);
  if (!enable_) {
    enable_ = true;
    std::unique_lock<std::shared_mutex> result_lock(result_mutex_);
    result_ = false;
    MS_VLOG(VL_ASCEND_SILENT_CHECK) << "CheckSum result is reset to false";
    return;
  }
  MS_VLOG(VL_ASCEND_SILENT_CHECK) << "CheckSum is already running";
}

void CheckSumMgr::CheckSumStop() {
  std::unique_lock<std::shared_mutex> lock(enable_mutex_);
  if (enable_) {
    MS_LOG(WARNING) << "CheckSum stop";
    enable_ = false;
    return;
  }
  MS_VLOG(VL_ASCEND_SILENT_CHECK) << "CheckSum is not running";
}

bool CheckSumMgr::GetCheckSumResult() const {
  std::shared_lock<std::shared_mutex> lock(result_mutex_);
  MS_VLOG(VL_ASCEND_SILENT_CHECK) << "CheckSum result is " << result_;
  return result_;
}

void CheckSumMgr::SetCheckSumResult(bool result) {
  std::unique_lock<std::shared_mutex> lock(result_mutex_);
  result_ = result;
}

bool NeedEnableCheckSum() {
  auto &checkSumMgr = CheckSumMgr::GetInstance();
  return checkSumMgr.NeedEnableCheckSum();
}

}  // namespace checksum
}  // namespace mindspore
