/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "common/string_util.h"
#include "common/op_attr.h"
#include "manager/custom_config_manager.h"
#include "common/log_util.h"
#include "include/errorcode.h"
namespace mindspore {
namespace lite {
int CustomConfigManager::Init(const std::map<std::string, std::string> &dpico_config) {
  if (inited_) {
    MS_LOG(INFO) << "device only needs to init once.";
    return RET_OK;
  }
  if (UpdateConfig(dpico_config) != RET_OK) {
    MS_LOG(ERROR) << "init custom config failed.";
    return RET_ERROR;
  }
  return RET_OK;
}
int CustomConfigManager::UpdateConfig(const std::map<std::string, std::string> &dpico_config) {
  // parse float params
  std::map<std::string, float *> float_params = {{kNmsThreshold, &nms_threshold_},
                                                 {kScoreThreshold, &score_threshold_},
                                                 {kMinHeight, &min_height_},
                                                 {kMinWidth, &min_width_}};
  for (const auto &param : float_params) {
    if (dpico_config.find(param.first) != dpico_config.end()) {
      if (IsValidDoubleNum(dpico_config.at(param.first))) {
        *param.second = std::stof(dpico_config.at(param.first));
      } else {
        MS_LOG(WARNING) << param.first
                        << " param in config is invalid, will use default or last value:" << *param.second;
      }
    } else {
      MS_LOG(INFO) << param.first << " param isn't configured, will use default or last value:" << *param.second;
    }
  }

  // parse size_t params
  std::map<std::string, size_t *> unsigned_params = {{kMaxRoiNum, &max_roi_num_}, {kGTotalT, &g_total_t_}};
  for (const auto &param : unsigned_params) {
    if (dpico_config.find(param.first) != dpico_config.end()) {
      if (IsValidUnsignedNum(dpico_config.at(param.first))) {
        *param.second = std::stoul(dpico_config.at(param.first));
      } else {
        MS_LOG(WARNING) << param.first
                        << " param in config is invalid, will use default or last value:" << *param.second;
      }
    } else {
      MS_LOG(INFO) << param.first << " param isn't configured, will use default or last value:" << *param.second;
    }
  }

  // parse bool params
  if (dpico_config.find(kDetectionPostProcess) != dpico_config.end()) {
    if (dpico_config.at(kDetectionPostProcess) == "on") {
      detect_post_process_ = true;
    } else if (dpico_config.at(kDetectionPostProcess) == "off") {
      detect_post_process_ = false;
    } else {
      MS_LOG(WARNING) << kDetectionPostProcess
                      << " param in config is invalid, will use default or last value: " << detect_post_process_;
    }
  } else {
    MS_LOG(INFO) << kDetectionPostProcess
                 << " param isn't configured, will use default or last value: " << detect_post_process_;
  }

  // parse string params
  if (dpico_config.find(kAclConfigPath) != dpico_config.end()) {
    acl_config_file_ = dpico_config.at(kAclConfigPath);
  }
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
