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

#include <string>
#include <fstream>
#include <climits>
#include <memory>
#include <cstring>
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
std::string CustomConfigManager::RealPath(const char *path) {
  if (path == nullptr) {
    MS_LOG(ERROR) << "path is nullptr";
    return "";
  }
  if ((std::strlen(path)) >= PATH_MAX) {
    MS_LOG(ERROR) << "path is too long";
    return "";
  }
  auto resolved_path = std::make_unique<char[]>(PATH_MAX);
  if (resolved_path == nullptr) {
    MS_LOG(ERROR) << "new resolved_path failed";
    return "";
  }
  char *real_path = realpath(path, resolved_path.get());
  if (real_path == nullptr || strlen(real_path) == 0) {
    MS_LOG(ERROR) << "file path is not valid : " << path;
    return "";
  }
  std::string res = resolved_path.get();
  return res;
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
    if (AccessFile(acl_config_file_, F_OK) != 0) {
      MS_LOG(ERROR) << " AclConfigPath not exist, please check.";
      return RET_ERROR;
    }
    auto acl_config_file = RealPath(acl_config_file_.c_str());
    if (acl_config_file.empty()) {
      MS_LOG(ERROR) << "Get realpath failed, AclConfigPath is " << acl_config_file;
      return RET_ERROR;
    }
  }
  return RET_OK;
}
bool CustomConfigManager::IsEnableMultiModelSharingMemPrepare(
  const std::map<std::string, std::string> &model_share_config) {
  return model_share_config.find(kModelSharingPrepareKey) != model_share_config.end();
}
bool CustomConfigManager::IsEnableMultiModelSharingMem(const std::map<std::string, std::string> &model_share_config) {
  return model_share_config.find(kModelSharingKey) != model_share_config.end();
}
}  // namespace lite
}  // namespace mindspore
