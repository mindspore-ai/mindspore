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

#ifndef DPICO_MANAGER_CUSTOM_CONFIG_MANAGER_H_
#define DPICO_MANAGER_CUSTOM_CONFIG_MANAGER_H_

#include <map>
#include <string>
#include <memory>
namespace mindspore {
namespace lite {
class CustomConfigManager {
 public:
  CustomConfigManager() = default;
  ~CustomConfigManager() = default;

  int Init(const std::map<std::string, std::string> &dpico_config);
  int UpdateConfig(const std::map<std::string, std::string> &dpico_config);
  size_t MaxRoiNum() const { return max_roi_num_; }
  float NmsThreshold() const { return nms_threshold_; }
  float ScoreThreshold() const { return score_threshold_; }
  float MinHeight() const { return min_height_; }
  float MinWidth() const { return min_width_; }
  int GTotalT() const { return g_total_t_; }
  void SetGTotalT(size_t g_total_t) { g_total_t_ = g_total_t; }
  bool NeedDetectPostProcess() const { return detect_post_process_; }
  const std::string &AclConfigFile() const { return acl_config_file_; }

 private:
  float nms_threshold_{0.9f};
  float score_threshold_{0.08f};
  float min_height_{1.0f};
  float min_width_{1.0f};
  size_t g_total_t_{0};
  size_t max_roi_num_{300};
  bool detect_post_process_{false};
  std::string acl_config_file_;
  bool inited_{false};
};
using CustomConfigManagerPtr = std::shared_ptr<CustomConfigManager>;
}  // namespace lite
}  // namespace mindspore

#endif  // DPICO_MANAGER_CUSTOM_CONFIG_MANAGER_H_
