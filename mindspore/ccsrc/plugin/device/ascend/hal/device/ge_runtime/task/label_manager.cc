/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/hal/device/ge_runtime/task/label_manager.h"
#include <algorithm>
#include <string>
#include "runtime/mem.h"
#include "runtime/dev.h"
#include "runtime/rt_model.h"
#include "mindspore/core/utils/log_adapter.h"
#include "mindspore/core/utils/convert_utils_base.h"

namespace mindspore::ge::model_runner {
std::weak_ptr<LabelManager> LabelManager::instance_;
std::mutex LabelManager::instance_mutex_;

template <class T>
static std::string GetVectorString(const std::vector<T> &vec) {
  std::string ret;
  for (size_t i = 0; i < vec.size(); ++i) {
    if (i != 0) {
      ret.push_back(',');
    }
    ret += std::to_string(vec[i]);
  }
  return ret;
}

LabelGuard::~LabelGuard() {
  void *label_info = GetLabelInfo();
  if (label_info != nullptr) {
    rtError_t rt_ret = rtFree(label_info);
    if (rt_ret != RT_ERROR_NONE) {
      MS_LOG(ERROR) << "rtFree label_info failed! ret: " << rt_ret;
    }
  }
}

std::shared_ptr<LabelManager> LabelManager::GetInstance() {
  std::lock_guard<std::mutex> lock(instance_mutex_);
  auto instance = instance_.lock();
  if (instance != nullptr) {
    return instance;
  }

  instance = std::make_shared<LabelManager>();
  instance_ = instance;
  return instance;
}

std::shared_ptr<LabelGuard> LabelManager::GetLabelInfo(rtModel_t model, const std::vector<uint32_t> &label_ids,
                                                       const std::vector<void *> &all_label) {
  std::lock_guard<std::mutex> lock(model_info_mapping_mutex_);
  rtError_t rt_ret;
  auto model_iter = model_info_mapping_.find(model);
  if (model_iter == model_info_mapping_.end()) {
    model_info_mapping_.emplace(model, std::map<std::string, std::weak_ptr<LabelGuard>>());
    model_iter = model_info_mapping_.find(model);
  }

  std::string label_id_str = GetVectorString(label_ids);
  auto &label_map = model_iter->second;
  auto label_iter = label_map.find(label_id_str);
  if (label_iter != label_map.end()) {
    auto label_guard = label_iter->second.lock();
    if (label_guard != nullptr) {
      MS_LOG(INFO) << "model " << model << " find same label id " << label_id_str;
      return label_guard;
    }
  }

  MS_LOG(INFO) << "Alloc label id " << label_id_str << " for model " << model;
  void *label_info = nullptr;
  std::vector<void *> label_list;
  bool status = true;
  std::transform(label_ids.begin(), label_ids.end(), std::back_inserter(label_list),
                 [&all_label, &status](uint32_t idx) -> void * {
                   if (idx >= all_label.size()) {
                     MS_LOG(ERROR) << "Invalid label id " << idx << " all label list size " << all_label.size();
                     status = false;
                     return nullptr;
                   }
                   return all_label[idx];
                 });
  if (!status) {
    MS_LOG(ERROR) << "Get label info failed.";
    return nullptr;
  }
  uint32_t label_info_size = SizeToUint(sizeof(rtLabelDevInfo) * label_list.size());
  int64_t value = 0;
  rt_ret = rtGetRtCapability(FEATURE_TYPE_MEMORY, MEMORY_INFO_TS_LIMITED, &value);
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "Call rt api rtGetRtCapability failed, ret: " << rt_ret;
    return nullptr;
  }

  rt_ret = rtMalloc(&label_info, label_info_size, (value == RT_CAPABILITY_SUPPORT) ? RT_MEMORY_TS : RT_MEMORY_HBM, 0);
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "Call rt api rtMalloc failed, ret: " << rt_ret;
    return nullptr;
  }

  rt_ret = rtLabelListCpy(label_list.data(), SizeToUint(label_list.size()), label_info, label_info_size);
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "Call rt api rtLabelListCpy failed, ret: " << rt_ret;
    (void)rtFree(label_info);
    return nullptr;
  }

  auto label_guard = std::make_shared<LabelGuard>(label_info);
  label_map.emplace(label_id_str, label_guard);
  return label_guard;
}
}  // namespace mindspore::ge::model_runner
