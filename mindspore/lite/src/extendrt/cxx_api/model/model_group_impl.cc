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

#include "src/extendrt/cxx_api/model/model_group_impl.h"
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "include/api/types.h"
#include "include/api/context.h"
#include "src/common/log_adapter.h"
#include "src/extendrt/model_manager.h"

namespace mindspore {
Status ModelGroupImpl::AddModel(const std::vector<std::string> &model_path_list) {
  if (model_path_list.empty()) {
    MS_LOG(ERROR) << "Param model_path_list is empty.";
    return kLiteParamInvalid;
  }
  for (auto &model_path : model_path_list) {
    if (model_path.empty()) {
      continue;
    }
    (void)model_path_list_.emplace_back(model_path);
  }

  return kSuccess;
}

Status ModelGroupImpl::AddModel(const std::vector<std::pair<const void *, size_t>> &model_buff_list) {
  if (model_buff_list.empty()) {
    MS_LOG(ERROR) << "Param model_buff_list is empty.";
    return kLiteParamInvalid;
  }
  for (auto &model_buff : model_buff_list) {
    if (model_buff.first == nullptr || model_buff.second == 0) {
      continue;
    }
    (void)model_buff_list_.emplace_back(model_buff);
  }

  return kSuccess;
}

Status ModelGroupImpl::CalMaxSizeOfWorkspace(ModelType model_type, const std::shared_ptr<Context> &ms_context) {
  for (auto &model_path : model_path_list_) {
    Model model;
    std::string sharing_workspace_section = "inner_common";
    std::string calc_workspace_key = "inner_calc_workspace_size";
    std::string calc_workspace_value = "true";
    model.UpdateConfig(sharing_workspace_section, std::make_pair(calc_workspace_key, calc_workspace_value));
    auto ret = model.Build(model_path, model_type, ms_context);
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "model build failed.";
      ModelManager::GetInstance().ClearModel();
      return kLiteError;
    }
    ModelManager::GetInstance().AddModel(model_path);
  }

  for (auto &model_buff : model_buff_list_) {
    Model model;
    std::string sharing_workspace_section = "inner_common";
    std::string calc_workspace_key = "inner_calc_workspace_size";
    std::string calc_workspace_value = "true";
    model.UpdateConfig(sharing_workspace_section, std::make_pair(calc_workspace_key, calc_workspace_value));
    auto ret = model.Build(model_buff.first, model_buff.second, model_type, ms_context);
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "model build failed.";
      ModelManager::GetInstance().ClearModel();
      return kLiteError;
    }
    ModelManager::GetInstance().AddModel(model_buff);
  }
  return kSuccess;
}
}  // namespace mindspore
