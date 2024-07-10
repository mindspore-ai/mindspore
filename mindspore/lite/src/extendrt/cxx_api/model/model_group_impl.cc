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
#include "extendrt/cxx_api/model/model_impl.h"
#include "src/common/common.h"

namespace mindspore {
ModelGroupImpl::ModelGroupImpl(ModelGroupFlag flags) : flags_(flags) {
  static uint32_t g_model_group_id = 0;
  model_group_id_ = ++g_model_group_id;
}

Status ModelGroupImpl::AddModel(const std::vector<std::vector<char>> &model_path_list) {
  if (model_path_list.empty()) {
    MS_LOG(ERROR) << "Param model_path_list is empty.";
    return kLiteParamInvalid;
  }
  std::unique_lock<std::mutex> local_path(mtx_path_list_);
  for (auto &model_path : model_path_list) {
    if (model_path.empty()) {
      continue;
    }
    (void)model_path_list_.emplace_back(CharToString(model_path));
  }
  return kSuccess;
}

Status ModelGroupImpl::AddModel(const std::vector<std::pair<const void *, size_t>> &model_buff_list) {
  if (flags_ != ModelGroupFlag::kShareWorkspace) {
    MS_LOG(ERROR)
      << "Only support share workspace for ModelGroup::AddModel(const std::vector<std::pair<const void *, size_t>> &)";
    return kLiteError;
  }
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

Status ModelGroupImpl::AddModel(const std::vector<std::shared_ptr<ModelImpl>> &model_list) {
  if (flags_ != ModelGroupFlag::kShareWeight) {
    MS_LOG(ERROR) << "Only support share weight for ModelGroup::AddModel(const std::vector<Model>&)";
    return kLiteError;
  }
  for (auto &impl : model_list) {
    if (impl == nullptr) {
      MS_LOG(ERROR) << "model impl cannot be nullptr.";
      return kLiteError;
    }
    auto old_val = impl->GetConfig(lite::kLiteInnerGroupSection, lite::kLiteInnerGroupId);
    if (!old_val.empty()) {
      MS_LOG(ERROR) << "model has been in another group, group id: " << old_val;
      return kLiteError;
    }
    impl->UpdateConfig(lite::kLiteInnerGroupSection, {lite::kLiteInnerGroupId, std::to_string(model_group_id_)});
  }
  MS_LOG(INFO) << "Update config " << lite::kLiteInnerGroupId << " to " << model_group_id_ << ", section "
               << lite::kLiteInnerGroupSection;
  return kSuccess;
}

Status ModelGroupImpl::CalMaxSizeOfWorkspace(ModelType model_type, const std::shared_ptr<Context> &ms_context) {
  if (ms_context->MutableDeviceInfo().size() > 0 && ms_context->MutableDeviceInfo()[0]->GetProvider() == "ge") {
    MS_LOG(ERROR) << "Not Support GE model to ModelGroup::CalMaxSizeOfWorkspace!";
    return kLiteError;
  }
  std::unique_lock<std::mutex> local_path(mtx_path_list_);
  for (auto &model_path : model_path_list_) {
    Model model;
    model.UpdateConfig(lite::kInnerCommon, std::make_pair(lite::kInnerCalcWorkspaceSize, "true"));
    model.UpdateConfig(lite::kInnerCommon, std::make_pair(lite::kInnerModelPath, model_path));
    auto ret = model.Build(model_path, model_type, ms_context);
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "model build failed.";
      ModelManager::GetInstance().ClearModel();
      return kLiteError;
    }
    if (flags_ == ModelGroupFlag::kShareWorkspace) {
      ModelManager::GetInstance().AddModel(model_path, ModelGroupFlag::kShareWorkspace);
    } else if (flags_ == ModelGroupFlag::kShareWeight) {
      ModelManager::GetInstance().AddModel(model_path, ModelGroupFlag::kShareWeight);
    } else if (flags_ == ModelGroupFlag::kShareWeightAndWorkspace) {
      ModelManager::GetInstance().AddModel(model_path, ModelGroupFlag::kShareWeightAndWorkspace);
    } else {
      MS_LOG(ERROR) << "Only Support share weightspace and share workspace!";
      return kLiteError;
    }
  }
  model_path_list_.clear();
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
