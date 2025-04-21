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

#include "src/litert/cxx_api/model/model_group_impl.h"
#include <memory>
#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include "include/api/types.h"
#include "include/api/context.h"
#include "src/litert/cxx_api/converters.h"
#include "src/common/log_adapter.h"
#include "src/litert/lite_session.h"
#include "src/litert/model_manager.h"
#include "src/common/config_file.h"

namespace mindspore {
using mindspore::lite::RET_OK;
ModelGroupImpl::ModelGroupImpl(ModelGroupFlag flags) : flags_(flags) {
  static uint32_t g_model_group_id = 0;
  model_group_id_ = ++g_model_group_id;
}

Status ModelGroupImpl::AddModel(const std::vector<std::vector<char>> &model_path_list) {
  if (flags_ != ModelGroupFlag::kShareWorkspace) {
    MS_LOG(ERROR) << "Only support share workspace for ModelGroup::AddModel(const std::vector<std::string> &)";
    return kLiteError;
  }
  if (model_path_list.empty()) {
    MS_LOG(ERROR) << "Param model_path_list is empty.";
    return kLiteParamInvalid;
  }
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

lite::LiteSession *ModelGroupImpl::CreateLiteSession(const std::shared_ptr<Context> &ms_context) {
  auto session = new (std::nothrow) lite::LiteSession();
  if (session == nullptr) {
    return nullptr;
  }

  std::string sharing_workspace_section = "inner_common";
  std::string calc_workspace_key = "inner_calc_workspace_size";
  std::string calc_workspace_value = "true";
  std::map<std::string, std::string> model_sharing{{calc_workspace_key, calc_workspace_value}};
  config_info_[sharing_workspace_section] = model_sharing;
  session->SetConfigInfo(&config_info_);
  session->SetPrepareSessionFlag(true);
  auto ret = session->Init(ContextUtils::Convert(ms_context.get()));
  if (ret != mindspore::lite::RET_OK) {
    MS_LOG(ERROR) << "init session failed";
    delete session;
    return nullptr;
  }
  return session;
}

Status ModelGroupImpl::CalMaxSizeOfWorkspace(ModelType model_type, const std::shared_ptr<Context> &ms_context) {
  if (flags_ != ModelGroupFlag::kShareWorkspace) {
    MS_LOG(ERROR) << "Only support share workspace for ModelGroup::CalMaxSizeOfWorkspace";
    return kLiteError;
  }
  for (auto &model_path : model_path_list_) {
    auto *session = CreateLiteSession(ms_context);
    if (session == nullptr) {
      MS_LOG(ERROR) << "Calculate the maximum workspace size of the model " << model_path << " failed.";
      ModelManager::GetInstance().ClearModel();
      return kLiteError;
    }
    auto ret = session->LoadModelAndCompileByPath(model_path, model_type);
    if (ret != mindspore::lite::RET_OK) {
      MS_LOG(ERROR) << "Calculate the maximum workspace size of the model " << model_path << " failed.";
      delete session;
      session = nullptr;
      ModelManager::GetInstance().ClearModel();
      return kLiteError;
    }
    ModelManager::GetInstance().AddModel(model_path);
    delete session;
    session = nullptr;
  }

  for (auto &model_buff : model_buff_list_) {
    auto *session = CreateLiteSession(ms_context);
    if (session == nullptr) {
      MS_LOG(ERROR) << "Calculate the maximum workspace size of the model failed.";
      ModelManager::GetInstance().ClearModel();
      return kLiteError;
    }
    auto ret =
      session->LoadModelAndCompileByBuf(static_cast<const char *>(model_buff.first), model_type, model_buff.second);
    if (ret != mindspore::lite::RET_OK) {
      MS_LOG(ERROR) << "Calculate the maximum workspace size of the model failed.";
      delete session;
      session = nullptr;
      ModelManager::GetInstance().ClearModel();
      return kLiteError;
    }
    ModelManager::GetInstance().AddModel(model_buff);
    delete session;
    session = nullptr;
  }
  return kSuccess;
}
}  // namespace mindspore
