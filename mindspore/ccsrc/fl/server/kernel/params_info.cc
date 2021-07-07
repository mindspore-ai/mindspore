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

#include "fl/server/kernel/params_info.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace fl {
namespace server {
namespace kernel {
ParamsInfo &ParamsInfo::AddInputNameType(const std::string &name, TypeId type) {
  inputs_name_type_.push_back(std::make_pair(name, type));
  inputs_names_.push_back(name);
  return *this;
}

ParamsInfo &ParamsInfo::AddWorkspaceNameType(const std::string &name, TypeId type) {
  workspaces_name_type_.push_back(std::make_pair(name, type));
  workspace_names_.push_back(name);
  return *this;
}

ParamsInfo &ParamsInfo::AddOutputNameType(const std::string &name, TypeId type) {
  outputs_name_type_.push_back(std::make_pair(name, type));
  outputs_names_.push_back(name);
  return *this;
}

size_t ParamsInfo::inputs_num() const { return inputs_name_type_.size(); }

size_t ParamsInfo::outputs_num() const { return outputs_name_type_.size(); }

const std::pair<std::string, TypeId> &ParamsInfo::inputs_name_type(size_t index) const {
  if (index >= inputs_name_type_.size()) {
    MS_LOG(EXCEPTION) << "Index " << index << " is out of bound of inputs_name_type_.";
  }
  return inputs_name_type_[index];
}

const std::pair<std::string, TypeId> &ParamsInfo::outputs_name_type(size_t index) const {
  if (index >= outputs_name_type_.size()) {
    MS_LOG(EXCEPTION) << "Index " << index << " is out of bound of outputs_name_type_.";
  }
  return outputs_name_type_[index];
}

const std::vector<std::string> &ParamsInfo::inputs_names() const { return inputs_names_; }

const std::vector<std::string> &ParamsInfo::workspace_names() const { return workspace_names_; }

const std::vector<std::string> &ParamsInfo::outputs_names() const { return outputs_names_; }
}  // namespace kernel
}  // namespace server
}  // namespace fl
}  // namespace mindspore
