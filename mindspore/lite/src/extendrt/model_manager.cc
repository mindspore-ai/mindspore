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

#include "src/extendrt/model_manager.h"
#include <memory>
#include <algorithm>
#include <map>
#include <string>
#include "include/api/model_group.h"

namespace mindspore {
bool JudgeMergeFlag(ModelGroupFlag input_flag, ModelGroupFlag cur_flag) {
  return (input_flag == ModelGroupFlag::kShareWeight && cur_flag == ModelGroupFlag::kShareWorkspace) ||
         (cur_flag == ModelGroupFlag::kShareWeight && input_flag == ModelGroupFlag::kShareWorkspace);
}

void ModelManager::AddModel(const std::string model_path, ModelGroupFlag share_flag) {
  if (model_path_set_.find(model_path) != model_path_set_.end() &&
      JudgeMergeFlag(share_flag, model_path_set_.at(model_path))) {
    model_path_set_.at(model_path) = ModelGroupFlag::kShareWeightAndWorkspace;
  } else {
    (void)model_path_set_.insert(std::make_pair(model_path, share_flag));
  }
}

void ModelManager::AddModel(const std::pair<const void *, size_t> model_buff) {
  (void)model_buff_set_.insert(model_buff);
}

void ModelManager::ClearModel() {
  model_path_set_.clear();
  model_buff_set_.clear();
}

ModelManager::~ModelManager() {
  model_path_set_.clear();
  model_buff_set_.clear();
}
}  // namespace mindspore
