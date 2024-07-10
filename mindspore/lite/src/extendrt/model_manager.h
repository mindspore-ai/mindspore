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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_MODEL_MANAGER_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_MODEL_MANAGER_H_

#include <functional>
#include <set>
#include <map>
#include <string>
#include <memory>
#include <utility>
#include "include/api/model_group.h"

namespace mindspore {
class ModelManager {
 public:
  ModelManager() {}
  ~ModelManager();

  ModelManager(const ModelManager &) = delete;
  ModelManager &operator=(const ModelManager &) = delete;

  static ModelManager &GetInstance() {
    static ModelManager instance;
    return instance;
  }

  void AddModel(const std::string model_path, ModelGroupFlag share_flag);
  void AddModel(const std::pair<const void *, size_t> model_buff);
  const std::map<std::string, ModelGroupFlag> &GetModelPath() const { return model_path_set_; }
  const std::set<std::pair<const void *, size_t>> &GetModelBuff() const { return model_buff_set_; }
  void ClearModel();

 private:
  std::map<std::string, ModelGroupFlag> model_path_set_;
  std::set<std::pair<const void *, size_t>> model_buff_set_;
};
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_EXTENDRT_MODEL_MANAGER_H_
