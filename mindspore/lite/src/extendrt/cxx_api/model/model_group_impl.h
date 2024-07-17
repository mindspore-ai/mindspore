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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_CXX_API_MODEL_MODEL_GROUP_IMPL_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_CXX_API_MODEL_MODEL_GROUP_IMPL_H_

#include <functional>
#include <map>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <utility>
#include <unordered_map>
#include "include/api/model_group.h"
#include "include/api/context.h"

namespace mindspore {
class ModelGroupImpl {
 public:
  explicit ModelGroupImpl(ModelGroupFlag flags);
  ~ModelGroupImpl() = default;

  Status AddModel(const std::vector<std::vector<char>> &model_path_list);
  Status AddModel(const std::vector<std::pair<const void *, size_t>> &model_buff_list);
  Status AddModel(const std::vector<std::shared_ptr<ModelImpl>> &model_list);
  Status CalMaxSizeOfWorkspace(ModelType model_type, const std::shared_ptr<Context> &ms_context);

 private:
  friend class ModelGroup;
  std::vector<std::string> model_path_list_;
  std::mutex mtx_path_list_;
  std::vector<std::pair<const void *, size_t>> model_buff_list_;
  std::map<std::string, std::map<std::string, std::string>> config_info_;
  ModelGroupFlag flags_;
  uint32_t model_group_id_ = 0;
};
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_EXTENDRT_CXX_API_MODEL_MODEL_GROUP_IMPL_H_
