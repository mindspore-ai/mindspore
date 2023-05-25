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
#ifndef MINDSPORE_INCLUDE_API_MODEL_GROUP_H
#define MINDSPORE_INCLUDE_API_MODEL_GROUP_H

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <utility>
#include "include/api/model.h"
#include "include/api/status.h"
#include "include/api/types.h"
#include "include/api/context.h"

namespace mindspore {
class ModelGroupImpl;

/// \brief The ModelGroup class is used to define a MindSpore model group, facilitating
/// multiple models to share workspace memory.

enum class ModelGroupFlag : int {
  kShareWeight = 0x0001,
  kShareWorkspace = 0x0002,
};

class MS_API ModelGroup {
 public:
  explicit ModelGroup(ModelGroupFlag flags = ModelGroupFlag::kShareWorkspace);
  ~ModelGroup() = default;

  /// \brief Add models that require shared workspace memory.
  ///
  /// \param[in] model_path_list Define the list of model path.
  ///
  /// \return Status.
  Status AddModel(const std::vector<std::string> &model_path_list);

  /// \brief Add models that require shared workspace memory.
  ///
  /// \param[in] model_buff_list Define the list of model buff.
  ///
  /// \return Status.
  Status AddModel(const std::vector<std::pair<const void *, size_t>> &model_buff_list);

  /// \brief Add models that require shared weight memory.
  ///
  /// \param[in] model_list Define the list of model object.
  ///
  /// \return Status.
  Status AddModel(const std::vector<Model> &model_list);

  /// \brief Calculate the max workspace of the added models.
  ///
  /// \param[in] model_type Define The type of model file. Options: ModelType::kMindIR_Lite, ModelType::kMindIR. Only
  /// ModelType::kMindIR_Lite is valid for Lite.
  /// \param[in] ms_context A context used to store options.
  ///
  /// \return Status.
  Status CalMaxSizeOfWorkspace(ModelType model_type, const std::shared_ptr<Context> &ms_context);

 private:
  std::shared_ptr<ModelGroupImpl> impl_;
};
}  // namespace mindspore
#endif  // MINDSPORE_INCLUDE_API_MODEL_GROUP_H
