/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_CXX_API_ACL_MODEL_H
#define MINDSPORE_CCSRC_CXX_API_ACL_MODEL_H

#include <vector>
#include <string>
#include <unordered_map>
#include <utility>
#include <memory>
#include <map>

#include "include/api/cell.h"
#include "include/api/status.h"
#include "cxx_api/model/model_impl.h"
#include "cxx_api/model/acl/model_converter.h"
#include "cxx_api/model/acl/acl_model_options.h"
#include "ir/tensor.h"
#include "ir/anf.h"

namespace mindspore {
class AclModel : public ModelImpl {
 public:
  AclModel() : model_converter_(), options_(nullptr) {}
  ~AclModel() = default;

  Status Build() override;
  Status Resize(const std::vector<MSTensor> &inputs, const std::vector<std::vector<int64_t>> &dims) override;

  Status Predict(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs) override;

  std::vector<MSTensor> GetInputs() override;
  std::vector<MSTensor> GetOutputs() override;

 private:
  std::shared_ptr<GraphCell> graph_cell_;
  ModelConverter model_converter_;
  std::unique_ptr<AclModelOptions> options_;
  std::map<std::string, std::shared_ptr<Graph>> dynamic_size_graph_map_;
};
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_CXX_API_ACL_MODEL_H
