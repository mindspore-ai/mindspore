/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include <string>
#include <nlohmann/json.hpp>
#include "minddata/dataset/include/datasets.h"
#include "minddata/dataset/engine/opt/pre/input_validation_pass.h"

namespace mindspore {
namespace dataset {

Status InputValidationPass::Visit(std::shared_ptr<DatasetNode> node, bool *const modified) {
  *modified = false;
  RETURN_IF_NOT_OK(node->ValidateParams());

  // A data source node must be a leaf node
  if ((node->IsMappableDataSource() || node->IsNonMappableDataSource()) && !node->IsLeaf()) {
    std::string err_msg = node->Name() + " is a data source and must be a leaf node.";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  // A non-leaf node must not be a data source node
  if (node->IsNotADataSource() && node->IsLeaf()) {
    std::string err_msg = node->Name() + " is a dataset operator and must not be a leaf node.";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
