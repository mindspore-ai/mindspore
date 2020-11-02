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

#include "minddata/dataset/engine/ir/datasetops/zip_node.h"

#include <memory>
#include <string>
#include <vector>

#include "minddata/dataset/engine/datasetops/zip_op.h"

#include "minddata/dataset/util/status.h"
namespace mindspore {
namespace dataset {

ZipNode::ZipNode(const std::vector<std::shared_ptr<DatasetNode>> &datasets) : datasets_(datasets) {
  for (auto dataset : datasets_) {
    this->children.push_back(dataset);
  }
}

Status ZipNode::ValidateParams() {
  if (datasets_.empty()) {
    std::string err_msg = "ZipNode: datasets to zip are not specified.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  if (find(datasets_.begin(), datasets_.end(), nullptr) != datasets_.end()) {
    std::string err_msg = "ZipNode: zip datasets should not be null.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  return Status::OK();
}

std::vector<std::shared_ptr<DatasetOp>> ZipNode::Build() {
  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

  node_ops.push_back(std::make_shared<ZipOp>(rows_per_buffer_, connector_que_size_));
  return node_ops;
}

}  // namespace dataset
}  // namespace mindspore
