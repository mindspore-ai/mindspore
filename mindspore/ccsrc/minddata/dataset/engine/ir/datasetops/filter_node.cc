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

#include "minddata/dataset/engine/ir/datasetops/filter_node.h"

#include <memory>
#include <string>
#include <vector>

#include "minddata/dataset/engine/datasetops/filter_op.h"

#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {

// Constructor for FilterNode
FilterNode::FilterNode(std::shared_ptr<DatasetNode> child, std::shared_ptr<TensorOp> predicate,
                       std::vector<std::string> input_columns)
    : predicate_(predicate), input_columns_(input_columns) {
  this->children.push_back(child);
}

std::vector<std::shared_ptr<DatasetOp>> FilterNode::Build() {
  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

  node_ops.push_back(std::make_shared<FilterOp>(input_columns_, num_workers_, connector_que_size_, predicate_));
  return node_ops;
}

Status FilterNode::ValidateParams() {
  if (predicate_ == nullptr) {
    std::string err_msg = "FilterNode: predicate is not specified.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (!input_columns_.empty()) {
    RETURN_IF_NOT_OK(ValidateDatasetColumnParam("FilterNode", "input_columns", input_columns_));
  }
  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
