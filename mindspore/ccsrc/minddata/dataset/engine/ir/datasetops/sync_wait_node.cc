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

#include "minddata/dataset/engine/ir/datasetops/sync_wait_node.h"

#include <memory>
#include <string>
#include <vector>

#include "minddata/dataset/engine/datasetops/barrier_op.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {

// Constructor for SyncWaitNode
SyncWaitNode::SyncWaitNode(std::shared_ptr<DatasetNode> child, const std::string &condition_name, int32_t num_batch,
                           py::function callback)
    : condition_name_(condition_name), num_batch_(num_batch), callback_(callback) {
  this->children.push_back(child);
}

// Function to build the BarrierOp
std::vector<std::shared_ptr<DatasetOp>> SyncWaitNode::Build() {
  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

  node_ops.push_back(std::make_shared<BarrierOp>(num_batch_, connector_que_size_, condition_name_, callback_));
  return node_ops;
}

// Function to validate the parameters for SyncWaitNode
Status SyncWaitNode::ValidateParams() {
  if (num_batch_ <= 0) {
    std::string err_msg = "SyncWaitNode: num_batch must be greater than 0, num_batch: " + std::to_string(num_batch_);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  if (condition_name_.empty()) {
    std::string err_msg = "SyncWaitNode: condition_name must not be empty.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
