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

#include "minddata/dataset/engine/ir/datasetops/sync_wait_node.h"

#include <memory>
#include <string>
#include <vector>

#include "minddata/dataset/engine/datasetops/barrier_op.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {

// Constructor for SyncWaitNode
SyncWaitNode::SyncWaitNode(std::shared_ptr<DatasetNode> child, const std::string &condition_name, py::function callback)
    : condition_name_(condition_name), callback_(callback) {
  this->AddChild(child);
}

std::shared_ptr<DatasetNode> SyncWaitNode::Copy() {
  auto node = std::make_shared<SyncWaitNode>(nullptr, condition_name_, callback_);
  return node;
}

void SyncWaitNode::Print(std::ostream &out) const {
  out << Name() + "(cond_name:" + condition_name_ + "<pyfunc>" + ")";
}

// Function to build the BarrierOp
Status SyncWaitNode::Build(std::vector<std::shared_ptr<DatasetOp>> *const node_ops) {
  // Right now barrier should only take num_rows_per_buffer = 1
  // The reason for this is because having it otherwise can lead to blocking issues
  // See barrier_op.h for more details
  const int32_t rows_per_buffer = 1;
  auto op = std::make_shared<BarrierOp>(rows_per_buffer, connector_que_size_, condition_name_, callback_);
  op->set_total_repeats(GetTotalRepeats());
  op->set_num_repeats_per_epoch(GetNumRepeatsPerEpoch());
  node_ops->push_back(op);
  return Status::OK();
}

// Function to validate the parameters for SyncWaitNode
Status SyncWaitNode::ValidateParams() {
  RETURN_IF_NOT_OK(DatasetNode::ValidateParams());
  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
