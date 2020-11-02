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

#include "minddata/dataset/engine/ir/datasetops/transfer_node.h"

#include <memory>
#include <string>
#include <vector>

#include "minddata/dataset/engine/datasetops/device_queue_op.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {

// Constructor for TransferNode
TransferNode::TransferNode(std::shared_ptr<DatasetNode> child, bool send_epoch_end)
    : prefetch_size_(16), send_epoch_end_(send_epoch_end), total_batch_(0) {
  this->children.push_back(child);
}

// Validator for TransferNode
Status TransferNode::ValidateParams() {
  // Check if device_type_ is in {"CPU", "GPU", "Ascend"}
  RETURN_IF_NOT_OK(ValidateStringValue("TransferNode", device_type_, {"CPU", "GPU", "Ascend"}));
  return Status::OK();
}

// Function to build TransferNode
std::vector<std::shared_ptr<DatasetOp>> TransferNode::Build() {
  // Get a uuid for queue name
  queue_name_ = Services::GetUniqueID();
  // TODO(CRC):
  // Get device type from ms context
  device_type_ = "CPU";
  // Get device ID from children
  device_id_ = 0;
  RETURN_EMPTY_IF_ERROR(TransferNode::get_distribution(shared_from_this(), &device_id_));

  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

  // Convert device_type_ from string to DeviceType
  DeviceQueueOp::DeviceType type;
  if (device_type_ == "CPU") {
    type = DeviceQueueOp::DeviceType::CPU;
  } else if (device_type_ == "GPU") {
    type = DeviceQueueOp::DeviceType::GPU;
  } else if (device_type_ == "Ascend") {
    type = DeviceQueueOp::DeviceType::Ascend;
  }

  node_ops.push_back(
    std::make_shared<DeviceQueueOp>(queue_name_, type, device_id_, prefetch_size_, send_epoch_end_, total_batch_));
  return node_ops;
}

// Function to get the device_id
Status TransferNode::get_distribution(std::shared_ptr<DatasetNode> ds, int32_t *device_id) {
  // Get device id according to the type of dataset
  Status rc = ds->GetShardId(device_id);
  if (rc != Status::OK()) {
    // Get device id from the child node
    if (ds->Children().size()) {
      ds = ds->Children()[0];
      return TransferNode::get_distribution(ds, device_id);
    } else {
      std::string err_msg = "Unknown dataset type.";
      MS_LOG(ERROR) << err_msg;
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
  }

  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
