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
#include <utility>
#include <vector>

#include "minddata/dataset/engine/datasetops/device_queue_op.h"
#include "minddata/dataset/util/status.h"

#include "utils/ms_context.h"

namespace mindspore {
namespace dataset {

// Constructor for TransferNode
TransferNode::TransferNode(std::shared_ptr<DatasetNode> child, std::string queue_name, std::string device_type,
                           bool send_epoch_end, int32_t total_batch, bool create_data_info_queue)
    : prefetch_size_(16),
      queue_name_(std::move(queue_name)),
      device_type_(std::move(device_type)),
      send_epoch_end_(send_epoch_end),
      total_batch_(total_batch),
      create_data_info_queue_(create_data_info_queue),
      device_id_(0) {
  this->children.push_back(child);
}

// Validator for TransferNode
Status TransferNode::ValidateParams() {
  if (total_batch_ < 0) {
    std::string err_msg = "TransferNode: Total batches should be >= 0, value given: ";
    MS_LOG(ERROR) << err_msg << total_batch_;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

// Function to build TransferNode
std::vector<std::shared_ptr<DatasetOp>> TransferNode::Build() {
  if (queue_name_.empty()) {
    // Get a uuid for queue name
    queue_name_ = Services::GetUniqueID();
  }
  if (device_type_.empty()) {
    auto context = MsContext::GetInstance();
    if (context == nullptr) {
      device_type_ = kCPUDevice;
    } else {
      device_type_ = context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
    }
  }

  // Get device type from ms context
  // Convert device_type_ from string to DeviceType
  DeviceQueueOp::DeviceType type;
  if (device_type_ == kCPUDevice) {
    type = DeviceQueueOp::DeviceType::CPU;
  } else if (device_type_ == kGPUDevice) {
    type = DeviceQueueOp::DeviceType::GPU;
  } else if (device_type_ == kAscendDevice) {
    type = DeviceQueueOp::DeviceType::Ascend;
  } else {
    MS_LOG(ERROR) << "Unknown device target.";
    return {};
  }

  // Get device ID (shard ID) from children
  device_id_ = 0;
  build_status = this->GetShardId(&device_id_);  // remove me after changing return val of Build()
  RETURN_EMPTY_IF_ERROR(build_status);

  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

  node_ops.push_back(std::make_shared<DeviceQueueOp>(queue_name_, type, device_id_, prefetch_size_, send_epoch_end_,
                                                     total_batch_, create_data_info_queue_));
  return node_ops;
}

}  // namespace dataset
}  // namespace mindspore
