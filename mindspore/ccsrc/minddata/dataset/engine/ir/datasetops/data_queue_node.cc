/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/engine/ir/datasetops/data_queue_node.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/engine/datasetops/data_queue_op.h"
#include "minddata/dataset/engine/opt/pass.h"
#include "minddata/dataset/util/status.h"

#include "utils/ms_context.h"

namespace mindspore {
namespace dataset {
// Constructor for DataQueueNode
DataQueueNode::DataQueueNode(std::shared_ptr<DatasetNode> child, std::string queue_name, std::string device_type,
                             int32_t device_id, bool send_epoch_end, int32_t total_batch, bool create_data_info_queue)
    : queue_name_(std::move(queue_name)),
      device_id_(device_id),
      device_type_(std::move(device_type)),
      send_epoch_end_(send_epoch_end),
      total_batch_(total_batch),
      create_data_info_queue_(create_data_info_queue) {
  this->AddChild(child);
}

std::shared_ptr<DatasetNode> DataQueueNode::Copy() {
  auto node = std::make_shared<DataQueueNode>(nullptr, queue_name_, device_type_, device_id_, send_epoch_end_,
                                              total_batch_, create_data_info_queue_);
  return node;
}

void DataQueueNode::Print(std::ostream &out) const {
  out << (Name() + ",send_epoch_end:" + (send_epoch_end_ ? "true" : "false") +
          ",total_batch:" + std::to_string(total_batch_) + ")");
}

// Validator for DataQueueNode
Status DataQueueNode::ValidateParams() {
  RETURN_IF_NOT_OK(DatasetNode::ValidateParams());
  RETURN_IF_NOT_OK(ValidateScalar("Transfer", "Total batches", total_batch_, {0}, false));
  return Status::OK();
}

// Function to build DataQueueNode
Status DataQueueNode::Build(std::vector<std::shared_ptr<DatasetOp>> *const node_ops) {
  if (queue_name_.empty()) {
    // Get a uuid for queue name
    queue_name_ = Services::GetUniqueID();
  }

  // This is an issue from MindSpore C++ user
  // https://gitee.com/mindspore/mindspore/issues/I39J9A
  // Link _c_expression.so and _c_dataengine.so simultaneously will cause heap overflow because MindData uses MSContext.
  // We should find a new way to get device_type here.
  if (device_type_.empty()) {
    device_type_ = kCPUDevice;
  }

  // Get device type from ms context
  // Convert device_type_ from string to DeviceType
  DataQueueOp::DeviceType type;
  if (device_type_ == kCPUDevice) {
    type = DataQueueOp::DeviceType::CPU;
  } else if (device_type_ == kGPUDevice) {
    type = DataQueueOp::DeviceType::GPU;
  } else if (device_type_ == kAscendDevice) {
    type = DataQueueOp::DeviceType::Ascend;
  } else {
    std::string err_msg = "Unknown device target, support CPU, GPU or Ascend";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  // Ascend does not support receiving empty data at the moment, so send_epoch_end needs to be set to false.
  send_epoch_end_ = false;
  auto op = std::make_shared<DataQueueOp>(queue_name_, type, device_id_, send_epoch_end_, total_batch_,
                                          create_data_info_queue_);
  op->SetTotalRepeats(GetTotalRepeats());
  op->SetNumRepeatsPerEpoch(GetNumRepeatsPerEpoch());
  node_ops->push_back(op);
  return Status::OK();
}

// Visitor accepting method for IRNodePass
Status DataQueueNode::Accept(IRNodePass *const p, bool *const modified) {
  // Downcast shared pointer then call visitor
  return p->Visit(shared_from_base<DataQueueNode>(), modified);
}

// Visitor accepting method for IRNodePass
Status DataQueueNode::AcceptAfter(IRNodePass *const p, bool *const modified) {
  // Downcast shared pointer then call visitor
  return p->VisitAfter(shared_from_base<DataQueueNode>(), modified);
}

Status DataQueueNode::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["queue_name"] = queue_name_;
  args["device_type"] = device_type_;
  args["device_id"] = device_id_;
  args["send_epoch_end"] = send_epoch_end_;
  args["total_batch"] = total_batch_;
  args["create_data_info_queue"] = create_data_info_queue_;
  *out_json = args;
  return Status::OK();
}

Status DataQueueNode::from_json(nlohmann::json json_obj, std::shared_ptr<DatasetNode> ds,
                                std::shared_ptr<DatasetNode> *result) {
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "queue_name", kTransferNode));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "device_type", kTransferNode));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "device_id", kTransferNode));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "send_epoch_end", kTransferNode));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "total_batch", kTransferNode));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "create_data_info_queue", kTransferNode));
  std::string queue_name = json_obj["queue_name"];
  std::string device_type = json_obj["device_type"];
  int32_t device_id = json_obj["device_id"];
  bool send_epoch_end = json_obj["send_epoch_end"];
  int32_t total_batch = json_obj["total_batch"];
  bool create_data_info_queue = json_obj["create_data_info_queue"];
  *result = std::make_shared<DataQueueNode>(ds, queue_name, device_type, device_id, send_epoch_end, total_batch,
                                            create_data_info_queue);
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
