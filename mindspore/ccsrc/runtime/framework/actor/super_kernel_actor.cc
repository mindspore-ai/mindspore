/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "runtime/framework/actor/super_kernel_actor.h"
#include "runtime/framework/actor/output_actor.h"
#include "mindrt/include/async/async.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace runtime {
void SuperKernelActor::Init() {
  MS_EXCEPTION_IF_NULL(graph_);
  // Check device contexts number.
  if (device_contexts_.size() != device::kDeviceContextsNumOne) {
    MS_LOG(EXCEPTION) << "The device contexts number is wrong.";
  }

  // Set the number of actor running dependent messages.
  running_dependent_msg_num_ = SizeToInt(input_datas_num_ + input_controls_num_);

  if (output_data_arrows_.size() != output_data_nodes_.size()) {
    MS_LOG(EXCEPTION) << "The size of output data arrows is not equal to the output data nodes.";
  }
  // Init the output data.
  for (size_t i = 0; i < output_data_arrows_.size(); ++i) {
    auto &data_arrow = output_data_arrows_[i];
    auto &output_node = output_data_nodes_[i];
    MS_EXCEPTION_IF_NULL(data_arrow);
    MS_EXCEPTION_IF_NULL(output_node);

    auto device_address = AnfAlgo::GetMutableOutputAddr(output_node, data_arrow->from_output_index_, false);
    auto data =
      std::make_unique<OpData<DeviceTensor>>(data_arrow->to_op_id_, device_address.get(), data_arrow->to_input_index_);
    (void)output_data_.emplace_back(std::move(data));
  }
}

size_t SuperKernelActor::FetchInputNodePosition(const AnfNodePtr &intput_node) {
  MS_EXCEPTION_IF_NULL(intput_node);
  MS_EXCEPTION_IF_NULL(graph_);

  auto &input_nodes = graph_->input_nodes();
  const auto &iter = find(input_nodes.begin(), input_nodes.end(), intput_node);
  if (iter == input_nodes.end()) {
    MS_LOG(EXCEPTION) << "Invalid input node:" << intput_node->fullname_with_scope();
  }
  return iter - input_nodes.begin();
}

void SuperKernelActor::Run(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  MS_EXCEPTION_IF_NULL(graph_);
  MS_EXCEPTION_IF_NULL(device_contexts_[0]);
  MS_LOG(INFO) << "Super kernel actor(" << GetAID().Name()
               << ") launches graph: " << std::to_string(graph_->graph_id());
  if (!CopyInputData(context)) {
    std::string error_info = "Copy the input data failed, graph id: " + graph_->graph_id();
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
  }

  try {
    auto ret = device_contexts_[0]->LaunchGraph(graph_);
    if (!ret) {
      std::string error_info = "Launch graph failed, graph id: " + std::to_string(graph_->graph_id());
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
  } catch (const std::exception &e) {
    MsException::Instance().SetException();
    std::string error_info = "Launch graph exception, graph id: " + std::to_string(graph_->graph_id());
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
  }

  PostRun(context);
}

bool SuperKernelActor::CopyInputData(const OpContext<DeviceTensor> *context) {
  MS_EXCEPTION_IF_NULL(context);
  MS_EXCEPTION_IF_NULL(graph_);
  const auto &data_iter = input_op_datas_.find(context->sequential_num_);
  if (data_iter == input_op_datas_.end()) {
    return true;
  }

  auto &input_nodes = graph_->input_nodes();
  for (auto &input_data : data_iter->second) {
    MS_EXCEPTION_IF_NULL(input_data);
    if (IntToSize(input_data->index_) >= input_nodes.size()) {
      MS_LOG(ERROR) << "The input index:" << input_data->index_ << "is out of range:" << input_nodes.size() << ".";
      return false;
    }
    auto input_node = input_nodes[input_data->index_];
    MS_EXCEPTION_IF_NULL(input_node);
    auto device_address = AnfAlgo::GetMutableOutputAddr(input_node, 0, false);
    MS_EXCEPTION_IF_NULL(device_address);

    auto &input_device_tensor = input_data->data_;
    MS_EXCEPTION_IF_NULL(input_device_tensor);
    if (input_device_tensor->DeviceType() != device_address->DeviceType()) {
      MS_LOG(ERROR) << "The input data device type:" << input_device_tensor->DeviceType()
                    << " is not equal to the graph node device type:" << device_address->DeviceType() << ".";
      return false;
    }

    if (input_device_tensor->GetPtr() == device_address->GetPtr()) {
      continue;
    }
    MS_LOG(INFO) << "The input data of node:" << input_node->DebugString()
                 << " need copy from address:" << input_device_tensor->GetPtr()
                 << " to address:" << device_address->GetPtr() << ".";
    if (!device_address->SyncDeviceToDevice(trans::GetRuntimePaddingShape(input_node, 0),
                                            input_device_tensor->GetSize(), input_device_tensor->type_id(),
                                            input_device_tensor->GetPtr(), input_device_tensor->format())) {
      MS_LOG(ERROR) << "Sync device to device failed.";
      return false;
    }
  }

  return true;
}
}  // namespace runtime
}  // namespace mindspore
