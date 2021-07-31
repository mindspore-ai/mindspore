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

#include "runtime/framework/actor/copy_actor.h"
#include "runtime/framework/actor/memory_manager_actor.h"
#include "mindrt/include/async/async.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace runtime {
const size_t kDeviceTensorNum = 1;

void CopyActor::Init() {
  input_device_tensor_.resize(kDeviceTensorNum);
  output_device_tensor_.resize(kDeviceTensorNum);

  // Init output data.
  for (auto &data_arrow : output_data_arrows_) {
    MS_EXCEPTION_IF_NULL(data_arrow);
    if (IntToSize(data_arrow->from_output_index_) != 0) {
      MS_LOG(EXCEPTION) << "The output index is out of range: " << GetAID().Name();
    }
    auto data = std::make_unique<OpData<DeviceTensor>>(data_arrow->to_op_id_, nullptr, data_arrow->to_input_index_);
    (void)output_data_.emplace_back(std::move(data));
  }
}

void CopyActor::RunOpData(OpData<DeviceTensor> *const input_data, OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  auto &sequential_num = context->sequential_num_;
  (void)input_op_datas_[sequential_num].emplace_back(input_data);
  // When all the inputs are collected, then allocate memory and callback copy.
  if (CheckCopyCondition(context)) {
    FetchDeviceTensor(context);
    SendMemoryAllocReq(context);
  }
}

void CopyActor::RunOpControl(AID *const input_control, OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  auto &sequential_num = context->sequential_num_;
  (void)input_op_controls_[sequential_num].emplace_back(input_control);
  // When all the inputs are collected, then allocate memory and callback copy.
  if (CheckCopyCondition(context)) {
    FetchDeviceTensor(context);
    SendMemoryAllocReq(context);
  }
}

void CopyActor::SendMemoryAllocReq(OpContext<DeviceTensor> *const context) {
  Async(memory_manager_aid_, &MemoryManagerActor::AllocateMemory, &output_device_tensor_, output_device_context_,
        context, GetAID());
}

void CopyActor::SendMemoryFreeReq(OpContext<DeviceTensor> *const context) {
  Async(memory_manager_aid_, &MemoryManagerActor::FreeMemory, &input_device_tensor_, input_device_context_, context);
  Async(memory_manager_aid_, &MemoryManagerActor::FreeMemory, &output_device_tensor_, output_device_context_, context);
}

void CopyActor::OnMemoryAllocFinish(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  MS_EXCEPTION_IF_NULL(output_device_tensor_[0]);
  MS_EXCEPTION_IF_NULL(input_device_tensor_[0]);

  if (input_device_tensor_[0]->GetSize() != output_device_tensor_[0]->GetSize()) {
    MS_LOG(WARNING) << GetAID().Name() << " copy size is not equal, input size:" << input_device_tensor_[0]->GetSize()
                    << ", output size:" << output_device_tensor_[0]->GetSize();
  }

  if (!Copy(output_device_tensor_[0], input_device_tensor_[0])) {
    std::string error_info = "Copy device tensor failed: " + GetAID().Name();
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
  }

  // The input is invalid and needs to be erased when finish copy.
  EraseInput(context);

  // Note that SendMemoryFreeReq must be in front of SendOutput, because SendOutput will trigger SendMemoryAllocReq of
  // the next actor and the actor is asynchronous execution. So it is necessary to ensure that SendMemoryFreeReq of the
  // current actor is in front of SendMemoryAllocReq of the next actor.  One is to reuse the memory more fully, the
  // other is to ensure the execution order and avoid the illegal memory timing problem.
  SendMemoryFreeReq(context);
  SendOutput(context);
}

bool CopyActor::CheckCopyCondition(OpContext<DeviceTensor> *const context) const {
  MS_EXCEPTION_IF_NULL(context);
  if (input_datas_num_ != 0) {
    const auto &data_iter = input_op_datas_.find(context->sequential_num_);
    if (data_iter == input_op_datas_.end()) {
      return false;
    }
    if (data_iter->second.size() != input_datas_num_) {
      return false;
    }
  }

  if (input_controls_num_ != 0) {
    const auto &control_iter = input_op_controls_.find(context->sequential_num_);
    if (control_iter == input_op_controls_.end()) {
      return false;
    }
    if (control_iter->second.size() != input_controls_num_) {
      return false;
    }
  }
  return true;
}

void CopyActor::FetchDeviceTensor(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  MS_EXCEPTION_IF_NULL(input_device_context_);

  if (device_tensor_store_key_.second != nullptr) {
    input_device_tensor_[0] = DeviceTensorStore::GetInstance().Fetch(device_tensor_store_key_.second,
                                                                     input_device_context_->GetDeviceAddressType());
    if (input_device_tensor_[0] == nullptr) {
      std::string error_info =
        GetAID().Name() + " get device tensor store failed: " + device_tensor_store_key_.second->fullname_with_scope() +
        ", device type:" + std::to_string(static_cast<int>(input_device_context_->GetDeviceAddressType()));
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }

    output_device_tensor_[0] = DeviceTensorStore::GetInstance().Fetch(device_tensor_store_key_.second,
                                                                      output_device_context_->GetDeviceAddressType());
    if (output_device_tensor_[0] == nullptr) {
      std::string error_info =
        GetAID().Name() + " get device tensor store failed: " + device_tensor_store_key_.second->fullname_with_scope() +
        ", device type:" + std::to_string(static_cast<int>(output_device_context_->GetDeviceAddressType()));
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
  } else {
    const auto &data_iter = input_op_datas_.find(context->sequential_num_);
    if (data_iter == input_op_datas_.end()) {
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), "No input data.");
    }
    const auto &input_data = data_iter->second[0];
    MS_EXCEPTION_IF_NULL(input_data);
    input_device_tensor_[0] = input_data->data_;

    MS_EXCEPTION_IF_NULL(output_);
    output_device_tensor_[0] = output_.get();
  }
}

void CopyActor::SendOutput(OpContext<DeviceTensor> *const context) const {
  MS_EXCEPTION_IF_NULL(context);
  // No output.
  if ((output_data_arrows_.size() == 0) && (output_control_arrows_.size() == 0)) {
    SET_OPCONTEXT_SUCCESS_RET((*context));
  }

  // Send output data.
  for (auto &output_data : output_data_) {
    MS_EXCEPTION_IF_NULL(output_data);
    output_data->data_ = output_device_tensor_[0];
    Async(output_data->op_id_, &OpActor::RunOpData, output_data.get(), context);
  }

  // Send output control.
  if (output_control_arrows_.size() > 0) {
    auto source_aid = const_cast<AID *>(&GetAID());
    for (auto &output_control : output_control_arrows_) {
      Async(output_control, &OpActor::RunOpControl, source_aid, context);
    }
  }
}

void CopyActor::EraseInput(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  if (input_datas_num_ != 0) {
    auto ret = input_op_datas_.erase(context->sequential_num_);
    if (ret == 0) {
      std::string error_info = "Erase input data failed: " + GetAID().Name();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
  }

  if (input_controls_num_ != 0) {
    auto ret = input_op_controls_.erase(context->sequential_num_);
    if (ret == 0) {
      std::string error_info = "Erase input controls failed: " + GetAID().Name();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
  }
}
}  // namespace runtime
}  // namespace mindspore
