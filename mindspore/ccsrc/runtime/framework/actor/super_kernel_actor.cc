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
}

void SuperKernelActor::RunOpData(OpData<DeviceTensor> *const input_data, OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  MS_EXCEPTION_IF_NULL(device_contexts_[0]);

  auto &sequential_num = context->sequential_num_;
  (void)input_op_datas_[sequential_num].emplace_back(input_data);
  if (CheckRunningCondition(context)) {
    MS_LOG(INFO) << "Super kernel actor(" << GetAID().Name() << ") launches graph: " << graph_->graph_id();
    try {
      auto ret = device_contexts_[0]->LaunchGraph(graph_);
      if (!ret) {
        std::string error_info = "Launch graph failed, graph id: " + graph_->graph_id();
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
      }
    } catch (const std::exception &e) {
      MsException::Instance().SetException();
      std::string error_info = "Launch graph failed, graph id: " + graph_->graph_id();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }

    // The input is invalid and needs to be erased when finish kernel launch.
    EraseInput(context);
    SendOutput(context);
  }
}

void SuperKernelActor::RunOpControl(AID *const input_control, OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  MS_EXCEPTION_IF_NULL(device_contexts_[0]);

  auto &sequential_num = context->sequential_num_;
  (void)input_op_controls_[sequential_num].emplace_back(input_control);
  if (CheckRunningCondition(context)) {
    MS_LOG(INFO) << "Super kernel actor(" << GetAID().Name() << ") launches graph: " << graph_->graph_id();
    try {
      auto ret = device_contexts_[0]->LaunchGraph(graph_);
      if (!ret) {
        std::string error_info = "Launch graph failed, graph id: " + graph_->graph_id();
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
      }
    } catch (const std::exception &e) {
      MsException::Instance().SetException();
      std::string error_info = "Launch graph failed, graph id: " + graph_->graph_id();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }

    // The input is invalid and needs to be erased when finish kernel launch.
    EraseInput(context);
    SendOutput(context);
  }
}

void SuperKernelActor::SendOutput(OpContext<DeviceTensor> *const context) const {
  MS_EXCEPTION_IF_NULL(context);
  SendOutputResult(context);
  SendOutputControl(context);

  // No output.
  if ((output_data_arrows_.size() == 0) && (output_control_arrows_.size() == 0) &&
      (output_result_arrows_.size() == 0)) {
    SET_OPCONTEXT_SUCCESS_RET((*context));
  }
}
}  // namespace runtime
}  // namespace mindspore
