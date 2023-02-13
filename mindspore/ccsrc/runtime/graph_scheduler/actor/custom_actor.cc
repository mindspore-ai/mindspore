/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "runtime/graph_scheduler/actor/custom_actor.h"
#include "runtime/graph_scheduler/actor/memory_manager_actor.h"
#include "utils/log_adapter.h"
#include "utils/anf_utils.h"

namespace mindspore {
namespace runtime {
void CustomActor::Init() {
  auto kernel = kernel_.lock();
  MS_EXCEPTION_IF_NULL(kernel);
  auto base_node = AnfUtils::GetCustomActorBaseNode(kernel);
  MS_EXCEPTION_IF_NULL(base_node);
  if (base_node->isa<CNode>()) {
    const auto &cnode = base_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    input_device_tensors_.resize(common::AnfAlgo::GetInputNum(cnode));
  }
}

void CustomActor::Run(OpContext<DeviceTensor> *const ctx) {
  MS_EXCEPTION_IF_NULL(ctx);
  auto node = kernel_.lock();
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_ZERO("device_contexts_ size", device_contexts_.size());
  MS_EXCEPTION_IF_NULL(device_contexts_[0]);
  try {
    // Collect the input data for infer shape.
    const auto &data_iter = input_op_datas_.find(ctx->sequential_num_);
    if (data_iter != input_op_datas_.end()) {
      for (auto &input_data : data_iter->second) {
        MS_EXCEPTION_IF_NULL(input_data);
        if (IntToSize(input_data->index_) >= input_device_tensors_.size()) {
          SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(
            strategy_, (*ctx),
            "The input index:" + std::to_string(input_data->index_) + " is out of vector size:" +
              std::to_string(input_device_tensors_.size()) + " for node:" + node->DebugString());
          return;
        }
        MS_LOG(DEBUG) << "Collect input data index:" << input_data->index_ << " for custom actor:" << GetAID();
        input_device_tensors_[IntToSize(input_data->index_)] = input_data->data_;
      }
    }

    // Launch custom func
    MS_EXCEPTION_IF_NULL(node);
    auto custom_func = AnfUtils::GetCustomFunc(node);
    if (!device_contexts_[0]->device_res_manager_->BindDeviceToCurrentThread(false)) {
      std::string error_info = "BindDevice to current thread failed: " + node->fullname_with_scope();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, (*ctx), error_info);
    }
    custom_func(&input_device_tensors_);

    // Update the output addr size after inferop && updateop, because after the inferop & updateop, the shape of output
    // maybe changed.
    if (AnfUtils::GetCustomActorType(kernel_.lock()) == kInfer) {
      auto base_node = AnfUtils::GetCustomActorBaseNode(kernel_.lock());
      MS_EXCEPTION_IF_NULL(base_node);
      auto kernel_info = dynamic_cast<KernelInfo *>(base_node->kernel_info());
      AnfAlgo::UpdateOutputAddrSize(kernel_info, base_node);
      // Update the shape of internal parameter.
      AnfAlgo::UpdateInternalParameterShape(internal_parameters_, base_node);
    }
  } catch (const std::exception &e) {
    if (strategy_ == GraphExecutionStrategy::kPipeline) {
      MsException::Instance().SetException();
    }
    std::string error_info = "Launch custom kernel exception: " + node->fullname_with_scope();
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, (*ctx), error_info);
  }

  EraseInput(ctx);
  SendOutput(ctx);
}
}  // namespace runtime
}  // namespace mindspore
