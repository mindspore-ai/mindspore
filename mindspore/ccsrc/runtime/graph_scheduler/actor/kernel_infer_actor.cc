/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "runtime/graph_scheduler/actor/kernel_infer_actor.h"
#include "runtime/graph_scheduler/actor/memory_manager_actor.h"

namespace mindspore {
namespace runtime {
void KernelInferActor::Init() {
  KernelActor::Init();

  // Erase output and workspace device tensors which is released by kernel actor.
  size_t input_num = input_device_tensors_.size();
  // size_t output_num = output_device_tensors_.size();
  if (memory_free_list_.size() > input_num) {
    memory_free_list_.erase(memory_free_list_.begin() + input_num, memory_free_list_.end());
  }
}

void KernelInferActor::RunOpData(OpData<DeviceTensor> *const input_data, OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(input_data);
  MS_EXCEPTION_IF_NULL(input_data->data_);
  MS_EXCEPTION_IF_NULL(context);
  auto &sequential_num = context->sequential_num_;
  (void)input_op_datas_[sequential_num].emplace_back(input_data);
  // Without verifying that the device pointer for device tensor is empty, the kernel before the KernelResizeActor phase
  // may not have started memory allocate and launch.
  auto can_run = CheckRunningCondition(context);
  MS_LOG(DEBUG) << "Actor(" << GetAID().Name() << ") receive the input op data and check running condition:" << can_run
                << ", sequential num:" << sequential_num << ", the input data:" << input_data->data_
                << " input index:" << input_data->index_ << ", size:" << input_data->data_->GetSize()
                << ", origin ref count:" << input_data->data_->original_ref_count()
                << ", current ref count:" << input_data->data_->ref_count()
                << ", dynamic ref count:" << input_data->data_->dynamic_ref_count()
                << ", flag:" << input_data->data_->flag() << " user data:" << input_data->data_->user_data();

  if (can_run) {
    Run(context);
  }
}

void KernelInferActor::Run(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  MS_EXCEPTION_IF_ZERO("device_contexts_ size", device_contexts_.size());
  MS_EXCEPTION_IF_NULL(device_contexts_[0]);
  try {
    ProfilerRecorder profiler(ProfilerModule::kKernel, ProfilerEvent::kKernelInfer, GetAID().Name());
    // 1. Collect the inputs from input data.
    const auto &data_iter = input_op_datas_.find(context->sequential_num_);
    if (data_iter != input_op_datas_.end()) {
      for (auto &input_data : data_iter->second) {
        UpdateInputDeviceTensor(input_data, context);
      }
    }

    // Collect the inputs from device tensor store.
    FetchInputByTensorStore(&input_device_tensors_, &input_kernel_tensors_, &input_kernel_tensors_for_infer_,
                            &memory_free_list_, context);

    // 2. InferShape or InferShapeAndType and update output shape(and type).
    if (is_dynamic_type_) {
      // For dynamic type case, need Re-Infer shape and type.
      InferShapeAndType();
    } else if (is_dynamic_shape_) {
      // For dynamic shape case, need Re-Infer shape.
      InferShape();
    }
  } catch (const std::exception &e) {
    MsException::Instance().SetException();
    std::string error_info =
      "Run infer shape actor failed for kernel: " + kernel_->fullname_with_scope() + ", exception: " + e.what();
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(GraphExecutionStrategy::kPipeline, (*context), error_info);
  }

  PostRun(context);
}

void KernelInferActor::SendMemoryFreeReq(OpContext<DeviceTensor> *const context) {
  if (ActorDispatcher::is_memory_free_sync()) {
    ActorDispatcher::SendSync(memory_manager_aid_, &MemoryManagerActor::FreeMemory, &memory_free_list_,
                              device_contexts_[0], context, GetAID());
  } else {
    ActorDispatcher::Send(memory_manager_aid_, &MemoryManagerActor::FreeMemory, &memory_free_list_, device_contexts_[0],
                          context, GetAID());
  }
}
}  // namespace runtime
}  // namespace mindspore
