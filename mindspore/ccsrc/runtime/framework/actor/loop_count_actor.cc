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

#include "runtime/framework/actor/loop_count_actor.h"
#include "runtime/framework/actor/data_source_actor.h"
#include "runtime/framework/actor/kernel_actor.h"
#include "runtime/framework/actor/output_actor.h"
#include "runtime/framework/actor/memory_manager_actor.h"
#include "runtime/framework/actor/recorder_actor.h"
#include "runtime/framework/actor/debug_actor.h"
#include "mindrt/include/async/async.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace runtime {
namespace {
void FetchContinuousMemoryInfo(const CNodePtr &node, std::vector<DeviceTensorPtr> *const addr_list,
                               std::vector<size_t> *const size_list, size_t *const total_size, bool is_input) {
  MS_EXCEPTION_IF_NULL(node);
  const auto &kernel_mod = AnfAlgo::GetKernelMod(node);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  (*addr_list).clear();
  (*size_list).clear();
  *total_size = 0;

  if (is_input) {
    const auto &intput_sizes = kernel_mod->GetInputSizeList();
    for (size_t i = 0; i < intput_sizes.size(); ++i) {
      const auto &device_tensor = AnfAlgo::GetPrevNodeMutableOutputAddr(node, i, false);
      MS_EXCEPTION_IF_NULL(device_tensor);
      *total_size += intput_sizes[i];
      (void)size_list->emplace_back(intput_sizes[i]);
      (void)addr_list->emplace_back(device_tensor);
    }
  } else {
    const auto &output_sizes = kernel_mod->GetOutputSizeList();
    for (size_t i = 0; i < output_sizes.size(); ++i) {
      const auto &device_tensor = AnfAlgo::GetMutableOutputAddr(node, i, false);
      MS_EXCEPTION_IF_NULL(device_tensor);
      *total_size += output_sizes[i];
      (void)size_list->emplace_back(output_sizes[i]);
      (void)addr_list->emplace_back(device_tensor);
    }
  }
}
}  // namespace
void LoopCountActor::Init() {
  for (auto &iter : continuous_memory_nodes_) {
    size_t total_size = 0;
    std::vector<size_t> size_list;
    std::vector<DeviceTensorPtr> addr_list;
    // Inputs need continuous memory.
    if (iter.second.first == true) {
      FetchContinuousMemoryInfo(iter.first.first, &addr_list, &size_list, &total_size, true);
      (void)continuous_memory_alloc_list_list_.emplace_back(addr_list);
      (void)size_list_list_.emplace_back(size_list);
      (void)total_size_list_.emplace_back(total_size);
      (void)device_contexts_.emplace_back(iter.first.second);
    }

    // Outputs need continuous memory.
    if (iter.second.second == true) {
      FetchContinuousMemoryInfo(iter.first.first, &addr_list, &size_list, &total_size, false);
      (void)continuous_memory_alloc_list_list_.emplace_back(addr_list);
      (void)size_list_list_.emplace_back(size_list);
      (void)total_size_list_.emplace_back(total_size);
      (void)device_contexts_.emplace_back(iter.first.second);
    }
  }
}

void LoopCountActor::RunOpControl(AID *const input_control, OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  auto sequential_num = context->sequential_num_;
  (void)input_op_controls_[sequential_num].emplace_back(input_control);
  if (CheckLoopCountIncreaseCondition(context)) {
    IncreaseLoopCount(context);
  }
}

void LoopCountActor::SendDebugReq(OpContext<DeviceTensor> *const context) {
  Async(*debug_aid_, &DebugActor::DebugOnStepEnd, context, &GetAID());
}

void LoopCountActor::OnDebugFinish(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  SendOutput(context);
}

void LoopCountActor::IncreaseLoopCount(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  auto sequential_num = context->sequential_num_;
  auto ret = input_op_controls_.erase(sequential_num);
  if (ret == 0) {
    std::string error_info = "Erase input controls failed: " + GetAID().Name();
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
  }

  total_running_count_++;
  current_count_++;
  MS_LOG(INFO) << "Loop count actor(" << GetAID().Name() << ") running, loop count: " << loop_count_
               << ", current count: " << current_count_ << ", total running count: " << total_running_count_;

  // Debug actor is blocked, must wait debug actor callback message to process continue.
  if (debug_aid_ != nullptr) {
    SendDebugReq(context);
    return;
  }

  SendOutput(context);
}

void LoopCountActor::SendOutput(OpContext<DeviceTensor> *const context) {
  // Send recorder info.
  if (recorder_aid_ != nullptr) {
    Async(*recorder_aid_, &RecorderActor::RecordOnStepEnd, context);
  }
  SendMemoryAllocReq(context);
}

void LoopCountActor::SendMemoryAllocReq(OpContext<DeviceTensor> *const context) {
  if (current_count_ == loop_count_) {
    // Need wait MemoryManagerActor running finished to avoid the illegal memory timing problem before
    // LoopCountActor exits, because other processors which are not in actor also will allocate or free memory.
    Async(memory_manager_aid_, &MemoryManagerActor::Wait, context, GetAID());
  } else if (continuous_memory_alloc_list_list_.size() > 0) {
    // Allocate continuous memory in the begin of next step running.
    Async(memory_manager_aid_, &MemoryManagerActor::AllocateContinuousMemory, &continuous_memory_alloc_list_list_,
          &size_list_list_, &total_size_list_, &device_contexts_, context, GetAID());
  } else {
    OnMemoryAllocFinish(context);
  }
}

void LoopCountActor::OnMemoryAllocFinish(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  // Send loop count to output actor.
  Async(output_aid_, &OutputActor::CollectLoopCount, current_count_, context);

  // The LoopCountActor exits.
  if (current_count_ == loop_count_) {
    current_count_ = 0;
    return;
  }

  // Send output control to trigger next step running.
  for (auto &data_source_aid : data_source_aids_) {
    Async(data_source_aid, &DataSourceActor::FetchData, context);
  }
  auto source_aid = const_cast<AID *>(&GetAID());
  for (auto &kernel_aid : no_input_kernel_aids_) {
    Async(kernel_aid, &KernelActor::RunOpControl, source_aid, context);
  }
}

bool LoopCountActor::CheckLoopCountIncreaseCondition(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  auto sequential_num = context->sequential_num_;

  return input_op_controls_[sequential_num].size() == input_controls_num_;
}
}  // namespace runtime
}  // namespace mindspore
