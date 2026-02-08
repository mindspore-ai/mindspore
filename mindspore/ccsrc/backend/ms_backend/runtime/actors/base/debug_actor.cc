/**
 * Copyright 2021-2025 Huawei Technologies Co., Ltd
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

#include "backend/ms_backend/runtime/actors/base/debug_actor.h"
#include <vector>
#include <memory>
#include <string>
#include "utils/log_adapter.h"
#include "include/utils/callback.h"

namespace mindspore {
namespace runtime {
void DebugActor::DebugPreLaunch(const AnfNodePtr &node, const std::vector<KernelTensorPtr> &input_kernel_tensors,
                                const std::vector<KernelTensorPtr> &output_kernel_tensors,
                                const DeviceContext *device_context, OpContext<KernelTensor> *const op_context,
                                const AID *) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(op_context);
}

/*
 * Feature group: Dump, Online debugger.
 * Target device group: GPU.
 * Runtime category: MindRT.
 * Description: Load and read data for the given node if needed. Dump the node if dump is enabled and free the loaded
 * memory after the dump (for GPU and ascend kernel-by-kernel).
 */
void DebugActor::DebugPostLaunch(const AnfNodePtr &node, const std::vector<KernelTensorPtr> &input_kernel_tensors,
                                 const std::vector<KernelTensorPtr> &output_kernel_tensors,
                                 const DeviceContext *device_context, OpContext<KernelTensor> *const op_context,
                                 const AID *) {
  constexpr char kDebugPostLaunch[] = "DebugPostLaunch";
  static const auto debug_post_launch =
    callback::CommonCallback::GetInstance()
      .GetCallback<void, const AnfNodePtr &, const std::vector<KernelTensorPtr> &, const std::vector<KernelTensorPtr> &,
                   const DeviceContext *>(kDebugPostLaunch);
  MS_EXCEPTION_IF_CHECK_FAIL(debug_post_launch, "Failed to get DebugPostLaunch");
  debug_post_launch(node, input_kernel_tensors, output_kernel_tensors, device_context);
}

/*
 * Feature group: Dump, Online debugger.
 * Target device group: Ascend, GPU.
 * Runtime category: MindRT.
 * Description: Checks dataset_sink_mode and generates the related error if any exist and calls
 * PreExecuteGraphDebugger.
 */
void DebugActor::DebugOnStepBegin(const std::vector<KernelGraphPtr> &graphs,
                                  const std::vector<AnfNodePtr> &origin_parameters_order,
                                  std::vector<DeviceContext *> device_contexts,
                                  OpContext<KernelTensor> *const op_context, const AID *) {
  constexpr char kDebugOnStepBegin[] = "DebugOnStepBegin";
  static const auto debug_on_step_begin =
    callback::CommonCallback::GetInstance()
      .GetCallback<void, const std::vector<KernelGraphPtr> &, const std::vector<AnfNodePtr> &,
                   std::vector<DeviceContext *>>(kDebugOnStepBegin);
  MS_EXCEPTION_IF_CHECK_FAIL(debug_on_step_begin, "Failed to get DebugOnStepBegin");
  debug_on_step_begin(graphs, origin_parameters_order, device_contexts);
}

/*
 * Feature group: Dump, Online debugger.
 * Target device group: Ascend, GPU and CPU.
 * Runtime category: MindRT.
 * Description: Dump parameters and constants and update dump iter for CPU. Call PostExecuteGraph Debugger for GPU and
 * Ascend and update step number of online debugger GPU.
 */
void DebugActor::DebugOnStepEnd(OpContext<KernelTensor> *const, const AID *, int total_running_count,
                                std::vector<const DeviceContext *> device_contexts) {
  constexpr char kDebugOnStepEnd[] = "DebugOnStepEnd";
  static const auto debug_on_step_end =
    callback::CommonCallback::GetInstance().GetCallback<void, int, std::vector<const DeviceContext *>>(kDebugOnStepEnd);
  MS_EXCEPTION_IF_CHECK_FAIL(debug_on_step_end, "Failed to get DebugOnStepEnd");
  debug_on_step_end(total_running_count, device_contexts);
}

void DebugActor::Finalize() {
  constexpr char kDebugFinalize[] = "DebugFinalize";
  static const auto debug_finalize = callback::CommonCallback::GetInstance().GetCallback<void>(kDebugFinalize);
  MS_EXCEPTION_IF_CHECK_FAIL(debug_finalize, "Failed to get DebugFinalize");
  debug_finalize();
}
}  // namespace runtime
}  // namespace mindspore
