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

#ifndef MINDSPORE_MINDSPORE_CCSRC_DEVICE_KERNEL_ADJUST_H_
#define MINDSPORE_MINDSPORE_CCSRC_DEVICE_KERNEL_ADJUST_H_

#include <memory>
#include <map>
#include <string>
#include <vector>
#include <unordered_set>
#include "ir/anf.h"
#include "session/kernel_graph.h"
#include "kernel/kernel_build_info.h"
#include "session/session_context.h"
#include "ir/meta_tensor.h"
#include "device/ascend/profiling/profiling_utils.h"

using mindspore::device::ascend::ProfilingTraceInfo;
using mindspore::device::ascend::ProfilingUtils;
namespace mindspore {
namespace device {
class KernelAdjust {
 public:
  static KernelAdjust &GetInstance() {
    static KernelAdjust instance;
    return instance;
  }
  void Reorder(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr);
  void InsertSwitchLoop(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr);
  void SetStreamActiveOPs(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr,
                          const std::unordered_set<uint32_t> &ctrl_stream_list,
                          const std::unordered_set<uint32_t> &comm_stream_list,
                          const std::unordered_set<uint32_t> &momentum_stream_list);
  void SetStreamSwitchOps(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr);
  bool StepLoadCtrlInputs(const std::shared_ptr<session::Context> &context,
                          const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr);
  void Profiling(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr);
  static bool NeedInsertSwitch();
  CNodePtr CreateSteamActiveOp(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr);

 private:
  KernelAdjust() = default;
  ~KernelAdjust() = default;
  void CreateSwitchOpParameters(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr,
                                std::map<std::string, mindspore::ParameterPtr> *switch_loop_input);
  CNodePtr CreateStreamSwitchOp(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr,
                                const std::map<std::string, mindspore::ParameterPtr> &switch_loop_input);
  CNodePtr CreateStreamActiveSwitchOp(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr);
  CNodePtr CreateStreamActiveOtherOp(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr);
  CNodePtr CreateStreamAssignAddnOP(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr,
                                    const std::map<std::string, mindspore::ParameterPtr> &switch_loop_input);
  kernel::KernelBuildInfo::KernelBuildInfoBuilder CreateMngKernelBuilder(const std::vector<std::string> &formats,
                                                                         const std::vector<TypeId> &type_ids);
  void LoadSwitchInputs(std::vector<tensor::TensorPtr> *inputs);
  void InsertProfilingKernel(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr,
                             const ProfilingTraceInfo &profiling_trace_info);
};
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_DEVICE_KERNEL_ADJUST_H_
