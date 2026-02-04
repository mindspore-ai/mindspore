/**
 * Copyright 2023-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_HARDWARE_ASCEND_KERNEL_EXECUTOR_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_HARDWARE_ASCEND_KERNEL_EXECUTOR_H_

#include <vector>
#include <memory>
#include <string>
#include <set>
#include <map>
#include <utility>
#include <tuple>
#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "include/utils/anfalgo.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"

namespace mindspore {
namespace device {
namespace ascend {
class AscendKernelExecutor : public KernelExecutor {
 public:
  AscendKernelExecutor() = default;
  ~AscendKernelExecutor() override = default;

  void Initialize() override;
  void Destroy() override;

  // Optimize the kernel graph for graph mode.
  void OptimizeGraph(const FuncGraphPtr &graph) const override;

  // Generate 'KernelMod' for all kernels and set 'KernelMod' into kernel,
  // 'KernelMod' is real executive object of kernel.
  void CreateKernel(const std::vector<CNodePtr> &nodes) const override;

  // Generate 'KernelMod' by operator name.
  // Note: Only support generate aclnn kernel mod current.
  kernel::KernelModPtr CreateKernelMod(const std::string &op_name) const override;

  int StressDetect(const std::string &detect_type) const override;
  int CleanTdtChannel() const override;
  // Adjust kernel graph before run graph, used in Graph Mode.
  void PreprocessBeforeRun(const FuncGraphPtr &graph) const override;

  // Create event for graph from cache.
  void CreateEventForCache(const KernelGraphPtr &kernel_graph) const override;

  // Launch a kernel via 'KernelMod' of the kernel.
  bool LaunchKernel(const CNodePtr &kernel, const std::vector<KernelTensor *> &inputs,
                    const std::vector<KernelTensor *> &workspace, const std::vector<KernelTensor *> &outputs,
                    KernelMod *kernel_mod, void *stream) const override;
  // This is a high performance version of 'LaunchKernel', which will be called in performance-critical scenario.
  bool LaunchKernelHP(const CNodePtr &kernel, const std::vector<KernelTensor *> &inputs,
                      const std::vector<KernelTensor *> &workspace, const std::vector<KernelTensor *> &outputs,
                      KernelMod *kernel_mod, void *stream) const override;
  void AddMindIRPass(const KernelGraphPtr &graph) const override;
  void OptimizeExecutionOrder(const FuncGraphPtr &graph) const;

  bool ExecuteKernelTask(const runtime::KernelTaskType &task_type, const tensor::TensorPtrList &input_tensors,
                         const tensor::TensorPtrList &output_tensors, const size_t &stream_id) const override;

  bool ExecuteKernelTask(const runtime::KernelTaskType &task_type,
                         const std::vector<KernelTensor *> &input_kernel_tensors,
                         const std::vector<KernelTensor *> &output_kernel_tensors,
                         const size_t &stream_id) const override;

  std::vector<size_t> GetLaunchIgnoredInputAddressIdx(const AnfNodePtr &node) const {
    MS_EXCEPTION_IF_NULL(node);
    auto input_num = common::AnfAlgo::GetInputTensorNum(node);
    std::vector<size_t> ignore_input_list;
    for (size_t input_idx = 0; input_idx < input_num; ++input_idx) {
      if (AnfAlgo::IsLaunchIgnoredInputAddressIdx(node, input_idx)) {
        ignore_input_list.emplace_back(input_idx);
      }
    }
    return ignore_input_list;
  }

  bool IsLaunchIgnoredInputAddressIdx(const AnfNodePtr &node, size_t input_idx) const {
    return AnfAlgo::IsLaunchIgnoredInputAddressIdx(node, input_idx);
  }

 private:
  static void DoSomas(const FuncGraphPtr &graph);
  void DoStreamAssign(const KernelGraphPtr &kernel_graph) const;
  // launch
  bool MemoryCopyAsync(const CNodePtr &node, const std::vector<KernelTensor *> &inputs,
                       const std::vector<KernelTensor *> &outputs, void *stream) const;
  bool LaunchCallback(CallbackFunc callback_func, size_t stream_id, bool is_block) const;

  mutable std::set<CNodePtr> nop_op_to_memcpy_;
  DeviceResManager *res_manager_{nullptr};
  bool initialized_ = false;
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_HARDWARE_ASCEND_KERNEL_EXECUTOR_H_
