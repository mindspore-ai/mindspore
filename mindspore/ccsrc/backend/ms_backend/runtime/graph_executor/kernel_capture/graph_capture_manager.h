/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_KERNEL_CAPTURE_GRAPH_CAPTURE_MANAGER_H_
#define MINDSPORE_CCSRC_RUNTIME_KERNEL_CAPTURE_GRAPH_CAPTURE_MANAGER_H_

#include <vector>
#include <memory>
#include <utility>
#include <queue>
#include <map>
#include <tuple>
#include <string>
#include <sstream>
#include "backend/ms_backend/runtime/graph_executor/kernel_capture/capture_graph.h"
#include "backend/ms_backend/runtime/actors/base/kernel_runner.h"
#include "backend/ms_backend/runtime/graph_scheduler/base/graph_parameter_store.h"
#include "backend/ms_backend/runtime/actors/base/super_kernel_actor.h"

namespace mindspore {
namespace runtime {
// The GraphCaptureManager class is used to manage graph capture and replay functionality in kbk mode. It dynamically
// captures kernel launch operations during execution, translates them into a captured graph to sink execution.
// This class provides capabilities for graph capture, replay, and automatic graph partitioning.
using parameter_idx = std::pair<size_t, std::pair<KernelWithIndex, size_t>>;
using KernelRunnerWithIdx = std::pair<KernelRunnerPtr, size_t>;

struct CaptureKernelInfo {
  CaptureKernelInfo() : device_ptr(nullptr), size(0), shape(nullptr) {}
  CaptureKernelInfo(void *device_addr, size_t address_size, abstract::BaseShapePtr cur_shape)
      : device_ptr(device_addr), size(address_size), shape(cur_shape) {}
  void *device_ptr;
  size_t size;
  abstract::BaseShapePtr shape;
};
using CaptureKernelInfoPtr = std::shared_ptr<CaptureKernelInfo>;
using CaptureKernelInfoList = std::vector<CaptureKernelInfoPtr>;

class GraphCaptureManager {
 public:
  static GraphCaptureManager &GetInstance() noexcept;

  // Check whether enable graph capture.
  bool GetEnableGraphCapture() const;
  void SetEnableGraphCapture(bool enable_graph_capture);

  // Check a kernel can be captured or not.
  bool CheckKernelSupportCapture(const KernelRunnerPtr &kernel_runner, const DeviceContext *expected_device_context);

  // According to the execution order, find all operator interval and position that support capture.
  bool FindSupportCaptureKernelPositions(const std::vector<KernelRunnerPtr> &kernel_runners,
                                         const DeviceContext *expected_device_context);

  void Initialize(const DeviceContext *device_context);

  // Capture operators according to the execution order. Operators that are not supported for capture will be dispatched
  // immediately.
  bool LaunchAllKernelsWithCapture(OpContext<KernelTensor> *const context,
                                   const std::vector<KernelRunnerPtr> &kernel_runners,
                                   SuperKernelActor *super_kernel_actor, bool hp_mode);
  // Replay all captured sub graphs in series according to the execution order, or execute operators that cannot be
  // captured.
  bool LaunchAllKernelsWithReplayGraph(OpContext<KernelTensor> *const context,
                                       const std::vector<KernelRunnerPtr> &kernel_runners,
                                       SuperKernelActor *super_kernel_actor, bool hp_mode);

  void SetInReplay(bool in_replay) { in_replay_ = in_replay; }

  bool InReplay() const { return in_replay_; }

  void SetIncrementGraph(bool increment_graph) { increment_graph_ = increment_graph; }

  bool IncrementGraph() const { return increment_graph_; }

  // Before capture graph, process the inputs of all operators. For normal inputs, perform memory solidification
  // by constructing fix_addrs. Record the weights and kv_cache, which will be used during the subsequent replay phase
  // to verify whether there are any changes in the addresses.
  void FetchAllInputsBeforeCaptureGraph(OpContext<KernelTensor> *const context,
                                        const std::vector<KernelRunnerPtr> &kernel_runners,
                                        std::queue<std::vector<KernelTensorPtr>> *memory_free_lists);

  void FetchNonFixedInput(const KernelRunnerPtr &kernel_actor, OpContext<KernelTensor> *const context,
                          size_t stream_id);

  // Through D2D copy operations, update all the fixed ddresses recorded during the capture phase to ensure that
  // the addresses of all normal inputs are valid during the replay phase.
  void UpdateFixAddressBeforeReplayGraph(size_t stream_id, std::queue<std::vector<KernelTensorPtr>> *memory_free_lists);

  // In capture mode, record the op's outputs are graph outputs.
  void RecordGraphOutputKernelInfo(OpContext<KernelTensor> *const context, const KernelRunnerPtr &kernel_actor,
                                   size_t index);

  // In replay mode, recover the op's output device_ptr, shape and size.
  void RecoverGraphOutputKernelInfo();

  // Using the kv_cache and weight results recorded during the capture phase, verify whether the addresses
  // fetched during replay phase have changed.
  bool CheckParameterNotChange(size_t stream_id);

  void HandleFirstUserMemoryFree(const KernelTensorPtr &kernel_tensor, const KernelRunnerPtr &kernel_actor,
                                 std::queue<std::vector<KernelTensorPtr>> *memory_free_lists);

  bool IsNonFixedInput(GraphParameterStore *cur_graph_parameter_store, const AnfNodePtr &node, size_t parameter_idx);

  bool IsNonFixedInputInReplay(const KernelRunnerPtr &kernel_runner, size_t kernel_input_index);

  void SetShapeKey();

  const std::string &ShapeKey() const { return shape_key_; }

  bool IsInit() const { return init_; }

  bool HasCapturedGraph();

  // During the capture mode, before launch single op, record the state of its input, output, and workspace infos for
  // replay before its execution.
  void RecodeInfoForSingleOp(const KernelRunnerPtr &kernel_actor, size_t index);

  // During the replay mode, recover all the infos for single op, include input、output and workspace.
  void RecoverInfoForSingleOp(const KernelRunnerPtr &kernel_actor, size_t index);

  // During the replay mode, after launch single op, reset all the infos in related kernel tensors to avoid influence
  // in the next step.
  void ResetInfoForSingleOp(const std::vector<KernelRunnerPtr> &kernel_runners);

  void InitFixedInputInfoForSingleOp(const std::vector<KernelRunnerPtr> &kernel_runners);

  bool IsSingleOp(const std::vector<KernelRunnerPtr> &kernel_runners, size_t kernel_index);

  bool IsExceedMaxCaptureCount();

  void SetStreamId(size_t stream_id) {
    if (stream_id_ != 0) {
      MS_LOG(WARNING) << "Has set stream for capture graph";
      return;
    }
    stream_id_ = stream_id;
  }

  size_t GetStreamId() const { return stream_id_; }

  void Finalize();

 private:
  enum ExecutorType { CAPTURE_GRAPH = 0, KERNEL };

  GraphCaptureManager() = default;
  ~GraphCaptureManager() = default;
  DISABLE_COPY_AND_ASSIGN(GraphCaptureManager);

  CaptureGraphPtr capture_graph_{nullptr};
  std::map<std::string, std::vector<CaptureGraphPtr>> capture_graphs_;

  // Captured sub graph number.
  size_t capture_graph_num_ = 0;

  // Record all operator interval and position that support capture according to the execution order.
  std::vector<std::pair<size_t, size_t>> capture_kernel_range_positions_;
  // Record all captured sub graphs and kernels that don't support capture, according to the execution order.
  std::vector<std::pair<ExecutorType, size_t>> executors_;

  std::map<std::string, std::vector<std::tuple<parameter_idx, KernelTensorPtr, KernelRunnerPtr>>>
    fixed_addrs_for_update_;

  std::map<std::string, std::map<KernelWithIndex, KernelTensorPtr>> fixed_addrs_for_set_inputs_;

  std::map<std::string, std::map<KernelWithIndex, std::tuple<KernelTensorPtr, size_t, KernelRunnerPtr>>>
    weight_kv_addrs_;

  // Only used for the ops can not be captured.
  std::map<std::string, std::map<KernelRunnerWithIdx, CaptureKernelInfoList>> fix_single_op_input_info_;
  std::map<std::string, std::map<KernelRunnerWithIdx, CaptureKernelInfoList>> fix_single_op_output_info_;
  std::map<std::string, std::map<KernelRunnerWithIdx, CaptureKernelInfoList>> fix_single_op_workspace_info_;

  std::map<std::string, std::map<KernelRunnerWithIdx, CaptureKernelInfoList>> fix_replay_graph_output_info_;

  std::map<std::string, std::map<KernelRunnerWithIdx, std::vector<KernelTensorPtr>>>
    fix_network_input_for_replay_single_op_;

  mindspore::HashMap<std::string, std::vector<std::pair<KernelRunnerWithIdx, std::vector<size_t>>>>
    recorded_kernel_output_for_graph_output_;

  std::string shape_key_{""};

  bool init_{false};

  bool in_replay_{false};
  bool increment_graph_{false};
  size_t stream_id_{0};
};
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_KERNEL_CAPTURE_GRAPH_CAPTURE_MANAGER_H_
