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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_KERNEL_ADJUST_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_KERNEL_ADJUST_H_

#include <memory>
#include <map>
#include <string>
#include <vector>
#include <unordered_set>
#include "ir/anf.h"
#include "backend/common/session/kernel_graph.h"
#include "kernel/kernel_build_info.h"
#include "backend/common/session/session_context.h"
#include "ir/tensor.h"
#include "runtime/device/kernel_info.h"
#include "runtime/device/kernel_runtime_manager.h"

#ifndef ENABLE_SECURITY
#include "plugin/device/ascend/hal/device/profiling/profiling_utils.h"
using mindspore::device::ascend::ProfilingTraceInfo;
using mindspore::device::ascend::ProfilingUtils;
#endif
namespace mindspore {
// device loop control
constexpr auto kCurLoopCountName = "current_loop_count";
constexpr auto kNextLoopCountName = "next_loop_count";
constexpr auto kCurEpochCountName = "current_epoch_count";
constexpr auto kConstOneName = "const_one";
constexpr auto kConstLoopNumInEpochName = "const_loop_num_in_epoch";

constexpr auto kStreamNeedActivedFirst = "stream_need_active_first";
enum StreamSwitchKind {
  kFpBpStreamSwitch = 0,
  kGetNextStreamSwitch = 1,
  kEosStreamSwitch = 2,
  kIndependentStreamSwitch = 3
};

namespace device {
class KernelAdjust {
 public:
  static KernelAdjust &GetInstance() {
    static KernelAdjust instance;
    return instance;
  }
  // device loop control
  void InsertDeviceLoopCtrl(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr);
  void AssignLoopCtrlMemory(const session::KernelGraph &kernel_graph_ptr);
  void LoadDeviceLoopCtrlParameters(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr);

  void InsertOverflowCheckOperations(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr);
  void ProcessLoopSink(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr);
#ifndef ENABLE_SECURITY
  void Profiling(NotNull<session::KernelGraph *> kernel_graph_ptr);
#endif
  static bool NeedLoopSink();
  CNodePtr CreateStreamActiveOp(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr);
  CNodePtr CreateRecvApplyKernel(const std::shared_ptr<session::KernelGraph> &graph_ptr, uint32_t event_id);
  CNodePtr CreateSendApplyKernel(const std::shared_ptr<session::KernelGraph> &graph_ptr, uint32_t event_id);

 private:
  KernelAdjust() = default;
  ~KernelAdjust() = default;

  CNodePtr CreateNPUGetFloatStatus(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr,
                                   const CNodePtr &npu_cnode);
  CNodePtr CreateNPUClearStatus(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr,
                                const CNodePtr &npu_cnode);
  CNodePtr CreateNPUAllocStatus(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr);
  CNodePtr CreateAssignAdd(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr, const CNodePtr &npu_get_cnode,
                           const AnfNodePtr &specify_para);
  CNodePtr CreateAssign(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr, const AnfNodePtr &specify_para);
  void ReorderGetNext(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr);
  CNodePtr CreateStreamSwitchOp(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr,
                                const std::map<std::string, mindspore::ParameterPtr> &switch_loop_input,
                                StreamSwitchKind kind);

  CNodePtr CreatTupleGetItemNode(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr, const CNodePtr &node,
                                 size_t output_idx);
  CNodePtr CreateEndOfSequenceOP(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr,
                                 const CNodePtr &getnext_cnode);
  CNodePtr CreateStreamAssignAddnOP(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr,
                                    const std::map<std::string, mindspore::ParameterPtr> &switch_loop_input,
                                    bool cur_loop);
  kernel::KernelBuildInfo::KernelBuildInfoBuilder CreateMngKernelBuilder(const std::vector<std::string> &formats,
                                                                         const std::vector<TypeId> &type_ids);
#ifndef ENABLE_SECURITY
  void InsertProfilingKernel(const ProfilingTraceInfo &profiling_trace_info,
                             NotNull<session::KernelGraph *> kernel_graph_ptr);
#endif
  bool ExistIndependent(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr);
  bool ExistGetNext(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr);
  void InsertGetNextLoopStreamSwitch(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr,
                                     std::vector<CNodePtr> *exec_order, uint32_t *getnext_switch_stream_id,
                                     uint32_t *getnext_stream_id,
                                     const std::map<std::string, mindspore::ParameterPtr> &switch_loop_input);
  void SetBeforeGetNextStreamID(std::vector<CNodePtr> *exec_order, const std::vector<CNodePtr> &orders,
                                size_t *order_index, CNodePtr getnext_cnode, uint32_t getnext_stream_id);
  void InsertGetNextLoopFpBpStartSend(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr,
                                      std::vector<CNodePtr> *exec_order, uint32_t *fpbp_start_event_id,
                                      uint32_t getnext_stream_id);
  void InsertGetNextLoopEosStartSend(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr,
                                     std::vector<CNodePtr> *exec_order, uint32_t *eos_start_event_id,
                                     uint32_t getnext_stream_id);
  void InsertEosStreamSwitch(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr,
                             const std::map<std::string, mindspore::ParameterPtr> &switch_loop_input,
                             std::vector<CNodePtr> *exec_order, uint32_t *eos_switch_stream_id,
                             uint32_t *eos_stream_id);
  void InsertGetNextLoopEosStartRecv(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr,
                                     std::vector<CNodePtr> *exec_order, uint32_t eos_start_event_id,
                                     uint32_t eos_stream_id);
  void InsertEosOp(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr, std::vector<CNodePtr> *exec_order,
                   const CNodePtr &getnext_cnode, uint32_t eos_stream_id);
  void InsertEosDoneSend(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr,
                         std::vector<CNodePtr> *exec_order, uint32_t *eos_done_event_id, uint32_t eos_stream_id);
  void InsertIndepentParallel(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr,
                              const std::map<std::string, mindspore::ParameterPtr> &switch_loop_input,
                              std::vector<CNodePtr> *exec_order);
  void InsertFpBpLoopStreamSwitch(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr,
                                  const std::map<std::string, mindspore::ParameterPtr> &switch_loop_input,
                                  std::vector<CNodePtr> *exec_order, uint32_t *fpbp_stream_id,
                                  uint32_t *fpbp_switch_stream_id);
  void InsertFpBpStartRecv(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr,
                           std::vector<CNodePtr> *exec_order, uint32_t fpbp_start_event_id, uint32_t fpbp_stream_id);
  void InsertNextLoopAssignAdd(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr,
                               std::vector<CNodePtr> *exec_order,
                               const std::map<std::string, mindspore::ParameterPtr> &switch_loop_input,
                               uint32_t fpbp_stream_id);
  void CopyMemcpyList(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr,
                      const std::vector<CNodePtr> &orders, size_t order_index, std::vector<CNodePtr> *memcpy_list,
                      std::vector<CNodePtr> *other_list);
  void InsertEosDoneRecv(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr,
                         std::vector<CNodePtr> *exec_order, uint32_t eos_done_event_id, uint32_t fpbp_stream_id);
  void InsertGetNextLoopStreamActive(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr,
                                     std::vector<CNodePtr> *exec_order,
                                     const std::vector<uint32_t> &getnext_active_streams);
  void InsertCurrentLoopAssignAdd(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr,
                                  std::vector<CNodePtr> *exec_order,
                                  const std::map<std::string, mindspore::ParameterPtr> &switch_loop_input);
  void InsertFpBpAndEosLoopStreamActive(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr,
                                        std::vector<CNodePtr> *exec_order,
                                        const std::vector<uint32_t> &fpbp_active_streams);
  void SetDeviceLoopCtrlTensor(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr, const string name,
                               int32_t value);
  void AssignLoopCtrlTensorMem(const session::KernelGraph &kernel_graph, KernelRuntime *runtime_instance,
                               const string name);
  std::shared_ptr<Tensor> CreateTensor(int32_t initial_value);
  std::shared_ptr<Parameter> CreateParameter(const std::shared_ptr<session::KernelGraph> &kernel_graph_ptr,
                                             const string parameter_name);
};
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_KERNEL_ADJUST_H_
