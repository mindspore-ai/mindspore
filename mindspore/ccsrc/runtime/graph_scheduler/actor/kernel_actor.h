/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_KERNEL_ACTOR_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_KERNEL_ACTOR_H_

#include <vector>
#include <set>
#include <string>
#include <memory>
#include <utility>
#include "utils/hash_map.h"
#include "runtime/graph_scheduler/actor/actor_common.h"
#include "runtime/graph_scheduler/actor/debug_aware_actor.h"
#include "runtime/graph_scheduler/actor/kernel_async_launch_actor.h"
#include "runtime/graph_scheduler/actor/kernel_async_infer_actor.h"
#include "runtime/graph_scheduler/actor/kernel_async_resize_actor.h"
#include "runtime/hardware/device_context.h"
#include "runtime/graph_scheduler/device_tensor_store.h"
#include "kernel/kernel.h"
#include "ir/anf.h"
#include "ir/tensor.h"

namespace mindspore {
namespace runtime {
using mindspore::device::DeviceContext;
using mindspore::device::KernelInfo;
using mindspore::kernel::Address;
using mindspore::kernel::KernelLaunchAddr;
using mindspore::kernel::KernelMod;
using mindspore::kernel::KernelTensor;
using mindspore::kernel::KernelTensorPtr;
using mindspore::session::SomasInfo;
using mindspore::tensor::TensorPtr;

class SuperKernelActor;

struct InputDataInfo {
  InputDataInfo(const std::string &format, const ShapeVector &shape, size_t size, TypeId type_id)
      : format_(format), shape_(shape), size_(size), type_id_(type_id) {}
  std::string format_;
  ShapeVector shape_;
  size_t size_;
  TypeId type_id_;
};

// The kernel actor is used to receive the device tensors and control info to luanch kernel.
// The processing flow is RunOpData/RunOpControl -> CheckRunningCondition -> SendMemoryAllocReq
// -> OnMemoryAllocFinish -> LaunchKernel -> SendMemoryFreeReq -> SendOutput.
class KernelActor : public DebugAwareActor {
 public:
  KernelActor(const std::string &name, const CNodePtr &kernel, const DeviceContext *device_context,
              const AID &memory_manager_aid, const AID *debug_aid, const AID *recorder_aid,
              GraphExecutionStrategy strategy, const std::set<size_t> &modifiable_ref_input_indexes,
              const std::set<size_t> &modifiable_ref_output_indexes,
              const KernelTransformType &type = KernelTransformType::kKernelActor)
      : DebugAwareActor(name, type, recorder_aid, memory_manager_aid, debug_aid, nullptr),
        kernel_(kernel),
        is_dynamic_value_(false),
        is_dynamic_type_(false),
        has_dynamic_(false),
        enable_async_infer_(false),
        kernel_info_(nullptr),
        kernel_mod_(nullptr),
        somas_info_(nullptr),
        real_input_num_(0),
        strategy_(strategy),
        modifiable_ref_input_indexes_(modifiable_ref_input_indexes),
        modifiable_ref_output_indexes_(modifiable_ref_output_indexes),
        is_launch_skipped_(false),
        inputs_continuous_memory_(false) {
    (void)device_contexts_.emplace_back(device_context);
    is_dynamic_shape_ = common::AnfAlgo::IsDynamicShape(kernel_) || common::AnfAlgo::IsDynamicSequence(kernel_);

    kernel_async_infer_aid_ = KernelAsyncInferActor::GetInstance()->GetAID();
    kernel_async_resize_aid_ = KernelAsyncResizeActor::GetInstance()->GetAID();
    kernel_async_launch_aid_ = KernelAsyncLaunchActor::GetInstance()->GetAID();
  }

  ~KernelActor() override = default;

  // The memory related operation interface.
  void SendMemoryAllocReq(OpContext<DeviceTensor> *const context) override;
  void SendMemoryFreeReq(OpContext<DeviceTensor> *const context) override;
  // The callback after memory alloc finished.
  void OnMemoryAllocFinish(OpContext<DeviceTensor> *const context) override;

  const CNodePtr &kernel() const { return kernel_; }
  const std::set<size_t> &modifiable_ref_input_indexes() const { return modifiable_ref_input_indexes_; }
  const std::set<size_t> &modifiable_ref_output_indexes() const { return modifiable_ref_output_indexes_; }
  bool is_dynamic_shape() const { return is_dynamic_shape_; }
  bool is_launch_skipped() const { return is_launch_skipped_; }
  bool inputs_continuous_memory() const { return inputs_continuous_memory_; }
  SomasInfo *somas_info() const { return somas_info_; }
  const std::set<size_t> &somas_graph_output_indexes() const { return somas_graph_output_indexes_; }

  void set_enable_async_infer(bool enable_async_infer) { enable_async_infer_ = enable_async_infer; }

  // Really do infer shape and update kernel tensor shape.
  void ExecuteInferShapeTask(OpContext<DeviceTensor> *const context);
  // Really do resize kernel mod and update new size into output and workspace kernel tensors.
  void ExecuteResizeKernelModTask(OpContext<DeviceTensor> *const context);
  // Really do launch kernel with memory allocate and free.
  void ExecuteLaunchKernelTask(OpContext<DeviceTensor> *const context);

  void set_stream_send_actor(KernelActor *stream_send_actor) { stream_send_actor_ = stream_send_actor; }

  void SetInputDeviceTensor(DeviceTensor *input_device_tensor, size_t input_index);

  // Set the memory address for the tensors which use the somas.
  void SetSomasMemory(OpContext<DeviceTensor> *const context) const;

  bool skip_launch_shape_related_op() const { return skip_launch_shape_related_op_; }
  void set_skip_launch_shape_related_op(bool skip_launch_shape_related_op) {
    skip_launch_shape_related_op_ = skip_launch_shape_related_op;
  }

 protected:
  void Init() override;
  void Run(OpContext<DeviceTensor> *const context) override;
  void SendRecorderInfo(OpContext<DeviceTensor> *const context) const override;

  // Do kernel launching in this method after 'PreLaunchKernel' and 'PostLaunchKernel'.
  virtual bool LaunchKernel(OpContext<DeviceTensor> *const context);
  // Execute kernel actor multi stream produre to make sure safety of memory before kernel launch.
  virtual void ProcessMultiStreamBeforeKernelLaunch(OpContext<DeviceTensor> *const context);
  // Execute kernel actor multi stream produre to make sure safety of memory after kernel launch.
  virtual void ProcessMultiStreamAfterKernelLaunch(OpContext<DeviceTensor> *const context);

  // Execute infer shape, resize and launch kernel by runtime pipeline which executes by KernelAsyncInferActor,
  // KernelAsyncResizeActor and KernelAsyncLaunchActor.
  void RunWithMultiPipeline(OpContext<DeviceTensor> *const context);
  // Execute launch kernel asynchronously in KernelAsyncLaunchActor.
  void RunWithAsyncLaunchKernel(OpContext<DeviceTensor> *const context);

  // Infer shape(and type) and resize kernel mod.
  void InferAndResize(OpContext<DeviceTensor> *const context);

  // Re-Infer shape, type and resize before kernel launch in dynamic scenarios.
  void InferShapeAndType();

  // Re-InferShape and resize before kernel launch in dynamic scenarios.
  void InferShape();

  void ResizeKernelMod();

  // Update input_device_tensors by input op data.
  void UpdateInputDeviceTensor(const OpData<DeviceTensor> *input_data, OpContext<DeviceTensor> *const context);

  // Record the output and workspace memory pointer and size to optimize memory allocate/free performance in next step.
  // Note: only use in inference case.
  void TraceDynamicMemory();

  // The info of kernel.
  CNodePtr kernel_;
  bool is_dynamic_shape_;
  bool is_dynamic_value_;
  bool is_dynamic_type_;
  bool has_dynamic_;
  // Whether enable asynchronously infer shape and resize kernel mod by KernelInferActor and KernelResizeActor.
  bool enable_async_infer_;
  AID kernel_async_infer_aid_;
  AID kernel_async_resize_aid_;
  AID kernel_async_launch_aid_;
  KernelInfo *kernel_info_;
  KernelMod *kernel_mod_;

  // The device tensors for launch.
  std::vector<DeviceTensor *> input_device_tensors_;
  std::vector<DeviceTensor *> output_device_tensors_;
  std::vector<DeviceTensor *> workspace_device_tensors_;

  std::vector<DeviceTensor *> max_ref_cnt_output_list_;

  // The input kernel tensors for infer shape.
  std::vector<abstract::AbstractBasePtr> input_kernel_tensors_for_infer_;
  // The kernel tensors for resize and launch.
  std::vector<KernelTensor *> input_kernel_tensors_;
  std::vector<KernelTensor *> output_kernel_tensors_;
  std::vector<KernelTensor *> workspace_kernel_tensors_;

  // The received input device type and format may be different from the formal parameter in the control flow
  // scenarios, so it needs to be copied from the input data to real data that kernel launch needs.
  std::vector<DeviceTensorPtr> copy_input_device_tensors_;
  // Real data info that kernel launch needs, used to check the consistency of received input data.
  std::vector<std::shared_ptr<InputDataInfo>> real_input_data_infos_;

  // The device tensors for memory alloc and free.
  // output + workspace
  std::vector<DeviceTensor *> memory_alloc_list_;
  // input + output + workspace
  std::vector<DeviceTensor *> memory_free_list_;
  // depend shape input list
  std::vector<bool> depend_shape_input_list_;
  // The device tensor of external reference is not the real data of this kernel, but need add to the
  // memory_free_list_.
  std::vector<DeviceTensor *> external_reference_tensors_;

  // The information used for integration of dynamic and static memory.
  SomasInfo *somas_info_;
  // The graph output node and index use somas info.
  std::set<size_t> somas_graph_output_indexes_;
  // Task id on stream, use for events.
  std::shared_ptr<int64_t> task_id_on_stream_ = std::make_shared<int64_t>(0L);
  // Send actor ref, point to the send actor when current actor is recv actor.
  KernelActor *stream_send_actor_{nullptr};
  // Flag for stream recv actor.
  bool is_stream_recv_actor_{false};
  // Flag for indicating if current actor is multi-thread safe, which was generate at compile time.
  bool is_multi_stream_safe_{false};

 private:
  friend class GraphScheduler;
  friend class ControlNodeScheduler;
  friend class InlineControlFlowScheduler;
  friend class SchedulerHelper;
#ifdef ENABLE_RPC_ACTOR
  friend class RpcNodeScheduler;
#endif
  friend class SuperKernelActor;

  // Init the device tensors and kernel launch info.
  void InitInputInfo();
  void InitOutputInfo();
  void InitWorkspaceInfo();
  void InitShapeDependInfo();

  // Fetch the device tensor for launch.
  void FetchInputDeviceTensor(OpContext<DeviceTensor> *const context);
  void FetchOutputDeviceTensor(OpContext<DeviceTensor> *const context);
  void FetchWorkspaceDeviceTensor();
  // Need copy when the data type or format between real parameters and formal parameters are inconsistent.
  void CopyInputDeviceTensor(const OpData<DeviceTensor> *input_data, OpContext<DeviceTensor> *const context);

  // The processing before kernel launch: update the info of kernel launch.
  void PreLaunchKernel(OpContext<DeviceTensor> *const context);
  // The processing after kernel launch: 1.erase input, 2.free memory, 3.send output.
  void PostLaunchKernel(OpContext<DeviceTensor> *const context);
  // Back refresh the dynamic device tensor stores that have been triggered copy.
  void RefreshDeviceTensorCopyStore(OpContext<DeviceTensor> *const context);

  void *GetSomasDevicePtr(size_t offset) const;

  // Record mem info, because async send may free device info.
  void SetMemInfoForDebugAndRdr();

  // The real input number of kernel launch.
  size_t real_input_num_;

  // The execution strategy of kernel actor.
  // In pipeline mode, kernel actor executes asynchronously.
  // In step mode, kernel actor executes synchronously.
  GraphExecutionStrategy strategy_{GraphExecutionStrategy::kPipeline};

  // Record the modifiable ref indexes. Used to refresh the ref data which are modified in the running.
  std::set<size_t> modifiable_ref_input_indexes_;
  std::set<size_t> modifiable_ref_output_indexes_;

  // Whether skip the kernel launch.
  bool is_launch_skipped_;

  // Recoreded mem info.
  KernelLaunchAddr mem_info_;

  // The ignore input addresses when the kernel launch.
  std::vector<size_t> launch_ignored_inputs_;

  // Whether the inputs need continuous memory, used to check the inputs legitimacy.
  bool inputs_continuous_memory_;

  // The stream resource of the KernelActor to launch kernel.
  void *stream_{nullptr};

  bool is_multi_stream_process_skipped_{false};
  std::vector<std::pair<uint32_t, void *>> cross_stream_addresses_;

  // Flag for skipping launch shape related operator, such as RealMakeTuple.
  // RealMakeTuple --> ShapeCalc pattern: if ShapeCalc is not value depend for one input RealMakeTuple op, we can skip
  // launch this RealMakeTuple.
  bool skip_launch_shape_related_op_{false};

  bool is_output_kernel_{false};
};

using KernelActorPtr = std::shared_ptr<KernelActor>;
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_KERNEL_ACTOR_H_
