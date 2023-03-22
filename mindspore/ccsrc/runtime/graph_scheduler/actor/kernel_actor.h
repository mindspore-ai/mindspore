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
using mindspore::kernel::KernelLaunchInfo;
using mindspore::kernel::KernelMod;
using mindspore::session::SomasInfo;
using mindspore::tensor::TensorPtr;

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
      : DebugAwareActor(name, type, recorder_aid, memory_manager_aid, debug_aid),
        kernel_(kernel),
        is_dynamic_shape_(false),
        kernel_info_(nullptr),
        real_input_num_(0),
        strategy_(strategy),
        modifiable_ref_input_indexes_(modifiable_ref_input_indexes),
        modifiable_ref_output_indexes_(modifiable_ref_output_indexes),
        is_launch_skipped_(false),
        inputs_continuous_memory_(false),
        somas_info_(nullptr) {
    (void)device_contexts_.emplace_back(device_context);
  }
  ~KernelActor() override = default;

  // The memory related operation interface.
  void SendMemoryAllocReq(OpContext<DeviceTensor> *const context) override;
  void SendMemoryFreeReq(OpContext<DeviceTensor> *const context) override;
  // The callback after memory alloc finished.
  void OnMemoryAllocFinish(OpContext<DeviceTensor> *const context) override;

  // The debug related operation interface.
  void SendDebugReq(OpContext<DeviceTensor> *const context) override;
  // The callback after debug finished.
  void OnDebugFinish(OpContext<DeviceTensor> *const context) override;

  const CNodePtr &kernel() const { return kernel_; }
  const std::set<size_t> &modifiable_ref_input_indexes() const { return modifiable_ref_input_indexes_; }
  const std::set<size_t> &modifiable_ref_output_indexes() const { return modifiable_ref_output_indexes_; }
  bool is_dynamic_shape() const { return is_dynamic_shape_; }
  bool is_launch_skipped() const { return is_launch_skipped_; }
  bool inputs_continuous_memory() const { return inputs_continuous_memory_; }
  SomasInfo *somas_info() const { return somas_info_; }

 protected:
  void Init() override;
  void Run(OpContext<DeviceTensor> *const context) override;
  void SendRecorderInfo(OpContext<DeviceTensor> *const context) const override;

  // Do kernel launching in this method after 'PreLaunchKernel' and 'PostLaunchKernel'.
  virtual bool LaunchKernel(OpContext<DeviceTensor> *const context);

  // The info of kernel.
  CNodePtr kernel_;
  bool is_dynamic_shape_;
  KernelInfo *kernel_info_;
  KernelMod *kernel_mod_;
  // The kernel launch info is fetched by the device tensors.
  KernelLaunchInfo launch_info_;

  // The device tensors for launch.
  std::vector<DeviceTensor *> input_device_tensors_;
  std::vector<DeviceTensor *> output_device_tensors_;
  std::vector<DeviceTensor *> workspace_device_tensors_;
  // The received input device type and format may be different from the formal parameter in the control flow scenarios,
  // so it needs to be copied from the input data to real data that kernel launch needs.
  std::vector<DeviceTensorPtr> copy_input_device_tensors_;
  // Real data info that kernel launch needs, used to check the consistency of received input data.
  std::vector<std::shared_ptr<InputDataInfo>> real_input_data_infos_;

  // The device tensors for memory alloc and free.
  // output + workspace
  std::vector<DeviceTensor *> memory_alloc_list_;
  // input + output + workspace
  std::vector<DeviceTensor *> memory_free_list_;
  // The device tensor of external reference is not the real data of this kernel, but need add to the memory_free_list_.
  std::vector<DeviceTensor *> external_reference_tensors_;

 private:
  friend class GraphScheduler;
  friend class ControlNodeScheduler;
  friend class SchedulerHelper;
#ifdef ENABLE_RPC_ACTOR
  friend class RpcNodeScheduler;
#endif

  // Init the device tensors and kernel launch info.
  void InitInputInfo();
  void InitOutputInfo();
  void InitWorkspaceInfo();

  // Fetch the device tensor for launch.
  void FetchInputDeviceTensor(OpContext<DeviceTensor> *const context);
  void FetchOutputDeviceTensor(OpContext<DeviceTensor> *const context);
  void FetchWorkspaceDeviceTensor();
  // Need copy when the data type or format between real parameters and formal parameters are inconsistent.
  void CopyInputDeviceTensor(const OpData<DeviceTensor> *input_data, OpContext<DeviceTensor> *const context);
  // In step mode, push the input tensors which contain valid device address into input_device_tensors_ directly.
  void PushInputDeviceTensor(const std::vector<TensorPtr> *input_tensors);

  // The processing before kernel launch: update the info of kernel launch.
  void PreLaunchKernel(OpContext<DeviceTensor> *const context);
  // The processing after kernel launch: 1.erase input, 2.free memory, 3.send output.
  void PostLaunchKernel(OpContext<DeviceTensor> *const context);
  // Back refresh the dynamic device tensor stores that have been triggered copy.
  void RefreshDeviceTensorCopyStore(OpContext<DeviceTensor> *const context);

  // Set the memory address for the tensors which use the somas.
  void SetSomasMemory(OpContext<DeviceTensor> *const context) const;
  void *GetSomasDevicePtr(size_t offset) const;

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

  // Whether the inputs need continuous memory, used to check the inputs legitimacy.
  bool inputs_continuous_memory_;

  // The information used for integration of dynamic and static memory.
  SomasInfo *somas_info_;
};

using KernelActorPtr = std::shared_ptr<KernelActor>;
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_KERNEL_ACTOR_H_
