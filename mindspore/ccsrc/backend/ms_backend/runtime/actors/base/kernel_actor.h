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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_KERNEL_ACTOR_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_KERNEL_ACTOR_H_

#include <vector>
#include <set>
#include <string>
#include <map>
#include <memory>
#include <utility>

#include "utils/hash_map.h"
#include "backend/ms_backend/runtime/actors/base/actor_common.h"
#include "backend/ms_backend/runtime/actors/base/debug_aware_actor.h"
#include "backend/ms_backend/runtime/actors/base/kernel_async_launch_actor.h"
#include "backend/ms_backend/runtime/actors/dynamic_shape/kernel_async_infer_actor.h"
#include "backend/ms_backend/runtime/actors/dynamic_shape/kernel_async_resize_actor.h"
#include "backend/ms_backend/runtime/graph_scheduler/base/parameter_store.h"
#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "include/runtime/hardware_abstract/kernel_base/device_tensor_store.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"
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
using mindspore::runtime::InputDataInfo;
using mindspore::runtime::KernelLaunchInfoWithStream;
using mindspore::session::SomasInfo;
using mindspore::tensor::TensorPtr;

class SuperKernelActor;

// The kernel actor is used to receive the device tensors and control info to luanch kernel.
// The processing flow is RunOpData/RunOpControl -> CheckRunningCondition -> SendMemoryAllocReq
// -> OnMemoryAllocFinish -> LaunchKernel -> SendMemoryFreeReq -> SendOutput.
class KernelActor : public DebugAwareActor {
 public:
  KernelActor(const std::string &name, const CNodePtr &kernel, const DeviceContext *device_context,
              const AID &memory_manager_aid, const AID *debug_aid, const AID *recorder_aid,
              GraphExecutionStrategy strategy, const std::set<size_t> &modifiable_ref_input_indexes,
              const std::set<size_t> &modifiable_ref_output_indexes,
              const KernelTransformType &type = KernelTransformType::kKernelActor);

  ~KernelActor() override = default;

  // The memory related operation interface.
  void SendMemoryAllocReq(OpContext<KernelTensor> *const context) override;
  void SendMemoryFreeReq(OpContext<KernelTensor> *const context) override;
  // The callback after memory alloc finished.
  void OnMemoryAllocFinish(OpContext<KernelTensor> *const context) override;

  const CNodePtr &kernel() const { return kernel_; }
  KernelLaunchInfoWithStream kernel_launch_info() const {
    return KernelLaunchInfoWithStream(input_launch_tensors_, output_launch_tensors_, workspace_launch_tensors_,
                                      stream_);
  }
  const std::set<size_t> &modifiable_ref_input_indexes() const { return modifiable_ref_input_indexes_; }
  const std::set<size_t> &modifiable_ref_output_indexes() const { return modifiable_ref_output_indexes_; }
  bool is_dynamic_shape() const { return is_dynamic_shape_; }
  bool is_launch_skipped() const { return is_launch_skipped_; }
  bool inputs_continuous_memory() const { return inputs_continuous_memory_; }
  SomasInfo *somas_info() const { return somas_info_; }
  const std::set<size_t> &somas_graph_output_indexes() const { return somas_graph_output_indexes_; }
  size_t get_stream() const { return kernel_info_->stream_id(); }

  const mindspore::HashMap<size_t, size_t> &increase_ref_count_size() const { return increase_ref_count_size_; }
  const std::vector<bool> &is_output_kernel() const { return is_output_kernel_; }
  const std::vector<bool> &is_monad_input() const { return is_monad_input_; }
  const std::vector<size_t> &input_free_index() const { return input_free_index_; }
  const std::vector<size_t> &output_free_index() const { return output_free_index_; }
  const std::vector<bool> &depend_shape_input_list() const { return depend_shape_input_list_; }
  const std::vector<bool> &is_weight() const { return is_weight_; }

  void set_enable_async_infer(bool enable_async_infer) { enable_async_infer_ = enable_async_infer; }

  // Really do infer shape and update kernel tensor shape.
  virtual void ExecuteInferShapeTask(OpContext<KernelTensor> *const context);
  // Really do resize kernel mod and update new size into output and workspace kernel tensors.
  virtual void ExecuteResizeKernelModTask(OpContext<KernelTensor> *const context);
  // Really do launch kernel with memory allocate and free.
  virtual void ExecuteLaunchKernelTask(OpContext<KernelTensor> *const context);

  void set_stream_send_actor(KernelActor *stream_send_actor) { stream_send_actor_ = stream_send_actor; }

  void SetInputDeviceTensor(const KernelTensorPtr &input_kernel_tensor, size_t input_index);

  // Set the memory address for the tensors which use the somas.
  void SetSomasMemory(OpContext<KernelTensor> *const context) const;

  bool skip_launch_shape_related_op() const { return skip_launch_shape_related_op_; }
  void set_skip_launch_shape_related_op(bool skip_launch_shape_related_op) {
    skip_launch_shape_related_op_ = skip_launch_shape_related_op;
  }

  const std::map<size_t, std::pair<KernelTensorPtr, std::pair<const DeviceContext *, std::vector<KernelTensorPtr>>>> &
  copy_output_kernel_tensors() const {
    return copy_output_kernel_tensors_;
  }
  std::vector<KernelTensor *> GetOutputDeviceTensors() { return output_launch_tensors_; }

  void set_insert_input_event(bool insert_input_event) { insert_input_event_ = insert_input_event; }

  void set_is_first_used_param(bool is_first_used_param, size_t index) {
    if (index >= is_first_used_params_.size()) {
      MS_LOG(EXCEPTION) << "Out of range for setting first used param, index: " << index
                        << ", size: " << is_first_used_params_.size();
    }
    is_first_used_params_[index] = is_first_used_param;
  }

  // Reset state for UCE.
  void ResetState(OpContext<KernelTensor> *const context) override;

  const std::vector<KernelTensorPtr> &workspace_kernel_tensors() { return workspace_kernel_tensors_; }
  const std::vector<KernelTensorPtr> &output_kernel_tensors() { return output_kernel_tensors_; }
  const std::vector<KernelTensorPtr> &input_kernel_tensors() { return input_kernel_tensors_; }

 protected:
  void Init() override;
  void Run(OpContext<KernelTensor> *const context) override;
  void SendRecorderInfo(OpContext<KernelTensor> *const context) const override;

  // Do kernel launching in this method after 'PreLaunchKernel' and 'PostLaunchKernel'.
  virtual bool LaunchKernel(OpContext<KernelTensor> *const context, bool is_skip_launch = false);
  // Handle the ref op, set input addr to output addr.
  virtual void UpdateRefDeviceAddress(OpContext<KernelTensor> *const context, bool increase_ref_count);
  // Execute kernel actor multi stream produre to make sure safety of memory before kernel launch.
  void ProcessMultiStreamBeforeKernelLaunch(OpContext<KernelTensor> *const context);
  // Execute kernel actor multi stream produre to make sure safety of memory after kernel launch.
  void ProcessMultiStreamAfterKernelLaunch(OpContext<KernelTensor> *const context);
  // Update the output ref count of graph output kernel.
  void UpdateGraphOutputRefCount(OpContext<KernelTensor> *const context);
  // Update the input device tensors to the memory free list.
  void UpdateMemoryFreeList(OpContext<KernelTensor> *const context);
  // Execute infer shape, resize and launch kernel by runtime pipeline which executes by KernelAsyncInferActor,
  // KernelAsyncResizeActor and KernelAsyncLaunchActor.
  void RunWithMultiPipeline(OpContext<KernelTensor> *const context);
  // Execute launch kernel asynchronously in KernelAsyncLaunchActor.
  void RunWithAsyncLaunchKernel(OpContext<KernelTensor> *const context);

  // Infer shape(and type) and resize kernel mod for dynamic shape or dynamic value, and update memory size from kernel
  // mod to output/workspace device tensors.
  void InferAndUpdateDeviceTensorSize(OpContext<KernelTensor> *const context);

  // Infer shape(and type) and resize kernel mod.
  void InferAndResize(OpContext<KernelTensor> *const context);

  // Re-Infer shape, type and resize before kernel launch in dynamic scenarios.
  void InferShapeAndType();

  // Re-InferShape and resize before kernel launch in dynamic scenarios.
  void InferShape();

  void ResizeKernelMod();

  // Update input_device_tensors by input op data.
  void UpdateInputDeviceTensor(const OpData<KernelTensor> *input_data, OpContext<KernelTensor> *const context);

  // Record the output and workspace memory pointer and size to optimize memory allocate/free performance in next step.
  // Note: only use in inference case.
  void TraceDynamicMemory();

  // Only aclnn kernel support no-contiguous input, so need make input contiguous.
  void ConvertInputContiguous(OpContext<KernelTensor> *const context);

  // Recover the inputs with contiguous.
  void RecoverInputs();

  // The info of kernel.
  CNodePtr kernel_;
  bool is_dynamic_shape_;
  bool is_dynamic_value_;
  bool is_dynamic_type_;
  bool has_dynamic_;
  std::vector<size_t> rw_write_index_;
  // Whether enable asynchronously infer shape and resize kernel mod by KernelInferActor and KernelResizeActor.
  bool enable_async_infer_;
  AID kernel_async_infer_aid_;
  AID kernel_async_resize_aid_;
  AID kernel_async_launch_aid_;
  KernelInfo *kernel_info_;
  KernelMod *kernel_mod_;

  // Used for set kernel tensor to resize and launch list.
  std::vector<KernelTensor *> input_launch_tensors_;
  std::vector<KernelTensor *> output_launch_tensors_;
  std::vector<KernelTensor *> workspace_launch_tensors_;

  std::vector<KernelTensorPtr> max_ref_cnt_output_list_;

  // The input kernel tensors for infer shape.
  std::vector<abstract::AbstractBasePtr> input_kernel_tensors_for_infer_;
  // The kernel tensors for input, output and workspace.
  std::vector<KernelTensorPtr> input_kernel_tensors_;
  std::vector<KernelTensorPtr> output_kernel_tensors_;
  std::vector<KernelTensorPtr> workspace_kernel_tensors_;

  // The received input device type and format may be different from the formal parameter in the control flow
  // scenarios, so it needs to be copied from the input data to real data that kernel launch needs.
  // And if the kernel has a ref input and output, the ptr should be set into the output device addresses.
  std::vector<KernelTensorPtr> copy_input_kernel_tensors_;
  // This vector record the heter input device tensor which is the source of copy input device tensors, and the
  // size of this vector is same to copy input device tensors. It is used to free memory of input, when the pre
  // input is not empty, and the input should be free, the pre device tensors should be free too.
  std::vector<KernelTensorPtr> pre_input_kernel_tensors_;
  std::map<size_t, std::pair<KernelTensorPtr, std::pair<const DeviceContext *, std::vector<KernelTensorPtr>>>>
    copy_output_kernel_tensors_;
  // Real data info that kernel launch needs, used to check the consistency of received input data.
  std::vector<std::shared_ptr<InputDataInfo>> real_input_data_infos_;

  // Store the tensor generated by ConvertInputContiguous.
  std::vector<KernelTensorPtr> contiguous_tensors_;
  // Store the input infos for recover after launch.
  std::map<size_t, KernelTensorPtr> temp_input_kernel_tensors_;

  // The device tensors for memory alloc and free.
  // output + workspace
  std::vector<KernelTensorPtr> memory_alloc_list_;
  // input + output + workspace
  std::vector<KernelTensorPtr> memory_free_list_;

  std::vector<size_t> input_free_index_;
  std::vector<size_t> output_free_index_;
  std::vector<KernelTensorPtr> new_memory_free_list_;

  // depend shape input list
  std::vector<bool> depend_shape_input_list_;
  // The device tensor of external reference is not the real data of this kernel, but need add to the
  // memory_free_list_.
  std::vector<KernelTensorPtr> external_reference_tensors_;

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
  // Flag for record weight.
  std::vector<bool> is_weight_;
  // Flag for aclop or reshape that does not support non-contiguous input.
  bool need_check_tensor_contiguous_{false};
  // Flag for kernel actor should insert event for parameter.
  bool insert_input_event_{false};

 protected:
  friend class GraphScheduler;
  friend class ControlNodeScheduler;
  friend class SchedulerHelper;
  friend class SuperKernelActor;

  // Init the device tensors and kernel launch info.
  void InitInputInfo();
  void InitOutputInfo();
  void InitWorkspaceInfo();
  void InitMultiStreamInfo();
  void InitIsMonadInput();

  // Fetch the device tensor for launch.
  void FetchInputDeviceTensor(OpContext<KernelTensor> *const context);
  void FetchOutputDeviceTensor(OpContext<KernelTensor> *const context);
  void FetchWorkspaceDeviceTensor();
  // Need copy when the data type or format between real parameters and formal parameters are inconsistent.
  void CopyInputDeviceTensor(KernelTensorPtr kernel_tensor, size_t input_index, OpContext<KernelTensor> *const context,
                             bool inference_param);
  // The processing before kernel launch: update the info of kernel launch.
  void PreLaunchKernel(OpContext<KernelTensor> *const context);
  // The processing after kernel launch: 1.erase input, 2.free memory, 3.send output.
  void PostLaunchKernel(OpContext<KernelTensor> *const context);
  // Back refresh the dynamic device tensor stores that have been triggered copy.
  void RefreshKernelTensorCopyStore(OpContext<KernelTensor> *const context);

  void *GetSomasDevicePtr(size_t offset) const;

  // Record mem info, because async send may free device info.
  void SetMemInfoForRdr();
  void SetShapeDependInfo();
  void DispatchDebugActor(OpContext<KernelTensor> *const context);
  bool LaunchKernelWithDebug(OpContext<KernelTensor> *const context, const bool skip_launch);

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

  // If the kernel is output of kernel graph, the ref count of output user should be add. This element record the output
  // index to the ref count size.
  mindspore::HashMap<size_t, size_t> increase_ref_count_size_;
  std::vector<bool> is_output_kernel_;
  std::vector<bool> is_monad_input_;
  std::vector<bool> is_first_used_params_;
  bool is_mc2_kernel_{false};
  // This flag are only valid in control flow. In inline control flow, due to the existence of switch branches, some
  // kernel actors will not be executed, and the condition switch actor controls whether to execute. It points to
  // the flag of the control branch in the condition switch actor. When the switch confirms the execution of a branch,
  // it sets the flag of the branch to true to enable the actor in this branch.
  bool *is_enable_{nullptr};
  bool need_wait_pipeline_{false};
  bool need_ref_for_storage_info_{true};
};

using KernelActorPtr = std::shared_ptr<KernelActor>;
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_KERNEL_ACTOR_H_
