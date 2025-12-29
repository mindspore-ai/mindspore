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

#include "backend/ms_backend/runtime/actors/base/kernel_runner.h"

#include <mutex>
#include <algorithm>
#include <unordered_map>
#include <limits>
#include <set>
#include <memory>
#include <utility>
#include <vector>
#include <string>

#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "device_address/device_type.h"
#include "ir/dtype/tensor_type.h"
#include "backend/ms_backend/runtime/graph_executor/pipeline/runtime_pipeline.h"
#include "backend/ms_backend/runtime/actors/base/memory_manager_actor.h"
#include "backend/ms_backend/runtime/actors/base/output_actor.h"
#include "backend/ms_backend/runtime/actors/base/recorder_actor.h"
#include "backend/ms_backend/runtime/actors/base/debug_actor.h"
#include "include/backend/common/exec_order/kernel_cache.h"
#include "backend/ms_backend/runtime/graph_executor/kernel_capture/graph_capture_manager.h"
#include "include/runtime/hardware_abstract/stream/multi_stream_controller.h"
#include "async/async.h"
#include "include/utils/callback.h"
#include "utils/log_adapter.h"
#include "include/runtime/memory/mem_pool/mem_tracker.h"
#include "include/backend/common/exec_order/execute_order_tracker.h"
#include "include/runtime/hardware_abstract/collective/collective_manager.h"
#include "include/runtime/utils/runtime_conf/runtime_conf.h"
#include "include/backend/common/pass_manager/dynamic_shape_helper.h"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/framework_utils.h"
#include "primitive/framework_ops.h"
#include "utils/compile_config.h"
#include "primitive/structure_op_name.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"
#include "backend/ms_backend/runtime/actors/hyper_offload/mem_use_analyzer.h"

namespace mindspore {
namespace runtime {
namespace {
bool IsSomasEnable(const SomasInfo *somas_info) {
  return ((somas_info != nullptr) && (somas_info->whole_block_size_ != 0));
}

void CheckDryRun(const CNodePtr &kernel_) {
  static const bool is_dry_run_mode = (common::IsExecuteSimulation() || IsSkippedLaunch());
  static auto enabled_profile = common::GetCompileConfig("COMPILE_PROFILE") == "1";
  if (is_dry_run_mode && !enabled_profile) {
    MS_LOG_WITH_NODE(EXCEPTION, kernel_)
      << "The dry run mode can not support dynamic shape graph which contains value depend or computing depend "
         "kernel:"
      << kernel_->fullname_with_scope()
      << ", launch kernel is skipped for dry run mode, which leads to fail to GetValue for infer "
         "shape of these value depend or computing depend kernel. You can only simulate compile graph and not do "
         "InferShape and Resize by `export MS_SIMULATION_LEVEL=0` instead.";
  }
}
void TrackInputOutputMemory(const std::vector<KernelTensor *> &input_launch_tensors,
                            const std::vector<KernelTensor *> &output_launch_tensors, const std::string &actor_name,
                            const std::vector<bool> &depend_shape_input_list) {
  for (size_t i = 0, end = input_launch_tensors.size(); i < end; i++) {
    // Skip shape depend inputs.
    if (i < depend_shape_input_list.size() && depend_shape_input_list[i]) {
      continue;
    }
    auto device_addr = input_launch_tensors[i]->device_address().get();
    if (device_addr == nullptr || !device_addr->IsPtrValid()) {
      continue;
    }
    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(
      MarkTensorAsInput, actor_name, device::GetDeviceNameByType(device_addr->GetDeviceType()), device_addr->GetPtr(),
      input_launch_tensors[i]->dtype_id(), input_launch_tensors[i]->GetShapeVector(),
      device_addr->GetTensorStorageInfo());
  }
  for (size_t i = 0, end = output_launch_tensors.size(); i < end; i++) {
    auto device_addr = output_launch_tensors[i]->device_address().get();
    if (device_addr == nullptr || !device_addr->IsPtrValid()) {
      continue;
    }
    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(
      MarkTensorAsOutput, actor_name, device::GetDeviceNameByType(device_addr->GetDeviceType()), device_addr->GetPtr(),
      output_launch_tensors[i]->dtype_id(), output_launch_tensors[i]->GetShapeVector(),
      device_addr->GetTensorStorageInfo());
  }
}

void AddNodeMemTrackerInfo(const CNodePtr cnode, const std::string &actor_name, bool is_stream_recv_actor) {
  if (is_stream_recv_actor || IsPrimitiveCNode(cnode, prim::kPrimStreamSend)) {
    auto node_name = is_stream_recv_actor ? "WaitEvent" : "RecordEvent";
    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, node_name, node_name, "", true);
  } else {
    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, actor_name, cnode->fullname_with_scope(),
                                                   cnode->func_graph()->ToString(), true);
  }
}

void AddEventNodeToGraphTracker(const CNodePtr &cnode, const std::string &actor_name, const std::string &type,
                                const std::string &stream_id) {
  if (!common::AnfAlgo::HasNodeAttr(kAttrEventId, cnode)) {
    MS_LOG(EXCEPTION) << "StreamSend or StreamRecv ops does not have attribute kAttrEventId.";
  }
  const std::string event_id = std::to_string(common::AnfAlgo::GetNodeAttr<uint32_t>(cnode, kAttrEventId));
  const bool is_user_event = cnode->HasAttr(kAttrEventId);

  const std::string node_name =
    is_user_event ? cnode->fullname_with_scope() : (type == kStreamSendOpName ? "RecordEvent" : "WaitEvent");
  const std::string func_graph_str = is_user_event ? cnode->func_graph()->ToString() : "";
  const std::string task_owner = is_user_event ? actor_name : node_name;

  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, task_owner, node_name, func_graph_str, true);
  device::tracker::CALL_MEMORY_TRACKER(UpdateTask, task_owner,
                                       {{device::tracker::kStreamId, stream_id}, {device::tracker::kEvent, event_id}});
}

void AddNodeToGraphTracker(const CNodePtr cnode, const std::string &actor_name) {
  auto type = common::AnfAlgo::GetCNodeName(cnode);
  auto stream_id = std::to_string(AnfAlgo::GetStreamId(cnode));
  if (type == kStreamSendOpName || type == kStreamRecvOpName) {
    AddEventNodeToGraphTracker(cnode, actor_name, type, stream_id);
    return;
  }
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, actor_name, cnode->fullname_with_scope(),
                                                 cnode->func_graph()->ToString(), true);
  device::tracker::CALL_MEMORY_TRACKER(UpdateTask, actor_name, {{device::tracker::kStreamId, stream_id}});

  if (!(common::AnfAlgo::IsCommunicationOp(cnode) && common::AnfAlgo::HasNodeAttr(kAttrGroup, cnode))) {
    return;
  }

  auto group_name = common::AnfAlgo::GetNodeAttr<std::string>(cnode, kAttrGroup);
  std::vector<uint32_t> comm_ranks;
  if (group_name == "hccl_world_group") {
    uint32_t rank_size = 1;
    rank_size = distributed::collective::CollectiveManager::instance()->global_rank_size();
    comm_ranks.resize(rank_size);
    std::iota(comm_ranks.begin(), comm_ranks.end(), 0);
  } else {
    comm_ranks = distributed::collective::CollectiveManager::instance()->GetGroupRanks(group_name);
  }
  std::string comm_ranks_str = std::accumulate(
    comm_ranks.begin(), comm_ranks.end(), std::string(),
    [](const std::string &a, uint32_t b) { return a.empty() ? std::to_string(b) : a + " " + std::to_string(b); });
  std::unordered_map<std::string, std::string> attrs = {{device::tracker::kGroup, group_name},
                                                        {device::tracker::kCommRank, comm_ranks_str}};

  auto get_rank = [&](const std::string &attr_name) -> uint32_t {
    uint32_t rank_value = std::numeric_limits<uint32_t>::max();
    if (common::AnfAlgo::HasNodeAttr(attr_name, cnode)) {
      int64_t rank_attr = common::AnfAlgo::GetNodeAttr<int64_t>(cnode, attr_name);
      if (rank_attr >= 0 && static_cast<size_t>(rank_attr) < comm_ranks.size()) {
        rank_value = comm_ranks[static_cast<size_t>(rank_attr)];
      } else {
        MS_LOG(EXCEPTION) << "Invalid rank_attr value: " << rank_attr << ", or out of range for comm_ranks with size "
                          << comm_ranks.size() << ".";
      }
    }
    return rank_value;
  };
  auto src_rank = get_rank(device::tracker::kSrcRank);
  if (src_rank != std::numeric_limits<uint32_t>::max()) {
    attrs[device::tracker::kSrcRank] = std::to_string(src_rank);
  }
  uint32_t dst_rank;
  if (common::AnfAlgo::GetCNodeName(cnode) != device::tracker::kSend) {
    dst_rank = get_rank(device::tracker::kDstRank);
  } else {
    dst_rank = get_rank(device::tracker::kSendDstRank);
  }
  if (dst_rank != std::numeric_limits<uint32_t>::max()) {
    attrs[device::tracker::kDstRank] = std::to_string(dst_rank);
  }
  auto root_rank = get_rank(device::tracker::kRootRank);
  if (root_rank != std::numeric_limits<uint32_t>::max()) {
    attrs[device::tracker::kRootRank] = std::to_string(root_rank);
  }
  device::tracker::CALL_MEMORY_TRACKER(UpdateTask, actor_name, attrs);
}

void ResetNewRefCountForRefOutputInSomas(const CNodePtr &node, size_t index) {
  if (node == nullptr) {
    return;
  }
  auto kernel_info = dynamic_cast<KernelInfo *>(node->kernel_info());
  if (kernel_info == nullptr) {
    return;
  }
  const auto &ref_map = kernel_info->out_in_ref_map();
  const auto &iter = ref_map.find(index);
  if (iter == ref_map.end()) {
    return;
  }
  size_t input_index = iter->second;
  if (index >= common::AnfAlgo::GetInputTensorNum(node)) {
    return;
  }
  const auto &input_node_with_index = common::AnfAlgo::GetPrevNodeOutput(node, input_index, false);
  if (input_node_with_index.first == nullptr || !input_node_with_index.first->isa<CNode>() ||
      common::AnfAlgo::CheckPrimitiveType(input_node_with_index.first, prim::kPrimConditionGather) ||
      !AnfAlgo::OutputAddrExist(input_node_with_index.first, input_node_with_index.second, false)) {
    return;
  }
  const auto &input_kernel_tensor =
    AnfAlgo::GetOutputKernelTensor(input_node_with_index.first, input_node_with_index.second, false);
  input_kernel_tensor->set_new_ref_count(0);
  MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
    << "Set new ref count to 0 for kernel tensor:" << input_kernel_tensor->ToString()
    << " for node:" << input_node_with_index.first->fullname_with_scope()
    << " debug string:" << input_node_with_index.first->DebugString() << " index:" << input_node_with_index.second;
  ResetNewRefCountForRefOutputInSomas(input_node_with_index.first->cast<CNodePtr>(), input_node_with_index.second);
}

void InsertEventForInput(uint32_t stream_id, const DeviceContext *device_context) {
  // Insert record wait pair to ensure first used parameter async copy end before launch.
  if (stream_id != kDefaultStreamIndex) {
    MS_EXCEPTION_IF_NULL(device_context);
    MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);
    auto &multi_stream_controller = device::DeviceContextManager::GetInstance().GetMultiStreamController(
      device_context->device_context_key().device_type_);
    MS_EXCEPTION_IF_NULL(multi_stream_controller);
    multi_stream_controller->DispatchRecordWaitEvent(stream_id, kDefaultStreamIndex);
  }
}
}  // namespace

using distributed::collective::CollectiveManager;

KernelRunner::KernelRunner(const std::string &name, const CNodePtr &kernel, const DeviceContext *device_context,
                           const AID &memory_manager_aid, const AID *debug_aid, const AID *recorder_aid,
                           GraphExecutionStrategy strategy, const std::set<size_t> &modifiable_ref_input_indexes,
                           const std::set<size_t> &modifiable_ref_output_indexes, const KernelTransformType &type)
    : type_(type),
      id(name, ActorMgr::GetActorMgrRef()->GetUrl()),
      enable_input_optimize_(EnableInputOptimize()),
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
  debug_aid_ = debug_aid;
  recorder_aid_ = recorder_aid;
  (void)device_contexts_.emplace_back(device_context);
  real_output_device_context_ = device_context;
  is_dynamic_shape_ = common::AnfAlgo::IsDynamicShape(kernel_) || common::AnfAlgo::IsDynamicSequence(kernel_);

  kernel_async_infer_aid_ = KernelAsyncInferActor::GetInstance()->GetAID();
  kernel_async_resize_aid_ = KernelAsyncResizeActor::GetInstance()->GetAID();
  kernel_async_launch_aid_ = KernelAsyncLaunchActor::GetInstance()->GetAID();
  input_free_index_.resize(common::AnfAlgo::GetInputTensorNum(kernel));
  std::iota(input_free_index_.begin(), input_free_index_.end(), 0);
  output_free_index_.resize(AnfAlgo::GetOutputAddressNum(kernel));
  std::vector<bool> is_output_kernel(AnfAlgo::GetOutputAddressNum(kernel), false);
  is_output_kernel_.swap(is_output_kernel);
  std::iota(output_free_index_.begin(), output_free_index_.end(), 0);
  need_ref_for_storage_info_ = (!common::AnfAlgo::IsViewNode(kernel));
  MS_LOG(DEBUG) << "Input free index:" << input_free_index_ << " output free index:" << output_free_index_
                << " need ref storage info:" << need_ref_for_storage_info_ << " for actor:" << GetAID()
                << " kernel:" << kernel->DebugString();
  const auto &prim = GetCNodePrimitive(kernel_);
  if (prim) {
    rw_write_index_ = prim->rw_write_input_indexes();
  }
  // shape depend need kernel is cnode.
  SetShapeDependInfo();

  enable_hyper_offload_slide_ =
    common::GetEnv("MS_DEV_HYPER_OFFLOAD") == "1" && common::GetCompileConfig("ENABLE_HYPER_OFFLOAD_SLIDE") == "1";
}

void KernelRunner::Init() {
  // Check device contexts number.
  if (device_contexts_.size() != runtime::kDeviceContextsNumOne) {
    MS_LOG(EXCEPTION) << "The device contexts number is wrong.";
  }
  MS_EXCEPTION_IF_NULL(device_contexts_[0]);

  MS_EXCEPTION_IF_NULL(kernel_);
  real_input_num_ = common::AnfAlgo::GetInputTensorNum(kernel_);
  kernel_info_ = dynamic_cast<KernelInfo *>(kernel_->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info_);
  // monad
  InitIsMonadInput();
  kernel_mod_ = kernel_info_->MutableKernelMod();
  MS_EXCEPTION_IF_NULL(kernel_mod_);
  is_dynamic_value_ = common::AnfAlgo::IsDynamicValue(kernel_);
  if (is_dynamic_shape_ && IsSomasEnable(somas_info_)) {
    MS_LOG(EXCEPTION) << "Not support the somas for the dynamic shape: " << GetAID().Name();
  }
  is_dynamic_type_ = common::AnfAlgo::IsAnyTypeOutput(kernel_);
  has_dynamic_ = is_dynamic_shape_ || is_dynamic_type_ || is_dynamic_value_;
  bool is_value_dyn = (is_dynamic_value_ && (is_dynamic_shape_ || is_dynamic_type_));
  if (is_value_dyn || (kernel_mod_->IsNeedUpdateOutputShapeAndSize() &&
                       no_dyn_need_update_ops.find(kernel_mod_->kernel_name()) == no_dyn_need_update_ops.end())) {
    CheckDryRun(kernel_);
  }

  // Check whether the kernel has input node which is a computed depend kernel.
  MS_EXCEPTION_IF_NULL(device_contexts_[0]);
  auto kernel_executor = device_contexts_[0]->GetKernelExecutor();
  MS_EXCEPTION_IF_NULL(kernel_executor);
  launch_ignored_inputs_ = kernel_executor->GetLaunchIgnoredInputAddressIdx(kernel_);

  stream_ = device_contexts_[0]->device_res_manager_->GetStream(kernel_info_->stream_id());
  // Init the device tensors and kernel launch info.
  InitInputInfo();
  InitOutputInfo();
  InitWorkspaceInfo();
  if (!output_kernel_tensors_.empty() && output_kernel_tensors_[0] &&
      output_kernel_tensors_[0]->GetDeviceType() != device_contexts_[0]->GetDeviceType()) {
    real_output_device_context_ = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
      {output_kernel_tensors_[0]->GetDeviceType(), device_contexts_[0]->device_context_key().device_id_});
    MS_EXCEPTION_IF_NULL(real_output_device_context_);
    if (!real_output_device_context_->initialized()) {
      const_cast<DeviceContext *>(real_output_device_context_)->Initialize();
    }
  }

  // Set flag to check input contiguous
  if (NeedCheckInputContiguous(kernel_)) {
    need_check_tensor_contiguous_ = true;
  }

  this->InitMultiStreamInfo();
}

void KernelRunner::InitMultiStreamInfo() {
  auto device_context = device_contexts_[0];
  // cpu kernel does not need multi stream process, and gpu kernel has not adapt it currently.
  if (device_context->GetDeviceType() == device::DeviceType::kCPU ||
      device_context->GetDeviceType() == device::DeviceType::kGPU) {
    MS_VLOG(VL_RUNTIME_FRAMEWORK_KERNEL) << "Kernel : " << kernel_->fullname_with_scope() << " device type is "
                                         << device_context->GetDeviceType() << ", will skip multi stream process.";
    is_multi_stream_process_skipped_ = true;
  }

  // Share pointer of task id on stream with output kernel tensor.
  for (auto &output_kernel_tensor : output_kernel_tensors_) {
    output_kernel_tensor->set_task_id_on_stream(task_id_on_stream_);
  }
  is_stream_recv_actor_ = IsPrimitiveCNode(kernel_, prim::kPrimStreamRecv);
  // kernel_ may be ValueNode<FuncGraph>, skip exception situation.
  auto cnode = kernel_->cast<CNodePtr>();
  if (cnode == nullptr) {
    return;
  }
  bool match_mc2_pattern = std::string::npos != kernel_->fullname_with_scope().find("_all_gather_matmul") ||
                           std::string::npos != kernel_->fullname_with_scope().find("_matmul_reduce_scatter") ||
                           std::string::npos != kernel_->fullname_with_scope().find("MatmulReduceScatter-") ||
                           std::string::npos != kernel_->fullname_with_scope().find("AllGatherMatmul-") ||
                           std::string::npos != kernel_->fullname_with_scope().find("MatMulAllReduce-");
  is_mc2_kernel_ = (runtime::IsEnableRuntimeConfig(runtime::kRuntimeMultiStream)) &&
                   !runtime::IsDisableRuntimeConfig(runtime::kRuntimeMc2Event) && match_mc2_pattern;

  auto input0 = cnode->input(kAnfPrimitiveIndex);
  if (IsValueNode<FuncGraph>(input0)) {
    MS_LOG(INFO) << "Cnode is not a func graph value node : " << kernel_->fullname_with_scope() << ".";
    return;
  }

  auto multi_stream_safe_value = cnode->GetAttr(kAttrInputMultiStreamSafe);
  if (multi_stream_safe_value != nullptr) {
    is_multi_stream_safe_ = GetValue<bool>(multi_stream_safe_value);
    MS_LOG(DEBUG) << "cnode : " << cnode->DebugString() << " is thread safe.";
  }
}

void KernelRunner::InitIsMonadInput() {
  auto build_info = kernel_info_->GetMutableSelectKernelBuildInfo();
  MS_EXCEPTION_IF_NULL(build_info);
  is_monad_input_.resize(real_input_num_, false);
  for (size_t i = 0; i < real_input_num_; ++i) {
    if (common::AnfAlgo::IsMonadType(build_info->GetInputDeviceType(i))) {
      is_monad_input_[i] = true;
    }
  }
}

void KernelRunner::InitInputInfo() {
  for (size_t i = 0; i < real_input_num_; ++i) {
    if (is_monad_input_[i]) {
      auto build_info = kernel_info_->GetMutableSelectKernelBuildInfo();
      MS_EXCEPTION_IF_NULL(build_info);
      (void)real_input_data_infos_.emplace_back(
        std::make_shared<InputDataInfo>(kernel::GetFormatFromStrToEnum(build_info->GetInputFormat(i)), ShapeVector{}, 0,
                                        build_info->GetInputDeviceType(i)));
      continue;
    }
    const auto &input_kernel_tensor = AnfAlgo::GetPrevNodeOutputKernelTensor(kernel_, i, false);
    MS_EXCEPTION_IF_NULL(input_kernel_tensor);
    const auto &input_device_tensor = input_kernel_tensor->device_address();
    MS_EXCEPTION_IF_NULL(input_device_tensor);
    (void)real_input_data_infos_.emplace_back(
      std::make_shared<InputDataInfo>(input_kernel_tensor->format(), input_kernel_tensor->GetShapeVector(),
                                      input_device_tensor->GetSize(), input_kernel_tensor->dtype_id()));
  }

  copy_input_kernel_tensors_.resize(real_input_num_);
  pre_input_kernel_tensors_.resize(real_input_num_);
  contiguous_tensors_.resize(real_input_num_);
  input_launch_tensors_.resize(real_input_num_);
  input_kernel_tensors_.resize(real_input_num_);
  input_kernel_tensors_for_infer_.resize(real_input_num_);
  is_first_used_params_.resize(real_input_num_);
  for (auto &input_kernel_tensor : input_kernel_tensors_) {
    (void)memory_free_list_.emplace_back(input_kernel_tensor);
    if (recorder_aid_ != nullptr) {
      (void)mem_info_.inputs_.emplace_back(std::make_shared<Address>());
    }
  }
  for (size_t index : input_free_index_) {
    if (index >= input_kernel_tensors_.size()) {
      MS_LOG(EXCEPTION) << "Invalid input index:" << index << " in free index:" << input_free_index_
                        << " input size:" << input_kernel_tensors_.size() << " for actor:" << GetAID();
    }
    new_memory_free_list_.emplace_back(input_kernel_tensors_[index]);
  }
}

void KernelRunner::InitOutputInfo() {
  MS_EXCEPTION_IF_NULL(kernel_info_);
  const auto &output_kernel_tensors = kernel_info_->output_kernel_tensor_list();
  const auto &somas_outputs = kernel_info_->somas_output_result();
  bool output_need_somas = false;
  for (size_t i = 0; i < output_kernel_tensors.size(); ++i) {
    auto &output_kernel_tensor = output_kernel_tensors[i];
    MS_EXCEPTION_IF_NULL(output_kernel_tensor);
    auto &output_address = output_kernel_tensor->device_address();
    MS_EXCEPTION_IF_NULL(output_address);

    if (output_address->stream_id() != kernel_info_->stream_id()) {
      MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
        << "Output address : " << output_address << " stream id :" << output_address->stream_id()
        << " is not equal kernel info stream id : " << kernel_info_->stream_id() << ".";
    }

    (void)output_kernel_tensors_.emplace_back(output_kernel_tensor);
    (void)output_launch_tensors_.emplace_back(output_kernel_tensor.get());
    MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
      << "Init output[" << i << "] info for node:" << kernel_->fullname_with_scope()
      << ", kernel tensor: " << output_kernel_tensor->ToString();
    if (recorder_aid_ != nullptr) {
      (void)mem_info_.outputs_.emplace_back(std::make_shared<Address>());
    }
    // The output taken over by soma does not need to allocate memory.
    if (kernel_info_->IsTensorEnableSomas(somas_outputs, i)) {
      output_kernel_tensor->set_managed_by_somas(true);
      MS_LOG(INFO) << "Device address : " << output_address << ", kernel tensor : " << output_kernel_tensor
                   << " is managed by somas.";
      // Somas outputs use the info of kernelMod, and output address use the info of device address.
      if (somas_outputs[i].second < output_address->GetSize()) {
        MS_LOG(INFO) << GetAID().Name() << " check somas size warning, output index:" << i
                     << " somas aligned size:" << somas_outputs[i].second
                     << " is smaller than address size:" << output_address->GetSize();
      }
      // Used to keep graph output address when somas block memory free, and reused by the ref conut in other graphs.
      if (somas_graph_output_indexes_.count(i) > 0) {
        MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
          << "Somas keep output device address:" << output_address << " ptr:" << output_address->GetPtr();
        (void)somas_info_->InsertGraphOutputInfo(output_address.get(), somas_outputs[i].first, somas_outputs[i].second);
        ResetNewRefCountForRefOutputInSomas(kernel_, i);
      } else {
        output_kernel_tensor->set_new_ref_count(SIZE_MAX);
      }
      output_need_somas = true;
    } else {
      (void)memory_alloc_list_.emplace_back(output_kernel_tensor);
      if (is_output_kernel_[i]) {
        max_ref_cnt_output_list_.emplace_back(output_kernel_tensor);
        MS_LOG(DEBUG) << "Add output kernel tensor:" << output_kernel_tensor << " for trace in actor:" << GetAID();
      }
      (void)memory_free_list_.emplace_back(output_kernel_tensor);
    }
  }

  for (size_t index : output_free_index_) {
    if (index >= output_kernel_tensors.size()) {
      MS_LOG(EXCEPTION) << "Invalid output free index:" << index << " total size:" << output_kernel_tensors.size()
                        << " for actor:" << GetAID();
    }
    if (kernel_info_->IsTensorEnableSomas(somas_outputs, index) || output_kernel_tensors[index] == nullptr) {
      continue;
    }
    MS_LOG(DEBUG) << "Add output free kernel tensor:" << output_kernel_tensors[index] << " for actor:" << GetAID();
    new_memory_free_list_.emplace_back(output_kernel_tensors[index]);
  }
  if (output_need_somas && (!IsSomasEnable(somas_info_))) {
    MS_LOG(EXCEPTION) << "The somas is not enable for: " << GetAID().Name();
  }

  if (IsSomasEnable(somas_info_)) {
    MS_EXCEPTION_IF_CHECK_FAIL((output_kernel_tensors_.size() >= somas_outputs.size()), "The output num is wrong.");
  }

  for (auto &external_reference_tensor : external_reference_tensors_) {
    (void)memory_free_list_.emplace_back(external_reference_tensor);
  }
}

void KernelRunner::InitWorkspaceInfo() {
  MS_EXCEPTION_IF_NULL(kernel_info_);
  // The size of workspace maybe changed in dynamic shape, so put workspace_address in the end of memory_alloc_list_
  // and memory_free_list_, for the operation of dynamic_shape condition in FetchWorkspaceDeviceTensor.
  const auto &workspace_kernel_tensor_list = kernel_info_->workspace_kernel_tensor_list();
  const auto &somas_workspace = kernel_info_->somas_workspace_result();
  bool workspace_need_somas = false;
  for (size_t i = 0; i < workspace_kernel_tensor_list.size(); ++i) {
    auto &workspace_kernel_tensor = workspace_kernel_tensor_list[i];
    MS_EXCEPTION_IF_NULL(workspace_kernel_tensor);
    auto &workspace_address = workspace_kernel_tensor->device_address();
    MS_EXCEPTION_IF_NULL(workspace_address);
    (void)workspace_kernel_tensors_.emplace_back(workspace_kernel_tensor);
    (void)workspace_launch_tensors_.emplace_back(workspace_kernel_tensor.get());
    if (recorder_aid_ != nullptr) {
      (void)mem_info_.workspaces_.emplace_back(std::make_shared<Address>());
    }

    // The workspace taken over by soma does not need to allocate memory.
    if (kernel_info_->IsTensorEnableSomas(somas_workspace, i)) {
      if (somas_workspace[i].second < workspace_address->GetSize()) {
        MS_LOG(INFO) << GetAID().Name() << " check somas size warning, workspace index:" << i
                     << " somas aligned size:" << somas_workspace[i].second
                     << " is smaller than address size:" << workspace_address->GetSize();
      }
      workspace_kernel_tensor->set_new_ref_count(SIZE_MAX);
      workspace_need_somas = true;
    } else {
      (void)memory_alloc_list_.emplace_back(workspace_kernel_tensor);
      (void)memory_free_list_.emplace_back(workspace_kernel_tensor);
      (void)new_memory_free_list_.emplace_back(workspace_kernel_tensor);
    }
  }

  if (workspace_need_somas && (!IsSomasEnable(somas_info_))) {
    MS_LOG(EXCEPTION) << "The somas is not enable for: " << GetAID().Name();
  }

  if (IsSomasEnable(somas_info_)) {
    MS_EXCEPTION_IF_CHECK_FAIL((workspace_kernel_tensors_.size() >= somas_workspace.size()),
                               "The output num is wrong.");
  }
}

void KernelRunner::SetShapeDependInfo() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  static const bool enable_infer_boost = ms_context->IsEnableInferBoost();
  if (enable_infer_boost) {
    return;
  }
  // Shape kernel no need to decrease ref count.
  const auto &only_depend_shape_attr = common::AnfAlgo::GetCNodePrimitiveAttr(kernel_, kAttrOnlyDependShape);
  if (only_depend_shape_attr != nullptr) {
    auto only_depend_shape = GetValue<std::vector<bool>>(only_depend_shape_attr);
    MS_LOG(INFO) << "Init shape depend info, real_input_num_ : " << real_input_num_
                 << ", only_depend_shape size : " << only_depend_shape.size() << ".";
    for (size_t i = 0; i < only_depend_shape.size(); i++) {
      // shape depend, no need free this device tensor.
      MS_LOG(INFO) << "only_shape_depend[" << i << "] : " << only_depend_shape[i] << ".";
      depend_shape_input_list_.emplace_back(only_depend_shape[i]);
    }
  }
  if (depend_shape_input_list_.empty()) {
    return;
  }
  std::vector<size_t> need_free_input_index;
  for (size_t index : input_free_index_) {
    if (index < depend_shape_input_list_.size() && depend_shape_input_list_[index]) {
      MS_LOG(DEBUG) << "Actor:" << GetAID() << " skip free input device tensor index:" << index;
      continue;
    }
    need_free_input_index.emplace_back(index);
  }
  input_free_index_.swap(need_free_input_index);
}

void KernelRunner::ConvertInputContiguous(OpContext<KernelTensor> *const context) {
  auto cur_stream_id = device_contexts_[0]->device_res_manager_->GetCurrentStreamId();
  auto stream_id = kernel_info_->stream_id();
  for (size_t i = 0; i < input_kernel_tensors_.size(); ++i) {
    if (input_kernel_tensors_[i] == nullptr) {
      continue;
    }
    auto input_device_tensor = input_kernel_tensors_[i]->device_address().get();
    if (input_device_tensor == nullptr) {
      continue;
    }
    if (i >= contiguous_tensors_.size()) {
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, (*context), " input out of range.");
    }
    const auto old_storage_info = input_device_tensor->GetTensorStorageInfo();
    if (old_storage_info) {
      if (IsContiguousStorage(old_storage_info)) {
        continue;
      }
      if (!launch_ignored_inputs_.empty() && (std::find(launch_ignored_inputs_.begin(), launch_ignored_inputs_.end(),
                                                        i) != launch_ignored_inputs_.end())) {
        MS_LOG(DEBUG) << GetAID().Name() << " ignore the input address for input index: " << i;
        continue;
      }

      // Check the inplace op not support the view input.
      if (std::find(rw_write_index_.begin(), rw_write_index_.end(), i) != rw_write_index_.end()) {
        std::string error_msg =
          kernel_->fullname_with_scope() +
          " is an inplace op and does not support view input. Please use other inplace op that support view "
          "input instead, or convert the view input to continuous input in advance. The input index is " +
          std::to_string(i) + trace::DumpSourceLines(kernel_);
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_msg);
      }

      MS_LOG(INFO) << "Make input [" << i << "] contiguous for kernel " << kernel_->DebugString();
      if (contiguous_tensors_[i] == nullptr) {
        // Make new device tensor and run InplaceCopy to make contiguous.
        MS_EXCEPTION_IF_NULL(old_storage_info);
        auto address_size =
          GetTypeByte(TypeIdToType(input_kernel_tensors_[i]->dtype_id())) * SizeOf(old_storage_info->shape);
        auto kernel_tensor = AnfAlgo::CreateKernelTensor(
          nullptr, address_size, Format::DEFAULT_FORMAT, input_kernel_tensors_[i]->dtype_id(), old_storage_info->shape,
          device::GetDeviceNameByType(device_contexts_[0]->device_context_key().device_type_),
          device_contexts_[0]->device_context_key().device_id_);
        kernel_tensor->SetType(std::make_shared<TensorType>(TypeIdToType(input_kernel_tensors_[i]->dtype_id())));
        kernel_tensor->SetShape(std::make_shared<abstract::TensorShape>(old_storage_info->shape));
        kernel_tensor->set_stream_id(stream_id);

        auto new_device_address = kernel_tensor->device_address();
        MS_EXCEPTION_IF_NULL(new_device_address);
        // Store the temp device address
        contiguous_tensors_[i] = kernel_tensor;
        MS_LOG(DEBUG) << "Create kernel tensor:" << kernel_tensor->ToString();
      }
      auto &new_kernel_tensor = contiguous_tensors_[i];
      MS_EXCEPTION_IF_NULL(new_kernel_tensor);
      auto &new_device_address = new_kernel_tensor->device_address();
      MS_EXCEPTION_IF_NULL(new_device_address);
      if (is_dynamic_shape_) {
        auto input_tensor = input_kernel_tensors_[i];
        MS_EXCEPTION_IF_NULL(input_tensor);
        MS_EXCEPTION_IF_NULL(input_tensor->GetShape());
        new_kernel_tensor->SetShape(input_tensor->GetShape()->Clone());
        MS_EXCEPTION_IF_NULL(input_tensor->device_address());
        auto address_size = GetTypeByte(TypeIdToType(input_tensor->dtype_id())) * SizeOf(old_storage_info->shape);
        new_kernel_tensor->set_size(address_size);
      }
      new_device_address->set_tensor_storage_info(nullptr);

      if (ActorDispatcher::enable_trace_dynamic_memory() || ActorDispatcher::enable_use_trace_memory()) {
        MS_LOG(EXCEPTION)
          << "The input[" << i << "] of kernel(" << kernel_->fullname_with_scope()
          << ") got a non-contiguous memory layout tensor, and framework will automatically convert it "
             "to contiguous memory layout, which involves temporary device memory allocation."
             " The trace memory feature can not work in this case, please find the source of "
             "non-contiguous input and convert it to contiguous memory layout, or disable trace memory "
             "feature by export MS_ENABLE_TRACE_MEMORY=off. Note: Disabling the trace "
             "memory feature will degrade memory management performance. Additionally, it will automatically disable "
             "the kernel group launch(parallel launch) and graph capture features, which may reduce network execution "
             "performance.";
      }
      // Launch CopyInplace to make tensor contiguous.
      if (i >= depend_shape_input_list_.size() || !depend_shape_input_list_[i]) {
        if (!device_contexts_[0]->GetKernelExecutor()->ExecuteKernelTask(runtime::KernelTaskType::kCONTIGUOUS_TASK,
                                                                         {input_kernel_tensors_[i].get()},
                                                                         {new_kernel_tensor.get()}, stream_id)) {
          MS_LOG(EXCEPTION) << "Graph mode executeKernelTask Contiguous failed.";
        }
        // Store the old tensor storage info , input device tensor and input kernel tensor.
        // Recover them when launch finished.
        if (cur_stream_id != stream_id) {
          cross_stream_addresses_.emplace_back(0, input_kernel_tensors_[i]->device_ptr());
          cross_stream_addresses_.emplace_back(0, new_kernel_tensor->device_ptr());
        }
      }
      temp_input_kernel_tensors_[i] = input_kernel_tensors_[i];
      MS_LOG(DEBUG) << "Replace input kernel tensor from:" << input_kernel_tensors_[i]->ToString()
                    << " to:" << new_kernel_tensor->ToString() << " input index:" << i << " for actor:" << GetAID();
      input_kernel_tensors_[i] = new_kernel_tensor;
      input_launch_tensors_[i] = new_kernel_tensor.get();
    }
  }
}

void KernelRunner::RecoverInputs() {
  if (!temp_input_kernel_tensors_.empty()) {
    for (const auto &pair : temp_input_kernel_tensors_) {
      input_kernel_tensors_[pair.first] = pair.second;
      input_launch_tensors_[pair.first] = pair.second.get();
    }
    temp_input_kernel_tensors_.clear();
  }
}

void KernelRunner::FetchWorkspaceDeviceTensor() {
  auto workspace_sizes = kernel_mod_->GetWorkspaceSizeList();
  // Resize of workspace_kernel_tensors_, memory_alloc_list_ and memory_free_list_, because of
  // the dynamic size of workspace.
  if (workspace_kernel_tensors_.size() > workspace_sizes.size()) {
    size_t size = workspace_kernel_tensors_.size() - workspace_sizes.size();
    (void)workspace_kernel_tensors_.erase(workspace_kernel_tensors_.end() - size, workspace_kernel_tensors_.end());
    if (recorder_aid_ != nullptr) {
      (void)mem_info_.workspaces_.erase(mem_info_.workspaces_.end() - size, mem_info_.workspaces_.end());
    }

    MS_EXCEPTION_IF_CHECK_FAIL((memory_alloc_list_.size() >= size), "The memory alloc list size is wrong.");
    MS_EXCEPTION_IF_CHECK_FAIL((memory_free_list_.size() >= size), "The memory free list size is wrong.");
    (void)memory_alloc_list_.erase(memory_alloc_list_.end() - size, memory_alloc_list_.end());
    (void)memory_free_list_.erase(memory_free_list_.end() - size, memory_free_list_.end());
    (void)new_memory_free_list_.erase(new_memory_free_list_.end() - size, new_memory_free_list_.end());
  } else if (workspace_kernel_tensors_.size() < workspace_sizes.size()) {
    if (device_contexts_.empty() || device_contexts_[0] == nullptr) {
      MS_LOG(ERROR) << "Invalid device context for kernel actor:" + GetAID().Name();
      return;
    }
    for (size_t i = workspace_kernel_tensors_.size(); i < workspace_sizes.size(); ++i) {
      auto kernel_tensor =
        AnfAlgo::CreateKernelTensor(nullptr, workspace_sizes[i], Format::DEFAULT_FORMAT, kTypeUnknown, ShapeVector(),
                                    device::GetDeviceNameByType(device_contexts_[0]->device_context_key().device_type_),
                                    device_contexts_[0]->device_context_key().device_id_);
      auto stream_id = AnfAlgo::GetStreamId(kernel_);
      kernel_tensor->set_stream_id(stream_id);
      AnfAlgo::SetAllocStreamId(kernel_tensor, stream_id, kernel_);
      auto device_address = kernel_tensor->device_address();
      MS_EXCEPTION_IF_NULL(device_address);
      MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS) << "Create kernel tensor for node:" << kernel_->fullname_with_scope()
                                                   << " kernel tensor:" << kernel_tensor->ToString();
      AnfAlgo::SetWorkspaceAddr(device_address, i, kernel_);  // set to kernel_info
      (void)workspace_kernel_tensors_.emplace_back(kernel_tensor);
      if (recorder_aid_ != nullptr) {
        (void)mem_info_.workspaces_.emplace_back(std::make_shared<Address>());
      }
      (void)memory_alloc_list_.emplace_back(kernel_tensor);
      (void)memory_free_list_.emplace_back(kernel_tensor);
      (void)new_memory_free_list_.emplace_back(kernel_tensor);
    }
  }
  // Set workspace address new size
  for (size_t i = 0; i < workspace_sizes.size(); ++i) {
    auto workspace_kernel_tensor = workspace_kernel_tensors_[i].get();
    MS_EXCEPTION_IF_NULL(workspace_kernel_tensor);
    auto workspace_device_tensor = workspace_kernel_tensor->device_address();
    MS_EXCEPTION_IF_NULL(workspace_device_tensor);
    workspace_device_tensor->SetSize(workspace_sizes[i]);
  }

  // Update workspace kernel tensors.
  workspace_launch_tensors_.resize(workspace_kernel_tensors_.size());
  for (size_t i = 0; i < workspace_sizes.size(); ++i) {
    workspace_launch_tensors_[i] = workspace_kernel_tensors_[i].get();
  }
}

void KernelRunner::SetSomasMemory(OpContext<KernelTensor> *const context) const {
  if (!IsSomasEnable(somas_info_)) {
    return;
  }

  // Set the memory address for the output tensors which use the somas.
  const auto &somas_outputs = kernel_info_->somas_output_result();
  for (size_t i = 0; i < somas_outputs.size(); ++i) {
    if (somas_outputs[i].second > 0) {
      auto device_ptr = GetSomasDevicePtr(somas_outputs[i].first);
      // In this scenario, the Init function can ensure that the pointer of the relevant operation is not nullptr.
      // In order to perform performance, the pointer validity is not checked here.
      // Check the graph output address need free.
      auto output_device_tensor = output_kernel_tensors_[i]->device_address().get();
      MS_EXCEPTION_IF_NULL(output_device_tensor);
      if (somas_graph_output_indexes_.count(i) && (output_device_tensor->GetPtr() != nullptr)) {
        if (device_ptr != output_device_tensor->GetPtr()) {
          // The device memory may not be released here and free it directly. When sink size>1, due to some problem on
          // ref count, when the step is not the last step of the sink size, it cannot be released according to the ref
          // count in the output actor. When sink size=1, there is no such problem, and ptr will be directly swapped to
          // the new device address. The problem of ref count  will be fixed in next releases.
          MS_LOG(INFO) << GetAID().Name() << " does not free address for graph output index: " << i
                       << " kernel tensor:" << output_kernel_tensors_[i]->ToString();
          device_contexts_[0]->device_res_manager_->FreeMemory(output_device_tensor);
        }
      }
      MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
        << "Set ptr:" << device_ptr << " to device address:" << output_device_tensor << " in actor:" << GetAID();
      output_device_tensor->set_ptr(device_ptr);
      if (somas_graph_output_indexes_.count(i) || output_kernel_tensors_[i]->new_ref_count() != SIZE_MAX) {
        output_kernel_tensors_[i]->IncreaseNewRefCount(GetAID().Name());
        MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
          << "Add new ref count for somas output kernel tensor:" << output_kernel_tensors_[i]->ToString()
          << " in kernel actor:" << GetAID();
      }
    }
  }

  // Set the memory address for the workspace tensors which use the somas.
  const auto &somas_workspace = kernel_info_->somas_workspace_result();
  for (size_t i = 0; i < somas_workspace.size(); ++i) {
    if (somas_workspace[i].second > 0) {
      auto device_ptr = GetSomasDevicePtr(somas_workspace[i].first);
      // In this scenario, the Init function can ensure that the pointer of the relevant operation is not nullptr.
      // In order to perform performance, the pointer validity is not checked here.
      auto &workspace_device_tensor = workspace_kernel_tensors_[i]->device_address();
      MS_EXCEPTION_IF_NULL(workspace_device_tensor);
      workspace_device_tensor->set_ptr(device_ptr);
    }
  }
}

void *KernelRunner::GetSomasDevicePtr(size_t offset) const {
  // Get the ptr from the whole block.
  if (somas_info_->base_address_ != nullptr) {
    return AddressOffset(somas_info_->base_address_, offset);
  }

  // Get the ptr from the merged blocks.
  auto iter = somas_info_->merged_base_addresses_.upper_bound(offset);
  if (iter == somas_info_->merged_base_addresses_.begin()) {
    MS_LOG(ERROR) << GetAID().Name() << " can't find the merged block for offset: " << offset;
    return nullptr;
  }
  --iter;
  size_t real_offset = offset - iter->first;
  void *real_base_address = iter->second;
  if (real_base_address == nullptr) {
    MS_LOG(ERROR) << GetAID().Name() << " doesn't allocate the merged block base address for offset: " << iter->first;
    return nullptr;
  }
  return AddressOffset(real_base_address, real_offset);
}

void KernelRunner::TraceDynamicMemory() {
  bool enable_capture_graph = GraphCaptureManager::GetInstance().GetEnableGraphCapture();
  for (size_t i = 0; i < output_kernel_tensors_.size(); i++) {
    const auto &kernel_tensor = output_kernel_tensors_[i];
    // If enable kernel launch capture, the kernel output as graph output will be captured and can not changed, so need
    // trace the graph output kernel tensor device address, which device memory will be allocated and released with the
    // whole graph.
    if ((is_output_kernel_[i] && !enable_capture_graph) || kernel_tensor->new_ref_count() == SIZE_MAX) {
      continue;
    }
    MemoryTraceManager::GetInstance().AddKernelMemoryTraceBlock(
      std::make_shared<KernelMemoryTraceBlock>(kernel_, kernel_tensor->device_ptr(), kernel_tensor->size(), kOutputMem,
                                               i, kernel_tensor.get(), real_output_device_context_),
      real_output_device_context_);
  }

  for (size_t i = 0; i < workspace_kernel_tensors_.size(); i++) {
    const auto &kernel_tensor = workspace_kernel_tensors_[i];
    MemoryTraceManager::GetInstance().AddKernelMemoryTraceBlock(
      std::make_shared<KernelMemoryTraceBlock>(kernel_, kernel_tensor->device_ptr(), kernel_tensor->size(),
                                               kWorkspaceMem, i, kernel_tensor.get(), device_contexts_[0]),
      device_contexts_[0]);
  }
}

void KernelRunner::SendMemoryAllocReq(OpContext<KernelTensor> *const context) {
  MemoryManagerActor::GetInstance()->AllocateMemory(&memory_alloc_list_, device_contexts_[0], context, GetAID());

  if (ActorDispatcher::enable_trace_dynamic_memory()) {
    if (IsRunningFailed(context)) {
      return;
    }
    TraceDynamicMemory();
  }
}

void KernelRunner::SendMemoryAllocReqHP(OpContext<KernelTensor> *const context, uint32_t mem_alloc_stream_id) {
  if (device_contexts_[0]->device_res_manager_->swap_manager() != nullptr) {
    MS_EXCEPTION_IF_NULL(kernel_info_);
    for (const auto &out_in : kernel_info_->out_in_ref_map()) {
      const auto &input_kernel_tensor = input_kernel_tensors_[out_in.second];
      MS_EXCEPTION_IF_NULL(input_kernel_tensor);
      const auto &ptr = input_kernel_tensor->GetValidPtr(kDefaultStreamIndex);
      MS_EXCEPTION_IF_NULL(output_kernel_tensors_[out_in.first]);
      const auto &output_device_tensor = output_kernel_tensors_[out_in.first]->device_address();
      if (ptr == nullptr || output_device_tensor == nullptr || output_device_tensor->GetPtr() != nullptr) {
        continue;
      }
      // Pointer in DeviceAddress which is reference output may not be updated to the same as the reference input
      // which is swapped out.
      MS_LOG(DEBUG) << "Set device ptr of " << out_in.first << "th ref output the same as input " << out_in.second
                    << ": " << ptr;
      output_device_tensor->set_ptr(ptr);
    }
  }
  MemoryManagerActor::GetInstance()->AllocateMemoryHP(&memory_alloc_list_, device_contexts_[0], context, GetAID(),
                                                      mem_alloc_stream_id);

  if (ActorDispatcher::enable_trace_dynamic_memory()) {
    if (IsRunningFailed(context)) {
      return;
    }
    TraceDynamicMemory();
  }
}

void KernelRunner::SendMemoryFreeReq(OpContext<KernelTensor> *const context) {
  MemoryManagerActor::GetInstance()->FreeMemory(&new_memory_free_list_, device_contexts_[0], context, GetAID());
  // Free the address that is the temp store for kernel input copy.
  for (auto &copy_input_kernel_tensor : copy_input_kernel_tensors_) {
    if (copy_input_kernel_tensor == nullptr) {
      continue;
    }
    const auto &copy_input_device_tensor = copy_input_kernel_tensor->device_address();
    if ((copy_input_device_tensor != nullptr) && (copy_input_device_tensor->GetPtr() != nullptr)) {
      MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
        << "Free memory by ref count for kernel tensor:" << copy_input_kernel_tensor->ToString()
        << " for actor:" << GetAID();
      MemoryManagerActor::GetInstance()->FreeMemoryByRefCount(copy_input_kernel_tensor.get(), device_contexts_[0],
                                                              GetAID().Name());
    }
  }
  // Free the address that is the temp store for kernel contiguous copy.
  for (auto &contiguous_kernel_tensor : contiguous_tensors_) {
    if (contiguous_kernel_tensor == nullptr) {
      continue;
    }
    auto &contiguous_device_tensor = contiguous_kernel_tensor->device_address();
    if ((contiguous_device_tensor != nullptr) && (contiguous_device_tensor->GetPtr() != nullptr)) {
      device_contexts_[0]->device_res_manager_->FreeMemory(contiguous_device_tensor.get());
    }
  }
}

void KernelRunner::SetMemInfoForRdr() {
  for (size_t i = 0; i < input_kernel_tensors_.size(); ++i) {
    if (is_monad_input_[i]) {
      continue;
    }
    MS_EXCEPTION_IF_NULL(input_kernel_tensors_[i]);
    MS_EXCEPTION_IF_NULL(input_kernel_tensors_[i]->device_address());
    mem_info_.inputs_[i]->addr = input_kernel_tensors_[i]->device_address()->GetMutablePtr();
    mem_info_.inputs_[i]->size = input_kernel_tensors_[i]->device_address()->GetSize();
  }
  for (size_t i = 0; i < output_kernel_tensors_.size(); ++i) {
    MS_EXCEPTION_IF_NULL(output_kernel_tensors_[i]->device_address());
    mem_info_.outputs_[i]->addr = output_kernel_tensors_[i]->device_address()->GetMutablePtr();
    mem_info_.outputs_[i]->size = output_kernel_tensors_[i]->device_address()->GetSize();
  }
  for (size_t i = 0; i < workspace_kernel_tensors_.size(); ++i) {
    MS_EXCEPTION_IF_NULL(workspace_kernel_tensors_[i]->device_address());
    mem_info_.workspaces_[i]->addr = workspace_kernel_tensors_[i]->device_address()->GetMutablePtr();
    mem_info_.workspaces_[i]->size = workspace_kernel_tensors_[i]->device_address()->GetSize();
  }
}

void KernelRunner::CopyInputDeviceTensor(KernelTensorPtr kernel_tensor, size_t input_index,
                                         OpContext<KernelTensor> *const context, bool in_increment) {
  // The ignored input address that is not used in the kernel launch and no need copy.
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  auto device_tensor = kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(device_tensor);
  if (!launch_ignored_inputs_.empty() && (std::find(launch_ignored_inputs_.begin(), launch_ignored_inputs_.end(),
                                                    input_index) != launch_ignored_inputs_.end())) {
    MS_LOG(DEBUG) << GetAID().Name() << " ignore the input address for input index: " << input_index;
    return;
  }
  if (skip_launch_shape_related_op_) {
    return;
  }
  if (input_index >= real_input_data_infos_.size()) {
    std::stringstream ofs;
    ofs << "Invalid input index:" << input_index << " size:" << real_input_data_infos_.size()
        << " for actor:" << GetAID();
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, *context, ofs.str());
  }
  auto &real_input_info = real_input_data_infos_[input_index];
  if ((device_tensor->GetDeviceType() == device_contexts_[0]->GetDeviceType()) &&
      AnfAlgo::IsEquivalentFormat(kernel_tensor->format(), real_input_info->format_) &&
      kernel_tensor->dtype_id() == real_input_info->type_id_) {
    return;
  }
  if (in_increment) {
    MS_LOG(EXCEPTION) << GetAID().Name() << " got an unexpected input type for input[" << input_index << "]:\n"
                      << "Got format: " << kernel::GetFormatFromEnumToStr(kernel_tensor->format())
                      << ", expected format: " << kernel::GetFormatFromEnumToStr(real_input_info->format_)
                      << ", they are equivalent format: " << std::boolalpha
                      << AnfAlgo::IsEquivalentFormat(kernel_tensor->format(), real_input_info->format_) << ".\n"
                      << "Got device type: " << device::GetDeviceNameByType(kernel_tensor->GetDeviceType())
                      << ", expected device type: " << device::GetDeviceNameByType(device_contexts_[0]->GetDeviceType())
                      << ".\n"
                      << "Got data type: " << TypeIdToString(kernel_tensor->dtype_id())
                      << ", expected data type: " << TypeIdToString(real_input_info->type_id_) << ".\n"
                      << "Full input kernel tensor information: " << kernel_tensor->ToString();
  }
  uint64_t start_time = 0;
  PROFILER_START(start_time);
  if (!WaitRuntimePipelineFinish(context, GetAID().Name())) {
    MS_LOG(INFO) << "Run failed and early stop for kernel: " << kernel_->fullname_with_scope();
    return;
  }
  if (inputs_continuous_memory_) {
    std::string error_info = GetAID().Name() + " inputs must be continuous memory and can't be copied for index " +
                             std::to_string(input_index);
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, *context, error_info);
  }
  if (!IsContiguousStorage(device_tensor->GetTensorStorageInfo())) {
    std::stringstream error_info;
    error_info << "Not support non-contiguous heter input:" << kernel_tensor->ToString() << " for actor:" << GetAID()
               << " input index:" << input_index;
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info.str());
  }
  if (input_index >= copy_input_kernel_tensors_.size()) {
    std::stringstream ofs;
    ofs << "Invalid input index:" << input_index
        << " copy input device tensor size:" << copy_input_kernel_tensors_.size() << " for actor:" << GetAID();
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, *context, ofs.str());
  }
  if (copy_input_kernel_tensors_[input_index] == nullptr) {
    const auto &pre_kernel_tensor = kernel_tensor;
    auto new_kernel_tensor = AnfAlgo::CreateKernelTensor(
      pre_kernel_tensor->GetShape(), pre_kernel_tensor->GetType(), pre_kernel_tensor->GetValueTrack(), nullptr,
      real_input_info->size_, kernel::GetFormatFromEnumToStr(real_input_info->format_), real_input_info->type_id_,
      real_input_info->shape_, device::GetDeviceNameByType(device_contexts_[0]->device_context_key().device_type_),
      device_contexts_[0]->device_context_key().device_id_, kernel_tensor->user_data());
    auto pre_stream_id = pre_kernel_tensor->stream_id();
    if (pre_stream_id == UINT32_MAX) {
      auto stream_id = kernel_info_->stream_id();
      MS_LOG(DEBUG) << "Rewrite kernel tensor : " << new_kernel_tensor
                    << " stream id with kernel info stream id : " << stream_id << ".";
      new_kernel_tensor->set_stream_id(stream_id);
    } else {
      MS_LOG(DEBUG) << "Rewrite kernel tensor : " << new_kernel_tensor
                    << " stream id with pre kernel tensor stream id : " << pre_stream_id << ".";
      new_kernel_tensor->set_stream_id(pre_stream_id);
    }

    copy_input_kernel_tensors_[input_index] = new_kernel_tensor;
    MS_LOG(DEBUG) << "Create copy kernel tensor:" << copy_input_kernel_tensors_[input_index]->ToString()
                  << " index:" << input_index << " for actor:" << GetAID();
  }
  auto &new_kernel_tensor = copy_input_kernel_tensors_[input_index];
  MS_EXCEPTION_IF_NULL(new_kernel_tensor);
  auto &new_device_tensor = new_kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(new_device_tensor);
  new_kernel_tensor->set_need_sync_user_data(kernel_tensor->need_sync_user_data());
  MS_LOG(DEBUG) << "Prev stream id : " << input_kernel_tensors_[input_index]->device_address()->stream_id()
                << " new stream id : " << new_device_tensor->stream_id() << ".";
  // Update the input kernel tensor.
  input_launch_tensors_[input_index] = new_kernel_tensor.get();
  pre_input_kernel_tensors_[input_index] = kernel_tensor;
  input_kernel_tensors_[input_index] = new_kernel_tensor;
  if (is_dynamic_shape_) {
    // Need update shape and size for dynamic shape case.
    input_kernel_tensors_for_infer_[input_index] = input_kernel_tensors_[input_index];
    MS_EXCEPTION_IF_NULL(input_kernel_tensors_[input_index]);
    MS_EXCEPTION_IF_NULL(kernel_tensor);
    MS_EXCEPTION_IF_NULL(kernel_tensor->GetShape());
    input_kernel_tensors_[input_index]->SetShape(kernel_tensor->GetShape()->Clone());
    input_kernel_tensors_[input_index]->set_size(device_tensor->GetSize());
  }

  if (new_device_tensor->GetPtr() == nullptr) {
    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, GetAID().Name(), memory::mem_pool::MemType::kOther,
                                                   new_device_tensor->GetSize(), new_device_tensor.get());
    if (!device_contexts_[0]->device_res_manager_->AllocateMemory(new_device_tensor.get(), kDefaultStreamIndex)) {
      SET_OPCONTEXT_MEMORY_ALLOC_FAIL_BY_STRATEGY(strategy_, *context, *(device_contexts_[0]), GetAID().Name(),
                                                  new_device_tensor->GetSize());
    }
    static std::string name = "Alloc memory";
    new_kernel_tensor->IncreaseNewRefCount(name);
    MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
      << "Increase new ref count for kernel tensor:" << new_kernel_tensor->ToString() << " in actor:" << GetAID();
  }

  MS_LOG(INFO) << GetAID().Name() << " the input position:" << input_index
               << " copy from device address:" << kernel_tensor->ToString()
               << " to device address:" << new_kernel_tensor->ToString();
  // Copy from the real parameter to formal parameter and insert the device tensor copy store.
  if (!SyncAllStreamForDeviceAddress(new_device_tensor, device_tensor) ||
      !SyncCopy(new_kernel_tensor.get(), kernel_tensor.get(), kDefaultStreamIndex)) {
    std::string error_info = "Copy device tensor failed: " + GetAID().Name();
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, *context, error_info);
  }
  if (modifiable_ref_input_indexes_.count(input_index) > 0) {
    MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
      << "Add device tensor copy store for kernel tensor:" << new_kernel_tensor->ToString() << " and "
      << kernel_tensor->ToString() << " for copy actor:" << GetAID();
    KernelTensorCopyStore::GetInstance().Insert(new_kernel_tensor.get(), kernel_tensor.get());
  }
  PROFILER_END(start_time, runtime::ProfilerModule::kKernel, runtime::ProfilerEvent::kPreLaunch,
               "CopyInputDeviceTensor", false);
}

void KernelRunner::UpdateGraphOutputRefCount(OpContext<KernelTensor> *const context) {
  for (const auto &pair : increase_ref_count_size_) {
    if (pair.first >= output_kernel_tensors_.size() || output_kernel_tensors_[pair.first] == nullptr) {
      MS_LOG(EXCEPTION) << "Invalid output index:" << pair.first << " total size:" << output_kernel_tensors_.size()
                        << " for actor:" << GetAID();
    }
    const auto &output_kernel_tensor = output_kernel_tensors_[pair.first];
    MS_EXCEPTION_IF_NULL(output_kernel_tensor);
    output_kernel_tensor->IncreaseNewRefCount(GetAID().Name(), pair.second);
    MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
      << "Add new ref count size:" << pair.second << " for kernel tensor:" << output_kernel_tensor->ToString()
      << " for kernel actor:" << GetAID();
  }
}

void KernelRunner::UpdateMemoryFreeList(OpContext<KernelTensor> *const context) {
  // Set input device address to memory free list by free index.
  for (size_t free_list_index = 0; free_list_index < input_free_index_.size(); ++free_list_index) {
    size_t input_list_index = input_free_index_[free_list_index];
    if (free_list_index >= new_memory_free_list_.size() || input_list_index >= input_kernel_tensors_.size() ||
        input_list_index >= pre_input_kernel_tensors_.size()) {
      MS_LOG(EXCEPTION) << "Invalid free position:" << free_list_index
                        << " free list size:" << new_memory_free_list_.size() << " or input index:" << input_list_index
                        << " input size:" << input_kernel_tensors_.size()
                        << " pre input size:" << pre_input_kernel_tensors_.size() << " for actor:" << GetAID();
    }
    new_memory_free_list_[free_list_index] =
      (pre_input_kernel_tensors_[input_list_index] == nullptr ? input_kernel_tensors_[input_list_index]
                                                              : pre_input_kernel_tensors_[input_list_index]);
    MS_LOG(DEBUG) << "Add new memory free list for input index:" << input_list_index
                  << " input kernel tensor:" << input_kernel_tensors_[input_list_index]
                  << " and pre input kernel tensor:" << pre_input_kernel_tensors_[input_list_index]
                  << " for kernel actor:" << GetAID();
    pre_input_kernel_tensors_[input_list_index] = nullptr;
  }
}

void KernelRunner::UpdateRefDeviceAddress(OpContext<KernelTensor> *const context, bool increase_ref_count) {
  MS_EXCEPTION_IF_NULL(kernel_info_);
  for (const auto &pair : kernel_info_->out_in_ref_map()) {
    if (pair.first >= output_kernel_tensors_.size() || pair.second >= input_kernel_tensors_.size()) {
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(
        strategy_, (*context),
        "Invalid ref index pair [" + std::to_string(pair.first) + ", " + std::to_string(pair.second) +
          "] input size:" + std::to_string(input_kernel_tensors_.size()) + " output size:" +
          std::to_string(output_kernel_tensors_.size()) + " for kernel:" + kernel_->fullname_with_scope());
    }
    if (output_kernel_tensors_[pair.first] == nullptr || input_kernel_tensors_[pair.second] == nullptr) {
      std::stringstream error_info;
      error_info << "Invalid ref input kernel tensor:" << input_kernel_tensors_[pair.second]
                 << "and output kernel tensor:" << output_kernel_tensors_[pair.first]
                 << " for kernel:" + kernel_->fullname_with_scope();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, (*context), error_info.str());
    }
    auto input_device_tensor = input_kernel_tensors_[pair.second]->device_address().get();
    MS_EXCEPTION_IF_NULL(input_device_tensor);
    auto output_device_tensor = output_kernel_tensors_[pair.first]->device_address().get();
    MS_EXCEPTION_IF_NULL(output_device_tensor);
    output_kernel_tensors_[pair.first]->set_pointer_ref_count(input_kernel_tensors_[pair.second].get());
    output_kernel_tensors_[pair.first]->IncreaseNewRefCount(GetAID().Name());
    if (input_device_tensor->GetTensorStorageInfo() != nullptr && need_ref_for_storage_info_) {
      output_device_tensor->set_tensor_storage_info(input_device_tensor->GetTensorStorageInfo());
    }
    MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
      << "Actor:" << GetAID()
      << " increase new ref count for kernel tensor:" << output_kernel_tensors_[pair.first]->ToString()
      << " and input kernel tensor:" << input_kernel_tensors_[pair.second]->ToString();
  }
}

void KernelRunner::FetchOutputDeviceTensor(OpContext<KernelTensor> *const context) {
  auto &output_kernel_tensors = kernel_info_->output_kernel_tensor_list();
  const auto &output_size_list = kernel_mod_->GetOutputSizeList();

  // May exist in the kernel which does not support the dynamic shape.
  if (output_kernel_tensors.size() != output_size_list.size()) {
    std::string error_info =
      "For " + GetAID().Name() + ", the expected outputs number: " + std::to_string(output_kernel_tensors.size()) +
      ", but the number of output size list after Resize: " + std::to_string(output_size_list.size()) +
      ", this kernel may not support the dynamic shape, please check.";
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, (*context), error_info);
  }

  // Update the size of output device tensor.
  for (size_t i = 0; i < output_kernel_tensors.size(); ++i) {
    MS_EXCEPTION_IF_NULL(output_kernel_tensors[i]);
    auto &output_address = output_kernel_tensors[i]->device_address();
    if (output_size_list[i] == output_address->GetSize()) {
      continue;
    }
    output_address->SetSize(output_size_list[i]);
  }
}

void KernelRunner::ExecuteInferShapeTask(OpContext<KernelTensor> *const context, bool high_perf) {
  ProfilerRecorder profiler(ProfilerModule::kKernel, ProfilerEvent::kKernelInfer, GetAID().Name());
  if (IsRunningFailed(context)) {
    MS_VLOG(VL_RUNTIME_FRAMEWORK_KERNEL) << "Run failed and early stop infer shape for kernel: "
                                         << kernel_->fullname_with_scope();
    return;
  }

  if (is_dynamic_type_) {
    InferShapeAndType();
  } else if (is_dynamic_shape_) {
    device_contexts_[0]->device_res_manager_->BindDeviceToCurrentThread(false);
    InferShape();
  }

  if (EnableRuntimeNewPipeline()) {
    auto resize_task = [context, this, high_perf]() {
      KernelAsyncResizeActor::GetInstance()->ResizeKernelModV2(context, this, high_perf);
    };
    RuntimePipeline::GetInstance().resize_queue()->Push(std::move(resize_task));
  } else {
    Async(kernel_async_resize_aid_, &KernelAsyncResizeActor::ResizeKernelModV2, context, this, high_perf);
  }
}

void KernelRunner::ExecuteResizeKernelModTask(OpContext<KernelTensor> *const context, bool high_perf) {
  ProfilerRecorder profiler(ProfilerModule::kKernel, ProfilerEvent::kKernelResize, GetAID().Name());
  if (IsRunningFailed(context)) {
    MS_VLOG(VL_RUNTIME_FRAMEWORK_KERNEL) << "Run failed and early stop resize for kernel: "
                                         << kernel_->fullname_with_scope();
    return;
  }
  bool view_input = false;
  if (!need_check_tensor_contiguous_) {
    auto it = std::find_if(input_kernel_tensors_.begin(), input_kernel_tensors_.end(),
                           [](const KernelTensorPtr &tensor) { return tensor->tensor_storage_info() != nullptr; });
    if (it != input_kernel_tensors_.end()) {
      view_input = true;
    }
  }

  if (has_dynamic_ || view_input) {
    device_contexts_[0]->device_res_manager_->BindDeviceToCurrentThread(false);
    ResizeKernelMod();

    FetchOutputDeviceTensor(context);
    FetchWorkspaceDeviceTensor();
  } else {
    FetchOutputDeviceTensor(context);
  }

  if (high_perf) {
    if (EnableRuntimeNewPipeline()) {
      auto launch_task = [context, this]() { KernelAsyncLaunchActor::GetInstance()->LaunchKernelV2HP(context, this); };
      RuntimePipeline::GetInstance().launch_queue()->Push(std::move(launch_task));
    } else {
      Async(kernel_async_launch_aid_, &KernelAsyncLaunchActor::LaunchKernelV2HP, context, this);
    }
  } else {
    if (EnableRuntimeNewPipeline()) {
      auto launch_task = [context, this]() { KernelAsyncLaunchActor::GetInstance()->LaunchKernelV2(context, this); };
      RuntimePipeline::GetInstance().launch_queue()->Push(std::move(launch_task));
    } else {
      Async(kernel_async_launch_aid_, &KernelAsyncLaunchActor::LaunchKernelV2, context, this);
    }
  }
}

void KernelRunner::ExecuteLaunchKernelTask(OpContext<KernelTensor> *const context) {
  if (MS_UNLIKELY(IsRunningFailed(context))) {
    MS_VLOG(VL_RUNTIME_FRAMEWORK_KERNEL) << "Run failed and early stop launch kernel: "
                                         << kernel_->fullname_with_scope();
    return;
  }

  if (enable_hyper_offload_slide_) {
    MemUseAnalyzer::GetInstance().LaunchTaskBefore(this, device_contexts_[0]);
  }

  // 1. Allocate memory.
  if (!ActorDispatcher::enable_use_trace_memory()) {
    if (!memory_alloc_list_.empty()) {
      SendMemoryAllocReq(context);
    }
  } else if (!max_ref_cnt_output_list_.empty()) {
    // Allocate dynamic memory for graph output.
    MemoryManagerActor::GetInstance()->AllocateMemory(&max_ref_cnt_output_list_, device_contexts_[0], context,
                                                      GetAID());
  }

  if (MS_UNLIKELY(IsRunningFailed(context))) {
    MS_VLOG(VL_RUNTIME_FRAMEWORK_KERNEL) << "Run failed and early stop launch kernel: "
                                         << kernel_->fullname_with_scope();
    return;
  }

  // For performance, Only kernel need user data (such as PyExecute op) need call 'PreLaunchKernel', the
  // 'PreLaunchKernel' will be removed in the future.
  if (MS_UNLIKELY(ActorDispatcher::has_kernel_need_user_data())) {
    PreLaunchKernel(context);
  }

  if (MS_UNLIKELY(debug_aid_ != nullptr)) {
    ActorDispatcher::SendSync(*debug_aid_, &DebugActor::DebugPreLaunch, kernel_, input_kernel_tensors_,
                              output_kernel_tensors_, device_contexts_[0], context, &GetAID());
  }

  if (!LaunchKernel(context, IsSkippedLaunch(kernel_, nullptr))) {
    MS_LOG_WITH_NODE(EXCEPTION, kernel_) << "#umsg#Kernel error:#umsg#Launch kernel failed: " +
                                              kernel_->fullname_with_scope()
                                         << trace::DumpSourceLines(kernel_);
  }

  if (MS_UNLIKELY(recorder_aid_ != nullptr)) {
    SetMemInfoForRdr();
    ActorDispatcher::Send(*recorder_aid_, &RecorderActor::RecordInfo, kernel_->fullname_with_scope(), &mem_info_,
                          device_contexts_[0], context);
  }

  if (is_dynamic_shape_ && kernel_mod_->IsNeedUpdateOutputShapeAndSize()) {
    kernel_mod_->UpdateOutputShapeAndSize(input_launch_tensors_, output_launch_tensors_);
  }

  if (MS_UNLIKELY(kernel_mod_->need_user_data())) {
    for_each(output_kernel_tensors_.begin(), output_kernel_tensors_.end(),
             [](auto &kernel_tensor) { kernel_tensor->set_need_sync_user_data(true); });
  }

  if ((modifiable_ref_input_indexes_.size() != 0) || (modifiable_ref_output_indexes_.size() != 0)) {
    RefreshDeviceTensorCopyStore(context);
  }

  // 3. Fix ref count.
  if (!ActorDispatcher::enable_use_trace_memory()) {
    if (new_memory_free_list_.size() > 0 && copy_output_kernel_tensors_.empty()) {
      SendMemoryFreeReq(context);
    }
  }

  if (enable_hyper_offload_slide_) {
    MemUseAnalyzer::GetInstance().LaunchTaskAfter(this, device_contexts_[0]);
  }
}

void KernelRunner::ExecuteLaunchKernelTaskHP(OpContext<KernelTensor> *const context) {
  if (MS_UNLIKELY(IsRunningFailed(context))) {
    MS_VLOG(VL_RUNTIME_FRAMEWORK_KERNEL) << "Run failed and early stop launch kernel: "
                                         << kernel_->fullname_with_scope();
    return;
  }

  if (enable_hyper_offload_slide_) {
    MemUseAnalyzer::GetInstance().LaunchTaskBefore(this, device_contexts_[0]);
  }

  // 1. Allocate memory.
  if (!ActorDispatcher::enable_use_trace_memory()) {
    if (!memory_alloc_list_.empty()) {
      SendMemoryAllocReqHP(context);
    }
  } else if (!max_ref_cnt_output_list_.empty()) {
    // Allocate dynamic memory for graph output.
    MemoryManagerActor::GetInstance()->AllocateMemory(&max_ref_cnt_output_list_, device_contexts_[0], context,
                                                      GetAID());
  }

  if (MS_UNLIKELY(IsRunningFailed(context))) {
    MS_VLOG(VL_RUNTIME_FRAMEWORK_KERNEL) << "Run failed and early stop launch kernel: "
                                         << kernel_->fullname_with_scope();
    return;
  }

  // For performance, Only kernel need user data (such as PyExecute op) need call 'PreLaunchKernel', the
  // 'PreLaunchKernel' will be removed in the future.
  if (MS_UNLIKELY(ActorDispatcher::has_kernel_need_user_data())) {
    PreLaunchKernel(context);
  }

  if (!LaunchKernelHP(context, IsSkippedLaunch(kernel_, nullptr))) {
    MS_LOG_WITH_NODE(EXCEPTION, kernel_) << "#umsg#Kernel error:#umsg#Launch kernel failed: " +
                                              kernel_->fullname_with_scope()
                                         << trace::DumpSourceLines(kernel_);
  }

  if (is_dynamic_shape_ && kernel_mod_->IsNeedUpdateOutputShapeAndSize()) {
    kernel_mod_->UpdateOutputShapeAndSize(input_launch_tensors_, output_launch_tensors_);
  }

  if (MS_UNLIKELY(kernel_mod_->need_user_data())) {
    for_each(output_kernel_tensors_.begin(), output_kernel_tensors_.end(),
             [](auto &kernel_tensor) { kernel_tensor->set_need_sync_user_data(true); });
  }

  if ((modifiable_ref_input_indexes_.size() != 0) || (modifiable_ref_output_indexes_.size() != 0)) {
    RefreshDeviceTensorCopyStore(context);
  }

  // 3. Fix ref count.
  if (!ActorDispatcher::enable_use_trace_memory()) {
    if (new_memory_free_list_.size() > 0 && copy_output_kernel_tensors_.empty()) {
      SendMemoryFreeReq(context);
    }
  }

  if (enable_hyper_offload_slide_) {
    MemUseAnalyzer::GetInstance().LaunchTaskAfter(this, device_contexts_[0]);
  }
}

void KernelRunner::InferAndUpdateDeviceTensorSize(OpContext<KernelTensor> *const context) {
  // For static shape, aclnn kernel with view input need to get input tensor storage info by resize.
  bool view_input = false;
  if (!need_check_tensor_contiguous_) {
    auto it = std::find_if(input_kernel_tensors_.begin(), input_kernel_tensors_.end(),
                           [](const KernelTensorPtr &tensor) { return tensor->tensor_storage_info() != nullptr; });
    if (it != input_kernel_tensors_.end()) {
      view_input = true;
    }
  }
  if (has_dynamic_) {
    // Infer shape and resize for dynamic shape or dynamice value case when disable runtime multi pipeline.
    InferAndResize(context);
    FetchOutputDeviceTensor(context);
    FetchWorkspaceDeviceTensor();
  } else if (view_input) {
    ResizeKernelMod();
    FetchOutputDeviceTensor(context);
    FetchWorkspaceDeviceTensor();
  } else {
    FetchOutputDeviceTensor(context);
  }
}

void KernelRunner::InferAndResize(OpContext<KernelTensor> *const context) {
  if (!enable_async_infer_) {
    // If the kernel need user data and is dynamic, maybe need input kernel's output user data to infer shape, this
    // value depend case can not handle in KernelTensor auto sync phase currently.
    if (ActorDispatcher::enable_async_launch_kernel() && kernel_mod_->need_user_data() &&
        !WaitRuntimePipelineFinish(context, GetAID().Name())) {
      MS_LOG(INFO) << "Run failed and early stop for kernel: " << kernel_->fullname_with_scope();
      return;
    }

    if (is_dynamic_type_) {
      ProfilerRecorder profiler(ProfilerModule::kKernel, ProfilerEvent::kKernelInferAndResize, GetAID().Name());
      // For dynamic shape case, need Re-InferShape and Resize kernel mod.
      InferShapeAndType();
      ResizeKernelMod();
    } else if (is_dynamic_shape_) {
      ProfilerRecorder profiler(ProfilerModule::kKernel, ProfilerEvent::kKernelInferAndResize, GetAID().Name());
      // For dynamic shape case, need Re-InferShape and Resize kernel mod.
      InferShape();
      ResizeKernelMod();
    } else if (is_dynamic_value_) {
      ProfilerRecorder profiler(ProfilerModule::kKernel, ProfilerEvent::kKernelResize, GetAID().Name());
      ResizeKernelMod();
    }

    return;
  }

  if (is_dynamic_value_ && !is_dynamic_shape_ && !is_dynamic_type_) {
    ProfilerRecorder profiler(ProfilerModule::kKernel, ProfilerEvent::kKernelResize, GetAID().Name());
    ResizeKernelMod();
  }
}

void KernelRunner::InferShapeAndType() {
  MS_VLOG(VL_RUNTIME_FRAMEWORK_KERNEL) << "Begin InferShapeAnyType for kernel: " << kernel_->fullname_with_scope()
                                       << ", inputs: " << input_kernel_tensors_for_infer_;
  // 1. Infer operator's output's Shape and Type.
  auto abstract = opt::dynamic_shape::InferShapeAndType(kernel_mod_->primitive(), input_kernel_tensors_for_infer_);
  MS_EXCEPTION_IF_NULL(abstract);
  MS_VLOG(VL_RUNTIME_FRAMEWORK_KERNEL) << "End InferShapeAnyType for kernel: " << kernel_->fullname_with_scope()
                                       << ", abstract: " << abstract->ToString();
  // 2. Update shape of output kernel tensor.
  opt::dynamic_shape::UpdateKernelTensorType(abstract->GetType(), output_launch_tensors_);
  opt::dynamic_shape::UpdateKernelTensorShape(abstract->GetShape(), output_launch_tensors_);
}

void KernelRunner::InferShape() {
  MS_VLOG(VL_RUNTIME_FRAMEWORK_KERNEL) << "Begin InferShape for kernel: " << kernel_->fullname_with_scope()
                                       << ", inputs: " << input_kernel_tensors_for_infer_;
  // 1. Infer operator's output's Shape.
  auto base_shape = opt::dynamic_shape::InferShape(kernel_mod_->primitive(), input_kernel_tensors_for_infer_);
  MS_EXCEPTION_IF_NULL(base_shape);
  MS_VLOG(VL_RUNTIME_FRAMEWORK_KERNEL) << "End InferShape for kernel: " << kernel_->fullname_with_scope()
                                       << ", shape: " << base_shape->ToString();

  // 2. Update shape of output kernel tensor.
  opt::dynamic_shape::UpdateKernelTensorShape(base_shape, output_launch_tensors_);
}

void KernelRunner::ResizeKernelMod() {
  ProfilerRecorder profiler(ProfilerModule::kKernel, ProfilerEvent::kKernelResizeInner, GetAID().Name(), true);
  MS_VLOG(VL_RUNTIME_FRAMEWORK_KERNEL) << "Begin Resize kernel mod for kernel: " << kernel_->fullname_with_scope();
  int ret = kernel_mod_->Resize(input_launch_tensors_, output_launch_tensors_);
  MS_VLOG(VL_RUNTIME_FRAMEWORK_KERNEL) << "End Resize kernel mod for kernel: " << kernel_->fullname_with_scope()
                                       << ", the output size list: " << kernel_mod_->GetOutputSizeList()
                                       << ", workspace size list: " << kernel_mod_->GetWorkspaceSizeList();
  if (ret != kernel::KRET_OK) {
    MS_LOG_WITH_NODE(EXCEPTION, kernel_) << "Resize failed for kernel: " << kernel_->fullname_with_scope();
  }
}

void KernelRunner::DispatchDebugActor(OpContext<KernelTensor> *const context) {
  // Debug actor is blocked, must wait debug actor callback message to process continue.
  if (debug_aid_ != nullptr) {
    ActorDispatcher::SendSync(*debug_aid_, &DebugActor::DebugPostLaunch, kernel_, input_kernel_tensors_,
                              output_kernel_tensors_, device_contexts_[0], context, &GetAID());
  }
}

bool KernelRunner::PreLaunchKernel() {
  auto &callbacks = device::KernelExecutor::GetLaunchKernelCallbacks<device::PreLaunchKernelFunc>();
  for (auto cb_obj : callbacks) {
    // skip callbacks not for the device
    if (cb_obj->device_type_ != device::DeviceType::kNone &&
        cb_obj->device_type_ != device_contexts_[0]->GetDeviceType()) {
      continue;
    }
    auto skip_launch = cb_obj->fn_callback_(kernel_, input_launch_tensors_, output_launch_tensors_, kernel_mod_,
                                            stream_, device_contexts_[0]);
    if (skip_launch) {
      MS_LOG(INFO) << "Callback " << cb_obj->name_ << " skip launch kernel " << kernel_->fullname_with_scope();
      return true;
    }
  }
  return false;
}

void KernelRunner::PostLaunchKernel(bool launch_success) {
  auto &callbacks = device::KernelExecutor::GetLaunchKernelCallbacks<device::PostLaunchKernelFunc>();
  for (auto cb_obj : callbacks) {
    // skip callbacks not for the device
    if (cb_obj->device_type_ != device::DeviceType::kNone &&
        cb_obj->device_type_ != device_contexts_[0]->GetDeviceType()) {
      continue;
    }
    cb_obj->fn_callback_(kernel_, input_launch_tensors_, output_launch_tensors_, kernel_mod_, stream_,
                         device_contexts_[0], launch_success);
  }
}

bool KernelRunner::LaunchKernelWithDebug(OpContext<KernelTensor> *const context, bool skip_launch) {
  MS_VLOG(VL_RUNTIME_FRAMEWORK_KERNEL) << "Begin launch kernel: " << kernel_->fullname_with_scope();
  static bool is_enable_mem_tracker = device::tracker::MemTrackerManager::GetInstance().IsEnabled();
  if (MS_UNLIKELY(is_enable_mem_tracker)) {
    AddNodeToGraphTracker(kernel_, GetAID().Name());
    TrackInputOutputMemory(input_launch_tensors_, output_launch_tensors_, GetAID().Name(), depend_shape_input_list_);
  } else {
    if (MS_UNLIKELY(device::tracker::MemTrackerManager::GetInstance().enable_memory_debug_info())) {
      AddNodeMemTrackerInfo(kernel_, GetAID().Name(), is_stream_recv_actor_);
    }
  }
  bool ret = true;
  if (!skip_launch) {
    skip_launch = PreLaunchKernel();
  }
  if (!skip_launch) {
    ret = device_contexts_[0]->GetKernelExecutor()->LaunchKernel(
      kernel_, input_launch_tensors_, workspace_launch_tensors_, output_launch_tensors_, kernel_mod_, stream_);
    PostLaunchKernel(ret);
  }
  MS_VLOG(VL_RUNTIME_FRAMEWORK_KERNEL) << "End launch kernel: " << kernel_->fullname_with_scope();
  DispatchDebugActor(context);
  return ret;
}

bool KernelRunner::LaunchKernel(OpContext<KernelTensor> *const context, bool is_skip_launch) {
  static KernelCache &cache = KernelCache::GetInstance();
  if (cache.need_add) {
    cache.Add(kernel_);
  }

  if (MS_UNLIKELY(EnableExecuteOrderDump())) {
    auto &execute_order_tracker = ExecuteOrderTracker::GetInstance();
    execute_order_tracker.ProcessNode(kernel_);
  }
  static bool is_enable_mem_tracker = device::tracker::MemTrackerManager::GetInstance().IsEnabled();
  if (MS_UNLIKELY(skip_launch_shape_related_op_)) {
    MS_VLOG(VL_RUNTIME_FRAMEWORK_KERNEL) << "Skip launch real make tuple kernel: " << kernel_->fullname_with_scope()
                                         << " input kernel tensor: " << input_kernel_tensors_;
    if (is_enable_mem_tracker) {
      AddNodeToGraphTracker(kernel_, GetAID().Name());
      TrackInputOutputMemory(input_launch_tensors_, output_launch_tensors_, GetAID().Name(), depend_shape_input_list_);
    } else {
      if (device::tracker::MemTrackerManager::GetInstance().enable_memory_debug_info()) {
        AddNodeMemTrackerInfo(kernel_, GetAID().Name(), is_stream_recv_actor_);
      }
    }
    return true;
  }
  // Check the skipped launch condition.
  if (is_launch_skipped_) {
    MS_EXCEPTION_IF_CHECK_FAIL((input_kernel_tensors_.size() >= 1), "The inputs size is wrong.");
    MS_EXCEPTION_IF_CHECK_FAIL((output_kernel_tensors_.size() >= 1), "The outputs size is wrong.");
    MS_EXCEPTION_IF_NULL(input_kernel_tensors_[0]);
    MS_EXCEPTION_IF_NULL(output_kernel_tensors_[0]);
    auto &input_device_tensor = input_kernel_tensors_[0]->device_address();
    MS_EXCEPTION_IF_NULL(input_device_tensor);
    auto &output_device_tensor = output_kernel_tensors_[0]->device_address();
    if (input_device_tensor->GetPtr() == output_device_tensor->GetPtr()) {
      MS_VLOG(VL_RUNTIME_FRAMEWORK_KERNEL) << "Skipped launch kernel: " << kernel_->fullname_with_scope();
      DispatchDebugActor(context);
      if (is_enable_mem_tracker) {
        AddNodeToGraphTracker(kernel_, GetAID().Name());
        TrackInputOutputMemory(input_launch_tensors_, output_launch_tensors_, GetAID().Name(),
                               depend_shape_input_list_);
      } else {
        if (device::tracker::MemTrackerManager::GetInstance().enable_memory_debug_info()) {
          AddNodeMemTrackerInfo(kernel_, GetAID().Name(), is_stream_recv_actor_);
        }
      }
      return true;
    } else {
      MS_LOG(ERROR) << "Input address:" << input_device_tensor->GetPtr()
                    << " and output address:" << output_device_tensor->GetPtr()
                    << " are not equal of skipped launch actor: " << GetAID().Name();
      return false;
    }
  }
  // Make tensor contiguous if needed
  if (need_check_tensor_contiguous_) {
    ConvertInputContiguous(context);
  }

  // Cpu not support stream lock with LaunchKernel.
  if (!ActorDispatcher::enable_multi_stream() || is_multi_stream_process_skipped_) {
    auto ret = LaunchKernelWithDebug(context, is_skip_launch);
    RecoverInputs();
    return ret;
  }

  auto &multi_stream_controller = device::DeviceContextManager::GetInstance().GetMultiStreamController(
    device_contexts_[0]->device_context_key().device_type_);
  bool ret = false;
  if (!ActorDispatcher::enable_async_launch_kernel()) {
    std::lock_guard<std::mutex> lock(multi_stream_controller->GetStreamMutex(kernel_info_->stream_id()));
    ProcessMultiStreamBeforeKernelLaunch(context);
    ret = LaunchKernelWithDebug(context, is_skip_launch);
    ProcessMultiStreamAfterKernelLaunch(context);
  } else {
    ProcessMultiStreamBeforeKernelLaunch(context);
    ret = LaunchKernelWithDebug(context, is_skip_launch);
    ProcessMultiStreamAfterKernelLaunch(context);
  }
  RecoverInputs();
  return ret;
}

bool KernelRunner::LaunchKernelHP(OpContext<KernelTensor> *const context, bool is_skip_launch) {
  static KernelCache &cache = KernelCache::GetInstance();
  if (cache.need_add) {
    cache.Add(kernel_);
  }

  if (MS_UNLIKELY(skip_launch_shape_related_op_)) {
    MS_VLOG(VL_RUNTIME_FRAMEWORK_KERNEL) << "Skip launch real make tuple kernel: " << kernel_->fullname_with_scope()
                                         << " input kernel tensor: " << input_kernel_tensors_;
    return true;
  }

  // Check the skipped launch condition.
  if (is_launch_skipped_) {
    MS_EXCEPTION_IF_CHECK_FAIL((input_kernel_tensors_.size() >= 1), "The inputs size is wrong.");
    MS_EXCEPTION_IF_CHECK_FAIL((output_kernel_tensors_.size() >= 1), "The outputs size is wrong.");
    auto &input_device_tensor = input_kernel_tensors_[0]->device_address();
    auto &output_device_tensor = output_kernel_tensors_[0]->device_address();
    if (input_device_tensor->GetPtr() == output_device_tensor->GetPtr()) {
      MS_VLOG(VL_RUNTIME_FRAMEWORK_KERNEL) << "Skipped launch kernel: " << kernel_->fullname_with_scope();
      return true;
    } else {
      MS_LOG(ERROR) << "Input address:" << input_device_tensor->GetPtr()
                    << " and output address:" << output_device_tensor->GetPtr()
                    << " are not equal of skipped launch actor: " << GetAID().Name();
      return false;
    }
  }
  // Make tensor contiguous if needed
  if (need_check_tensor_contiguous_) {
    ConvertInputContiguous(context);
  }

  // Cpu not support stream lock with LaunchKernel.
  MS_VLOG(VL_RUNTIME_FRAMEWORK_KERNEL) << "Begin launch kernel: " << kernel_->fullname_with_scope();
  bool ret = true;
  if (!ActorDispatcher::enable_multi_stream() || is_multi_stream_process_skipped_) {
    if (!is_skip_launch) {
      ret = kernel_mod_->Launch(input_launch_tensors_, workspace_launch_tensors_, output_launch_tensors_, stream_);
    }
  } else {
    auto &multi_stream_controller = device::DeviceContextManager::GetInstance().GetMultiStreamController(
      device_contexts_[0]->device_context_key().device_type_);
    if (!ActorDispatcher::enable_async_launch_kernel()) {
      std::lock_guard<std::mutex> lock(multi_stream_controller->GetStreamMutex(kernel_info_->stream_id()));
      ProcessMultiStreamBeforeKernelLaunch(context);
      if (!is_skip_launch) {
        ret = kernel_mod_->Launch(input_launch_tensors_, workspace_launch_tensors_, output_launch_tensors_, stream_);
      }
      ProcessMultiStreamAfterKernelLaunch(context);
    } else {
      ProcessMultiStreamBeforeKernelLaunch(context);
      if (!is_skip_launch) {
        ret = kernel_mod_->Launch(input_launch_tensors_, workspace_launch_tensors_, output_launch_tensors_, stream_);
      }
      ProcessMultiStreamAfterKernelLaunch(context);
    }
  }

  MS_VLOG(VL_RUNTIME_FRAMEWORK_KERNEL) << "End launch kernel: " << kernel_->fullname_with_scope();
  RecoverInputs();
  return ret;
}

void KernelRunner::ProcessMultiStreamBeforeKernelLaunch(OpContext<KernelTensor> *const context) {
  ProfilerRecorder profiler(ProfilerModule::kKernel, ProfilerEvent::kProcessMultiStream, GetAID().Name());
  auto device_context = device_contexts_[0];
  auto stream_id = kernel_info_->stream_id();
  // Update output_kernel_tensors_ with task id on stream.
  auto &multi_stream_controller = device::DeviceContextManager::GetInstance().GetMultiStreamController(
    device_context->device_context_key().device_type_);
  auto task_id_on_stream = multi_stream_controller->LaunchTaskIdOnStream(stream_id);
  // Adapter for mc2 kernel, need more process later.
  if (is_mc2_kernel_) {
    multi_stream_controller->DispatchRecordWaitEvent(kDefaultStreamIndex, kWorldGroupStreamIndex);
  }
  MS_LOG(DEBUG) << "device context : " << device_context
                << ", type : " << device_context->device_context_key().device_type_ << ", stream id : " << stream_id
                << ", actor name : " << GetAID().Name() << ", task_id_on_stream : " << task_id_on_stream << ".";
  if (INT64_MAX == task_id_on_stream) {
    // Cpu kernel task id on stream is meanless.
    *task_id_on_stream_ = 0;
    MS_LOG(DEBUG) << "Skip ProcessMultiStreamBeforeKernelLaunch since kernel type is CPU.";
    return;
  }
  *task_id_on_stream_ = task_id_on_stream;

  if (enable_input_optimize_ && insert_input_event_) {
    InsertEventForInput(stream_id, device_contexts_[0]);
  }

  // Process wait stream.
  if (is_stream_recv_actor_) {
    // Note: wait node start to launch. Event was record on send node, so, we can releases events on send node stream.
    // Release events on send node means memory stream id is recv node stream id and user stream id is send node
    // stream id.
    auto user_stream_id = kernel_mod_->record_stream_id();
    auto memory_stream_id = stream_id;
    if (stream_send_actor_ == nullptr) {
      // Gpu not add stream send/recv pair, nullptr is normal case.
      MS_LOG(DEBUG) << "Stream_send_actor_ is nullptr.";
      return;
    }
    MS_LOG(DEBUG) << "Process wait stream start, memory_stream_id : " << memory_stream_id
                  << ", send task id on stream : " << *(stream_send_actor_->task_id_on_stream_) << ".";
    // Here, need get task id on stream from send node.
    (void)multi_stream_controller->WaitEvent(*(stream_send_actor_->task_id_on_stream_), user_stream_id,
                                             memory_stream_id);
    return;
  }

  // Process inputs.
  if (input_kernel_tensors_.empty()) {
    return;
  }

  std::vector<KernelTensor *> cross_stream_kernel_tensors;
  size_t index = 0;
  for (const auto &input_kernel_tensor : input_kernel_tensors_) {
    if (is_monad_input_[index++]) {
      continue;
    }
    MS_EXCEPTION_IF_NULL(input_kernel_tensor);
    if (input_kernel_tensor->stream_id() == stream_id) {
      continue;
    }
    if (input_kernel_tensor->task_id_on_stream() == nullptr) {
      input_kernel_tensor->set_task_id_on_stream(task_id_on_stream_);
      MS_LOG(DEBUG) << "Input_kernel_tensor : " << input_kernel_tensor
                    << " task id on stream is nullptr, set task id for it.";
    }
    if (input_kernel_tensor->managed_by_somas()) {
      MS_LOG(DEBUG) << "Input_kernel_tensor : " << input_kernel_tensor << " is managed by somas.";
      continue;
    }
    // Nullptr device ptr is normal case, here need skip these inputs.
    if (input_kernel_tensor->device_ptr() == nullptr) {
      MS_LOG(DEBUG) << "Input kernel tensor device ptr is nullptr.";
      continue;
    }
    (void)cross_stream_addresses_.emplace_back(kDefaultStreamIndex, input_kernel_tensor->device_ptr());
    if (!is_multi_stream_safe_) {
      (void)cross_stream_kernel_tensors.emplace_back(input_kernel_tensor.get());
    }
  }

  // Dispatch record/wait.
  if (!is_multi_stream_safe_) {
    for (const auto &cross_stream_kernel_tensor : cross_stream_kernel_tensors) {
      // Nullptr of task id on stream is normal case.
      // If cross_stream_kernel_tensor's task id on stream is nullptr, kernel tensor must be safe.
      // Data prepare actor, data source actor and so on has prepare device tensors without task id on stream, and
      // those device tensors is multi-stream safe.
      if (cross_stream_kernel_tensor->task_id_on_stream() == nullptr) {
        continue;
      }
      // Input kernel tensor is memory stream id, this is important.
      auto user_stream_id = stream_id;
      auto memory_stream_id = cross_stream_kernel_tensor->stream_id();
      auto memory_task_id_on_stream = *cross_stream_kernel_tensor->task_id_on_stream();
      auto safe_task_id_on_stream = multi_stream_controller->QueryTaskIdOnStream(user_stream_id, memory_stream_id);
      if (safe_task_id_on_stream >= memory_task_id_on_stream) {
        MS_LOG(DEBUG) << "Safe_task_id_on_stream : " << safe_task_id_on_stream
                      << " is bigger than memory_task_id_on_stream : " << memory_task_id_on_stream << ".";
        continue;
      }
      multi_stream_controller->DispatchRecordWaitEvent(user_stream_id, memory_stream_id);
      // Add recv process.
      user_stream_id = memory_stream_id;
      memory_stream_id = stream_id;
      auto last_task_id_on_stream = multi_stream_controller->GetTaskIdOnStream(user_stream_id);
      MS_LOG(DEBUG) << "Dispatch wait stream start, user_stream_id : " << user_stream_id
                    << ", memory_stream_id : " << memory_stream_id
                    << ", last_task_id_on_stream : " << last_task_id_on_stream << ".";
      // Here, need get task id on stream from send node.
      (void)multi_stream_controller->WaitEvent(last_task_id_on_stream, user_stream_id, memory_stream_id);
    }
  }
}

void KernelRunner::ProcessMultiStreamAfterKernelLaunch(OpContext<KernelTensor> *const context) {
  auto stream_id = kernel_info_->stream_id();
  // Record event when op is not in default stream and not in multi stream case.
  if (stream_id != kDefaultStreamIndex && !kernel_->HasAttr(kAttrStreamId)) {
    for (const auto &workspace_kernel_tensor : workspace_kernel_tensors_) {
      cross_stream_addresses_.emplace_back(kDefaultStreamIndex, workspace_kernel_tensor->device_ptr());
    }
    for (const auto &input_kernel_tensor : input_kernel_tensors_) {
      if (input_kernel_tensor->stream_id() == stream_id) {
        cross_stream_addresses_.emplace_back(kDefaultStreamIndex, input_kernel_tensor->device_ptr());
      }
    }
    for (const auto &output_kernel_tensor : output_kernel_tensors_) {
      cross_stream_addresses_.emplace_back(kDefaultStreamIndex, output_kernel_tensor->device_ptr());
    }

    // Record event.
    if (!cross_stream_addresses_.empty()) {
      MS_LOG(DEBUG) << "Record event for kernel : " << kernel_->fullname_with_scope()
                    << ", addresses size : " << cross_stream_addresses_.size() << ".";
      // Record event on stream.
      auto device_context = device_contexts_[0];
      auto &multi_stream_controller = device::DeviceContextManager::GetInstance().GetMultiStreamController(
        device_context->device_context_key().device_type_);
      multi_stream_controller->RecordEvent(*task_id_on_stream_, stream_id, cross_stream_addresses_);
    }
  }
  // Reset cross stream addresses.
  cross_stream_addresses_.clear();
  // Add ref processes for sync stream on demand.
  // For 2.7.1 version. Broadcast inplace tuple input is not supported, so we add hard code here.
  // Remove this hard code after inplace input of AbstractTuple type is supported.
  if ((!rw_write_index_.empty() || common::AnfAlgo::GetCNodeName(kernel_) == kBroadcastOpName) &&
      stream_id != kDefaultStreamIndex) {
    auto &multi_stream_controller = device::DeviceContextManager::GetInstance().GetMultiStreamController(
      device_contexts_[0]->device_context_key().device_type_);
    multi_stream_controller->DispatchRecordWaitEvent(kDefaultStreamIndex, stream_id);
  }
}

void KernelRunner::PreLaunchKernel(OpContext<KernelTensor> *) {
  for (size_t i = 0; i < input_kernel_tensors_.size(); ++i) {
    if (input_kernel_tensors_[i] == nullptr) {
      continue;
    }
    auto &input_device_tensor = input_kernel_tensors_[i]->device_address();
    if (input_device_tensor == nullptr || !input_kernel_tensors_[i]->GetValidPtr(kernel_info_->stream_id())) {
      MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
        << "For kernel: " << kernel_->fullname_with_scope() << ", input device tensor " << input_device_tensor
        << " has no device ptr.";
    }
  }

  for (size_t i = 0; i < output_kernel_tensors_.size(); ++i) {
    if (output_kernel_tensors_[i] == nullptr) {
      continue;
    }
    auto &output_device_tensor = output_kernel_tensors_[i]->device_address();
    if (!output_kernel_tensors_[i]->GetValidPtr(kernel_info_->stream_id())) {
      MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
        << "For kernel: " << kernel_->fullname_with_scope() << ", output device tensor " << output_device_tensor
        << " has no device ptr.";
    }
  }

  for (size_t i = 0; i < workspace_kernel_tensors_.size(); ++i) {
    if (workspace_kernel_tensors_[i] == nullptr) {
      continue;
    }
    auto workspace_device_tensor = workspace_kernel_tensors_[i]->device_address().get();
    if (!workspace_kernel_tensors_[i]->GetValidPtr(kernel_info_->stream_id())) {
      MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
        << "For kernel: " << kernel_->fullname_with_scope() << ", workspace device tensor " << workspace_device_tensor
        << " has no device ptr.";
    }
  }
}

void KernelRunner::RefreshDeviceTensorCopyStore(OpContext<KernelTensor> *const context) {
  uint64_t start_time = 0;
  PROFILER_START(start_time);

  for (auto &ref_input_index : modifiable_ref_input_indexes_) {
    if (ref_input_index >= input_kernel_tensors_.size()) {
      std::stringstream ofs;
      ofs << "Invalid ref input index:" << ref_input_index
          << " input device tensor size:" << input_kernel_tensors_.size() << " for actor:" << GetAID();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, *context, ofs.str());
    }
    auto &input_kernel_tensor = input_kernel_tensors_[ref_input_index];
    MS_EXCEPTION_IF_NULL(input_kernel_tensor);
    auto need_refreshed_kernel_tensors = KernelTensorCopyStore::GetInstance().Fetch(input_kernel_tensor.get());
    MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
      << "Fetch input copy kernel tensor:" << input_kernel_tensor->ToString() << " for actor:" << GetAID();
    if (need_refreshed_kernel_tensors == nullptr) {
      continue;
    }
    for (auto &new_kernel_tensor : *need_refreshed_kernel_tensors) {
      MS_EXCEPTION_IF_NULL(new_kernel_tensor);
      MS_LOG(INFO) << GetAID().Name() << " the input position:" << ref_input_index
                   << " refresh from kernel tensor:" << input_kernel_tensor->ToString()
                   << " to kernel tensor:" << new_kernel_tensor->ToString();

      if (new_kernel_tensor->device_ptr() == nullptr ||
          new_kernel_tensor->device_ptr() == input_kernel_tensor->device_ptr()) {
        continue;
      }

      if (!SyncAllStreamForDeviceAddress(new_kernel_tensor->device_address(), input_kernel_tensor->device_address(),
                                         kDefaultStreamIndex, false) ||
          !SyncCopy(new_kernel_tensor, input_kernel_tensor.get(), kDefaultStreamIndex)) {
        std::string error_info = "Copy input device tensor failed: " + GetAID().Name();
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, *context, error_info);
      }
    }
  }

  for (auto &ref_output_index : modifiable_ref_output_indexes_) {
    if (ref_output_index >= output_kernel_tensors_.size()) {
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, *context, "The output index is of range.");
    }
    auto &output_kernel_tensor = output_kernel_tensors_[ref_output_index];
    MS_EXCEPTION_IF_NULL(output_kernel_tensor);
    auto need_refreshed_kernel_tensors = KernelTensorCopyStore::GetInstance().Fetch(output_kernel_tensor.get());
    MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
      << "Fetch output copy kernel tensor:" << output_kernel_tensor->ToString() << " for actor:" << GetAID();
    if (need_refreshed_kernel_tensors == nullptr) {
      continue;
    }
    for (auto &new_kernel_tensor : *need_refreshed_kernel_tensors) {
      MS_EXCEPTION_IF_NULL(new_kernel_tensor);
      MS_LOG(INFO) << GetAID().Name() << " the output position:" << ref_output_index
                   << " refresh from kernel tensor:" << output_kernel_tensor->ToString()
                   << " to:" << new_kernel_tensor->ToString();
      if (new_kernel_tensor->device_ptr() == nullptr ||
          new_kernel_tensor->device_ptr() == output_kernel_tensor->device_ptr()) {
        continue;
      }

      if (!SyncAllStreamForDeviceAddress(new_kernel_tensor->device_address(), output_kernel_tensor->device_address(),
                                         kDefaultStreamIndex, false) ||
          !SyncCopy(new_kernel_tensor, output_kernel_tensor.get(), kDefaultStreamIndex)) {
        std::string error_info = "Copy output device tensor failed: " + GetAID().Name();
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, *context, error_info);
      }
    }
  }

  PROFILER_END(start_time, ProfilerModule::kRuntime, ProfilerEvent::kPostLaunch, GetAID().Name(), false);
}

void KernelRunner::SendRecorderInfo(OpContext<KernelTensor> *const context) const {
  if (recorder_aid_ != nullptr && !ActorDispatcher::enable_async_launch_kernel()) {
    MS_EXCEPTION_IF_NULL(kernel_);
    ActorDispatcher::Send(*recorder_aid_, &RecorderActor::RecordInfo, kernel_->fullname_with_scope(), &mem_info_,
                          device_contexts_[0], context);
  }
}

void KernelRunner::SetInputDeviceTensor(const KernelTensorPtr &input_kernel_tensor, size_t input_index) {
  MS_EXCEPTION_IF_NULL(input_kernel_tensor);
  input_launch_tensors_[input_index] = input_kernel_tensor.get();
  input_kernel_tensors_[input_index] = input_kernel_tensor;
  input_kernel_tensors_for_infer_[input_index] = input_kernel_tensor;
}

void KernelRunner::ResetState() {
  MS_EXCEPTION_IF_NULL(kernel_);
  MS_LOG(INFO) << "Kernel actor " << kernel_->fullname_with_scope() << " start to reset state.";
  auto device_context = const_cast<DeviceContext *>(device_contexts_[0]);
  MS_LOG(INFO) << "Free output_device_tensor, list size: " << output_kernel_tensors_.size();
  for (auto kernel_tensor : output_kernel_tensors_) {
    if (kernel_tensor == nullptr || kernel_tensor->device_address() == nullptr) {
      continue;
    }
    auto device_tensor = kernel_tensor->device_address();
    if (kernel_tensor->new_ref_count() == SIZE_MAX) {
      continue;
    }
    if (device_tensor != nullptr && device_tensor->IsPtrValid()) {
      auto held_by_nodes = device_tensor->held_by_nodes();
      if (held_by_nodes.empty()) {
        FreeMemoryByDeviceContext(kernel_tensor->device_address().get(), device_context);
      } else {
        FreeMemoryByValueNode(held_by_nodes, device_tensor.get());
      }
    }
  }
  MS_LOG(INFO) << "Free workspace_device_tensor, list size: " << workspace_kernel_tensors_.size();
  for (auto kernel_tensor : workspace_kernel_tensors_) {
    if (kernel_tensor == nullptr || kernel_tensor->device_address() == nullptr) {
      continue;
    }
    auto device_tensor = kernel_tensor->device_address();
    if (device_tensor != nullptr && device_tensor->IsPtrValid()) {
      auto held_by_nodes = device_tensor->held_by_nodes();
      if (held_by_nodes.empty()) {
        FreeMemoryByDeviceContext(kernel_tensor->device_address().get(), device_context);
      } else {
        FreeMemoryByValueNode(held_by_nodes, device_tensor.get());
      }
    }
  }
  MS_LOG(INFO) << "Kernel actor " << kernel_->fullname_with_scope() << " end to reset state.";
}

void KernelRunner::FetchInputByTensorStore(std::vector<KernelTensor *> *const input_launch_tensors,
                                           std::vector<KernelTensorPtr> *const input_kernel_tensors,
                                           std::vector<abstract::AbstractBasePtr> *const input_kernel_tensors_for_infer,
                                           std::vector<KernelTensorPtr> *const memory_free_tensors,
                                           OpContext<KernelTensor> *const context) const {
  for (auto &device_tensor_store_key : device_tensor_store_keys_) {
    const auto &kernel_tensor = DeviceTensorStore::GetInstance().Fetch(device_tensor_store_key.second.get(),
                                                                       device_contexts_[0]->GetDeviceType());
    if (kernel_tensor == nullptr || kernel_tensor->device_address() == nullptr) {
      std::string error_info =
        GetAID().Name() + " get device tensor store failed: " + device_tensor_store_key.second->DebugString() +
        ", device type:" + std::to_string(static_cast<int>(device_contexts_[0]->GetDeviceType()));
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }

    // Collect the input kernel tensor.
    if (input_launch_tensors && input_kernel_tensors && input_kernel_tensors_for_infer &&
        ((*input_kernel_tensors)[device_tensor_store_key.first] != kernel_tensor)) {
      (*input_launch_tensors)[device_tensor_store_key.first] = kernel_tensor.get();
      (*input_kernel_tensors)[device_tensor_store_key.first] = kernel_tensor;
      (*input_kernel_tensors_for_infer)[device_tensor_store_key.first] = kernel_tensor;
      (*memory_free_tensors)[device_tensor_store_key.first] = kernel_tensor;
    }
  }
}

bool KernelRunner::IsRunningFailed(const OpContext<KernelTensor> *context) {
  static auto task_fail_cb = GET_COMMON_CALLBACK(TaskFailCallback, void, std::string *);
  if (task_fail_cb != nullptr) {
    task_fail_cb(&const_cast<OpContext<KernelTensor> *>(context)->error_info_);
  }

  return context->is_error_;
}

}  // namespace runtime
}  // namespace mindspore
