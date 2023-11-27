/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/kernel/hccl/hccl_kernel.h"

#include <map>
#include "ops/ascend_op_name.h"
#include "ops/other_op_name.h"
#include "ops/array_op_name.h"
#include "ops/framework_op_name.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/utils.h"
#include "utils/ms_context.h"
#include "runtime/device/kernel_runtime.h"
#include "plugin/device/ascend/hal/hccl_adapter/hccl_adapter.h"
#include "plugin/device/ascend/hal/hardware/ascend_collective_comm_lib.h"
#include "plugin/device/ascend/hal/device/ascend_memory_manager.h"

using AscendCollectiveCommLib = mindspore::device::ascend::AscendCollectiveCommLib;
namespace {
static std::map<std::string, std::string> kMsOpNameToHcomHcclType = {
  {mindspore::kAllReduceOpName, mindspore::kHcomOpTypeAllReduce},
  {mindspore::kReduceOpName, mindspore::kHcomOpTypeReduce},
  {mindspore::kAllGatherOpName, mindspore::kHcomOpTypeAllGather},
  {mindspore::kBroadcastOpName, mindspore::kHcomOpTypeBroadcast},
  {mindspore::kSendOpName, mindspore::kHcomOpTypeSend},
  {mindspore::kReceiveOpName, mindspore::kHcomOpTypeReceive},
  {mindspore::kReduceScatterOpName, mindspore::kHcomOpTypeReduceScatter},
  {mindspore::kBarrierOpName, mindspore::kHcomOpTypeBarrier}};
std::string MsOpNameToHcomOpType(const std::string &ms_op_type) {
  auto iter = kMsOpNameToHcomHcclType.find(ms_op_type);
  if (iter == kMsOpNameToHcomHcclType.end()) {
    MS_LOG(EXCEPTION) << "Invalid MsOpType:" << ms_op_type;
  }
  return iter->second;
}
}  // namespace

namespace mindspore {
namespace kernel {
void HcclKernelFactory::Register(const std::string &name, HcclKernelCreater &&fun) {
  hccl_kernel_map_.emplace(name, fun);
}

std::shared_ptr<HcclKernel> HcclKernelFactory::Get(const std::string &name) {
  const auto &map = Get().hccl_kernel_map_;
  auto it = map.find(name);
  if (it != map.end() && it->second) {
    return (it->second)();
  }
  return nullptr;
}

HcclKernelFactory &HcclKernelFactory::Get() {
  static HcclKernelFactory _this{};
  return _this;
}

HcclKernel::HcclKernel()
    : hccl_count_(0),
      op_type_(::HcclReduceOp::HCCL_REDUCE_SUM),
      root_id_(0),
      src_rank_(0),
      dest_rank_(0),
      comm_(nullptr) {}

bool HcclKernel::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  // set source/destination rank
  if (kernel_name_ == kSendOpName || kernel_name_ == kReduceOpName || kernel_name_ == kMuxSendOpName) {
    if (!HcomUtil::GetHcomAttr<uint32_t, int64_t>(primitive_, kAttrDestRank, &dest_rank_)) {
      MS_LOG(ERROR) << "GetHcomDestRank fail!";
      return false;
    }
  } else if (kernel_name_ == kReceiveOpName) {
    if (!HcomUtil::GetHcomAttr<uint32_t, int64_t>(primitive_, kAttrSrcRank, &src_rank_)) {
      MS_LOG(ERROR) << "GetHcomSrcRank fail!";
      return false;
    }
  }

  if (!CalcTypeShapeAndCount(inputs, outputs)) {
    return false;
  }

  if (kernel_name_ == kAllReduceOpName || kernel_name_ == kReduceScatterOpName || kernel_name_ == kReduceOpName) {
    if (!HcomUtil::GetHcomOperationType(primitive_, &op_type_)) {
      MS_LOG(ERROR) << "GetHcomOperationType fail!";
      return false;
    }
  } else if (kernel_name_ == kBroadcastOpName) {
    if (!HcomUtil::GetHcomAttr<uint32_t, int64_t>(primitive_, kAttrRootRank, &root_id_)) {
      MS_LOG(ERROR) << "GetHcomRootId fail!";
      return false;
    }
  }

  if (!HcomUtil::GetHcomAttr<std::string>(primitive_, kAttrGroup, &group_)) {
    return false;
  }
  // pynative with ranktable also need hccl_comm
  comm_ = AscendCollectiveCommLib::GetInstance().HcclCommunicator(group_);
  if (common::UseHostCollective() && !hccl::HcclAdapter::GetInstance().UseHcclCM()) {
    MS_EXCEPTION_IF_NULL(comm_);
    primitive_->set_attr(kAttrComm, MakeValue<int64_t>(reinterpret_cast<int64_t>(comm_)));
  }
  CalLoopSize();

  return true;
}

HcclDataType HcclKernel::GetHcclDataType() const {
  if (hccl_data_type_list_.empty()) {
    MS_LOG(EXCEPTION) << "list hccl_data_type_list_ is empty.";
  }
  return hccl_data_type_list_[0];
}

void HcclKernel::CalLoopSize() {
  int64_t rank_size = 1;
  int64_t fusion = 0;

  (void)HcomUtil::GetHcomAttr<int64_t>(primitive_, kAttrRankSize, &rank_size);
  (void)HcomUtil::GetHcomAttr<int64_t>(primitive_, kAttrFusion, &fusion);

  if (hccl_data_type_list_.size() != hccl_kernel_input_shape_list_.size()) {
    MS_LOG(EXCEPTION) << "Invalid data type size " << hccl_data_type_list_.size() << " diff shape size "
                      << hccl_kernel_input_shape_list_.size();
  }
  loop_size_ = hccl_data_type_list_.size();
  if (hccl_kernel_input_shape_list_.size() > 1 && kernel_name_ == kAllGatherOpName && fusion >= 1) {
    loop_size_ *= static_cast<ulong>(rank_size);
  }
  if (kernel_name_ == kReduceScatterOpName && fusion >= 1) {
    loop_size_ = hccl_kernel_output_shape_list_.size();
  }
}

void HcclKernel::CalcWorkspaceSize(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &outputs) {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);

  bool is_task_sink = context_ptr->get_param<bool>(MS_CTX_ENABLE_TASK_SINK);
  auto mode = context_ptr->get_param<int>(MS_CTX_EXECUTION_MODE);
  // Not task sink mode.
  if (!workspace_size_list_.empty() || hccl_data_type_list_.empty() || (!is_task_sink && mode == kGraphMode) ||
      (mode == kPynativeMode && !is_graph_mode_)) {
    return;
  }

  // Task sink mode.
  workspace_size_list_.emplace_back(
    hccl::HcclAdapter::GetInstance().CalcWorkspaceSize(primitive_, inputs, outputs, hccl_data_type_list_[0]));
  return;
}

bool HcclKernel::CalcTypeShapeAndCount(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &outputs) {
  hccl_kernel_input_shape_list_.clear();
  hccl_kernel_output_shape_list_.clear();

  // set hccl kernel input/output shape
  std::transform(inputs.cbegin(), inputs.cend(), std::back_inserter(hccl_kernel_input_shape_list_),
                 [](KernelTensor *kernel_tensor) { return kernel_tensor->GetShapeVector(); });
  std::transform(outputs.cbegin(), outputs.cend(), std::back_inserter(hccl_kernel_output_shape_list_),
                 [](KernelTensor *kernel_tensor) { return kernel_tensor->GetShapeVector(); });

  // set hccl data_type and count
  if (!HcomUtil::GetHcomDataType(kernel_name_, inputs, outputs, &hccl_data_type_list_)) {
    MS_LOG(ERROR) << "GetHcomDataType fail!";
    return false;
  }
  if (!HcomUtil::GetHcomCount(
        primitive_, hccl_data_type_list_,
        HcomUtil::IsReceiveOp(kernel_name_) ? hccl_kernel_output_shape_list_ : hccl_kernel_input_shape_list_,
        inputs.size(), &hccl_count_)) {
    MS_LOG(ERROR) << "GetHcomCount fail!";
    return false;
  }

  return true;
}

bool HcclKernel::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                        const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);

  if (inputs.empty() && outputs.empty()) {
    MS_LOG(ERROR) << "Hccl kernel input or output is empty.";
    return false;
  }
  if (hccl_data_type_list_.empty()) {
    MS_LOG(ERROR) << "Hccl data type list is empty.";
    return false;
  }

  MS_LOG(INFO) << "Start Execute: " << kernel_name_;
  std::string hccl_type = MsOpNameToHcomOpType(kernel_name_);
  HcclDataType data_type = hccl_data_type_list_[0];

  ::HcomOperation op_info;
  op_info.hcclType = hccl_type;
  op_info.inputPtr = inputs[0]->device_ptr();
  op_info.outputPtr = outputs[0]->device_ptr();
  op_info.dataType = static_cast<HcclDataType>(data_type);
  op_info.opType = static_cast<HcclReduceOp>(op_type_);
  op_info.root = root_id_;
  op_info.count = hccl_count_;

  auto callback = [this](HcclResult status) {
    if (status != HCCL_SUCCESS) {
      MS_LOG(ERROR) << "HcomExcutorInitialize failed, ret:" << status;
    }
    std::lock_guard<std::mutex> lock(this->hccl_mutex_);
    this->cond_.notify_all();
    MS_LOG(INFO) << "Hccl callback success.";
  };

  auto hccl_ret = hccl::HcclAdapter::GetInstance().HcclExecEnqueueOp(op_info, callback);
  if (hccl_ret != HCCL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Call EnqueueHcomOperation failed, node info: " << kernel_name_;
  }

  std::unique_lock<std::mutex> ulock(hccl_mutex_);
  cond_.wait(ulock);
  MS_LOG(INFO) << "Execute " << kernel_name_ << " success.";
  return true;
}

int HcclKernel::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (!CalcTypeShapeAndCount(inputs, outputs)) {
    return KRET_RESIZE_FAILED;
  }

  // update output_size_list_
  output_size_list_.clear();
  for (ulong i = 0; i < loop_size_; ++i) {
    size_t size = 0;
    if (!HcomUtil::GetHcclOpSize(GetHcclDataType(), hccl_kernel_output_shape_list_[i], &size)) {
      MS_LOG(INTERNAL_EXCEPTION) << "GetHcclOpOutputSize failed";
    }
    output_size_list_.push_back(size);
  }

  CalcWorkspaceSize(inputs, outputs);

  return KRET_OK;
}
}  // namespace kernel
}  // namespace mindspore
