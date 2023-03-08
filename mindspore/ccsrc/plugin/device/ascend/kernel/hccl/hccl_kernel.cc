/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/utils.h"
#include "utils/ms_context.h"
#include "runtime/device/kernel_runtime.h"
#include "plugin/device/ascend/hal/hccl_adapter/hccl_adapter.h"
#include "plugin/device/ascend/hal/device/ascend_memory_adapter.h"
#include "plugin/device/ascend/hal/hardware/ascend_collective_comm_lib.h"

using HcclTaskInfoPtr = std::shared_ptr<mindspore::ge::model_runner::HcclTaskInfo>;
using mindspore::ge::model_runner::HcclTaskInfo;
using AscendCollectiveCommLib = mindspore::device::ascend::AscendCollectiveCommLib;

namespace {
static std::map<std::string, std::string> kMsOpNameToHcomHcclType = {
  {mindspore::kAllReduceOpName, mindspore::kHcomOpTypeAllReduce},
  {mindspore::kAllGatherOpName, mindspore::kHcomOpTypeAllGather},
  {mindspore::kBroadcastOpName, mindspore::kHcomOpTypeBroadcast},
  {mindspore::kHcomSendOpName, mindspore::kHcomOpTypeSend},
  {mindspore::kReceiveOpName, mindspore::kHcomOpTypeReceive},
  {mindspore::kReduceScatterOpName, mindspore::kHcomOpTypeReduceScatter}};
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
HcclKernel::HcclKernel(const AnfNodePtr &anf_node)
    : AscendKernelMod(),
      hccl_count_(0),
      op_type_(::HcclReduceOp::HCCL_REDUCE_SUM),
      root_id_(0),
      src_rank_(0),
      dest_rank_(0),
      comm_(nullptr) {}
HcclKernel::~HcclKernel() {
  hccl_kernel_input_shape_list_.clear();
  hccl_kernel_output_shape_list_.clear();
  hccl_data_type_list_.clear();
  hccl_count_ = 0;
  op_type_ = ::HcclReduceOp::HCCL_REDUCE_SUM;
  root_id_ = 0;
  mutable_input_size_list_.clear();
  mutable_output_size_list_.clear();
  mutable_workspace_size_list_.clear();
}

bool HcclKernel::Init(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  op_name_ = common::AnfAlgo::GetCNodeName(anf_node);
  if (op_name_ == kHcomSendOpName) {
    if (!HcomUtil::GetHcomDestRank(anf_node, &dest_rank_)) {
      MS_LOG(ERROR) << "GetHcomDestRank fail!";
      return false;
    }
  }
  if (op_name_ == kReceiveOpName) {
    if (!HcomUtil::GetHcomSrcRank(anf_node, &src_rank_)) {
      MS_LOG(ERROR) << "GetHcomSrcRank fail!";
      return false;
    }
  }
  if (!HcomUtil::GetKernelInputShape(anf_node, &hccl_kernel_input_shape_list_)) {
    MS_LOG(ERROR) << "GetKernelInputShape fail!";
    return false;
  }
  if (!HcomUtil::GetKernelOutputShape(anf_node, &hccl_kernel_output_shape_list_)) {
    MS_LOG(ERROR) << "GetKernelOutputShape fail!";
    return false;
  }
  if (!HcomUtil::GetHcomDataType(anf_node, &hccl_data_type_list_)) {
    MS_LOG(ERROR) << "GetHcomDataType fail!";
    return false;
  }
  if (op_name_ == kReceiveOpName) {
    if (!HcomUtil::GetHcomCount(anf_node, hccl_data_type_list_, hccl_kernel_output_shape_list_, &hccl_count_)) {
      MS_LOG(ERROR) << "GetHcomCount fail!";
      return false;
    }
  } else {
    if (!HcomUtil::GetHcomCount(anf_node, hccl_data_type_list_, hccl_kernel_input_shape_list_, &hccl_count_)) {
      MS_LOG(ERROR) << "GetHcomCount fail!";
      return false;
    }
  }
  if (op_name_ == kAllReduceOpName || op_name_ == kReduceScatterOpName) {
    if (!HcomUtil::GetHcomOperationType(anf_node, &op_type_)) {
      MS_LOG(ERROR) << "GetHcomOperationType fail!";
      return false;
    }
  }
  if (op_name_ == kBroadcastOpName) {
    if (!HcomUtil::GetHcomRootId(anf_node, &root_id_)) {
      MS_LOG(ERROR) << "GetHcomRootId fail!";
      return false;
    }
  }
  HcomUtil::GetHcomGroup(NOT_NULL(anf_node), NOT_NULL(&group_));
  // pynative with ranktable also need hccl_comm
  comm_ = AscendCollectiveCommLib::GetInstance().HcclCommunicator(group_);
  if (common::UseHostCollective()) {
    MS_EXCEPTION_IF_NULL(comm_);
    common::AnfAlgo::SetNodeAttr(kAttrComm, MakeValue<int64_t>(reinterpret_cast<int64_t>(comm_)), anf_node);
  }
  anf_node_ = anf_node;
  CalLoopSize();
  return true;
}

void HcclKernel::SetInputSizeList(const std::vector<size_t> &size_list) { mutable_input_size_list_ = size_list; }
void HcclKernel::SetOutputSizeList(const std::vector<size_t> &size_list) { mutable_output_size_list_ = size_list; }
void HcclKernel::SetWorkspaceSizeList(const std::vector<size_t> &size_list) {
  mutable_workspace_size_list_ = size_list;
}

const std::vector<size_t> &HcclKernel::GetInputSizeList() const {
  size_t size = 0;
  if (!mutable_input_size_list_.empty()) {
    return mutable_input_size_list_;
  }
  if (hccl_data_type_list_.size() != hccl_kernel_input_shape_list_.size()) {
    MS_LOG(EXCEPTION) << "Invalid data type size " << hccl_data_type_list_.size() << " diff shape size "
                      << hccl_kernel_input_shape_list_.size();
  }
  for (ulong i = 0; i < hccl_data_type_list_.size(); ++i) {
    if (!HcomUtil::GetHcclOpSize(hccl_data_type_list_[i], hccl_kernel_input_shape_list_[i], &size)) {
      MS_LOG(ERROR) << "GetHcclOpInputSize failed";
    }
    mutable_input_size_list_.push_back(size);
  }
  return mutable_input_size_list_;
}

void HcclKernel::CalLoopSize() {
  auto anf_node = anf_node_.lock();
  if (!anf_node) {
    MS_LOG(EXCEPTION) << "anf_node pointer is expired.";
  }
  auto cnode = anf_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto op_name = common::AnfAlgo::GetCNodeName(cnode);
  int64_t rank_size = 1;
  if (common::AnfAlgo::HasNodeAttr(kAttrRankSize, cnode)) {
    rank_size = common::AnfAlgo::GetNodeAttr<int64_t>(cnode, kAttrRankSize);
  }

  int64_t fusion = 0;
  if (common::AnfAlgo::HasNodeAttr(kAttrFusion, cnode)) {
    fusion = common::AnfAlgo::GetNodeAttr<int64_t>(cnode, kAttrFusion);
  }
  if (hccl_data_type_list_.size() != hccl_kernel_input_shape_list_.size()) {
    MS_LOG(EXCEPTION) << "Invalid data type size " << hccl_data_type_list_.size() << " diff shape size "
                      << hccl_kernel_input_shape_list_.size();
  }
  loop_size_ = hccl_data_type_list_.size();
  if (common::AnfAlgo::GetInputTensorNum(anf_node) > 1 && op_name == kAllGatherOpName && fusion >= 1) {
    loop_size_ *= static_cast<ulong>(rank_size);
  }
  if (op_name == kReduceScatterOpName && fusion >= 1) {
    loop_size_ = AnfAlgo::GetOutputTensorNum(anf_node);
  }
}

const std::vector<size_t> &HcclKernel::GetOutputSizeList() const {
  size_t size = 0;
  if (!mutable_output_size_list_.empty()) {
    return mutable_output_size_list_;
  }

  for (ulong i = 0; i < loop_size_; ++i) {
    if (!HcomUtil::GetHcclOpSize(hccl_data_type_list_[0], hccl_kernel_output_shape_list_[i], &size)) {
      MS_LOG(ERROR) << "GetHcclOpOutputSize failed";
    }
    mutable_output_size_list_.push_back(size);
  }
  return mutable_output_size_list_;
}

const std::vector<size_t> &HcclKernel::GetWorkspaceSizeList() const {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  auto hccl_node = anf_node_.lock();
  MS_EXCEPTION_IF_NULL(hccl_node);
  auto func_graph = hccl_node->func_graph();
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto graph_run_mode = kernel_graph->is_graph_run_mode();

  bool is_task_sink = context_ptr->get_param<bool>(MS_CTX_ENABLE_TASK_SINK);
  auto mode = context_ptr->get_param<int>(MS_CTX_EXECUTION_MODE);
  // Not task sink mode.
  if (!mutable_workspace_size_list_.empty() || hccl_data_type_list_.empty() || (!is_task_sink && mode == kGraphMode) ||
      (mode == kPynativeMode && !graph_run_mode)) {
    return mutable_workspace_size_list_;
  }
  // Task sink mode.
  mutable_workspace_size_list_.emplace_back(
    hccl::HcclAdapter::GetInstance().CalcWorkspaceSize(anf_node_.lock(), hccl_data_type_list_[0]));
  return mutable_workspace_size_list_;
}

std::vector<TaskInfoPtr> HcclKernel::GenTask(const std::vector<AddressPtr> &inputs,
                                             const std::vector<AddressPtr> &workspace,
                                             const std::vector<AddressPtr> &outputs, uint32_t stream_id) {
  auto anf_node = anf_node_.lock();
  if (!anf_node) {
    MS_LOG(EXCEPTION) << "anf_node pointer is expired.";
  }
  std::string hccl_type = common::AnfAlgo::GetCNodeName(anf_node);
  if (hccl_type == kReceiveOpName) {
    if (outputs.empty()) {
      MS_LOG(EXCEPTION) << "Outputs is empty";
    }
  } else if (inputs.empty() || outputs.empty()) {
    MS_LOG(EXCEPTION) << "Inputs or outputs is empty";
  }
  stream_id_ = stream_id;
  void *input_data_addr = nullptr;
  if (hccl_type != kReceiveOpName) {
    MS_EXCEPTION_IF_NULL(inputs.at(0));
    input_data_addr = inputs.at(0)->addr;
  }
  MS_EXCEPTION_IF_NULL(outputs.at(0));
  auto output_data_addr = outputs.at(0)->addr;
  std::vector<uint8_t> private_def;
  if (hccl_data_type_list_.empty()) {
    MS_LOG(EXCEPTION) << "Hccl data type list is empty";
  }
  HcclDataType data_type = hccl_data_type_list_[0];
  std::vector<hccl::HcclTaskInfo> task_info;
  bool ret = hccl::HcclAdapter::GetInstance().GenTask(anf_node, data_type, &task_info);
  if (!ret) {
    MS_LOG(EXCEPTION) << "Gen Task for " << anf_node->DebugString() << " failed.";
  }

  std::vector<TaskInfoPtr> results;
  for (auto &task : task_info) {
    MS_LOG(INFO) << "HCCL Task : stream_id=" << stream_id << ", count=" << hccl_count_ << ", root_id=" << root_id_
                 << ", op_type=" << static_cast<int>(op_type_) << ", data_type=" << static_cast<int>(data_type)
                 << ", workspace_size=" << task.workspace_size << ", stream_num=" << task.stream_num
                 << ", private_def_size=" << task.private_def.size();

    private_def.resize(task.private_def.size());
    auto sec_ret = memcpy_s(private_def.data(), private_def.size(), task.private_def.data(), task.private_def.size());
    if (sec_ret != 0) {
      MS_LOG(EXCEPTION) << "Set data memcpy_s failed, ret = " << sec_ret;
    }

    void *workspace_addr = nullptr;
    if (task.workspace_size != 0) {
      if (workspace.empty()) {
        MS_LOG(EXCEPTION) << "Workspace size list of " << anf_node->DebugString() << " is empty";
      }
      MS_EXCEPTION_IF_NULL(workspace.at(0));
      workspace_addr = workspace.at(0)->addr;
    }

    std::vector<void *> global_workspace_addr;
    auto overflow_memory_ptr = device::ascend::AscendMemAdapter::GetInstance().MallocOverflowMem();
    MS_EXCEPTION_IF_NULL(overflow_memory_ptr);
    global_workspace_addr.push_back(reinterpret_cast<void *>(overflow_memory_ptr));
    MS_LOG(DEBUG) << "Assign overflow memory for node " << anf_node->fullname_with_scope() << ", addr is "
                  << reinterpret_cast<void *>(overflow_memory_ptr);

    HcclTaskInfoPtr hcclTaskInfo =
      std::make_shared<HcclTaskInfo>(unique_name_, stream_id, hccl::HcclAdapter::GetHcclType(anf_node), input_data_addr,
                                     output_data_addr, workspace_addr, task.workspace_size, task.stream_num,
                                     private_def, hccl::HcclAdapter::GetInstance().GetHcclOpsKernelInfoStore(),
                                     hccl_count_, root_id_, op_type_, data_type, group_, NeedDump());
    hcclTaskInfo->SetGlobalWorkspaceAddr(global_workspace_addr);
    results.emplace_back(hcclTaskInfo);
  }

  return results;
}

bool HcclKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                        const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  auto node = anf_node_.lock();
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    MS_LOG(EXCEPTION) << "The anfnode is not a cnode.";
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);

  if (inputs.empty() && outputs.empty()) {
    MS_LOG(ERROR) << "Hccl kernel input or output is empty.";
    return false;
  }
  if (hccl_data_type_list_.empty()) {
    MS_LOG(ERROR) << "Hccl data type list is empty.";
    return false;
  }

  MS_EXCEPTION_IF_NULL(stream_ptr);

  MS_LOG(INFO) << "Start Execute: " << cnode->DebugString();
  std::string hccl_type = MsOpNameToHcomOpType(common::AnfAlgo::GetCNodeName(anf_node_.lock()));
  HcclDataType data_type = hccl_data_type_list_[0];

  ::HcomOperation op_info;
  op_info.hcclType = hccl_type;
  op_info.inputPtr = inputs[0]->addr;
  op_info.outputPtr = outputs[0]->addr;
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
    MS_LOG(EXCEPTION) << "Call EnqueueHcomOperation failed, node info: " << cnode->DebugString();
  }

  std::unique_lock<std::mutex> ulock(hccl_mutex_);
  cond_.wait(ulock);
  MS_LOG(INFO) << "Execute " << cnode->DebugString() << " success.";
  return true;
}

void HcclKernel::UpdateOutputSizeList() {
  auto anf_node = anf_node_.lock();
  if (!anf_node) {
    MS_LOG(EXCEPTION) << "anf_node pointer is expired.";
  }
  size_t size = 0;
  hccl_kernel_output_shape_list_.clear();
  mutable_output_size_list_.clear();
  if (!HcomUtil::GetKernelOutputShape(anf_node, &hccl_kernel_output_shape_list_)) {
    MS_LOG(EXCEPTION) << "GetKernelOutputShape fail!";
  }

  for (ulong i = 0; i < loop_size_; ++i) {
    if (!HcomUtil::GetHcclOpSize(hccl_data_type_list_[0], hccl_kernel_output_shape_list_[i], &size)) {
      MS_LOG(EXCEPTION) << "GetHcclOpOutputSize failed";
    }
    mutable_output_size_list_.push_back(size);
  }
}

int HcclKernel::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                       const std::vector<KernelTensorPtr> &outputs,
                       const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  auto node = anf_node_.lock();
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);

  if (!common::AnfAlgo::IsDynamicShape(cnode)) {
    MS_LOG(DEBUG) << "The node is not dynamic shape: " << cnode->fullname_with_scope();
    return 0;
  }

  MS_LOG(INFO) << "Start to InitOp. Node info: " << cnode->DebugString();
  UpdateOutputSizeList();
  std::vector<ShapeVector> hccl_kernel_input_shape_list;
  if (!HcomUtil::GetKernelInputShape(cnode, &hccl_kernel_input_shape_list)) {
    MS_LOG(EXCEPTION) << "GetKernelInputShape fail! Node info: " << cnode->DebugString();
  }

  std::vector<HcclDataType> hccl_data_type_list;
  if (!HcomUtil::GetHcomDataType(cnode, &hccl_data_type_list)) {
    MS_LOG(EXCEPTION) << "GetHcomDataType fail! Node info: " << cnode->DebugString();
  }

  // Update Hccl count
  if (!HcomUtil::GetHcomCount(cnode, hccl_data_type_list, hccl_kernel_input_shape_list, &hccl_count_)) {
    MS_LOG(EXCEPTION) << "GetHcomCount fail! Node info: " << cnode->DebugString();
  }
  MS_LOG(INFO) << "Update Hccl count:" << hccl_count_;
  return 0;
}
}  // namespace kernel
}  // namespace mindspore
