/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/kernel/aicpu/aicpu_kernel_mod.h"

#include <memory>
#include <vector>
#include <string>
#include <algorithm>

#include "runtime/mem.h"
#include "acl/acl_rt.h"
#include "include/common/utils/convert_utils.h"
#include "plugin/device/ascend/kernel/aicpu/aicpu_util.h"
#include "plugin/device/ascend/hal/device/ascend_data_queue.h"
#include "utils/ms_context.h"
#include "runtime/device/kernel_runtime.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "plugin/device/ascend/hal/device/ascend_memory_manager.h"
#include "include/backend/data_queue/data_queue_mgr.h"
#include "aicpu/common/aicpu_task_struct.h"
#include "external/graph/types.h"

using AicpuTaskInfoPtr = std::shared_ptr<mindspore::ge::model_runner::AicpuTaskInfo>;
using EventWaitTaskInfoPtr = std::shared_ptr<mindspore::ge::model_runner::EventWaitTaskInfo>;

namespace mindspore {
namespace kernel {
namespace {
// todo: delete when tansdata in libcpu_kernel.so is fixed
bool IsTransDataGroupsMoreThanOne(const AnfNodePtr &anf_node) {
  if (anf_node == nullptr) {
    return false;
  }

  if (!IsPrimitiveCNode(anf_node, prim::kPrimTransData)) {
    return false;
  }

  if (common::AnfAlgo::GetAttrGroups(anf_node, 0) == 1) {
    return false;
  }

  return true;
}
}  // namespace

AicpuOpKernelMod::AicpuOpKernelMod() : AscendKernelMod(), unknow_type_(::ge::UnknowShapeOpType::DEPEND_IN_SHAPE) {}

AicpuOpKernelMod::AicpuOpKernelMod(const AnfNodePtr &anf_node_ptr) : AscendKernelMod(anf_node_ptr) {
  if (common::AnfAlgo::GetCNodeName(anf_node_ptr) == kGetNextOpName && !common::AnfAlgo::IsDynamicShape(anf_node_ptr)) {
    device::CloseTdtWingManQueue(anf_node_ptr);
  }
  unknow_type_ = ::ge::UnknowShapeOpType::DEPEND_IN_SHAPE;
  is_blocking_ = false;
  auto cnode = anf_node_ptr->cast<CNodePtr>();
  if (cnode != nullptr) {
    auto op_name = common::AnfAlgo::GetCNodeName(cnode);
    if (IsOneOfComputeDepend(op_name)) {
      unknow_type_ = ::ge::UnknowShapeOpType::DEPEND_COMPUTE;
    }
    is_blocking_ = (common::AnfAlgo::GetCNodeName(cnode) == kGetNextOpName);
  }
}

AicpuOpKernelMod::~AicpuOpKernelMod() {
  FreeExtInfoDeviceAddr();
  args_.clear();
  input_list_.clear();
  output_list_.clear();
  input_size_list_.clear();
  output_size_list_.clear();
  workspace_size_list_.clear();
  ext_info_.clear();
}

void AicpuOpKernelMod::SetInputList(const std::vector<int64_t> &input_list) { input_list_ = input_list; }
void AicpuOpKernelMod::SetOutputList(const std::vector<int64_t> &output_list) { output_list_ = output_list; }
void AicpuOpKernelMod::SetNodeDef(const std::string &node_def) { (void)node_def_str_.assign(node_def); }
void AicpuOpKernelMod::SetNodeName(const std::string &node_name) { node_name_ = node_name; }
void AicpuOpKernelMod::SetCustSo(const std::string &cust_so) {
  node_so_ = cust_so;
  cust_kernel_ = true;
}

void AicpuOpKernelMod::SetAnfNode(const mindspore::AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  anf_node_ = anf_node;
}

void AicpuOpKernelMod::CreateAsyncWaitEventAndUpdateEventInfo(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  if (rt_event_ != nullptr) {
    MS_LOG(INFO) << "The event is already created! node: " << cnode->fullname_with_scope();
    return;
  }
  if (is_blocking_ && CheckDeviceSupportBlockingAicpuOpProcess()) {
    device::ascend::AscendStreamMng &resource_manager = device::ascend::AscendStreamMng::GetInstance();
    rt_event_ = resource_manager.ApplyRtEventWithFlag(RT_EVENT_WITH_FLAG);

    uint32_t rt_event_id = resource_manager.GetRtEventId(rt_event_);

    MS_EXCEPTION_IF_NULL(ext_info_handler_);
    MS_LOG(DEBUG) << "Call UpdateEventId, device event id: " << rt_event_id
                  << ", blocking node: " << cnode->fullname_with_scope();
    if (!ext_info_handler_->UpdateEventId(rt_event_id)) {
      MS_LOG(EXCEPTION) << "Aicpu ext_info_handler update event id failed.";
    }
  } else {
    MS_LOG(DEBUG) << "The node is not blocking op, no need to create event, node: " << cnode->fullname_with_scope();
  }
}

void AicpuOpKernelMod::ParseNodeNameAndNodeSo() {
  if (!cust_kernel_) {
    if (kCpuKernelOps.find(node_name_) != kCpuKernelOps.end() || IsTransDataGroupsMoreThanOne(anf_node_.lock())) {
      node_so_ = kLibCpuKernelSoName;
      node_name_ = kCpuRunApi;
    } else if (kCacheKernelOps.find(node_name_) != kCacheKernelOps.end()) {
      node_so_ = kLibAicpuKernelSoName;
      node_name_ = kCpuRunApi;
    } else {
      if (node_so_ != kLibCpuKernelSoName) {
        node_so_ = kLibAicpuKernelSoName;
      }
    }
  } else if (kCpuKernelBaseOps.find(node_name_) == kCpuKernelBaseOps.end()) {
    node_name_ = kCpuRunApi;
  }

  if (node_name_ == kStack) {
    node_name_ = kPack;
  }
}

void AicpuOpKernelMod::SetExtInfo(const std::string &ext_info) {
  ext_info_ = ext_info;

  // Initialize ext_info_handler_
  if (ext_info_handler_ == nullptr) {
    auto node = anf_node_.lock();
    MS_EXCEPTION_IF_NULL(node);
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    auto input_num = common::AnfAlgo::GetInputTensorNum(cnode);
    auto output_num = AnfAlgo::GetOutputTensorNum(cnode);

    ext_info_handler_ = std::make_shared<device::ascend::AicpuExtInfoHandler>(
      cnode->fullname_with_scope(), static_cast<uint32_t>(input_num), static_cast<uint32_t>(output_num), unknow_type_);
    MS_EXCEPTION_IF_NULL(ext_info_handler_);
  }
  // Parse ext_info_
  if (!ext_info_handler_->Parse(ext_info_)) {
    MS_LOG(EXCEPTION) << "Parse AiCpu ext_info_handler failed";
  }
}

void AicpuOpKernelMod::AllocateExtInfoDeviceAddr(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  if (ext_info_addr_dev_ != nullptr) {
    return;
  }
  // Allocate ext info addr in device
  if (!ext_info_.empty()) {
    auto mem_manager = std::make_shared<device::ascend::AscendMemoryManager>();
    ext_info_addr_dev_ = mem_manager->MallocMemFromMemPool(ext_info_.size(), false);
    if (ext_info_addr_dev_ == nullptr) {
      MS_LOG(EXCEPTION) << "Call MemoryPool to allocate ext_info_addr_dev_ failed. Op name: "
                        << cnode->fullname_with_scope();
    }
  }
  ext_info_size_ = ext_info_.size();
}

void AicpuOpKernelMod::FreeExtInfoDeviceAddr() {
  if (ext_info_addr_dev_ != nullptr) {
    auto mem_manager = std::make_shared<device::ascend::AscendMemoryManager>();
    mem_manager->FreeMemFromMemPool(ext_info_addr_dev_);
    ext_info_addr_dev_ = nullptr;
  }
}

bool AicpuOpKernelMod::CheckDeviceSupportBlockingAicpuOpProcess() const {
  int32_t device_id = 0;
  auto ret = rtGetDevice(&device_id);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Call rtGetDevice failed, ret: " << ret;
  }
  int32_t value = 0;
  ret = rtGetDeviceCapability(device_id, FEATURE_TYPE_BLOCKING_OPERATOR, RT_MODULE_TYPE_AICPU, &value);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Call rtGetDeviceCapability failed, ret: " << ret;
  }
  if ((value != RT_AICPU_BLOCKING_OP_NOT_SUPPORT) && (value != RT_AICPU_BLOCKING_OP_SUPPORT)) {
    MS_LOG(EXCEPTION)
      << "The value should be RT_AICPU_BLOCKING_OP_NOT_SUPPORT or RT_AICPU_BLOCKING_OP_SUPPORT, but got " << value;
  }

  return (value == RT_AICPU_BLOCKING_OP_SUPPORT);
}

void AicpuOpKernelMod::CreateCpuKernelInfo(const std::vector<AddressPtr> &inputs,
                                           const std::vector<AddressPtr> &outputs) {
  MS_LOG(DEBUG) << "CreateCpuKernelInfoOffline start";

  ParseNodeNameAndNodeSo();

  // InputOutputAddr
  vector<void *> io_addrs;
  (void)std::transform(std::begin(inputs), std::end(inputs), std::back_inserter(io_addrs),
                       [](const AddressPtr &input) -> void * { return input->addr; });
  (void)std::transform(std::begin(outputs), std::end(outputs), std::back_inserter(io_addrs),
                       [](const AddressPtr &output) -> void * { return output->addr; });

  auto io_addrs_num = io_addrs.size();
  // calculate paramLen: AicpuParamHead.len + ioAddrsSize + notifyId.len + customizedAttr.len
  auto param_len = sizeof(aicpu::AicpuParamHead);

  // get input and output addrs size, no need to check overflow
  auto io_addrs_size = io_addrs_num * sizeof(uint64_t);
  // refresh paramLen, no need to check overflow
  param_len += io_addrs_size;

  auto node_def_len = node_def_str_.length();
  param_len += node_def_len;
  param_len += sizeof(uint32_t);

  aicpu::AicpuParamHead aicpu_param_head{};
  aicpu_param_head.length = SizeToUint(param_len);
  aicpu_param_head.ioAddrNum = SizeToUint(io_addrs_num);

  if (ext_info_.empty()) {
    aicpu_param_head.extInfoLength = 0;
    aicpu_param_head.extInfoAddr = 0;
  } else {
    MS_LOG(INFO) << "Dynamic Kernel Ext Info size:" << ext_info_.size();
    aicpu_param_head.extInfoLength = SizeToUint(ext_info_.size());
    aicpu_param_head.extInfoAddr = reinterpret_cast<uint64_t>(ext_info_addr_dev_);
  }

  args_.clear();
  (void)args_.append(reinterpret_cast<const char *>(&aicpu_param_head), sizeof(aicpu::AicpuParamHead));
  // TaskArgs append ioAddrs
  if (io_addrs_size != 0) {
    (void)args_.append(reinterpret_cast<const char *>(io_addrs.data()), io_addrs_size);
  }

  // size for node_def
  (void)args_.append(reinterpret_cast<const char *>(&node_def_len), sizeof(uint32_t));

  // When it's aicpu customized ops, taskArgs should append customized attr
  if (node_def_len != 0) {
    (void)args_.append(reinterpret_cast<const char *>(node_def_str_.data()), node_def_len);
  }

  MS_LOG(DEBUG) << "CreateCpuKernelInfoOffline end";
}

bool AicpuOpKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                              const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  if (stream_ptr == nullptr) {
    MS_LOG(ERROR) << "stream_ptr should not be nullptr.";
    return false;
  }

  stream_ = stream_ptr;

  auto node = anf_node_.lock();
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  MS_LOG(DEBUG) << "Start launch of node: " << cnode->fullname_with_scope();

  // create asyncflag_op's event
  CreateAsyncWaitEventAndUpdateEventInfo(cnode);

  // alloc extinfo device address memory
  AllocateExtInfoDeviceAddr(cnode);

  // copy extinfo to device
  if (ext_info_handler_ != nullptr) {
    auto ret = aclrtMemcpyAsync(ext_info_addr_dev_, ext_info_size_, ext_info_handler_->GetExtInfo(),
                                ext_info_handler_->GetExtInfoLen(), ACL_MEMCPY_HOST_TO_DEVICE, stream_ptr);
    if (ret != RT_ERROR_NONE) {
      MS_LOG(ERROR) << "UpdateExtInfo aclrtMemcpy failed. Node info: " << cnode->fullname_with_scope();
      return false;
    }
  } else if (common::AnfAlgo::IsDynamicShape(cnode)) {
    MS_LOG(ERROR) << "The node is dynamic, but the ext_info_handler_ is nullptr. Node info: "
                  << cnode->fullname_with_scope();
    return false;
  }

  // create kernelinfo
  CreateCpuKernelInfo(inputs, outputs);

  // launch kernel
  auto flag = RT_KERNEL_DEFAULT;
  if (cust_kernel_) {
    flag = RT_KERNEL_CUSTOM_AICPU;
  }
  MS_LOG(DEBUG) << "Aicpu launch, node_so_:" << node_so_ << ", node name:" << node_name_
                << ", args_size:" << args_.length();
  // cppcheck-suppress unreadVariable
  auto lock = device::KernelRuntime::LockRuntime(stream_ptr);
  rtArgsEx_t argsInfo = {};
  argsInfo.args = args_.data();
  argsInfo.argsSize = static_cast<uint32_t>(args_.length());
  if (rtCpuKernelLaunchWithFlag(reinterpret_cast<const void *>(node_so_.c_str()),
                                reinterpret_cast<const void *>(node_name_.c_str()), 1, &argsInfo, nullptr, stream_ptr,
                                flag) != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "Aicpu op launch failed! node: " << cnode->fullname_with_scope();
    return false;
  }

  // for asyncflag op, create event wait op
  if (is_blocking_ && CheckDeviceSupportBlockingAicpuOpProcess()) {
    MS_LOG(INFO) << "Insert EventWait, stream: " << stream_ptr << ", event: " << rt_event_
                 << ", node: " << cnode->fullname_with_scope();

    rtError_t rt_ret = rtStreamWaitEvent(stream_ptr, rt_event_);
    if (rt_ret != RT_ERROR_NONE) {
      MS_LOG(ERROR) << "Call rt api rtStreamWaitEvent failed, ret: " << rt_ret;
      return false;
    }

    rt_ret = rtEventReset(rt_event_, stream_ptr);
    if (rt_ret != RT_ERROR_NONE) {
      MS_LOG(ERROR) << "Call rt api rtEventReset failed, ret: " << rt_ret;
      return false;
    }
  }

  return true;
}

int AicpuOpKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                             const std::vector<KernelTensorPtr> &outputs,
                             const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  auto node = anf_node_.lock();
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (!common::AnfAlgo::IsDynamicShape(cnode)) {
    MS_LOG(EXCEPTION) << "The node is not dynamic shape: " << cnode->fullname_with_scope();
  }
  if (common::AnfAlgo::GetCNodeName(cnode) == kGetNextOpName) {
    auto wingman_queue = device::GetTdtWingManQueue(cnode);
    std::vector<device::DataQueueItem> data;
    RetryPeakItemFromDataQueue(cnode, wingman_queue, &data);
    (void)wingman_queue->Pop();
    UpdateGetNextWithDataQueueItems(cnode, data);
  } else {
    // update output size after InferShape.
    AscendKernelMod::UpdateOutputSizeList();
  }

  MS_LOG(DEBUG) << "UpdateExtInfo of " << cnode->fullname_with_scope() << " start";
  auto input_num = common::AnfAlgo::GetInputTensorNum(cnode);
  auto output_num = AnfAlgo::GetOutputTensorNum(cnode);
  if (input_num == 0 && output_num == 0) {
    MS_LOG(INFO) << "Node:" << cnode->fullname_with_scope() << " no need to update output shape";
    return 0;
  }

  if (ext_info_handler_ == nullptr || ext_info_.empty()) {
    MS_LOG(EXCEPTION) << "The ext_info_handler_ is nullptr or  ext_info_ is empty.";
    return 0;
  }

  for (uint32_t i = 0; i < input_num; ++i) {
    if (!ext_info_handler_->UpdateInputShapeAndType(i, NOT_NULL(cnode))) {
      MS_LOG(EXCEPTION) << "Update input shape failed, cnode:" << cnode->fullname_with_scope() << " input:" << i;
    }
  }

  if (unknow_type_ != ::ge::UnknowShapeOpType::DEPEND_COMPUTE ||
      common::AnfAlgo::GetCNodeName(cnode) == kGetNextOpName) {
    for (uint32_t i = 0; i < output_num; ++i) {
      if (!ext_info_handler_->UpdateOutputShapeAndType(i, NOT_NULL(cnode))) {
        MS_LOG(EXCEPTION) << "Update output shape failed, cnode:" << cnode->fullname_with_scope() << " output:" << i;
      }
    }
  }

  return 0;
}

void AicpuOpKernelMod::SyncData() {
  auto node = anf_node_.lock();
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  MS_LOG(INFO) << "Aicpu " << cnode->fullname_with_scope() << " PostExecute";
  // is dynamic shape
  if (!common::AnfAlgo::IsDynamicShape(cnode)) {
    MS_LOG(EXCEPTION) << "The cnode is not dynamic shape:" << cnode->fullname_with_scope();
  }

  if (unknow_type_ != ::ge::UnknowShapeOpType::DEPEND_COMPUTE ||
      common::AnfAlgo::GetCNodeName(cnode) == kGetNextOpName) {
    MS_LOG(INFO) << "Node " << node->fullname_with_scope() << " update op skip.";
    return;
  }
  // cppcheck-suppress unreadVariable
  auto lock = device::KernelRuntime::LockRuntime(stream_);
  auto ret = aclrtMemcpyAsync(ext_info_handler_->GetExtInfo(), ext_info_handler_->GetExtInfoLen(), ext_info_addr_dev_,
                              ext_info_size_, ACL_MEMCPY_DEVICE_TO_HOST, stream_);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "AclrtMemcpyAsync output shape failed. Op name: " << cnode->fullname_with_scope();
  }
  ret = rtStreamSynchronize(stream_);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Call runtime rtStreamSynchronize failed. Op name: " << cnode->fullname_with_scope();
  }

  MS_LOG(INFO) << "Update aicpu kernel output shape from ext_info. Op name: " << cnode->fullname_with_scope();
  UpdateOutputShapeFromExtInfo(cnode);
}

void AicpuOpKernelMod::UpdateOutputShapeFromExtInfo(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_LOG(INFO) << "UpdateOutputShapeFromExtInfo start. Op name " << cnode->fullname_with_scope();
  MS_EXCEPTION_IF_NULL(ext_info_handler_);

  std::vector<TypeId> type_ids;
  std::vector<ShapeVector> shapes;
  auto output_num = AnfAlgo::GetOutputTensorNum(cnode);
  for (size_t i = 0; i < output_num; ++i) {
    std::vector<int64_t> shape;
    TypeId type_id;
    (void)ext_info_handler_->GetOutputShapeAndType(SizeToUint(i), NOT_NULL(&shape), NOT_NULL(&type_id));
    (void)type_ids.emplace_back(type_id);
    (void)shapes.emplace_back(shape);
  }

  common::AnfAlgo::SetOutputInferTypeAndShape(type_ids, shapes, cnode.get());
}

std::vector<TaskInfoPtr> AicpuOpKernelMod::GenTask(const std::vector<AddressPtr> &inputs,
                                                   const std::vector<AddressPtr> &,
                                                   const std::vector<AddressPtr> &outputs, uint32_t stream_id) {
  MS_LOG(INFO) << "AicpuOpKernelMod GenTask start";

  stream_id_ = stream_id;

  ParseNodeNameAndNodeSo();
  std::vector<void *> input_data_addrs;
  (void)std::transform(std::begin(inputs), std::end(inputs), std::back_inserter(input_data_addrs),
                       [](const AddressPtr &input) -> void * { return input->addr; });

  std::vector<void *> output_data_addrs;
  (void)std::transform(std::begin(outputs), std::end(outputs), std::back_inserter(output_data_addrs),
                       [](const AddressPtr &output) -> void * { return output->addr; });

  std::vector<TaskInfoPtr> ret_task_info;

  uint32_t ms_event_id = 0;
  bool is_blocking = (is_blocking_ && CheckDeviceSupportBlockingAicpuOpProcess());
  if (is_blocking) {
    device::ascend::AscendStreamMng &resource_manager = device::ascend::AscendStreamMng::GetInstance();
    ms_event_id = resource_manager.ApplyNewEvent();
  }

  // op task
  AicpuTaskInfoPtr task_info_ptr = std::make_shared<mindspore::ge::model_runner::AicpuTaskInfo>(
    unique_name_, stream_id, node_so_, node_name_, node_def_str_, ext_info_, input_data_addrs, output_data_addrs,
    NeedDump(), cust_kernel_, is_blocking, ms_event_id, unknow_type_);
  (void)ret_task_info.emplace_back(task_info_ptr);

  if (is_blocking) {
    EventWaitTaskInfoPtr wait_task_info_ptr =
      std::make_shared<mindspore::ge::model_runner::EventWaitTaskInfo>(unique_name_ + "_wait", stream_id, ms_event_id);
    (void)ret_task_info.emplace_back(wait_task_info_ptr);
  }

  MS_LOG(INFO) << "AicpuOpKernelMod GenTask end";
  return ret_task_info;
}
}  // namespace kernel
}  // namespace mindspore
