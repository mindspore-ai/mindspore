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

#include "ops/structure_op_name.h"
#include "ops/array_ops.h"
#include "ops/math_op_name.h"
#include "ops/lite_op_name.h"
#include "runtime/mem.h"
#include "acl/acl_rt.h"
#include "plugin/device/ascend/kernel/aicpu/aicpu_util.h"
#include "plugin/device/ascend/kernel/aicpu/aicpu_proto_util.h"
#include "plugin/device/ascend/hal/device/ascend_data_queue.h"
#include "utils/ms_context.h"
#include "runtime/device/kernel_runtime.h"
#include "runtime/pynative/op_runtime_info.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "plugin/device/ascend/hal/device/ascend_memory_manager.h"
#include "include/backend/data_queue/data_queue_mgr.h"
#include "aicpu/common/aicpu_task_struct.h"
#include "external/graph/types.h"

namespace mindspore {
namespace kernel {
namespace {
// todo: delete when tansdata in libcpu_kernel.so is fixed
bool IsTransDataGroupsMoreThanOne(const PrimitivePtr &prim) {
  if (!IsPrimitiveEquals(prim, prim::kPrimTransData)) {
    return false;
  }

  // Get AttrFracZGroup value
  int64_t fz_group = 1;
  if (prim->HasAttr(kAttrFracZGroupIdx)) {
    auto fz_group_idx = GetValue<std::vector<int64_t>>(prim->GetAttr(kAttrFracZGroupIdx));
    if (fz_group_idx.empty()) {
      MS_LOG(INTERNAL_EXCEPTION) << "Attr fracz_group_idx of kernel [" << prim->name() << "] is an empty vector";
      fz_group = fz_group_idx[0];
    }
  } else if (prim->HasAttr(kAttrFracZGroup)) {
    fz_group = GetValue<int64_t>(prim->GetAttr(kAttrFracZGroup));
  }

  return fz_group != 1;
}

bool IsGetNextOp(const std::string &op_name) { return op_name == kGetNextOpName || op_name == kDynamicGetNextV2OpName; }
}  // namespace

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

void AicpuOpKernelMod::SetNodeName(const std::string &node_name) { node_name_ = node_name; }
void AicpuOpKernelMod::SetCustSo(const std::string &cust_so) {
  node_so_ = cust_so;
  cust_kernel_ = true;
}

void AicpuOpKernelMod::CreateAsyncWaitEventAndUpdateEventInfo() {
  if (rt_event_ != nullptr) {
    MS_LOG(INFO) << "The event is already created! node: " << node_scope_name_;
    return;
  }
  if (is_blocking_ && CheckDeviceSupportBlockingAicpuOpProcess()) {
    device::ascend::AscendStreamMng &resource_manager = device::ascend::AscendStreamMng::GetInstance();
    rt_event_ = resource_manager.ApplyRtEventWithFlag(ACL_EVENT_SYNC);

    uint32_t rt_event_id = resource_manager.GetRtEventId(rt_event_);

    MS_EXCEPTION_IF_NULL(ext_info_handler_);
    MS_LOG(DEBUG) << "Call UpdateEventId, device event id: " << rt_event_id << ", blocking node: " << node_scope_name_;
    if (!ext_info_handler_->UpdateEventId(rt_event_id)) {
      MS_LOG(EXCEPTION) << "Aicpu ext_info_handler update event id failed.";
    }
  } else {
    MS_LOG(DEBUG) << "The node is not blocking op, no need to create event, node: " << node_scope_name_;
  }
}

void AicpuOpKernelMod::ParseNodeNameAndNodeSo() {
  if (!cust_kernel_) {
    if (kCpuKernelOps.find(node_name_) != kCpuKernelOps.end() || IsTransDataGroupsMoreThanOne(primitive_)) {
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

void AicpuOpKernelMod::SetExtInfo(const std::string &ext_info, size_t input_num, size_t output_num) {
  ext_info_ = ext_info;

  // Initialize ext_info_handler_
  if (ext_info_handler_ == nullptr) {
    ext_info_handler_ = std::make_shared<device::ascend::AicpuExtInfoHandler>(
      node_scope_name_, static_cast<uint32_t>(input_num), static_cast<uint32_t>(output_num), unknow_type_);
  }
  // Parse ext_info_
  if (!ext_info_handler_->Parse(ext_info_)) {
    MS_LOG(EXCEPTION) << "Parse AiCpu ext_info_handler failed";
  }
}

void AicpuOpKernelMod::AllocateExtInfoDeviceAddr() {
  if (ext_info_addr_dev_ != nullptr) {
    return;
  }
  // Allocate ext info addr in device
  if (!ext_info_.empty()) {
    auto mem_manager = std::make_shared<device::ascend::AscendMemoryManager>();
    ext_info_addr_dev_ = mem_manager->MallocMemFromMemPool(ext_info_.size(), false);
    if (ext_info_addr_dev_ == nullptr) {
      MS_LOG(EXCEPTION) << "Call MemoryPool to allocate ext_info_addr_dev_ failed. Op name: " << node_scope_name_;
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
  auto ret = aclrtGetDevice(&device_id);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Call aclrtGetDevice failed, ret: " << ret;
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

void AicpuOpKernelMod::CloseTdtWingManQueue() {
  if (IsGetNextOp(kernel_name_) && !is_dynamic_shape_) {
    device::CloseTdtWingManQueue(primitive_);
  }
}

bool AicpuOpKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  unknow_type_ = IsOneOfComputeDepend(kernel_name_) ? ::ge::UnknowShapeOpType::DEPEND_COMPUTE
                                                    : ::ge::UnknowShapeOpType::DEPEND_IN_SHAPE;
  is_blocking_ = IsGetNextOp(kernel_name_);
  return true;
}

int AicpuOpKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (IsGetNextOp(kernel_name_)) {
    auto wingman_queue = device::GetTdtWingManQueue(primitive_);
    std::vector<device::DataQueueItem> data;
    RetryPeakItemFromDataQueue(nullptr, wingman_queue, &data);
    (void)wingman_queue->Pop();
    MS_EXCEPTION_IF_CHECK_FAIL(outputs.size() == data.size(), "Size of output is not equal to size of data");
    output_size_list_.clear();
    for (size_t i = 0; i < outputs.size(); i++) {
      outputs[i]->SetShapeVector(data[i].shapes);
      output_size_list_.push_back(data[i].data_len);
    }
  } else {
    // update output size after InferShape.
    auto ret = KernelMod::Resize(inputs, outputs);
    if (ret != KRET_OK) {
      return ret;
    }
  }

  need_skip_execute_ = [this, &inputs]() -> bool {
    if ((kernel_name_ != kReduceSumOpName) && (kernel_name_ != kReduceSumDOpName)) {
      return false;
    }
    constexpr size_t kAxisIndex{1};
    bool skip_mode = inputs[kIndex3]->GetValueWithCheck<bool>();
    if (inputs.size() > kAxisIndex &&
        AnfAlgo::IsDynamicShapeSkipExecute(skip_mode, inputs[kAxisIndex]->GetShapeVector())) {
      return true;
    }

    return false;
  }();

  if (need_skip_execute_) {
    return KRET_OK;
  }
  if (IsOutputAllEmptyTensor(outputs)) {
    return KRET_OK;
  }

  if (!CreateNodeDefBytes(primitive_, inputs, outputs, &node_def_str_)) {
    return KRET_RESIZE_FAILED;
  }

  MS_LOG(DEBUG) << "UpdateExtInfo of " << node_scope_name_ << " start";
  auto input_num = inputs.size();
  auto output_num = outputs.size();
  if (input_num == 0 && output_num == 0) {
    MS_LOG(INFO) << "Node:" << node_scope_name_ << " no need to update output shape";
    return KRET_OK;
  }

  if (ext_info_handler_ == nullptr || ext_info_.empty()) {
    MS_LOG(EXCEPTION) << "The ext_info_handler_ is nullptr or  ext_info_ is empty.";
  }

  for (uint32_t i = 0; i < input_num; ++i) {
    if (!ext_info_handler_->UpdateInputShapeAndType(i, inputs[i])) {
      MS_LOG(EXCEPTION) << "Update input shape failed, node:" << node_scope_name_ << " input:" << i;
    }
  }

  for (uint32_t i = 0; i < output_num; ++i) {
    if (!ext_info_handler_->UpdateOutputShapeAndType(i, outputs[i])) {
      MS_LOG(EXCEPTION) << "Update output shape failed, node:" << node_scope_name_ << " output:" << i;
    }
  }

  return KRET_OK;
}

bool AicpuOpKernelMod::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                              const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  if (stream_ptr == nullptr) {
    MS_LOG(ERROR) << "stream_ptr should not be nullptr.";
    return false;
  }

  stream_ = stream_ptr;
  MS_LOG(DEBUG) << "Start launch of node: " << node_scope_name_;

  // need skip, for reducesum empty input axis
  if (need_skip_execute_) {
    // Skip reduce if axis is a empty Tensor (shape = 0)
    MS_LOG(INFO) << "For AICPU ,The node " << node_scope_name_ << " Need Skip.";
    // cppcheck-suppress unreadVariable
    auto lock = device::KernelRuntime::LockRuntime(stream_ptr);
    rtError_t status = aclrtMemcpyAsync(outputs[0]->device_ptr(), inputs[0]->size(), inputs[0]->device_ptr(),
                                        inputs[0]->size(), ACL_MEMCPY_DEVICE_TO_DEVICE, stream_ptr);
    if (status != RT_ERROR_NONE) {
      MS_LOG(EXCEPTION) << "AclrtMemcpyAsync failed for " << node_scope_name_;
    }

    MS_LOG(INFO) << "AICPU Execute node:" << node_scope_name_ << " success.";
    return true;
  }
  // skip execute if all outputs are empty tensor
  if (is_output_all_empty_tensor_) {
    MS_LOG(INFO) << "Outputs are all empty tensors, skip launch node " << node_scope_name_;
    return true;
  }

  // create asyncflag_op's event
  CreateAsyncWaitEventAndUpdateEventInfo();

  // alloc extinfo device address memory
  AllocateExtInfoDeviceAddr();

  // copy extinfo to device
  if (ext_info_handler_ != nullptr) {
    // cppcheck-suppress unreadVariable
    auto lock = device::KernelRuntime::LockRuntime(stream_ptr);
    auto ret = aclrtMemcpyAsync(ext_info_addr_dev_, ext_info_size_, ext_info_handler_->GetExtInfo(),
                                ext_info_handler_->GetExtInfoLen(), ACL_MEMCPY_HOST_TO_DEVICE, stream_ptr);
    if (ret != RT_ERROR_NONE) {
      MS_LOG(ERROR) << "UpdateExtInfo aclrtMemcpy failed. Node info: " << node_scope_name_;
      return false;
    }
  } else if (is_dynamic_shape_) {
    MS_LOG(ERROR) << "The node is dynamic, but the ext_info_handler_ is nullptr. Node info: " << node_scope_name_;
    return false;
  }

  // create kernelinfo
  auto get_addrs = [](const std::vector<KernelTensor *> &kernel_tensors) -> std::vector<AddressPtr> {
    std::vector<AddressPtr> addr_ptrs;
    (void)std::transform(
      kernel_tensors.begin(), kernel_tensors.end(), std::back_inserter(addr_ptrs),
      [](KernelTensor *ptr) { return std::make_shared<mindspore::kernel::Address>(ptr->device_ptr(), ptr->size()); });
    return addr_ptrs;
  };
  CreateCpuKernelInfo(get_addrs(inputs), get_addrs(outputs));

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
    MS_LOG(ERROR) << "Aicpu op launch failed! node: " << node_scope_name_;
    return false;
  }
  // for asyncflag op, create event wait op
  if (is_blocking_ && CheckDeviceSupportBlockingAicpuOpProcess()) {
    MS_LOG(INFO) << "Insert EventWait, stream: " << stream_ptr << ", event: " << rt_event_
                 << ", node: " << node_scope_name_;

    auto rt_ret = aclrtStreamWaitEvent(stream_ptr, rt_event_);
    if (rt_ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "Call rt api aclrtStreamWaitEvent failed, ret: " << rt_ret;
      return false;
    }

    rt_ret = aclrtResetEvent(rt_event_, stream_ptr);
    if (rt_ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "Call rt api aclrtResetEvent failed, ret: " << rt_ret;
      return false;
    }
  }

  if (unknow_type_ != ::ge::UnknowShapeOpType::DEPEND_COMPUTE) {
    FreeExtInfoDeviceAddr();
  }
  return true;
}

bool AicpuOpKernelMod::IsNeedUpdateOutputShapeAndSize() {
  if (IsOneOfComputeDepend(kernel_name_)) {
    return true;
  }
  return false;
}

void AicpuOpKernelMod::UpdateOutputShapeAndSize(const std::vector<KernelTensor *> &inputs,
                                                const std::vector<KernelTensor *> &outputs) {
  MS_LOG(INFO) << "Aicpu " << node_scope_name_ << " PostExecute";
  // is dynamic shape
  if (!is_dynamic_shape_) {
    MS_LOG(EXCEPTION) << "The node is not dynamic shape:" << node_scope_name_;
  }

  // cppcheck-suppress unreadVariable
  auto lock = device::KernelRuntime::LockRuntime(stream_);
  auto ret = aclrtMemcpyAsync(ext_info_handler_->GetExtInfo(), ext_info_handler_->GetExtInfoLen(), ext_info_addr_dev_,
                              ext_info_size_, ACL_MEMCPY_DEVICE_TO_HOST, stream_);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "AclrtMemcpyAsync output shape failed. Op name: " << node_scope_name_;
  }
  ret = aclrtSynchronizeStreamWithTimeout(stream_, -1);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Call runtime aclrtSynchronizeStreamWithTimeout failed. Op name: " << node_scope_name_;
  }

  MS_LOG(INFO) << "Update aicpu kernel output shape from ext_info. Op name: " << node_scope_name_;
  auto output_num = outputs.size();
  for (size_t i = 0; i < output_num; ++i) {
    std::vector<int64_t> shape;
    TypeId type_id;
    (void)ext_info_handler_->GetOutputShapeAndType(SizeToUint(i), NOT_NULL(&shape), NOT_NULL(&type_id));
    if (std::any_of(shape.begin(), shape.end(), [](int64_t x) { return x < 0; })) {
      MS_LOG(EXCEPTION) << node_scope_name_ << ": output[" << i << "] shape = " << ShapeVectorToStr(shape)
                        << " contains negative value.";
    }
    outputs[i]->SetShapeVector(shape);
    size_t dtype_byte = GetTypeByte(TypeIdToType(outputs[i]->dtype_id()));
    size_t update_size =
      LongToSize(std::accumulate(shape.begin(), shape.end(), dtype_byte, std::multiplies<int64_t>()));
    outputs[i]->set_size(update_size);
  }
}

bool AicpuOpKernelMod::IsOutputAllEmptyTensor(const std::vector<KernelTensor *> &outputs) {
  for (auto ptr : outputs) {
    auto &output_shape = ptr->GetShapeVector();
    if (std::none_of(output_shape.cbegin(), output_shape.cend(), [](int64_t dim) { return dim == 0; })) {
      is_output_all_empty_tensor_ = false;
      return false;
    }
  }
  is_output_all_empty_tensor_ = true;
  return true;
}
}  // namespace kernel
}  // namespace mindspore
