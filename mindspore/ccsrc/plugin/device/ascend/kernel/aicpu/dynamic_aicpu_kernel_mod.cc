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

#include "plugin/device/ascend/kernel/aicpu/dynamic_aicpu_kernel_mod.h"

#include <memory>
#include <vector>
#include <string>
#include <algorithm>
#include "runtime/mem.h"
#include "acl/acl_rt.h"
#include "include/common/utils/convert_utils.h"
#include "plugin/device/ascend/kernel/aicpu/aicpu_util.h"
#include "plugin/device/ascend/hal/device/ascend_memory_manager.h"
#include "utils/ms_context.h"
#include "runtime/device/kernel_runtime.h"
#include "runtime/kernel.h"
#include "include/common/utils/utils.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/backend/data_queue/data_queue_mgr.h"
#include "plugin/device/ascend/hal/device/ascend_data_queue.h"

namespace mindspore {
namespace kernel {
DynamicAicpuOpKernelMod::DynamicAicpuOpKernelMod(const AnfNodePtr &anf_node_ptr) : AicpuOpKernelMod(anf_node_ptr) {
  unknow_type_ = device::ascend::UnknowShapeOpType::DEPEND_IN_SHAPE;
  auto cnode = anf_node_ptr->cast<CNodePtr>();
  if (cnode != nullptr) {
    auto op_name = common::AnfAlgo::GetCNodeName(cnode);
    if (IsOneOfComputeDepend(op_name)) {
      unknow_type_ = device::ascend::UnknowShapeOpType::DEPEND_COMPUTE;
    }
  }
}
DynamicAicpuOpKernelMod::~DynamicAicpuOpKernelMod() noexcept {
  // free dev ptr
  if (ext_info_addr_dev_ != nullptr) {
    auto mem_manager = std::make_shared<device::ascend::AscendMemoryManager>();
    mem_manager->FreeMemFromMemPool(ext_info_addr_dev_);
  }
}

int DynamicAicpuOpKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
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
  MS_LOG(INFO) << "UpdateExtInfo of " << cnode->fullname_with_scope() << " start";
  auto input_num = common::AnfAlgo::GetInputTensorNum(cnode);
  auto output_num = common::AnfAlgo::GetOutputTensorNum(cnode);
  if (input_num == 0 && output_num == 0) {
    MS_LOG(INFO) << "Node:" << cnode->fullname_with_scope() << " no need to update output shape";
    return 0;
  }

  // Parse aicpu ext info
  ext_info_handler_ = std::make_shared<device::ascend::AicpuExtInfoHandler>(
    cnode->fullname_with_scope(), static_cast<uint32_t>(input_num), static_cast<uint32_t>(output_num), unknow_type_);
  MS_EXCEPTION_IF_NULL(ext_info_handler_);
  if (!ext_info_handler_->Parse(ext_info_)) {
    MS_LOG(EXCEPTION) << "Parse AiCpu ext_info_handler failed";
  }

  if (ext_info_.empty()) {
    MS_LOG(INFO) << "No need to copy to device, ext_info_ is empty. ";
    return 0;
  }

  for (uint32_t i = 0; i < input_num; ++i) {
    if (!ext_info_handler_->UpdateInputShapeAndType(i, NOT_NULL(cnode))) {
      MS_LOG(EXCEPTION) << "Update input shape failed, cnode:" << cnode->fullname_with_scope() << " input:" << i;
    }
  }

  if (unknow_type_ != device::ascend::UnknowShapeOpType::DEPEND_COMPUTE) {
    for (uint32_t i = 0; i < output_num; ++i) {
      if (!ext_info_handler_->UpdateOutputShapeAndType(i, NOT_NULL(cnode))) {
        MS_LOG(EXCEPTION) << "Update output shape failed, cnode:" << cnode->fullname_with_scope() << " output:" << i;
      }
    }
  }

  return 0;
}

void DynamicAicpuOpKernelMod::AllocateExtInfoDeviceAddr(const CNodePtr &cnode) {
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

bool DynamicAicpuOpKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
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
  MS_LOG(INFO) << "Start launch of node: " << cnode->fullname_with_scope();

  // is dynamic shape
  if (!common::AnfAlgo::IsDynamicShape(cnode)) {
    MS_LOG(EXCEPTION) << "The cnode is not dynamic shape:" << cnode->fullname_with_scope();
  }

  // copy extinfo to device
  AllocateExtInfoDeviceAddr(cnode);
  MS_EXCEPTION_IF_NULL(ext_info_handler_);
  auto ret = aclrtMemcpyAsync(ext_info_addr_dev_, ext_info_size_, ext_info_handler_->GetExtInfo(),
                              ext_info_handler_->GetExtInfoLen(), ACL_MEMCPY_HOST_TO_DEVICE, stream_ptr);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "UpdateExtInfo aclrtMemcpy failed. Node info: " << cnode->fullname_with_scope();
    return false;
  }

  AicpuOpKernelMod::CreateCpuKernelInfo(inputs, outputs);
  MS_LOG(INFO) << "Aicpu launch, node_so_:" << node_so_ << ", node name:" << node_name_
               << ", args_size:" << args_.length();
  auto flag = RT_KERNEL_DEFAULT;
  if (cust_kernel_) {
    flag = RT_KERNEL_CUSTOM_AICPU;
  }
  // cppcheck-suppress unreadVariable
  auto lock = device::KernelRuntime::LockRuntime(stream_ptr);
  rtArgsEx_t argsInfo = {};
  argsInfo.args = args_.data();
  argsInfo.argsSize = static_cast<uint32_t>(args_.length());
  ret = rtCpuKernelLaunchWithFlag(reinterpret_cast<const void *>(node_so_.c_str()),
                                  reinterpret_cast<const void *>(node_name_.c_str()), 1, &argsInfo, nullptr, stream_ptr,
                                  flag);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "Aicpu op launch failed!";
    return false;
  }

  return true;
}

void DynamicAicpuOpKernelMod::SyncData() {
  auto node = anf_node_.lock();
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  MS_LOG(INFO) << "Aicpu " << cnode->fullname_with_scope() << " PostExecute";
  // is dynamic shape
  if (!common::AnfAlgo::IsDynamicShape(cnode)) {
    MS_LOG(EXCEPTION) << "The cnode is not dynamic shape:" << cnode->fullname_with_scope();
  }

  if (unknow_type_ != device::ascend::UnknowShapeOpType::DEPEND_COMPUTE ||
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

void DynamicAicpuOpKernelMod::UpdateOutputShapeFromExtInfo(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_LOG(INFO) << "UpdateOutputShapeFromExtInfo start. Op name " << cnode->fullname_with_scope();
  MS_EXCEPTION_IF_NULL(ext_info_handler_);

  std::vector<TypeId> type_ids;
  std::vector<ShapeVector> shapes;
  auto output_num = common::AnfAlgo::GetOutputTensorNum(cnode);
  for (size_t i = 0; i < output_num; ++i) {
    std::vector<int64_t> shape;
    TypeId type_id;
    (void)ext_info_handler_->GetOutputShapeAndType(SizeToUint(i), NOT_NULL(&shape), NOT_NULL(&type_id));
    (void)type_ids.emplace_back(type_id);
    (void)shapes.emplace_back(shape);
  }

  common::AnfAlgo::SetOutputInferTypeAndShape(type_ids, shapes, cnode.get());
}
}  // namespace kernel
}  // namespace mindspore
