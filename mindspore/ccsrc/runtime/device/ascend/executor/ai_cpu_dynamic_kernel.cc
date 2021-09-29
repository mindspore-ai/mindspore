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

#include "runtime/device/ascend/executor/ai_cpu_dynamic_kernel.h"
#include "runtime/mem.h"
#include "runtime/kernel.h"
#include "utils/utils.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "backend/kernel_compiler/aicpu/aicpu_util.h"

namespace mindspore {
namespace device {
namespace ascend {
AiCpuDynamicKernel::~AiCpuDynamicKernel() {
  // free dev ptr
  if (ext_info_addr_dev_ == nullptr) {
    return;
  }
  auto ret = rtFree(ext_info_addr_dev_);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "rtFree failed";
  }
}

void AiCpuDynamicKernel::UpdateArgs() {
  if (!UpdateInputOutputAddr()) {
    MS_LOG(EXCEPTION) << "Update input output failed";
  }

  if (is_dynamic_shape_ && !UpdateExtInfo()) {
    MS_LOG(EXCEPTION) << "Update ExtInfo failed";
  }
}

void AiCpuDynamicKernel::Execute() {
  auto cnode = cnode_ptr_.lock();
  MS_EXCEPTION_IF_NULL(cnode);
  MS_LOG(INFO) << "Execute AiCpuDynamicKerenl Start, op name: " << cnode->fullname_with_scope();
  auto ret = rtCpuKernelLaunchWithFlag(
    reinterpret_cast<const void *>(so_name_.c_str()), reinterpret_cast<const void *>(kernel_name_.c_str()), 1,
    reinterpret_cast<const void *>(args_.data()), SizeToUint(args_.size()), nullptr, stream_, RT_KERNEL_DEFAULT);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Call rtCpuKernelLaunchWithFlag Failed";
  }
}

void AiCpuDynamicKernel::Initialize() {
  // is dynamic
  auto cnode = cnode_ptr_.lock();
  MS_EXCEPTION_IF_NULL(cnode);
  MS_LOG(INFO) << "Initialize node:" << cnode->fullname_with_scope();
  DynamicKernel::Initialize();

  input_num_ = AnfAlgo::GetInputTensorNum(cnode);
  output_num_ = AnfAlgo::GetOutputTensorNum(cnode);

  UnknowShapeOpType shape_type = UnknowShapeOpType::DEPEND_IN_SHAPE;
  auto op_name = AnfAlgo::GetCNodeName(cnode);
  if (kComputeDepend.find(op_name) != kComputeDepend.end()) {
    shape_type = UnknowShapeOpType::DEPEND_COMPUTE;
  }
  unknow_type_ = shape_type;
  // Parse aicpu ext info
  if (is_dynamic_shape_) {
    ext_info_handler_ =
      std::make_shared<AicpuExtInfoHandler>(cnode->fullname_with_scope(), input_num_, output_num_, shape_type);
    MS_EXCEPTION_IF_NULL(ext_info_handler_);
    if (!ext_info_handler_->Parse(ext_info_data_)) {
      MS_LOG(EXCEPTION) << "Parse AiCpu ext_info_handler failed";
    }
  }

  if (ext_info_data_.empty()) {
    MS_LOG(INFO) << "No need to copy to device, ext_info_data_ is empty. ";
    return;
  }

  // Allocate ext info addr in device
  auto ret = rtMalloc(&ext_info_addr_dev_, ext_info_data_.size(), RT_MEMORY_HBM);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Call rtMalloc ext_info_addr_dev_ failed. Op name: " << cnode->fullname_with_scope();
  }
  ext_info_size_ = ext_info_data_.size();

  ret = rtMemcpy(ext_info_addr_dev_, ext_info_size_, ext_info_data_.data(), ext_info_data_.size(),
                 RT_MEMCPY_HOST_TO_DEVICE);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Call rtMemcpy ext_info_addr_dev_ failed. Op name: " << cnode->fullname_with_scope();
  }

  auto aicpu_param_head = reinterpret_cast<kernel::AicpuParamHead *>(args_.data());
  MS_EXCEPTION_IF_NULL(aicpu_param_head);
  aicpu_param_head->extInfoLength = SizeToUint(ext_info_size_);
  aicpu_param_head->extInfoAddr = reinterpret_cast<uint64_t>(ext_info_addr_dev_);
}

bool AiCpuDynamicKernel::UpdateInputOutputAddr() {
  std::vector<uint64_t> io_addrs;
  io_addrs.reserve(input_num_ + output_num_);
  auto cnode = cnode_ptr_.lock();
  MS_EXCEPTION_IF_NULL(cnode);
  for (size_t i = 0; i < input_num_; ++i) {
    auto input_addr = AnfAlgo::GetPrevNodeOutputAddr(cnode, i);
    MS_EXCEPTION_IF_NULL(input_addr);
    io_addrs.emplace_back(reinterpret_cast<uintptr_t>(input_addr->GetMutablePtr()));
  }

  for (size_t i = 0; i < output_num_; ++i) {
    auto output_addr = AnfAlgo::GetOutputAddr(cnode, i);
    MS_EXCEPTION_IF_NULL(output_addr);
    io_addrs.emplace_back(reinterpret_cast<uintptr_t>(output_addr->GetMutablePtr()));
  }

  if (args_.empty()) {
    MS_LOG(ERROR) << "Parameter args_ is empty. Op name " << cnode->fullname_with_scope();
    return false;
  }

  auto io_ptr = args_.data() + sizeof(kernel::AicpuParamHead);
  if (io_addrs.empty()) {
    MS_LOG(ERROR) << "The io_addrs is empty";
    return false;
  }
  auto ret =
    memcpy_s(io_ptr, args_.size() - sizeof(kernel::AicpuParamHead), &io_addrs[0], sizeof(uint64_t) * io_addrs.size());
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "Memcpy input output addr failed. Op name: " << cnode->fullname_with_scope();
  }

  return true;
}

bool AiCpuDynamicKernel::UpdateExtInfo() {
  auto cnode = cnode_ptr_.lock();
  MS_EXCEPTION_IF_NULL(cnode);
  MS_LOG(INFO) << "UpdateExtInfo of " << cnode->fullname_with_scope() << " start";
  if (input_num_ == 0 && output_num_ == 0) {
    MS_LOG(INFO) << "Node:" << cnode->fullname_with_scope() << " no need to update output shape";
    return true;
  }

  MS_EXCEPTION_IF_NULL(ext_info_handler_);
  for (size_t i = 0; i < input_num_; ++i) {
    if (!ext_info_handler_->UpdateInputShapeAndType(i, NOT_NULL(cnode))) {
      MS_LOG(ERROR) << "Update input shape failed, cnode:" << cnode->fullname_with_scope() << " input:" << i;
      return false;
    }
  }

  if (AnfAlgo::IsDynamicShape(cnode) && unknow_type_ != DEPEND_COMPUTE) {
    for (size_t i = 0; i < output_num_; ++i) {
      if (!ext_info_handler_->UpdateOutputShapeAndType(i, NOT_NULL(cnode))) {
        MS_LOG(ERROR) << "Update output shape failed, cnode:" << cnode->fullname_with_scope() << " output:" << i;
        return false;
      }
    }
  }

  auto ret = rtMemcpy(ext_info_addr_dev_, ext_info_size_, ext_info_handler_->GetExtInfo(),
                      ext_info_handler_->GetExtInfoLen(), RT_MEMCPY_HOST_TO_DEVICE);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "UpdateExtInfo rtMemcpy failed. Node info: " << cnode->fullname_with_scope();
    return false;
  }

  MS_LOG(INFO) << "UpdateExtInfo of " << cnode->fullname_with_scope() << " end";
  return true;
}

bool AiCpuDynamicKernel::UpdateOutputShapeFromExtInfo() {
  auto cnode = cnode_ptr_.lock();
  MS_EXCEPTION_IF_NULL(cnode);
  MS_LOG(INFO) << "UpdateOutputShapeFromExtInfo start. Op name " << cnode->fullname_with_scope();
  MS_EXCEPTION_IF_NULL(ext_info_handler_);
  auto ret = rtMemcpy(ext_info_handler_->GetExtInfo(), ext_info_handler_->GetExtInfoLen(), ext_info_addr_dev_,
                      ext_info_size_, RT_MEMCPY_DEVICE_TO_HOST);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "rtMemcpy output shape failed. Op name: " << cnode->fullname_with_scope();
    return false;
  }

  MS_LOG(INFO) << "rtMemcpy from device to host success";
  std::vector<TypeId> type_ids;
  std::vector<std::vector<size_t>> shapes;

  for (size_t i = 0; i < output_num_; ++i) {
    MS_LOG(INFO) << "Get output:" << output_num_ << " Shape";
    std::vector<int64_t> shape;
    TypeId type_id;
    (void)ext_info_handler_->GetOutputShapeAndType(SizeToUint(i), NOT_NULL(&shape), NOT_NULL(&type_id));
    type_ids.emplace_back(type_id);
    std::vector<size_t> size_t_shape;
    std::transform(shape.begin(), shape.end(), std::back_inserter(size_t_shape), LongToSize);
    shapes.emplace_back(size_t_shape);
  }

  AnfAlgo::SetOutputInferTypeAndShape(type_ids, shapes, cnode_ptr_.lock().get());
  return true;
}

void AiCpuDynamicKernel::PostExecute() {
  auto cnode = cnode_ptr_.lock();
  MS_EXCEPTION_IF_NULL(cnode);
  MS_LOG(INFO) << "Aicpu " << cnode->fullname_with_scope() << " PostExecute";
  if (unknow_type_ != DEPEND_COMPUTE) {
    return;
  }
  if (RT_ERROR_NONE != rtStreamSynchronize(stream_)) {
    MS_LOG(EXCEPTION) << "Call runtime rtStreamSynchronize failed. Op name: " << cnode->fullname_with_scope();
  }
  if (AnfAlgo::IsDynamicShape(cnode)) {
    MS_LOG(INFO) << "Update aicpu kernel output shape from ext_info. Op name: " << cnode->fullname_with_scope();
    UpdateOutputShapeFromExtInfo();
  }
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
