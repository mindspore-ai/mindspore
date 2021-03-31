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
#include <vector>
#include <memory>
#include <set>
#include <algorithm>
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
  MS_LOG(INFO) << "Execute AiCpuDynamicKerenl Start";
  auto ret = rtCpuKernelLaunchWithFlag(
    reinterpret_cast<const void *>(so_name_.c_str()), reinterpret_cast<const void *>(kernel_name_.c_str()), 1,
    reinterpret_cast<const void *>(args_.data()), args_.size(), nullptr, stream_, RT_KERNEL_DEFAULT);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Call rtCpuKernelLaunchWithFlag Failed";
  }
}

void AiCpuDynamicKernel::Initialize() {
  // is dynamic
  auto cnode = cnode_ptr_.lock();
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
    MS_EXCEPTION_IF_NULL(cnode);
    ext_info_handler_ =
      std::make_shared<AicpuExtInfoHandler>(cnode->fullname_with_scope(), input_num_, output_num_, shape_type);
    ext_info_handler_->Parse(ext_info_data_);
  }

  if (ext_info_data_.empty()) {
    MS_LOG(INFO) << "No need to copy to device, ext_info_data_ is empty. ";
    return;
  }

  // Allocate ext info addr in device
  auto ret = rtMalloc(&ext_info_addr_dev_, ext_info_data_.size(), RT_MEMORY_HBM);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Call rtMalloc ext_info_addr_dev_ failed";
  }
  ext_info_size_ = ext_info_data_.size();

  ret = rtMemcpy(ext_info_addr_dev_, ext_info_size_, ext_info_data_.data(), ext_info_data_.size(),
                 RT_MEMCPY_HOST_TO_DEVICE);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "Call rtMemcpy ext_info_addr_dev_ failed";
  }

  auto aicpu_param_head = reinterpret_cast<kernel::AicpuParamHead *>(args_.data());
  aicpu_param_head->extInfoLength = ext_info_size_;
  aicpu_param_head->extInfoAddr = reinterpret_cast<uint64_t>(ext_info_addr_dev_);
}

bool AiCpuDynamicKernel::UpdateInputOutputAddr() {
  std::vector<uint64_t> io_addrs;
  io_addrs.reserve(input_num_ + output_num_);
  auto cnode = cnode_ptr_.lock();
  for (size_t i = 0; i < input_num_; ++i) {
    auto input_addr = AnfAlgo::GetPrevNodeOutputAddr(cnode, i);
    io_addrs.emplace_back(reinterpret_cast<uintptr_t>(input_addr->GetMutablePtr()));
  }

  for (size_t i = 0; i < output_num_; ++i) {
    auto output_addr = AnfAlgo::GetOutputAddr(cnode, i);
    io_addrs.emplace_back(reinterpret_cast<uintptr_t>(output_addr->GetMutablePtr()));
  }

  if (args_.empty()) {
    MS_LOG(ERROR) << "args_ is empty";
    return false;
  }

  auto io_ptr = args_.data() + sizeof(kernel::AicpuParamHead);
  auto ret =
    memcpy_s(io_ptr, args_.size() - sizeof(kernel::AicpuParamHead), &io_addrs[0], sizeof(uint64_t) * io_addrs.size());
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "Memcpy input output addr failed";
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

  for (size_t i = 0; i < input_num_; ++i) {
    ext_info_handler_->UpdateInputShapeAndType(i, NOT_NULL(cnode));
  }

  if (AnfAlgo::IsDynamicShape(cnode) && unknow_type_ != DEPEND_COMPUTE) {
    for (size_t i = 0; i < output_num_; ++i) {
      ext_info_handler_->UpdateOutputShapeAndType(i, NOT_NULL(cnode));
    }
  }

  auto ret = rtMemcpy(ext_info_addr_dev_, ext_info_size_, ext_info_handler_->GetExtInfo(),
                      ext_info_handler_->GetExtInfoLen(), RT_MEMCPY_HOST_TO_DEVICE);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "UpdateExtInfo rtMemcpy failed";
    return false;
  }

  MS_LOG(INFO) << "UpdateExtInfo of " << cnode->fullname_with_scope() << " end";
  return true;
}

bool AiCpuDynamicKernel::UpdateOutputShapeFromExtInfo() {
  if (input_num_ == 0) {
    MS_LOG(WARNING) << "input num is 0";
    return true;
  }
  MS_LOG(INFO) << "UpdateOutputShapeFromExtInfo start";
  auto ret = rtMemcpy(ext_info_handler_->GetExtInfo(), ext_info_handler_->GetExtInfoLen(), ext_info_addr_dev_,
                      ext_info_size_, RT_MEMCPY_DEVICE_TO_HOST);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "rtMemcpy output shape failed";
    return false;
  }

  MS_LOG(INFO) << "rtMemcpy from device to host success";

  std::vector<TypeId> type_ids;
  std::vector<std::vector<size_t>> shapes;

  for (size_t i = 0; i < output_num_; ++i) {
    MS_LOG(INFO) << "Get output:" << output_num_ << " Shape";
    std::vector<int64_t> shape;
    TypeId type_id;
    ext_info_handler_->GetOutputShapeAndType(i, NOT_NULL(&shape), NOT_NULL(&type_id));

    for (auto x : shape) {
      MS_LOG(INFO) << "Update output:" << i << " shape:" << x;
    }

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
    MS_LOG(ERROR) << "Call runtime rtStreamSynchronize error.";
    return;
  }
  if (AnfAlgo::IsDynamicShape(cnode) && unknow_type_ == DEPEND_COMPUTE) {
    MS_LOG(INFO) << "Update aicpu kernel output shape from ext_info";
    UpdateOutputShapeFromExtInfo();
  }
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
