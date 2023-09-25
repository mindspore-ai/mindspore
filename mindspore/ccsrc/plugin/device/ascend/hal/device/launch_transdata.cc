/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/hal/device/launch_transdata.h"
#include <algorithm>
#include "abstract/utils.h"
#include "backend/common/session/single_kernel_graph.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "runtime/device/memory_manager.h"
#include "plugin/device/ascend/hal/device/ascend_memory_pool.h"
#include "plugin/device/ascend/kernel/acl/acl_kernel_build.h"
#include "acl/acl_rt.h"
#include "ops/array_op_name.h"

namespace mindspore::device::ascend {
std::vector<uint8_t *> LaunchTransData::GetKernelOutputAddr() { return outputs_addr_; }

void LaunchTransData::SetInputAddr(void *input_addr) { input_addr_ = input_addr; }

void LaunchTransData::FreeDeviceMem() {
  input_addr_ = nullptr;
  for (size_t i = 0; i < outputs_addr_.size(); ++i) {
    if (outputs_addr_[i] != nullptr) {
      AscendMemoryPool::GetInstance().FreeTensorMem(outputs_addr_[i]);
      outputs_addr_[i] = nullptr;
    }
  }
  outputs_addr_.clear();
}

void LaunchTransData::SetKernelBuildInfo() {
  if (!kernel_graph_->execution_order().empty()) {
    auto new_op = kernel_graph_->execution_order()[0];
    std::vector<TypeId> device_type = {dtype_};
    auto input_format = (src_format_ == kOpFormat_NCHW) ? kOpFormat_DEFAULT : src_format_;
    auto output_format = (dst_format_ == kOpFormat_NCHW) ? kOpFormat_DEFAULT : dst_format_;
    std::vector<std::string> inputs_format = {input_format};
    std::vector<std::string> outputs_format = {output_format};
    std::vector<kernel::KernelObjectType> input_object_types = {kernel::KernelObjectType::TENSOR};
    std::vector<kernel::KernelObjectType> output_object_types{kernel::KernelObjectType::TENSOR};
    // set build info
    auto builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
    builder->SetKernelType(KernelType::ACL_KERNEL);
    builder->SetInputsDeviceType(device_type);
    builder->SetOutputsDeviceType(device_type);
    builder->SetInputsFormat(inputs_format);
    builder->SetOutputsFormat(outputs_format);
    builder->SetInputsKernelObjectType(input_object_types);
    builder->SetOutputsKernelObjectType(output_object_types);
    builder->SetInputsReshapeType({});
    builder->SetOutputsReshapeType({});
    AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), new_op.get());
    // set attr
    bool in_def_flag = IsOneOfDefaultFormat(input_format);
    bool out_def_flag = IsOneOfDefaultFormat(output_format);
    common::AnfAlgo::SetNodeAttr(kAttrInputDefaultFormat, MakeValue(in_def_flag), new_op);
    common::AnfAlgo::SetNodeAttr(kAttrOutputDefaultFormat, MakeValue(out_def_flag), new_op);
    common::AnfAlgo::SetNodeAttr(kAttrSrcFormat, MakeValue(src_format_), new_op);
    common::AnfAlgo::SetNodeAttr(kAttrDstFormat, MakeValue(dst_format_), new_op);
    common::AnfAlgo::SetNodeAttr(kAttrGroups, MakeValue(groups_), new_op);
    common::AnfAlgo::SetNodeAttr(kAttrFracZGroup, MakeValue(groups_), new_op);
  }
}

void LaunchTransData::ConstructKernelGraph() {
  std::vector<TypeId> input_dtypes = {dtype_};
  std::vector<TypeId> output_dtypes = {dtype_};
  // obtain input & output shape
  std::vector<ShapeVector> input_shapes = {{shape_}};
  std::vector<ShapeVector> output_shapes = {{shape_}};
  kernel_graph_ = session::SingleKernelGraph::ConstructKernelGraphBasedOnSingleOp(
    kIdentityOpName, input_dtypes, input_shapes, output_dtypes, output_shapes);
  MS_EXCEPTION_IF_NULL(kernel_graph_);
}

uint8_t *LaunchTransData::AllocDeviceMem(size_t size) {
  auto device_memory = AscendMemoryPool::GetInstance().AllocTensorMem(size);
  if (device_memory == nullptr) {
    MS_LOG(EXCEPTION) << "Fail to alloc memory, size: " << size << "B.";
  }
  return static_cast<uint8_t *>(device_memory);
}

std::vector<kernel::AddressPtr> LaunchTransData::CreateOutputAddr(const std::vector<size_t> &outputs_list) {
  // init output_addr_
  outputs_addr_ = std::vector<uint8_t *>(outputs_list.size(), nullptr);
  std::vector<kernel::AddressPtr> output_address;
  if (outputs_addr_.size() < outputs_list.size()) {
    MS_LOG_EXCEPTION << "Error addr size!";
  }
  for (size_t i = 0; i < outputs_list.size(); ++i) {
    auto size = MemoryManager::GetCommonAlignSize(outputs_list[i]);
    outputs_addr_[i] = AllocDeviceMem(size);
    auto address = std::make_shared<kernel::Address>();
    MS_EXCEPTION_IF_NULL(address);
    address->addr = outputs_addr_[i];
    MS_EXCEPTION_IF_NULL(address->addr);
    address->size = size;
    output_address.push_back(address);
  }
  return output_address;
}

void LaunchTransData::AclKernelBuild() {
  auto kernel = kernel_graph_->execution_order()[0];
  kernel_mod_ = kernel::AclOpBuild(kernel);
  MS_EXCEPTION_IF_NULL(kernel_mod_);
  AnfAlgo::SetKernelMod(kernel_mod_, kernel.get());
}

void LaunchTransData::LaunchOpKernel() {
  // construct graph
  if (kernel_graph_ == nullptr) {
    ConstructKernelGraph();
  }
  SetKernelBuildInfo();
  AclKernelBuild();
  // inputs
  std::vector<kernel::AddressPtr> kernel_inputs;
  auto input = std::make_shared<kernel::Address>();
  MS_EXCEPTION_IF_NULL(input);
  input->addr = input_addr_;
  MS_EXCEPTION_IF_NULL(input->addr);
  input->size = total_size_;
  kernel_inputs.push_back(input);
  // outputs
  auto kernel_outputs = CreateOutputAddr(kernel_mod_->GetOutputSizeList());
  // workspaces
  std::vector<kernel::AddressPtr> kernel_workspace;
  // launch
  auto ret_status = kernel_mod_->Launch(kernel_inputs, kernel_workspace, kernel_outputs, stream_);
  if (!ret_status) {
    MS_LOG(EXCEPTION) << "Launch transdata single kernel failed";
  }
}
}  // namespace mindspore::device::ascend
