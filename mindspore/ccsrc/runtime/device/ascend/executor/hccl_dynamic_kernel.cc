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

#include "runtime/device/ascend/executor/hccl_dynamic_kernel.h"

#include "hccl/hcom.h"
#include "common/opskernel/ge_task_info.h"
#include "utils/log_adapter.h"
#include "runtime/device/kernel_runtime.h"
#include "backend/kernel_compiler/hccl/hcom_util.h"
#include "runtime/hccl_adapter/hccl_adapter.h"

namespace {
// Find so in RPATH or LD_LIBRARY_PATH (/usr/local/Ascend/fwkacllib/lib64/)
constexpr auto kHcomGraphAdaptorPath = "libhcom_graph_adaptor.so";
}  // namespace

namespace mindspore {
namespace device {
namespace ascend {
void HcclDynamicKernel::UpdateArgs() {
  if (!is_dynamic_shape_) {
    MS_LOG(INFO) << "Not Dynamic Shape";
    return;
  }
  auto cnode = cnode_ptr_.lock();
  MS_EXCEPTION_IF_NULL(cnode);
  MS_LOG(INFO) << "Start to UpdateArgs. Node info: " << cnode->DebugString();
  auto kernel_mod = AnfAlgo::GetKernelMod(cnode);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  // Update input, output, count
  AddressPtrList kernel_inputs;
  AddressPtrList kernel_workspaces;
  AddressPtrList kernel_outputs;
  KernelRuntime::GenLaunchArgs(*kernel_mod, cnode, &kernel_inputs, &kernel_workspaces, &kernel_outputs);
  if (kernel_inputs.empty() || kernel_outputs.empty()) {
    MS_LOG(EXCEPTION) << "Inputs or outputs is empty. Node info: " << cnode->DebugString();
  }
  auto input0 = kernel_inputs.at(0);
  auto output0 = kernel_outputs.at(0);
  MS_EXCEPTION_IF_NULL(input0);
  MS_EXCEPTION_IF_NULL(output0);

  // Update Hccl input and output
  input_ptr_ = input0->addr;
  output_ptr_ = output0->addr;

  std::vector<std::vector<size_t>> hccl_kernel_input_shape_list;
  if (!HcomUtil::GetKernelInputShape(cnode, &hccl_kernel_input_shape_list)) {
    MS_LOG(EXCEPTION) << "GetKernelInputShape fail! Node info: " << cnode->DebugString();
  }

  std::vector<HcclDataType> hccl_data_type_list;
  if (!HcomUtil::GetHcomDataType(cnode, &hccl_data_type_list)) {
    MS_LOG(EXCEPTION) << "GetHcomDataType fail! Node info: " << cnode->DebugString();
  }

  // Update Hccl count
  if (!HcomUtil::GetHcomCount(cnode, hccl_data_type_list, hccl_kernel_input_shape_list, &count_)) {
    MS_LOG(EXCEPTION) << "GetHcomCount fail! Node info: " << cnode->DebugString();
  }
  MS_LOG(INFO) << "Update Hccl count:" << count_;
}

void HcclDynamicKernel::StaticShapeExecute() {
  auto cnode = cnode_ptr_.lock();
  MS_EXCEPTION_IF_NULL(cnode);
  auto kernel_mod = AnfAlgo::GetKernelMod(cnode);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  AddressPtrList kernel_inputs;
  AddressPtrList kernel_workspaces;
  AddressPtrList kernel_outputs;
  KernelRuntime::GenLaunchArgs(*kernel_mod, cnode, &kernel_inputs, &kernel_workspaces, &kernel_outputs);
  kernel_mod->Launch(kernel_inputs, kernel_workspaces, kernel_outputs, stream_);
}

void HcclDynamicKernel::Execute() {
  auto cnode = cnode_ptr_.lock();
  MS_EXCEPTION_IF_NULL(cnode);
  MS_LOG(INFO) << "Start Execute: " << cnode->DebugString();
  ::HcomOperation op_info;
  op_info.hcclType = hccl_type_;
  op_info.inputPtr = input_ptr_;
  op_info.outputPtr = output_ptr_;
  op_info.dataType = static_cast<HcclDataType>(data_type_);
  op_info.opType = static_cast<HcclReduceOp>(op_type_);
  op_info.root = IntToUint(root_);
  op_info.count = count_;

  auto callback = [this](HcclResult status) {
    if (status != HCCL_SUCCESS) {
      MS_LOG(ERROR) << "HcomExcutorInitialize failed, ret:" << status;
    }
    std::lock_guard<std::mutex> lock(this->hccl_mutex_);
    this->cond_.notify_all();
    MS_LOG(INFO) << "hccl callback success.";
  };

  auto hccl_ret = hccl::HcclAdapter::GetInstance().HcclExecEnqueueOp(op_info, callback);
  if (hccl_ret != HCCL_SUCCESS) {
    MS_LOG(EXCEPTION) << "Call EnqueueHcomOperation failed, node info: " << cnode->DebugString();
  }

  std::unique_lock<std::mutex> ulock(hccl_mutex_);
  cond_.wait(ulock);
  MS_LOG(INFO) << "Execute " << cnode->DebugString() << " success";
}

void HcclDynamicKernel::PostExecute() {}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
