/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#include "plugin/ascend/kernel_executor/rts/move_to.h"

#include <map>
#include <string>
#include <vector>
#include <utility>

#include "include/utils/anfalgo.h"
#include "include/utils/utils.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/runtime/hardware_abstract/memory_manager/swap_manager.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "utils/log_adapter.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"
#include "include/runtime/hardware_abstract/kernel_base/graph_fusion/framework_utils.h"

namespace mindspore {
namespace kernel {
constexpr size_t kToInputIndex = 2;
constexpr size_t kBlockInputIndex = 3;
constexpr int64_t kNpuInt = 0;
constexpr int64_t kCpuInt = 1;

static const std::map<std::string, int64_t> ToStrMap{{kToNpu, kNpuInt}, {kToCpu, kCpuInt}};

std::map<std::pair<int64_t, int64_t>, MoveFunc> MoveTo::func_map_ = {{{kNpuInt, kCpuInt}, &MoveTo::MoveFromDToH},
                                                                     {{kNpuInt, kNpuInt}, &MoveTo::EmptyMove},
                                                                     {{kCpuInt, kNpuInt}, &MoveTo::MoveFromHToD},
                                                                     {{kCpuInt, kCpuInt}, &MoveTo::EmptyMove}};

bool MoveTo::GetToFromValue(const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(value);
  if (!value->isa<StringImm>()) {
    MS_LOG(ERROR) << "The value of the second input of MoveTo[" << value->ToString() << "] is not a string.";
    return false;
  }
  const auto &str_value = value->cast<StringImmPtr>()->value();
  const auto &iter = ToStrMap.find(str_value);
  if (iter == ToStrMap.end()) {
    MS_LOG(ERROR) << "Invalid value for second input of MoveTo: " << str_value;
    return false;
  }
  to_ = iter->second;
  return true;
}

bool MoveTo::GetToValue(const AnfNodePtr &anf_node, size_t to_input_index) {
  MS_EXCEPTION_IF_NULL(anf_node);
  auto cnode = anf_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  const auto &kernel_with_index = common::AnfAlgo::VisitKernelWithReturnType(cnode->input(to_input_index), 0, true);
  auto to_input = kernel_with_index.first;
  MS_EXCEPTION_IF_NULL(to_input);
  if (!to_input->isa<ValueNode>()) {
    MS_LOG(ERROR) << "Get to value failed, the second input of MoveTo is not a ValueNode.";
    return false;
  }
  auto to_value_node = to_input->cast<ValueNodePtr>();
  auto to_value = to_value_node->value();
  if (!GetToFromValue(to_value)) {
    MS_LOG(ERROR) << anf_node->fullname_with_scope()
                  << ": GetToValue failed, second input value: " << to_value->ToString();
    return false;
  }
  return true;
}

bool MoveTo::GetBlockingValue(const mindspore::AnfNodePtr &anf_node, size_t block_input_index) {
  MS_EXCEPTION_IF_NULL(anf_node);
  auto cnode = anf_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  const auto &kernel_with_index = common::AnfAlgo::VisitKernelWithReturnType(cnode->input(block_input_index), 0, true);
  auto block_input = kernel_with_index.first;
  MS_EXCEPTION_IF_NULL(block_input);
  if (!block_input->isa<ValueNode>()) {
    MS_LOG(ERROR) << "Get to value failed, the second input of MoveTo is not a ValueNode.";
    return false;
  }
  auto block_value_node = block_input->cast<ValueNodePtr>();
  auto block_value = block_value_node->value();
  if (!block_value->isa<BoolImm>()) {
    MS_LOG(ERROR) << "The value of the third input of MoveTo[" << block_value->ToString() << "] is not a bool.";
    return false;
  }
  blocking_ = block_value->cast<BoolImmPtr>()->value();
  return true;
}

bool MoveTo::UpdateSizeList(const AnfNodePtr &anf_node) {
  std::vector<KernelTensor *> input_kernel_tensors = AnfAlgo::GetOrCreateAllInputKernelTensors(anf_node);
  std::vector<KernelTensor *> output_kernel_tensors = AnfAlgo::GetOrCreateAllOutputKernelTensors(anf_node);
  auto prim = common::AnfAlgo::GetCNodePrimitive(anf_node);
  MS_EXCEPTION_IF_NULL(prim);
  primitive_ = prim;
  kernel_name_ = prim->name();
  auto cnode = anf_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (CheckResizeCondition(cnode)) {
    if (Resize(input_kernel_tensors, output_kernel_tensors) == KRET_RESIZE_FAILED) {
      MS_LOG(ERROR) << "#dmsg#Kernel build failed:#dmsg#rts kernel op[" << cnode->fullname_with_scope()
                    << "] Resize failed.";
      return false;
    }
  }
  return true;
}

bool MoveTo::Init(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  if (!GetToValue(anf_node, kToInputIndex) || !GetBlockingValue(anf_node, kBlockInputIndex)) {
    return false;
  }
  return UpdateSizeList(anf_node);
}

int64_t MoveTo::GetTensorDevice(const KernelTensor *tensor) {
  MS_EXCEPTION_IF_NULL(tensor->device_address());
  const auto device_type = tensor->device_address()->GetDeviceType();
  if (device_type == device::DeviceType::kCPU) {
    return kCpuInt;
  } else if (device_type == device::DeviceType::kAscend) {
    return kNpuInt;
  } else {
    MS_LOG(EXCEPTION) << "MoveTo only support CPU or NPU DeviceAddress input, but get " << device_type;
  }
}

device::SwapManagerPtr MoveTo::GetSwapManager(const KernelTensor *tensor) {
  MS_EXCEPTION_IF_NULL(tensor);
  const auto device_id = tensor->device_id();
  const auto device_context =
    device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext({tensor->GetDeviceType(), device_id});
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);
  return device_context->device_res_manager_->swap_manager();
}

bool MoveTo::SyncStream(void *stream_ptr) { return CALL_ASCEND_API(aclrtSynchronizeStream, stream_ptr) == ACL_SUCCESS; }

bool MoveTo::D2H(void *host_ptr, const void *device_ptr, void *stream_ptr, size_t size) {
  const auto status =
    CALL_ASCEND_API(aclrtMemcpyAsync, host_ptr, size, device_ptr, size, ACL_MEMCPY_DEVICE_TO_HOST, stream_ptr);
  if (status != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Moveto kernel aclrtMemcpyAsync device to host failed! src ptr: " << device_ptr
                  << ", dst ptr: " << host_ptr << ", size: " << size << ", stream: " << stream_ptr;
    return false;
  }
  return true;
}

bool MoveTo::H2D(void *device_ptr, const void *host_ptr, void *stream_ptr, size_t size) {
  const auto status =
    CALL_ASCEND_API(aclrtMemcpyAsync, device_ptr, size, host_ptr, size, ACL_MEMCPY_HOST_TO_DEVICE, stream_ptr);
  if (status != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Moveto kernel aclrtMemcpyAsync host to device failed! src ptr: " << device_ptr
                  << ", dst ptr: " << host_ptr << ", size: " << size << ", stream: " << stream_ptr;
    return false;
  }
  return true;
}

bool MoveTo::MoveFromDToH(const KernelTensor *dst_tensor, const KernelTensor *src_tensor, void *stream_ptr) {
  // Get src device ptr.
  const auto device_ptr = src_tensor->device_ptr();
  MS_EXCEPTION_IF_NULL(device_ptr);
  // Get dst host ptr.
  const auto &host_ptr = dst_tensor->device_ptr();
  MS_EXCEPTION_IF_NULL(host_ptr);

  // Memory copy.
  const auto size = src_tensor->size();
  return D2H(host_ptr, device_ptr, stream_ptr, size);
}

bool MoveTo::MoveFromHToD(const KernelTensor *dst_tensor, const KernelTensor *src_tensor, void *stream_ptr) {
  // Get src host ptr.
  const auto &host_ptr = src_tensor->device_ptr();
  MS_EXCEPTION_IF_NULL(host_ptr);
  // Get dst device ptr.
  const auto device_ptr = dst_tensor->device_ptr();
  MS_EXCEPTION_IF_NULL(device_ptr);

  // Memory copy.
  const auto size = src_tensor->size();
  return H2D(device_ptr, host_ptr, stream_ptr, size);
}

bool MoveTo::EmptyMove(const KernelTensor *, const KernelTensor *, void *) {
  MS_LOG(INFO) << "Kernel tensor has already been stored in target device, skip moving it.";
  return true;
}

bool MoveTo::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  const auto input = inputs[0];
  MS_EXCEPTION_IF_NULL(input);
  const auto output = outputs[0];
  MS_EXCEPTION_IF_NULL(output);
  MS_EXCEPTION_IF_NULL(stream_ptr);

  const auto from = GetTensorDevice(input);
  const auto &func_iter = func_map_.find(std::make_pair(from, to_));
  if (func_iter == func_map_.end()) {
    MS_LOG(ERROR) << "Not supported moving, from: " << from << ", to " << to_;
    return false;
  }
  auto func = func_iter->second;
  if (!(this->*func)(output, input, stream_ptr)) {
    MS_LOG(ERROR) << "Launch MoveTo kernel failed.";
    return false;
  }
  if (blocking_) {
    const auto status = CALL_ASCEND_API(aclrtSynchronizeStreamWithTimeout, stream_ptr, -1);
    if (status != ACL_SUCCESS) {
      MS_LOG(ERROR) << "Failed to synchronize stream, ret = " << status << ".";
      return false;
    }
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
