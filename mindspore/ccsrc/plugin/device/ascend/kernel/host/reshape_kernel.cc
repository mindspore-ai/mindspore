/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/kernel/host/reshape_kernel.h"

#include <algorithm>
#include <functional>
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "abstract/utils.h"
#include "runtime/device/kernel_runtime_manager.h"
#include "utils/check_convert_utils.h"
#include "utils/trace_base.h"
#include "runtime/mem.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kInputNum = 2;

static std::vector<int64_t> GetInputValue(const CNodePtr &cnode, size_t index) {
  auto address_x = AnfAlgo::GetPrevNodeMutableOutputAddr(cnode, index);
  MS_EXCEPTION_IF_NULL(address_x);
  auto shape_x = common::AnfAlgo::GetPrevNodeOutputInferShape(cnode, index);
  if (shape_x.size() != 1) {
    MS_LOG(EXCEPTION) << "Input" << index << " must be [1-D], but got " << shape_x.size() << "-D."
                      << trace::DumpSourceLines(cnode);
  }
  session::KernelWithIndex kernel_with_index = common::AnfAlgo::GetPrevNodeOutput(cnode, index);
  auto type_x = common::AnfAlgo::GetOutputInferDataType(kernel_with_index.first, kernel_with_index.second);
  if (type_x != TypeId::kNumberTypeInt64 && type_x != TypeId::kNumberTypeInt32) {
    MS_LOG(EXCEPTION) << "Input x type must be int64 or int32, but got " << TypeIdToType(type_x)
                      << trace::DumpSourceLines(cnode);
  }

  auto x_num = shape_x[0];
  std::vector<int64_t> x{x_num};

  auto x_shape_value = std::make_shared<tensor::Tensor>(type_x, x);
  x_shape_value->set_device_address(address_x);
  x_shape_value->data_sync();

  std::vector<int64_t> input_shape;
  if (type_x == TypeId::kNumberTypeInt64) {
    auto x_value = static_cast<int64_t *>(x_shape_value->data_c());
    MS_EXCEPTION_IF_NULL(x_value);
    input_shape = {x_value, x_value + x_num};
  } else {
    auto x_value = static_cast<int *>(x_shape_value->data_c());
    MS_EXCEPTION_IF_NULL(x_value);
    for (int64_t i = 0; i < x_num; i++) {
      input_shape.push_back(static_cast<int64_t>(*x_value));
      ++x_value;
    }
  }
  return input_shape;
}

static int64_t GetArrProd(const CNodePtr &cnode) {
  auto shape_x = common::AnfAlgo::GetPrevNodeOutputInferShape(cnode, 0);
  auto arr_prod = std::accumulate(shape_x.begin(), shape_x.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());
  return arr_prod;
}

static std::vector<int64_t> GetOutputShapesFromCNode(const CNodePtr &cnode) {
  std::vector<int64_t> output_shapes;
  auto input_num = common::AnfAlgo::GetInputTensorNum(cnode);
  if (input_num != kInputNum) {
    MS_LOG(DEBUG) << "Reshape has one input";
    auto prim = common::AnfAlgo::GetCNodePrimitive(cnode);
    ValuePtr sh = prim->GetAttr("shape");
    if (sh == nullptr) {
      auto un_output_shapes = common::AnfAlgo::GetOutputInferShape(cnode, 0);
      (void)std::transform(std::begin(un_output_shapes), std::end(un_output_shapes), std::back_inserter(output_shapes),
                           [](const uint64_t &i) -> int64_t { return static_cast<int64_t>(i); });
    } else if (sh->isa<ValueTuple>()) {
      auto reshape_value_tuple = sh->cast<ValueTuplePtr>();
      MS_EXCEPTION_IF_NULL(reshape_value_tuple);
      auto reshape_tuple = reshape_value_tuple->value();
      (void)std::transform(std::begin(reshape_tuple), std::end(reshape_tuple), std::back_inserter(output_shapes),
                           [](const ValuePtr &e) -> int64_t {
                             if (e->isa<UInt64Imm>()) {
                               return SizeToLong(GetValue<uint64_t>(e));
                             } else {
                               return GetValue<int64_t>(e);
                             }
                           });
    } else if (sh->isa<tensor::Tensor>()) {
      output_shapes = CheckAndConvertUtils::CheckTensorIntValue("shape", sh, "Reshape");
    } else {
      MS_EXCEPTION(ValueError) << "shape must be a tuple or constant Tensor";
    }
  } else {
    MS_LOG(DEBUG) << "Reshape has dynamic shape";
    output_shapes = GetInputValue(cnode, 1);
  }

  auto arr_prod = GetArrProd(cnode);
  int64_t dim_prod = 1;
  int64_t neg_index = -1;

  for (size_t i = 0; i < output_shapes.size(); ++i) {
    if (output_shapes[i] == -1) {
      neg_index = SizeToLong(i);
    } else {
      dim_prod *= output_shapes[i];
    }
  }
  if (neg_index != -1) {
    output_shapes[LongToSize(neg_index)] = arr_prod / dim_prod;
  }
  return output_shapes;
}
}  // namespace

void ReshapeKernelMod::Execute() const {
  auto node = anf_node_.lock();
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);

  auto address_x = AnfAlgo::GetPrevNodeMutableOutputAddr(cnode, 0);
  MS_EXCEPTION_IF_NULL(address_x);

  std::vector<int64_t> output_shapes = GetOutputShapesFromCNode(cnode);
  auto type_x = common::AnfAlgo::GetOutputInferDataType(cnode, 0);

  size_t input_size_byte = LongToSize(GetArrProd(cnode)) * abstract::TypeIdSize(type_x);
  // At execute reshape is noOpNode as all shapes are known so set skipNoOpNode false
  auto output_addr = AnfAlgo::GetOutputAddr(cnode, 0, false);
  MS_EXCEPTION_IF_NULL(output_addr);
  if (address_x->GetDeviceType() == device::DeviceType::kCPU) {
    auto ret =
      memcpy_s(const_cast<void *>(output_addr->GetPtr()), output_addr->GetSize(), address_x->GetPtr(), input_size_byte);
    if (ret != EOK) {
      MS_LOG(EXCEPTION) << "Execute ReshapeKernel memcpy_s failed";
    }
  } else {
    if (!output_addr->AsyncDeviceToDevice(output_shapes, input_size_byte, address_x->type_id(), address_x->GetPtr(),
                                          address_x->format())) {
      MS_LOG(EXCEPTION) << "Host Reshape sync device to device failed.";
    }
  }
}

void ReshapeKernelMod::Execute(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs,
                               void *stream_ptr) const {
  auto node = anf_node_.lock();
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);

  if (inputs.empty() || outputs.empty()) {
    MS_LOG(EXCEPTION) << "Inputs or outputs address of Reshape kernel is empty";
  }
  auto address_x = inputs[0]->addr;
  MS_EXCEPTION_IF_NULL(address_x);
  auto output_addr = outputs[0]->addr;
  MS_EXCEPTION_IF_NULL(output_addr);

  auto type_x = common::AnfAlgo::GetOutputInferDataType(cnode, 0);

  size_t input_size_byte = LongToSize(GetArrProd(cnode)) * abstract::TypeIdSize(type_x);
  // cppcheck-suppress unreadVariable
  auto lock = device::KernelRuntime::LockRuntime(stream_ptr);
  auto status =
    rtMemcpyAsync(output_addr, outputs[0]->size, address_x, input_size_byte, RT_MEMCPY_DEVICE_TO_DEVICE, stream_ptr);
  if (status != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "Call rtMemcpyAsync failed, ret = 0x" << status;
  }
}

bool ReshapeKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                              const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  auto node = anf_node_.lock();
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  try {
    Execute(inputs, outputs, stream_ptr);
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "ReshapeKernelMod Launch failed. node: " << cnode->fullname_with_scope() << ", Error message is "
                  << e.what();
    return false;
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
