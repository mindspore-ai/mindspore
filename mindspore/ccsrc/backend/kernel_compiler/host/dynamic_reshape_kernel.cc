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

#include "backend/kernel_compiler/host/dynamic_reshape_kernel.h"

#include <functional>
#include "backend/session/anf_runtime_algorithm.h"
#include "abstract/utils.h"
#include "utils/trace_base.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kInputNum = 2;

std::vector<int64_t> GetInputValue(const CNodePtr &cnode, size_t index) {
  auto address_x = AnfAlgo::GetPrevNodeMutableOutputAddr(cnode, index);
  auto shape_x = AnfAlgo::GetPrevNodeOutputInferShape(cnode, index);
  if (shape_x.size() != 1) {
    MS_LOG(EXCEPTION) << "Input" << index << " must be [1-D], but got " << shape_x.size() << "-D."
                      << trace::DumpSourceLines(cnode);
  }
  session::KernelWithIndex kernel_with_index = AnfAlgo::GetPrevNodeOutput(cnode, index);
  auto type_x = AnfAlgo::GetOutputInferDataType(kernel_with_index.first, kernel_with_index.second);
  if (type_x != TypeId::kNumberTypeInt64 && type_x != TypeId::kNumberTypeInt32) {
    MS_LOG(EXCEPTION) << "Input x type must be int64 or int32, but got " << TypeIdToType(type_x)
                      << trace::DumpSourceLines(cnode);
  }

  size_t x_num = shape_x[0];
  std::vector<int64_t> x{SizeToLong(x_num)};

  auto x_shape_value = std::make_shared<tensor::Tensor>(type_x, x);
  x_shape_value->set_device_address(address_x);
  x_shape_value->data_sync();

  std::vector<int64_t> input_shape;
  if (type_x == TypeId::kNumberTypeInt64) {
    auto x_value = reinterpret_cast<int64_t *>(x_shape_value->data_c());
    MS_EXCEPTION_IF_NULL(x_value);
    input_shape = {x_value, x_value + x_num};
  } else {
    auto x_value = reinterpret_cast<int *>(x_shape_value->data_c());
    MS_EXCEPTION_IF_NULL(x_value);
    for (size_t i = 0; i < x_num; i++) {
      input_shape.push_back(static_cast<int64_t>(*x_value));
      ++x_value;
    }
  }
  return input_shape;
}
}  // namespace

void DynamicReshapeKernel::Execute() {
  MS_LOG(INFO) << "Execute host ReshapeKernel Start";
  auto cnode = cnode_ptr_.lock();
  MS_EXCEPTION_IF_NULL(cnode);
  auto input_num = AnfAlgo::GetInputTensorNum(cnode);
  if (input_num != kInputNum) {
    MS_LOG(EXCEPTION) << "Invalid input num, should be " << kInputNum << ", but got " << input_num
                      << trace::DumpSourceLines(cnode);
  }

  auto address_x = AnfAlgo::GetPrevNodeMutableOutputAddr(cnode, 0);
  MS_EXCEPTION_IF_NULL(address_x);
  auto type_x = AnfAlgo::GetOutputInferDataType(cnode, 0);
  auto shape_x = AnfAlgo::GetPrevNodeOutputInferShape(cnode, 0);
  std::vector<int64_t> output_shapes = GetInputValue(cnode, 1);

  int64_t dim_prod = 1;
  int64_t neg_index = -1;
  auto arr_prod = std::accumulate(shape_x.begin(), shape_x.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());
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

  size_t input_size_byte = LongToSize(arr_prod) * abstract::TypeIdSize(type_x);
  auto output_addr = AnfAlgo::GetOutputAddr(cnode, 0);
  MS_EXCEPTION_IF_NULL(output_addr);
  if (address_x->DeviceType() == device::DeviceAddressType::kCPU) {
    auto ret =
      memcpy_s(const_cast<void *>(output_addr->GetPtr()), output_addr->GetSize(), address_x->GetPtr(), input_size_byte);
    if (ret != EOK) {
      MS_LOG(EXCEPTION) << "Execute DynamicReshapeKernel memcpy_s failed";
    }
  } else {
    if (!output_addr->AsyncDeviceToDevice(output_shapes, input_size_byte, address_x->type_id(), address_x->GetPtr(),
                                          address_x->format())) {
      MS_LOG(EXCEPTION) << "Host Reshape sync device to device failed.";
    }
    MS_LOG(INFO) << "Execute host ReshapeKernel End";
  }
}
device::DynamicKernelPtr DynamicReshapeKernelMod::GenDynamicKernel(const CNodePtr &cnode_ptr, void *stream_ptr) {
  return std::make_shared<DynamicReshapeKernel>(stream_ptr, cnode_ptr);
}
}  // namespace kernel
}  // namespace mindspore
