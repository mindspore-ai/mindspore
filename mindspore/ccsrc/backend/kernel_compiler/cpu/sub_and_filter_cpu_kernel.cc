/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/cpu/sub_and_filter_cpu_kernel.h"
#include <string>
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kSubAndFilterInputsNum = 3;
constexpr size_t kSubAndFilterOutputNum = 2;
}  // namespace

void SubAndFilterCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  node_wpt_ = kernel_node;
  input_x_dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
}

bool SubAndFilterCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                   const std::vector<kernel::AddressPtr> &,
                                   const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSubAndFilterInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSubAndFilterOutputNum, kernel_name_);
  if (input_x_dtype_ == kNumberTypeInt32) {
    LaunchKernel<int>(inputs, outputs);
  } else if (input_x_dtype_ == kNumberTypeInt64) {
    LaunchKernel<int64_t>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "Unsupported input data type: " << input_x_dtype_;
  }
  return true;
}

template <typename T>
void SubAndFilterCPUKernel::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                         const std::vector<kernel::AddressPtr> &outputs) {
  auto node = node_wpt_.lock();
  MS_EXCEPTION_IF_NULL(node);
  auto indices_shape = AnfAlgo::GetPrevNodeOutputInferShape(node, 0);

  batch_size_ = 1;
  for (size_t i = 0; i < indices_shape.size(); ++i) {
    batch_size_ *= indices_shape[i];
  }
  MS_LOG(INFO) << "SubAndFilter batch_size:" << batch_size_;

  T *input_x = reinterpret_cast<T *>(inputs[0]->addr);
  T max_num = *reinterpret_cast<T *>(inputs[1]->addr);
  T offset = *reinterpret_cast<T *>(inputs[2]->addr);
  T *filter_res = reinterpret_cast<T *>(outputs[0]->addr);
  T *filter_idx = reinterpret_cast<T *>(outputs[1]->addr);

  size_t count = 0;
  for (size_t i = 0; i < batch_size_; ++i) {
    T temp = input_x[i] - offset;
    if (temp < 0 || temp >= max_num) continue;
    filter_res[count] = temp;
    filter_idx[count] = static_cast<T>(i);
    count++;
  }
  MS_LOG(INFO) << "SubAndFilter output count is " << count;
  std::vector<size_t> out_shape;
  (void)out_shape.emplace_back(count);
  size_t output_num = AnfAlgo::GetOutputTensorNum(node);
  std::vector<TypeId> dtypes(output_num);
  for (size_t i = 0; i < output_num; i++) {
    dtypes[i] = AnfAlgo::GetOutputDeviceDataType(node, i);
  }
  AnfAlgo::SetOutputInferTypeAndShape(dtypes, {out_shape, out_shape}, node.get());
}
}  // namespace kernel
}  // namespace mindspore
