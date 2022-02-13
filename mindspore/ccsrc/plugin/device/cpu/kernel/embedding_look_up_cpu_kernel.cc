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

#include "plugin/device/cpu/kernel/embedding_look_up_cpu_kernel.h"
#include <thread>
#include <string>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "ir/primitive.h"
#include "common/thread_pool.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kBlockSize = 10000;
constexpr size_t kEmbeddingLookupInputsNum = 2;
constexpr size_t kEmbeddingLookupOutputsNum = 1;
constexpr size_t kEmbeddingLookupInputParamsMaxDim = 2;

template <typename T>
void LookUpTableTask(const float *input_addr, const T *indices_addr, float *output_addr, size_t indices_lens,
                     size_t outer_dim_size, T offset, size_t first_dim_size, std::string kernel_name_) {
  auto type_size = sizeof(float);
  size_t lens = outer_dim_size * type_size;
  for (size_t i = 0; i < indices_lens; ++i) {
    T index = indices_addr[i] - offset;
    if (index >= 0 && index < SizeToInt(first_dim_size)) {
      size_t pos = static_cast<size_t>(index) * outer_dim_size;
      auto ret = memcpy_s(output_addr, (indices_lens - i) * lens, input_addr + pos, lens);
      if (ret != EOK) {
        MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memcpy failed. Error no: " << ret;
      }
    } else {
      auto ret = memset_s(output_addr, (indices_lens - i) * lens, 0, lens);
      if (ret != EOK) {
        MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memset failed. Error no: " << ret;
      }
    }
    output_addr += outer_dim_size;
  }
}
}  // namespace

void EmbeddingLookUpCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  node_wpt_ = kernel_node;
  auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  if (input_shape.empty() || input_shape.size() > kEmbeddingLookupInputParamsMaxDim) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input should be 1-"
                      << kEmbeddingLookupInputParamsMaxDim << "D, but got " << input_shape.size() << "D.";
  }
  first_dim_size_ = input_shape[0];
  outer_dim_size_ = 1;
  for (size_t i = 1; i < input_shape.size(); ++i) {
    outer_dim_size_ *= input_shape[i];
  }
  indices_lens_ = 1;
  std::vector<size_t> indices_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
  for (const auto &shape : indices_shape) {
    indices_lens_ *= shape;
  }
  indices_data_type_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 1);
  if (AnfAlgo::HasNodeAttr(kAttrOffset, kernel_node)) {
    offset_ = AnfAlgo::GetNodeAttr<int64_t>(kernel_node, kAttrOffset);
  }
}

template <typename T>
void EmbeddingLookUpCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                               const std::vector<kernel::AddressPtr> &outputs) {
  if (!node_wpt_.expired()) {
    auto node = node_wpt_.lock();
    if (!node) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', node_wpt_(kernel_node) is expired. Error no: " << node;
    }
    std::vector<size_t> input_shape = AnfAlgo::GetPrevNodeOutputInferShape(node, 0);
    if (input_shape.empty()) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', the dimension of input should be at least 1D, but got empty input.";
    }
    first_dim_size_ = input_shape[0];
    outer_dim_size_ = 1;
    for (size_t i = 1; i < input_shape.size(); ++i) {
      outer_dim_size_ *= input_shape[i];
    }

    indices_lens_ = 1;
    std::vector<size_t> indices_shape = AnfAlgo::GetPrevNodeOutputInferShape(node, 1);
    for (const auto &shape : indices_shape) {
      indices_lens_ *= shape;
    }
  }
  const auto *input_addr = reinterpret_cast<float *>(inputs[0]->addr);
  const auto *indices_addr = reinterpret_cast<T *>(inputs[1]->addr);
  auto *output_addr = reinterpret_cast<float *>(outputs[0]->addr);
  auto task = [&](size_t start, size_t end) {
    size_t task_proc_lens = end - start;
    LookUpTableTask<T>(input_addr, indices_addr + start, output_addr + start * outer_dim_size_, task_proc_lens,
                       outer_dim_size_, static_cast<T>(offset_), first_dim_size_, kernel_name_);
  };
  ParallelLaunchAutoSearch(task, indices_lens_, this, &parallel_search_info_);
}

bool EmbeddingLookUpCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                         const std::vector<kernel::AddressPtr> &,
                                         const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kEmbeddingLookupInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kEmbeddingLookupOutputsNum, kernel_name_);
  if (indices_data_type_ == kNumberTypeInt32) {
    LaunchKernel<int>(inputs, outputs);
  } else {
    LaunchKernel<int64_t>(inputs, outputs);
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
