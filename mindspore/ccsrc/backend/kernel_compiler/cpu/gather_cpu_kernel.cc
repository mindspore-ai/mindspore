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

#include "backend/kernel_compiler/cpu/gather_cpu_kernel.h"
#include "runtime/device/cpu/cpu_device_address.h"
#include "nnacl/gather_parameter.h"
#include "nnacl/base/gather_base.h"
#include "common/thread_pool.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kGatherInputsNum = 2;
constexpr size_t kGatherOutputsNum = 1;
constexpr size_t kGatherInputParamsMaxDim = 4;
}  // namespace

template <typename T>
void GatherV2CPUKernel<T>::CheckParam(const CNodePtr &kernel_node) {
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_num == kGatherInputsNum + 1) {
    is_dynamic_shape_ = true;
    MS_LOG(DEBUG) << " GatherV2CPUKernel running in Dynamic Mode.";
  } else if (input_num == kGatherInputsNum) {
    MS_LOG(DEBUG) << " GatherV2CPUKernel running in Normal Mode.";
  } else {
    MS_LOG(EXCEPTION) << "Argument number is " << input_num << ", but GatherV2CPUKernel needs 2.";
  }
}

template <typename T>
void GatherV2CPUKernel<T>::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  CheckParam(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  input_shape_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  indices_shape_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
  output_shape_ = AnfAlgo::GetOutputInferShape(kernel_node, 0);
  if (input_shape_.size() > kGatherInputParamsMaxDim) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of 'input_params' should be "
                      << kGatherInputParamsMaxDim << "D or lower, but got " << input_shape_.size() << ".";
  }
  if (!is_dynamic_shape_) {
    axis_ = AnfAlgo::GetNodeAttr<int64_t>(kernel_node, AXIS);
  }
}

template <typename T>
bool GatherV2CPUKernel<T>::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                  const std::vector<kernel::AddressPtr> &,
                                  const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kGatherOutputsNum, kernel_name_);
  const auto *input_tensor = reinterpret_cast<int8_t *>(inputs[0]->addr);
  const auto *indices_data = reinterpret_cast<int32_t *>(inputs[1]->addr);
  auto *output_addr = reinterpret_cast<int8_t *>(outputs[0]->addr);
  if (is_dynamic_shape_) {
    axis_ = reinterpret_cast<int64_t *>(inputs[2]->addr)[0];
  }

  int dims = SizeToInt(input_shape_.size());
  if (axis_ < -dims || axis_ >= dims) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the 'axis' should be in the range [-" << dims << ", " << dims
                  << "), but got " << axis_ << ".";
    return false;
  } else if (axis_ < 0) {
    axis_ = axis_ + dims;
  }

  size_t outer_size = 1, inner_size = 1;
  auto axis = static_cast<size_t>(axis_);
  for (size_t i = 0; i < axis; ++i) {
    outer_size *= input_shape_.at(i);
  }
  for (size_t i = axis + 1; i < input_shape_.size(); ++i) {
    inner_size *= input_shape_.at(i);
  }
  size_t indices_element_size = 1;
  for (size_t i = 0; i < indices_shape_.size(); i++) {
    indices_element_size *= indices_shape_.at(i);
  }
  auto limit = input_shape_.at(axis);
  auto task = [&](size_t start, size_t end) {
    int count = SizeToInt(end - start);
    const int8_t *in = input_tensor + start * limit * inner_size * sizeof(T);
    int8_t *out = output_addr + start * indices_element_size * inner_size * sizeof(T);
    int ret = Gather(in, count, inner_size, limit, indices_data, indices_element_size, out, sizeof(T));
    if (ret != 0) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', error_code[" << ret << "]";
    }
  };
  ParallelLaunchAutoSearch(task, outer_size, this, &parallel_search_info_);
  return true;
}
}  // namespace kernel
}  // namespace mindspore
