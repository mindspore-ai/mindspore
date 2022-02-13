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

#include "plugin/device/cpu/kernel/gather_d_grad_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kGatherDGradInputsNum = 2;
constexpr size_t kGatherDGradOutputsNum = 1;

size_t get_element_num(const std::vector<size_t> &shape) {
  size_t size = 1;
  for (size_t i = 0; i < shape.size(); i++) {
    size *= shape[i];
  }
  return size;
}

template <typename I, typename T>
void GatherDGradCopyTask(size_t cur, std::vector<size_t> *pos, T *input, I *index, const int &dim, T *output,
                         const std::vector<size_t> &output_shape, const std::vector<size_t> &out_cargo_size,
                         const std::vector<size_t> &input_cargo_size) {
  for (size_t i = 0; i < output_shape[cur]; ++i) {
    (*pos)[cur] = i;
    if (cur == output_shape.size() - 1) {
      size_t input_offset = 0;
      size_t out_offset = 0;
      // out offset
      for (size_t j = 0; j < output_shape.size(); ++j) {
        out_offset += (*pos)[j] * out_cargo_size[j];
      }
      // input offset
      size_t cur_index = (*pos)[dim];
      (*pos)[dim] = index[out_offset];
      for (size_t j = 0; j < output_shape.size(); ++j) {
        input_offset += (*pos)[j] * input_cargo_size[j];
      }
      // do copy
      input[input_offset] += output[out_offset];
      (*pos)[dim] = cur_index;
    } else {
      // CopyTask
      GatherDGradCopyTask(cur + 1, pos, input, index, dim, output, output_shape, out_cargo_size, input_cargo_size);
    }
  }
}
}  // namespace

template <typename I, typename T>
void GatherDGradCpuKernelMod<I, T>::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  index_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  input_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
  if (input_shape_ != index_shape_) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', shape size of 'x' should be equal to 'index', but got shape size of 'x': "
                      << input_shape_.size() << ", and shape size of 'index': " << index_shape_.size();
  }
  axis_ = AnfAlgo::GetNodeAttr<int64_t>(kernel_node, DIM);
  output_shape_ = AnfAlgo::GetOutputInferShape(kernel_node, 0);
}

template <typename I, typename T>
bool GatherDGradCpuKernelMod<I, T>::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                           const std::vector<kernel::AddressPtr> &,
                                           const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kGatherDGradInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kGatherDGradOutputsNum, kernel_name_);
  size_t input_size = get_element_num(input_shape_) * sizeof(T);
  size_t index_size = get_element_num(index_shape_) * sizeof(I);
  size_t output_size = get_element_num(output_shape_) * sizeof(T);
  if (inputs[0]->size != index_size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the address size of 'x' should be " << index_size
                      << ", but got " << inputs[0]->size << ".";
  }
  if (inputs[1]->size != input_size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the address size of 'dim' should be " << input_size
                      << ", but got " << inputs[1]->size << ".";
  }
  if (outputs[0]->size != output_size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the address size of output should be " << output_size
                      << ", but got " << outputs[0]->size << ".";
  }

  auto *index = reinterpret_cast<I *>(inputs[0]->addr);
  auto *input = reinterpret_cast<T *>(inputs[1]->addr);
  auto out = reinterpret_cast<T *>(outputs[0]->addr);
  int output_rank = SizeToInt(output_shape_.size());
  if (axis_ >= output_rank || axis_ < -output_rank) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the value of 'dim' should be in [" << -output_rank << ", "
                      << output_rank << "), but got: " << axis_;
  }
  if (axis_ < 0) {
    axis_ = axis_ + SizeToInt(output_shape_.size());
  }

  // check index
  index_size = get_element_num(index_shape_);
  int max_index = SizeToInt(output_shape_[axis_]);
  for (size_t i = 0; i < index_size; ++i) {
    if (index[i] >= max_index || index[i] < -max_index) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the value of 'index' should be in [" << -max_index << ", "
                        << max_index << "), but got: " << index[i];
    }
    if (index[i] < 0) {
      index[i] = max_index + index[i];
    }
  }
  auto out_size = get_element_num(output_shape_);
  auto ret = memset_s(out, out_size * sizeof(T), 0x00, out_size * sizeof(T));
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memset failed. Error no: " << ret;
  }

  // out_cargo_size
  std::vector<size_t> out_cargo_size = std::vector<size_t>(output_shape_.size(), 1);
  for (int i = out_cargo_size.size() - 2; i >= 0; --i) {
    out_cargo_size[i] = output_shape_[i + 1] * out_cargo_size[i + 1];
  }
  // input_cargo_size
  std::vector<size_t> input_cargo_size = std::vector<size_t>(input_shape_.size(), 1);
  for (int i = input_cargo_size.size() - 2; i >= 0; --i) {
    input_cargo_size[i] = input_shape_[i + 1] * input_cargo_size[i + 1];
  }

  // copy task
  std::vector<size_t> pos(index_shape_.size(), 0);
  GatherDGradCopyTask<I, T>(0, &pos, out, index, axis_, input, index_shape_, input_cargo_size, out_cargo_size);
  return true;
}
}  // namespace kernel
}  // namespace mindspore
