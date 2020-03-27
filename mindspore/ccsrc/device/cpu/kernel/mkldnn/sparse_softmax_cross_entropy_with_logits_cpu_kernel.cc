/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "device/cpu/kernel/mkldnn/sparse_softmax_cross_entropy_with_logits_cpu_kernel.h"
#include <numeric>
#include <functional>
#include <cmath>
#include "device/cpu/kernel/mkldnn/mkl_kernel_engine.h"
#include "device/cpu/cpu_device_address.h"
#include "common/utils.h"

namespace mindspore {
namespace device {
namespace cpu {
void SparseSoftmaxCrossEntropyWithLogitsCPUKernel::InitInputOutputSize(const CNodePtr &kernel_node) {
  CPUKernel::InitInputOutputSize(kernel_node);
  MS_EXCEPTION_IF_NULL(kernel_node);
  size_t type_size = sizeof(float);
  std::vector<size_t> shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  size_t tensor_size = std::accumulate(shape.begin(), shape.end(), type_size, std::multiplies<size_t>());
  workspace_size_list_.emplace_back(tensor_size);
}

void SparseSoftmaxCrossEntropyWithLogitsCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  std::vector<size_t> shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  dnnl::memory::dims mem_dims;
  mem_dims.insert(mem_dims.end(), shape.begin(), shape.end());
  if (mem_dims.size() != 2) {
    MS_LOG(EXCEPTION) << "SparseSoftmaxCrossEntropyWithLogits kernel dims invalid " << mem_dims.size();
  }
  batch_size_ = shape[0];
  class_num_ = shape[1];
  if (batch_size_ == 0 || class_num_ == 0) {
    MS_LOG(EXCEPTION) << "invalid batch size or class num input!";
  }
  is_grad_ = AnfAlgo::GetNodeAttr<bool>(kernel_node, IS_GRAD);
  dnnl::memory::desc mem_desc(mem_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nc);

  dnnl::softmax_forward::desc desc = dnnl::softmax_forward::desc(dnnl::prop_kind::forward_training, mem_desc, 1);
  auto prim_desc = dnnl::softmax_forward::primitive_desc(desc, MKLKernelEngine::Get().engine());
  primitive_ = std::make_shared<dnnl::softmax_forward>(prim_desc);

  AddArgument(DNNL_ARG_SRC, mem_desc);
  AddArgument(DNNL_ARG_DST, mem_desc);
}

void SparseSoftmaxCrossEntropyWithLogitsCPUKernel::ForwardPostExecute(const int *labels, const float *losses,
                                                                      float *output) const {
  float total_loss = 0;
  for (size_t i = 0; i < batch_size_; ++i) {
    if (labels[i] < 0) {
      MS_LOG(EXCEPTION) << "label value must >= 0";
    }
    size_t label = IntToSize(labels[i]);
    if (label > class_num_) {
      MS_LOG(EXCEPTION) << "error label input!";
    }
    total_loss -= logf(losses[i * class_num_ + label]);
  }
  output[0] = total_loss / batch_size_;
}

void SparseSoftmaxCrossEntropyWithLogitsCPUKernel::GradPostExecute(const int *labels, const float *losses,
                                                                   float *output) const {
  size_t row_start = 0;
  for (size_t i = 0; i < batch_size_; ++i) {
    if (labels[i] < 0) {
      MS_LOG(EXCEPTION) << "label value must >= 0";
    }
    size_t label = IntToSize(labels[i]);
    if (label > class_num_) {
      MS_LOG(EXCEPTION) << "error label input!";
    }
    for (size_t j = 0; j < class_num_; ++j) {
      size_t index = row_start + j;
      if (j == label) {
        output[index] = (losses[index] - 1) / batch_size_;
      } else {
        output[index] = losses[index] / batch_size_;
      }
    }
    row_start += class_num_;
  }
}

bool SparseSoftmaxCrossEntropyWithLogitsCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                                          const std::vector<kernel::AddressPtr> &workspace,
                                                          const std::vector<kernel::AddressPtr> &outputs) {
  if (inputs.empty() || workspace.empty() || outputs.empty()) {
    MS_LOG(EXCEPTION) << "error input output size!";
  }
  size_t batch_float_size = batch_size_ * sizeof(float);
  size_t batch_class_float_size = class_num_ * batch_float_size;
  if (inputs[0]->size != workspace[0]->size || inputs[0]->size != batch_class_float_size ||
      inputs[1]->size != batch_float_size) {
    MS_LOG(EXCEPTION) << "error input data size!";
  }
  if (is_grad_ && outputs[0]->size != batch_class_float_size) {
    MS_LOG(EXCEPTION) << "error output data size!";
  } else if (!is_grad_ && outputs[0]->size != sizeof(float)) {
    MS_LOG(EXCEPTION) << "error output data size!";
  }
  SetArgumentHandle(DNNL_ARG_SRC, inputs[0]->addr);
  SetArgumentHandle(DNNL_ARG_DST, workspace[0]->addr);
  ExecutePrimitive();
  auto labels = reinterpret_cast<int *>(inputs[1]->addr);
  auto losses = reinterpret_cast<float *>(workspace[0]->addr);
  auto output = reinterpret_cast<float *>(outputs[0]->addr);
  if (is_grad_) {
    GradPostExecute(labels, losses, output);
  } else {
    ForwardPostExecute(labels, losses, output);
  }
  return true;
}
}  // namespace cpu
}  // namespace device
}  // namespace mindspore
