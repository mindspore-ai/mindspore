/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/cpu/mkldnn/sparse_softmax_cross_entropy_with_logits_cpu_kernel.h"
#include <numeric>
#include <limits>
#include <functional>
#include <cmath>
#include "backend/kernel_compiler/cpu/mkldnn/mkl_kernel_engine.h"
#include "runtime/device/cpu/cpu_device_address.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kSparseSoftmaxCrossEntropyWithLogitsInputsNum = 2;
constexpr size_t kSparseSoftmaxCrossEntropyWithLogitsOutputsNum = 1;
constexpr size_t kSparseSoftmaxCrossEntropyWithLogitsWorkspaceSize = 1;
}  // namespace

void SparseSoftmaxCrossEntropyWithLogitsCPUKernel::InitInputOutputSize(const CNodePtr &kernel_node) {
  CPUKernel::InitInputOutputSize(kernel_node);
  MS_EXCEPTION_IF_NULL(kernel_node);
  size_t type_size = sizeof(float);
  std::vector<size_t> shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  size_t tensor_size = std::accumulate(shape.begin(), shape.end(), type_size, std::multiplies<size_t>());
  (void)workspace_size_list_.emplace_back(tensor_size);
}

void SparseSoftmaxCrossEntropyWithLogitsCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  std::vector<size_t> shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  std::vector<size_t> label_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
  if (label_shape.size() > 1) {
    MS_LOG(EXCEPTION) << "Labels shape length should be equal to Logits shape length minus 1";
  }
  dnnl::memory::dims mem_dims;
  (void)mem_dims.insert(mem_dims.end(), shape.begin(), shape.end());
  if (mem_dims.size() != 2) {
    MS_LOG(EXCEPTION) << "SparseSoftmaxCrossEntropyWithLogits kernel dims invalid " << mem_dims.size();
  }
  batch_size_ = shape[0];
  class_num_ = shape[1];
  if (batch_size_ == 0 || class_num_ == 0) {
    MS_LOG(EXCEPTION) << "Invalid batch size or class num input!";
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
  float epsilon = std::numeric_limits<float>::min();
  for (size_t i = 0; i < batch_size_; ++i) {
    if (labels[i] < 0) {
      MS_LOG(EXCEPTION) << "Label value must >= 0";
    }
    size_t label = IntToSize(labels[i]);
    if (label > class_num_) {
      MS_LOG(EXCEPTION) << "Error label input!";
    }
    total_loss -= logf(losses[i * class_num_ + label] <= 0.0 ? epsilon : losses[i * class_num_ + label]);
  }
  output[0] = total_loss / batch_size_;
}

void SparseSoftmaxCrossEntropyWithLogitsCPUKernel::GradPostExecute(const int *labels, const float *losses,
                                                                   float *output) const {
  size_t row_start = 0;
  for (size_t i = 0; i < batch_size_; ++i) {
    if (labels[i] < 0) {
      MS_LOG(EXCEPTION) << "Label value must >= 0";
    }
    size_t label = IntToSize(labels[i]);
    if (label > class_num_) {
      MS_LOG(EXCEPTION) << "Error label input!";
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
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSparseSoftmaxCrossEntropyWithLogitsInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSparseSoftmaxCrossEntropyWithLogitsOutputsNum, kernel_name_);
  CHECK_KERNEL_WORKSPACE_SIZE(workspace.size(), kSparseSoftmaxCrossEntropyWithLogitsWorkspaceSize, kernel_name_);
  size_t batch_float_size = batch_size_ * sizeof(float);
  size_t batch_class_float_size = class_num_ * batch_float_size;
  if (inputs[0]->size != workspace[0]->size || inputs[0]->size != batch_class_float_size ||
      inputs[1]->size != batch_float_size) {
    MS_LOG(EXCEPTION) << "Error input data size!";
  }
  if (is_grad_ && outputs[0]->size != batch_class_float_size) {
    MS_LOG(EXCEPTION) << "Error output data size!";
  } else if (!is_grad_ && outputs[0]->size != sizeof(float)) {
    MS_LOG(EXCEPTION) << "Error output data size!";
  }
  SetArgumentHandle(DNNL_ARG_SRC, inputs[0]->addr);
  SetArgumentHandle(DNNL_ARG_DST, workspace[0]->addr);
  ExecutePrimitive();
  const auto *labels = reinterpret_cast<int *>(inputs[1]->addr);
  const auto *losses = reinterpret_cast<float *>(workspace[0]->addr);
  auto *output = reinterpret_cast<float *>(outputs[0]->addr);
  if (is_grad_) {
    GradPostExecute(labels, losses, output);
  } else {
    ForwardPostExecute(labels, losses, output);
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
