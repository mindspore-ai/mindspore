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

#include "backend/kernel_compiler/cpu/mkldnn/softmax_cross_entropy_with_logits_cpu_kernel.h"
#include <numeric>
#include <limits>
#include <functional>
#include "backend/kernel_compiler/cpu/mkldnn/mkl_kernel_engine.h"
#include "runtime/device/cpu/cpu_device_address.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kSoftmaxCrossEntropyWithLogitsInputsNum = 2;
constexpr size_t kSoftmaxCrossEntropyWithLogitsOutputsNum = 2;
constexpr size_t kSoftmaxCrossEntropyWithLogitsWorkspaceSize = 1;
}  // namespace

void SoftmaxCrossEntropyWithLogitsCPUKernel::InitInputOutputSize(const CNodePtr &kernel_node) {
  CPUKernel::InitInputOutputSize(kernel_node);
  MS_EXCEPTION_IF_NULL(kernel_node);
  size_t type_size = sizeof(float);
  std::vector<size_t> shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  size_t tensor_size = std::accumulate(shape.begin(), shape.end(), type_size, std::multiplies<size_t>());
  (void)workspace_size_list_.emplace_back(tensor_size);
}

void SoftmaxCrossEntropyWithLogitsCPUKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  std::vector<size_t> shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  dnnl::memory::dims mem_dims;
  (void)mem_dims.insert(mem_dims.end(), shape.begin(), shape.end());
  if (mem_dims.size() != 2) {
    MS_LOG(EXCEPTION) << "SoftmaxCrossEntropyWithLogits kernel dims invalid " << mem_dims.size();
  }
  batch_size_ = shape[0];
  class_num_ = shape[1];
  if (batch_size_ == 0 || class_num_ == 0) {
    MS_LOG(EXCEPTION) << "Invalid batch size or class num input!";
  }
  dnnl::memory::desc mem_desc(mem_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nc);

  dnnl::softmax_forward::desc desc = dnnl::softmax_forward::desc(dnnl::prop_kind::forward_training, mem_desc, 1);
  auto prim_desc = dnnl::softmax_forward::primitive_desc(desc, MKLKernelEngine::Get().engine());
  primitive_ = std::make_shared<dnnl::softmax_forward>(prim_desc);

  AddArgument(DNNL_ARG_SRC, mem_desc);
  AddArgument(DNNL_ARG_DST, mem_desc);
}

void SoftmaxCrossEntropyWithLogitsCPUKernel::ForwardPostExecute(const float *logits, const float *labels,
                                                                float *output1, float *output2) const {
  float epsilon = std::numeric_limits<float>::min();
  for (size_t i = 0; i < batch_size_; ++i) {
    output1[i] = 0;
    float loss = 0.0;
    for (size_t j = 0; j < class_num_; ++j) {
      float logit = logf(logits[i * class_num_ + j] <= 0.0 ? epsilon : logits[i * class_num_ + j]);
      output2[i * class_num_ + j] = logits[i * class_num_ + j] - labels[i * class_num_ + j];
      loss += labels[i * class_num_ + j] * logit;
    }
    output1[i] = -loss;
  }
}

bool SoftmaxCrossEntropyWithLogitsCPUKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                                    const std::vector<kernel::AddressPtr> &workspace,
                                                    const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSoftmaxCrossEntropyWithLogitsInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSoftmaxCrossEntropyWithLogitsOutputsNum, kernel_name_);
  CHECK_KERNEL_WORKSPACE_SIZE(workspace.size(), kSoftmaxCrossEntropyWithLogitsWorkspaceSize, kernel_name_);

  size_t batch_float_size = batch_size_ * sizeof(float);
  size_t batch_class_float_size = class_num_ * batch_float_size;
  if (inputs[0]->size != workspace[0]->size || inputs[0]->size != batch_class_float_size ||
      inputs[1]->size != batch_class_float_size) {
    MS_LOG(EXCEPTION) << "Error input data size!";
  }
  if (outputs[1]->size != batch_class_float_size || outputs[0]->size != batch_float_size) {
    MS_LOG(EXCEPTION) << "Error output data size!";
  }
  SetArgumentHandle(DNNL_ARG_SRC, inputs[0]->addr);
  SetArgumentHandle(DNNL_ARG_DST, workspace[0]->addr);
  ExecutePrimitive();
  const auto *labels = reinterpret_cast<float *>(inputs[1]->addr);
  const auto *logits = reinterpret_cast<float *>(workspace[0]->addr);
  auto *output1 = reinterpret_cast<float *>(outputs[0]->addr);
  auto *output2 = reinterpret_cast<float *>(outputs[1]->addr);
  ForwardPostExecute(logits, labels, output1, output2);
  return true;
}
}  // namespace kernel
}  // namespace mindspore
