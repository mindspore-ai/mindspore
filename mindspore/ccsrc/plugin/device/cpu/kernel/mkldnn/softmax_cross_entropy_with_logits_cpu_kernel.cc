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

#include "plugin/device/cpu/kernel/mkldnn/softmax_cross_entropy_with_logits_cpu_kernel.h"
#include <numeric>
#include <limits>
#include <functional>
#include "kernel/ops_utils.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kSoftmaxCrossEntropyWithLogitsInputsNum = 2;
constexpr size_t kSoftmaxCrossEntropyWithLogitsInputDim = 2;
constexpr size_t kSoftmaxCrossEntropyWithLogitsOutputsNum = 2;
constexpr size_t kSoftmaxCrossEntropyWithLogitsWorkspaceSize = 1;
}  // namespace

bool SoftmaxCrossEntropyWithLogitsCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                                     const std::vector<KernelTensorPtr> &inputs,
                                                     const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  return true;
}

int SoftmaxCrossEntropyWithLogitsCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                                      const std::vector<KernelTensorPtr> &inputs,
                                                      const std::vector<KernelTensorPtr> &outputs,
                                                      const std::map<uint32_t, tensor::TensorPtr> &) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  auto shape = inputs.at(0)->GetShapeVector();
  dnnl::memory::dims mem_dims;
  (void)mem_dims.insert(mem_dims.end(), shape.begin(), shape.end());
  if (mem_dims.size() != kSoftmaxCrossEntropyWithLogitsInputDim) {
    MS_LOG(ERROR) << "SoftmaxCrossEntropyWithLogits kernel dims invalid " << mem_dims.size();
    return KRET_RESIZE_FAILED;
  }
  batch_size_ = static_cast<size_t>(shape[0]);
  class_num_ = static_cast<size_t>(shape[1]);
  if (batch_size_ == 0 || class_num_ == 0) {
    MS_LOG(EXCEPTION) << "Invalid batch size or class num input!";
  }

  size_t tensor_size = std::accumulate(shape.begin(), shape.end(), sizeof(float), std::multiplies<size_t>());
  (void)workspace_size_list_.emplace_back(tensor_size);
  return KRET_OK;
}

void SoftmaxCrossEntropyWithLogitsCpuKernelMod::ForwardPostExecute(const float *input0, const float *input1,
                                                                   float *output0, float *output1, float *work) {
  float epsilon = std::numeric_limits<float>::min();
  auto task = [this, input0, input1, output0, output1, work, epsilon](size_t start, size_t end) {
    for (size_t batch_index = start; batch_index < end; batch_index++) {
      const float *logits = input0 + batch_index * class_num_;
      const float *labels = input1 + batch_index * class_num_;
      float *backprop = output1 + batch_index * class_num_;
      float *workspace = work + batch_index * class_num_;

      float maxv = logits[0];
      for (size_t i = 0; i < class_num_; i++) {
        maxv = maxv > logits[i] ? maxv : logits[i];
      }

      float sum = 0.0;
      for (size_t i = 0; i < class_num_; i++) {
        backprop[i] = logits[i] - maxv;
        workspace[i] = exp(backprop[i]);
        sum += workspace[i];
      }

      float logit = logf(sum);
      float loss = 0.0;

      for (size_t i = 0; i < class_num_; i++) {
        loss += labels[i] * (backprop[i] - logit);
        workspace[i] = workspace[i] / sum;
        backprop[i] = workspace[i] - labels[i];
      }
      output0[batch_index] = -loss;
    }
  };
  ParallelLaunchAutoSearch(task, batch_size_, this, &parallel_search_info_);
}

bool SoftmaxCrossEntropyWithLogitsCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
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

  const auto *logits = reinterpret_cast<float *>(inputs[0]->addr);
  const auto *labels = reinterpret_cast<float *>(inputs[1]->addr);
  auto *work = reinterpret_cast<float *>(workspace[0]->addr);
  auto *output1 = reinterpret_cast<float *>(outputs[0]->addr);
  auto *output2 = reinterpret_cast<float *>(outputs[1]->addr);
  ForwardPostExecute(logits, labels, output1, output2, work);
  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SoftmaxCrossEntropyWithLogits, SoftmaxCrossEntropyWithLogitsCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
