/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/mkldnn/sparse_softmax_cross_entropy_with_logits_cpu_kernel.h"
#include <numeric>
#include <limits>
#include <functional>
#include <cmath>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kSparseSoftmaxCrossEntropyWithLogitsInputsNum = 2;
constexpr size_t kSparseSoftmaxCrossEntropyWithLogitsOutputsNum = 1;
constexpr size_t kSparseSoftmaxCrossEntropyWithLogitsWorkspaceSize = 1;
}  // namespace

bool SparseSoftmaxCrossEntropyWithLogitsCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                                           const std::vector<KernelTensorPtr> &inputs,
                                                           const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
    return false;
  }
  auto kernel_ptr = std::dynamic_pointer_cast<ops::SparseSoftmaxCrossEntropyWithLogits>(base_operator);
  MS_ERROR_IF_NULL_W_RET_VAL(kernel_ptr, false);
  is_grad_ = kernel_ptr->get_is_grad();
  return true;
}

int SparseSoftmaxCrossEntropyWithLogitsCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                                            const std::vector<KernelTensorPtr> &inputs,
                                                            const std::vector<KernelTensorPtr> &outputs,
                                                            const std::map<uint32_t, tensor::TensorPtr> &) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  auto logits_shape = inputs.at(kIndex0)->GetDeviceShapeAdaptively();
  auto label_shape = inputs.at(kIndex1)->GetDeviceShapeAdaptively();
  size_t tensor_size =
    std::accumulate(logits_shape.begin(), logits_shape.end(), sizeof(float), std::multiplies<size_t>());
  (void)workspace_size_list_.emplace_back(tensor_size);
  auto logits_dims = logits_shape.size();
  if (logits_dims <= 1) {
    MS_LOG(EXCEPTION) << "Labels shape length must be greater to 1";
  }
  auto labels_dims = label_shape.size();
  if (labels_dims + 1 != logits_dims) {
    MS_LOG(EXCEPTION) << "Labels shape length must be equal to Logits shape length minus 1";
  }
  auto is_same_shape_value = std::equal(label_shape.begin(), label_shape.end(), logits_shape.begin());
  if (!is_same_shape_value) {
    MS_LOG(EXCEPTION) << "Labels shape value must be equal to the Logits except the last dimension of Logits";
  }

  batch_size_ =
    static_cast<size_t>(std::accumulate(label_shape.begin(), label_shape.end(), 1, std::multiplies<size_t>()));
  class_num_ = static_cast<size_t>(logits_shape.back());
  if (batch_size_ == 0 || class_num_ == 0) {
    MS_LOG(EXCEPTION) << "Invalid batch size or class num input!";
  }

  ShapeVector tiled_logits_shape = {static_cast<ShapeValueDType>(batch_size_),
                                    static_cast<ShapeValueDType>(class_num_)};
  dnnl::memory::dims mem_dims;
  (void)mem_dims.insert(mem_dims.end(), tiled_logits_shape.begin(), tiled_logits_shape.end());
  if (mem_dims.size() != 2) {
    MS_LOG(EXCEPTION) << "SparseSoftmaxCrossEntropyWithLogits kernel dims invalid " << mem_dims.size();
  }
  auto mem_desc = CreateDesc<dnnl::memory::desc>(mem_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nc);
  auto desc = CreateDesc<dnnl::softmax_forward::desc>(dnnl::prop_kind::forward_training, mem_desc, 1);
  auto prim_desc = CreateDesc<dnnl::softmax_forward::primitive_desc>(desc, engine_);
  primitive_ = CreatePrimitive<dnnl::softmax_forward>(prim_desc);

  AddArgument(DNNL_ARG_SRC, mem_desc);
  AddArgument(DNNL_ARG_DST, mem_desc);
  return KRET_OK;
}

void SparseSoftmaxCrossEntropyWithLogitsCpuKernelMod::ForwardPostExecute(const int *labels, const float *losses,
                                                                         float *output) const {
  float total_loss = 0;
  float epsilon = std::numeric_limits<float>::min();
  for (size_t i = 0; i < batch_size_; ++i) {
    if (labels[i] < 0) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "' got error label! label[index] must >= 0, but got  index = " << i
                        << " label[index] = " << labels[i];
    }
    size_t label = IntToSize(labels[i]);
    if (label > class_num_) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "' got error label! label[index] must <= class_num, but got index = " << i
                        << " label[index] = " << label << " and class_num = " << class_num_;
    }
    total_loss -= logf(losses[i * class_num_ + label] <= 0.0 ? epsilon : losses[i * class_num_ + label]);
  }
  output[0] = total_loss / batch_size_;
}

void SparseSoftmaxCrossEntropyWithLogitsCpuKernelMod::GradPostExecute(const int *labels, const float *losses,
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

bool SparseSoftmaxCrossEntropyWithLogitsCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
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

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SparseSoftmaxCrossEntropyWithLogits,
                      SparseSoftmaxCrossEntropyWithLogitsCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
