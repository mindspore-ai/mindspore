/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/nn/batch_norm_grad_grad_gpu_kernel.h"

#include <algorithm>
#include "kernel/ops_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kBatchNormGradGradInputsNum = 8;
constexpr size_t kBatchNormGradGradTrainingWorkSpacesNum = 7;
constexpr size_t kBatchNormGradGradInferenceWorkSpacesNum = 2;
constexpr size_t kBatchNormGradGradOutputsNum = 3;
}  // namespace

bool BatchNormGradGradGpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                         const std::vector<KernelTensorPtr> &inputs,
                                         const std::vector<KernelTensorPtr> &outputs) {
  auto op = std::dynamic_pointer_cast<ops::BatchNormGradGrad>(base_operator);
  kernel_name_ = op->name();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << kernel_name_ << " does not support this kernel data type: " << kernel_attr;
  }
  execute_func_ = func_list_[index].second;
  is_training_ = op->get_is_training();
  epsilon_ = op->get_epsilon();
  format_ = op->get_format() == kOpFormat_NCHW ? DataFormat::NCHW : DataFormat::NHWC;
  return true;
}

bool BatchNormGradGradGpuKernelMod::Launch(const std::vector<AddressPtr> &inputs,
                                           const std::vector<AddressPtr> &workspace,
                                           const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  return execute_func_(this, inputs, workspace, outputs, stream_ptr);
}

template <typename T>
bool BatchNormGradGradGpuKernelMod::Execute(const std::vector<AddressPtr> &inputs,
                                            const std::vector<AddressPtr> &workspace,
                                            const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kBatchNormGradGradInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kBatchNormGradGradOutputsNum, kernel_name_);
  CHECK_KERNEL_WORKSPACE_SIZE(
    workspace.size(), is_training_ ? kBatchNormGradGradTrainingWorkSpacesNum : kBatchNormGradGradInferenceWorkSpacesNum,
    kernel_name_);
  auto x = GetDeviceAddress<T>(inputs, kIndex0);
  auto dy = GetDeviceAddress<T>(inputs, kIndex1);
  auto scale = GetDeviceAddress<float>(inputs, kIndex2);
  auto mean = GetDeviceAddress<float>(inputs, kIndex3);
  auto variance = GetDeviceAddress<float>(inputs, kIndex4);
  auto dout_dx = GetDeviceAddress<T>(inputs, kIndex5);
  auto dout_dscale = GetDeviceAddress<float>(inputs, kIndex6);
  auto dout_dbias = GetDeviceAddress<float>(inputs, kIndex7);

  auto dx = GetDeviceAddress<T>(outputs, kIndex0);
  auto ddy = GetDeviceAddress<T>(outputs, kIndex1);
  auto dscale = GetDeviceAddress<float>(outputs, kIndex2);

  auto inv_std = GetDeviceAddress<float>(workspace, kIndex0);
  auto tmp = GetDeviceAddress<float>(workspace, kIndex1);
  cudaError_t status = cudaErrorNotReady;
  if (is_training_) {
    auto mean_dy = GetDeviceAddress<float>(workspace, kIndex2);
    auto mean_dout_dx = GetDeviceAddress<float>(workspace, kIndex3);
    auto mean_dy_mul_x_hat = GetDeviceAddress<float>(workspace, kIndex4);
    auto mean_dout_dx_mul_x_hat = GetDeviceAddress<float>(workspace, kIndex5);
    auto mean_dy_mul_dout_dx = GetDeviceAddress<float>(workspace, kIndex6);
    status = BatchNormGradGradTraining(dy, x, scale, mean, variance, dout_dx, dout_dscale, dout_dbias, ddy, dx, dscale,
                                       inv_std, tmp, mean_dy, mean_dout_dx, mean_dy_mul_x_hat, mean_dout_dx_mul_x_hat,
                                       mean_dy_mul_dout_dx, shape_info_, format_, epsilon_, device_id_,
                                       reinterpret_cast<cudaStream_t>(stream_ptr));
    CHECK_CUDA_STATUS(status, kernel_name_);
  } else {
    status = BatchNormGradGradInference(dy, x, scale, mean, variance, dout_dx, dout_dscale, dout_dbias, ddy, dx, dscale,
                                        inv_std, tmp, shape_info_, format_, epsilon_, device_id_,
                                        reinterpret_cast<cudaStream_t>(stream_ptr));
    CHECK_CUDA_STATUS(status, kernel_name_);
  }
  return true;
}

int BatchNormGradGradGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                          const std::vector<KernelTensorPtr> &inputs,
                                          const std::vector<KernelTensorPtr> &outputs,
                                          const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kBatchNormGradGradInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kBatchNormGradGradOutputsNum, kernel_name_);
  auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != KRET_OK) {
    return ret;
  }
  auto x_shape = inputs[kIndex0]->GetShapeVector();
  if (x_shape.size() != kDim2 && x_shape.size() != kDim4) {
    MS_EXCEPTION(ValueError) << "For BatchNormGradGrad, x should be a 2-D or 4-D tensor, but got x shape: " << x_shape;
  }
  auto dy_shape = inputs[kIndex1]->GetShapeVector();
  auto scale_shape = inputs[kIndex2]->GetShapeVector();
  auto mean_shape = inputs[kIndex3]->GetShapeVector();
  auto variance_shape = inputs[kIndex4]->GetShapeVector();
  auto dout_dx_shape = inputs[kIndex5]->GetShapeVector();
  auto dout_dscale_shape = inputs[kIndex6]->GetShapeVector();
  auto dout_dbias_shape = inputs[kIndex7]->GetShapeVector();
  auto c = format_ == DataFormat::NCHW ? x_shape[kIndex1] : x_shape[kIndex3];
  ShapeArray shape_array_1{x_shape, dy_shape, dout_dx_shape};
  ShapeArray shape_array_2{std::vector<int64_t>{c}, scale_shape,       mean_shape,
                           variance_shape,          dout_dscale_shape, dout_dbias_shape};
  if (!CheckShapesSame(shape_array_1)) {
    MS_LOG(EXCEPTION)
      << "For BatchNormGradGrad, dy shape and dout_dx shape should be same to x shape, but got x shape: " << x_shape
      << ", dy shape: " << dy_shape << ", dout_dx shape: " << dout_dx_shape;
  }
  if (!CheckShapesSame(shape_array_2)) {
    MS_LOG(EXCEPTION) << "For BatchNormGradGrad, scale shape, mean shape, variance shape, dout_dscale shape and "
                         "dout_dbias shape should be "
                      << std::vector<int64_t>{c} << ", but got scale shape: " << scale_shape
                      << ", mean shape: " << mean_shape << ", variance shape: " << variance_shape
                      << ", dout_dsacle shape: " << dout_dscale_shape << ", dout_dbias shape: " << dout_dbias_shape;
  }

  if (x_shape.size() == kDim2) {
    shape_info_ = ShapeInfo{LongToSize(x_shape[N]), LongToSize(x_shape[C]), 1, 1};
  } else {
    shape_info_ =
      format_ == DataFormat::NCHW
        ? ShapeInfo{LongToSize(x_shape[N]), LongToSize(x_shape[C]), LongToSize(x_shape[H]), LongToSize(x_shape[W])}
        : ShapeInfo{LongToSize(x_shape[N]), LongToSize(x_shape[W]), LongToSize(x_shape[C]), LongToSize(x_shape[H])};
  }
  size_t x_size = shape_info_.n * shape_info_.c * shape_info_.h * shape_info_.w * sizeof(float);
  size_t scale_size = shape_info_.c * sizeof(float);
  workspace_size_list_.clear();
  workspace_size_list_.push_back(scale_size);
  workspace_size_list_.push_back(x_size);
  if (is_training_) {
    const size_t workspace_num = 5;
    workspace_size_list_.insert(workspace_size_list_.end(), workspace_num, scale_size);
  }
  return KRET_OK;
}

std::vector<KernelAttr> BatchNormGradGradGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, ExecuteFunc> &pair) { return pair.first; });
  return support_list;
}

std::vector<std::pair<KernelAttr, BatchNormGradGradGpuKernelMod::ExecuteFunc>>
  BatchNormGradGradGpuKernelMod::func_list_ = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)    // dy
       .AddInputAttr(kNumberTypeFloat32)    // x
       .AddInputAttr(kNumberTypeFloat32)    // scale
       .AddInputAttr(kNumberTypeFloat32)    // save_mean
       .AddInputAttr(kNumberTypeFloat32)    // save_variance
       .AddInputAttr(kNumberTypeFloat32)    // dout_dx
       .AddInputAttr(kNumberTypeFloat32)    // dout_dscale
       .AddInputAttr(kNumberTypeFloat32)    // dout_dbias
       .AddOutputAttr(kNumberTypeFloat32)   // dx
       .AddOutputAttr(kNumberTypeFloat32)   // ddy
       .AddOutputAttr(kNumberTypeFloat32),  // dscale
     &BatchNormGradGradGpuKernelMod::Execute<float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)    // dy
       .AddInputAttr(kNumberTypeFloat16)    // x
       .AddInputAttr(kNumberTypeFloat32)    // scale
       .AddInputAttr(kNumberTypeFloat32)    // save_mean
       .AddInputAttr(kNumberTypeFloat32)    // save_variance
       .AddInputAttr(kNumberTypeFloat16)    // dout_dx
       .AddInputAttr(kNumberTypeFloat32)    // dout_dscale
       .AddInputAttr(kNumberTypeFloat32)    // dout_dbias
       .AddOutputAttr(kNumberTypeFloat16)   // dx
       .AddOutputAttr(kNumberTypeFloat16)   // ddy
       .AddOutputAttr(kNumberTypeFloat32),  // dscale
     &BatchNormGradGradGpuKernelMod::Execute<half>},
};

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, BatchNormGradGrad, BatchNormGradGradGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
