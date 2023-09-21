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

#include "plugin/device/cpu/kernel/eigen/extract_volume_patches_cpu_kernel.h"
#include <functional>
#include <memory>
#include <string>
#include "mindapi/base/type_id.h"
#include "ops/extract_volume_patches.h"
#include "Eigen/Core"
#include "unsupported/Eigen/CXX11/Tensor"
#include "unsupported/Eigen/CXX11/src/Tensor/TensorDeviceDefault.h"

namespace mindspore {
namespace kernel {
namespace {
using KernelRunFunc = ExtractVolumePatchesKernelMod::KernelRunFunc;

Eigen::PaddingType String2EigenPadding(const std::string &padding) {
  if (padding == "VALID") {
    return Eigen::PADDING_VALID;
  } else if (padding == "SAME") {
    return Eigen::PADDING_SAME;
  }
  return Eigen::PADDING_SAME;
}
}  // namespace

bool ExtractVolumePatchesKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                         const std::vector<KernelTensor *> &outputs) {
  kernel_size_ = GetValue<std::vector<int64_t>>(primitive_->GetAttr(ops::kKernelSize));
  strides_ = GetValue<std::vector<int64_t>>(primitive_->GetAttr(ops::kStrides));
  padding_ = GetValue<int>(primitive_->GetAttr(ops::kPadding));

  if (!MatchKernelFunc(kernel_name_, inputs, outputs)) {
    return false;
  }
  return true;
}

int ExtractVolumePatchesKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                          const std::vector<KernelTensor *> &outputs) {
  constexpr size_t x_dim_num = 5;
  constexpr size_t out_dim_num = 5;
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(EXCEPTION) << "Get empty inputs or outputs, inputs size: " << inputs.size()
                      << ", outputs size: " << outputs.size();
  }

  int ret = KernelMod::Resize(inputs, outputs);
  if (ret != 0) {
    return ret;
  }
  input_shape_ = inputs[0]->GetShapeVector();
  if (input_shape_.size() != x_dim_num) {
    MS_LOG(EXCEPTION) << "Incorrect input dim size: " << input_shape_.size() << ", which should be " << x_dim_num;
  }

  output_shape_ = outputs[0]->GetShapeVector();
  if (output_shape_.size() != out_dim_num) {
    MS_LOG(EXCEPTION) << "Incorrect output dim size: " << output_shape_.size() << ", which should be " << out_dim_num;
  }

  return static_cast<int>(KRET_OK);
}

template <typename T>
bool ExtractVolumePatchesKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                                 const std::vector<kernel::KernelTensor *> &workspace,
                                                 const std::vector<kernel::KernelTensor *> &outputs) {
  constexpr size_t dims = 5;
  constexpr size_t x_dim_num = 5;
  constexpr size_t out_dim_num = 5;
  constexpr size_t extract_dims = 6;
  constexpr size_t xn = 0, xc = 1, xd = 2, xh = 3, xw = 4;
  constexpr size_t on = 0, oc = 1, od = 2, oh = 3, ow = 4;
  constexpr size_t kd = 2, kh = 3, kw = 4;
  constexpr size_t sd = 2, sh = 3, sw = 4;
  constexpr int storage_option = static_cast<int>(Eigen::RowMajor);
  constexpr int alignment_type = static_cast<int>(Eigen::Aligned);

  if (input_shape_.size() != x_dim_num) {
    MS_LOG(EXCEPTION) << "For ExtractVolumePatches, incorrect input dim size: " << input_shape_.size()
                      << ", which should be " << x_dim_num;
  }
  if (output_shape_.size() != out_dim_num) {
    MS_LOG(EXCEPTION) << "For ExtractVolumePatches, incorrect output dim size: " << output_shape_.size()
                      << ", which should be " << out_dim_num;
  }
  if (kernel_size_.size() != dims) {
    MS_LOG(EXCEPTION) << "For ExtractVolumePatches, incorrect kernel_size_ dim size: " << kernel_size_.size()
                      << ", which should be " << dims;
  }
  if (strides_.size() != dims) {
    MS_LOG(EXCEPTION) << "For ExtractVolumePatches, incorrect strides_ dim size: " << strides_.size()
                      << ", which should be " << dims;
  }

  Eigen::TensorMap<Eigen::Tensor<T, dims, storage_option, Eigen::DenseIndex>, alignment_type> eigen_inputs(
    static_cast<T *>(inputs[0]->device_ptr()), input_shape_[xn], input_shape_[xc], input_shape_[xd], input_shape_[xh],
    input_shape_[xw]);
  Eigen::TensorMap<Eigen::Tensor<T, dims, storage_option, Eigen::DenseIndex>, alignment_type> eigen_outputs(
    static_cast<T *>(outputs[0]->device_ptr()), output_shape_[on], output_shape_[oc], output_shape_[od],
    output_shape_[oh], output_shape_[ow]);
  Eigen::Tensor<T, extract_dims, storage_option, Eigen::DenseIndex> extract_tensor =
    eigen_inputs.shuffle(Eigen::array<int, dims>{xn, xd, xh, xw, xc})
      .extract_volume_patches(kernel_size_[kw], kernel_size_[kh], kernel_size_[kd], strides_[sw], strides_[sh],
                              strides_[sd], String2EigenPadding(padding_));
  const int64_t output_size =
    std::accumulate(output_shape_.begin(), output_shape_.end(), static_cast<int64_t>(1), std::multiplies<>());
  const auto &extract_shape = extract_tensor.dimensions();
  const int64_t extract_size =
    std::accumulate(extract_shape.begin(), extract_shape.end(), static_cast<int64_t>(1), std::multiplies<>());
  if (extract_size != output_size) {
    MS_LOG(EXCEPTION) << "Incorrect output shape " << output_shape_ << " for ExtractVolumePatch. Input shape "
                      << input_shape_;
  }
  eigen_outputs.device(Eigen::DefaultDevice()) =
    extract_tensor
      .reshape(Eigen::array<int64_t, dims>{output_shape_[on], output_shape_[od], output_shape_[oh], output_shape_[ow],
                                           output_shape_[oc]})
      .shuffle(Eigen::array<int, dims>{on, ow, oc, od, oh});
  return true;
}

const std::vector<std::pair<KernelAttr, KernelRunFunc>> &ExtractVolumePatchesKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, KernelRunFunc>> func_list = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &ExtractVolumePatchesKernelMod::LaunchKernel<float16>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &ExtractVolumePatchesKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &ExtractVolumePatchesKernelMod::LaunchKernel<double>},
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
     &ExtractVolumePatchesKernelMod::LaunchKernel<int8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
     &ExtractVolumePatchesKernelMod::LaunchKernel<int16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
     &ExtractVolumePatchesKernelMod::LaunchKernel<int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
     &ExtractVolumePatchesKernelMod::LaunchKernel<int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
     &ExtractVolumePatchesKernelMod::LaunchKernel<uint8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
     &ExtractVolumePatchesKernelMod::LaunchKernel<uint16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
     &ExtractVolumePatchesKernelMod::LaunchKernel<uint32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
     &ExtractVolumePatchesKernelMod::LaunchKernel<uint64_t>},
  };
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ExtractVolumePatches, ExtractVolumePatchesKernelMod);
}  // namespace kernel
}  // namespace mindspore
