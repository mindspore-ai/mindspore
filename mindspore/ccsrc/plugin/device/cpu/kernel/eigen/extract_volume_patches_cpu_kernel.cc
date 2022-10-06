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

bool ExtractVolumePatchesKernelMod::Init(const BaseOperatorPtr &base_operator,
                                         const std::vector<KernelTensorPtr> &inputs,
                                         const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::ExtractVolumePatches>(base_operator);
  if (kernel_ptr == nullptr) {
    MS_LOG(EXCEPTION) << "cast ExtractVolumePatches ops failed!";
  }
  kernel_name_ = kernel_ptr->name();
  kernel_size_ = kernel_ptr->get_kernel_size();
  strides_ = kernel_ptr->get_strides();
  padding_ = kernel_ptr->get_padding();
  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }
  return true;
}

int ExtractVolumePatchesKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                          const std::vector<KernelTensorPtr> &inputs,
                                          const std::vector<KernelTensorPtr> &outputs,
                                          const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != 0) {
    return ret;
  }
  input_shape_ = inputs[0]->GetShapeVector();
  output_shape_ = outputs[0]->GetShapeVector();
  return static_cast<int>(KRET_OK);
}

template <typename T>
bool ExtractVolumePatchesKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                 const std::vector<kernel::AddressPtr> &workspace,
                                                 const std::vector<kernel::AddressPtr> &outputs) {
  constexpr size_t dims = 5;
  constexpr size_t xn = 0, xc = 1, xd = 2, xh = 3, xw = 4;
  constexpr size_t on = 0, oc = 1, od = 2, oh = 3, ow = 4;
  constexpr size_t kd = 2, kh = 3, kw = 4;
  constexpr size_t sd = 2, sh = 3, sw = 4;
  constexpr int storage_option = static_cast<int>(Eigen::RowMajor);
  constexpr int alignment_type = static_cast<int>(Eigen::Aligned);

  Eigen::TensorMap<Eigen::Tensor<T, dims, storage_option, Eigen::DenseIndex>, alignment_type> eigen_inputs(
    static_cast<T *>(inputs[0]->addr), input_shape_[xn], input_shape_[xc], input_shape_[xd], input_shape_[xh],
    input_shape_[xw]);
  Eigen::TensorMap<Eigen::Tensor<T, dims, storage_option, Eigen::DenseIndex>, alignment_type> eigen_outputs(
    static_cast<T *>(outputs[0]->addr), output_shape_[on], output_shape_[oc], output_shape_[od], output_shape_[oh],
    output_shape_[ow]);
  eigen_outputs.device(Eigen::DefaultDevice()) =
    eigen_inputs.shuffle(Eigen::array<int, dims>{xn, xd, xh, xw, xc})
      .extract_volume_patches(kernel_size_[kw], kernel_size_[kh], kernel_size_[kd], strides_[sw], strides_[sh],
                              strides_[sd], String2EigenPadding(padding_))
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
