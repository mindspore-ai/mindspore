/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/arrays/extract_image_patches_gpu_kernel.h"

#include <complex>
#include <functional>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"

namespace mindspore {
namespace kernel {
template <typename T>
using Complex = mindspore::utils::Complex<T>;
constexpr size_t extract_image_rank_size = 4;
void ExtractImagePatchesKernelMod::ResetResource() noexcept {
  input_size_ = 1;
  output_size_ = 1;
  ksize_col_ = 1;
  stride_row_ = 1;
  stride_col_ = 1;
  rate_row_ = 1;
  rate_col_ = 1;
  output_rows_ = 1;
  output_cols_ = 1;
  need_batch_ = 1;
  row_stride_ = 1;
  patch_stride_ = 1;
  other_stride_ = 1;
  input_row_size_ = 1;
  input_col_size_ = 1;
  row_padding_top_ = 1;
  col_padding_left_ = 1;
  col_input_stride_ = 1;
  row_input_stride_ = 1;
  patch_input_stride_ = 1;
  output_depth_ = 1;
  is_null_input_ = false;
  input_shape_.clear();
  t_output_shape_.clear();
}

bool ExtractImagePatchesKernelMod::Init(const BaseOperatorPtr &base_operator,
                                        const std::vector<KernelTensorPtr> &inputs,
                                        const std::vector<KernelTensorPtr> &outputs) {
  ResetResource();
  auto kernel_ptr = std::dynamic_pointer_cast<ops::ExtractImagePatches>(base_operator);
  if (kernel_ptr == nullptr) {
    MS_LOG(EXCEPTION) << "cast ExtractImagePatches ops failed!";
  }
  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }
  return true;
}

int ExtractImagePatchesKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                         const std::vector<KernelTensorPtr> &inputs,
                                         const std::vector<KernelTensorPtr> &outputs,
                                         const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  ResetResource();
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  auto kernel_ptr = std::dynamic_pointer_cast<ops::ExtractImagePatches>(base_operator);
  if (kernel_ptr == nullptr) {
    MS_LOG(EXCEPTION) << "cast ExtractImagePatches ops failed!";
  }
  kernel_name_ = kernel_ptr->name();
  auto input_shape = inputs[0]->GetShapeVector();
  auto output_shape = outputs[0]->GetShapeVector();
  is_null_input_ =
    CHECK_SHAPE_NULL(input_shape, kernel_name_, "input") || CHECK_SHAPE_NULL(output_shape, kernel_name_, "output");
  if (is_null_input_) {
    return true;
  }
  (void)std::transform(input_shape.begin(), input_shape.end(), std::back_inserter(input_shape_), LongToSize);
  input_size_ = std::accumulate(input_shape_.begin(), input_shape_.end(), size_t(1), std::multiplies<size_t>());
  output_size_ = static_cast<size_t>(
    std::accumulate(output_shape.begin(), output_shape.end(), int64_t(1), std::multiplies<int64_t>()));
  if (input_shape.size() != extract_image_rank_size || output_shape.size() != extract_image_rank_size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dimension of input and output must be 4, but got the dimension of input: "
                      << input_shape.size() << ", the dimension of output: " << output_shape.size();
  }
  // transposed NHWC shape
  t_output_shape_ = {static_cast<size_t>(output_shape[kIndex0]), static_cast<size_t>(output_shape[kIndex2]),
                     static_cast<size_t>(output_shape[kIndex3]), static_cast<size_t>(output_shape[kIndex1])};

  auto ksizes = kernel_ptr->get_kernel_size();
  auto strides = kernel_ptr->get_strides();
  auto rates = kernel_ptr->get_rates();
  auto padding = kernel_ptr->get_padding();
  if (ksizes.size() != extract_image_rank_size || strides.size() != extract_image_rank_size ||
      rates.size() != extract_image_rank_size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the size of 'ksizes', 'strides' and 'rates' must be 4, but got the size of 'ksizes': "
                      << ksizes.size() << ", the size of 'strides': " << strides.size()
                      << ", the size of 'rates': " << rates.size();
  }

  ksize_row_ = ksizes[kIndex2];
  ksize_col_ = ksizes[kIndex3];
  stride_row_ = strides[kIndex2];
  stride_col_ = strides[kIndex3];
  rate_row_ = rates[kIndex2];
  rate_col_ = rates[kIndex3];

  // transposed NHWC shape
  std::vector<size_t> t_input_shape = {input_shape_[kIndex0], input_shape_[kIndex2], input_shape_[kIndex3],
                                       input_shape_[kIndex1]};
  int64_t input_depth = static_cast<int64_t>(t_input_shape[kIndex3]);
  input_col_size_ = static_cast<int64_t>(t_input_shape[kIndex2]);
  input_row_size_ = static_cast<int64_t>(t_input_shape[kIndex1]);

  int64_t patch_rows_eff = ksize_row_ + (ksize_row_ - 1) * (rate_row_ - 1);
  int64_t patch_cols_eff = ksize_col_ + (ksize_col_ - 1) * (rate_col_ - 1);

  MS_EXCEPTION_IF_ZERO("stride row", stride_row_);
  MS_EXCEPTION_IF_ZERO("stride col", stride_col_);

  if (padding == "VALID") {
    output_rows_ = std::ceil((input_row_size_ - patch_rows_eff + 1.f) / static_cast<float>(stride_row_));
    output_cols_ = std::ceil((input_col_size_ - patch_cols_eff + 1.f) / static_cast<float>(stride_col_));
    constexpr int64_t zero_value = 0;
    row_padding_top_ =
      std::max(zero_value, ((output_rows_ - 1) * stride_row_ + patch_rows_eff - input_row_size_) / kMidDividend);
    col_padding_left_ =
      std::max(zero_value, ((output_cols_ - 1) * stride_col_ + patch_cols_eff - input_col_size_) / kMidDividend);
  } else if (padding == "SAME") {
    output_rows_ = std::ceil(input_row_size_ / static_cast<float>(stride_row_));
    output_cols_ = std::ceil(input_col_size_ / static_cast<float>(stride_col_));
    row_padding_top_ = ((output_rows_ - 1) * stride_row_ + patch_rows_eff - input_row_size_) / kMidDividend;
    col_padding_left_ = ((output_cols_ - 1) * stride_col_ + patch_cols_eff - input_col_size_) / kMidDividend;
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'padding' must be 'VALID' or 'SAME', but got " << padding;
  }

  row_stride_ = ksize_col_;
  patch_stride_ = row_stride_ * ksize_row_ * input_depth;
  other_stride_ = patch_stride_ * output_rows_ * output_cols_;
  col_input_stride_ = input_depth;
  row_input_stride_ = input_depth * input_col_size_;
  patch_input_stride_ = input_depth * input_col_size_ * input_row_size_;
  output_depth_ = input_depth;
  MS_EXCEPTION_IF_ZERO("other stride", other_stride_);
  need_batch_ = (output_size_ - 1) / other_stride_;

  size_t type_size = GetTypeByte(TypeIdToType(inputs[0]->GetDtype()));
  workspace_size_list_.push_back(input_size_ * type_size);
  workspace_size_list_.push_back(output_size_ * type_size);
  workspace_size_list_.push_back(extract_image_rank_size * sizeof(size_t));
  workspace_size_list_.push_back(extract_image_rank_size * sizeof(size_t));
  workspace_size_list_.push_back(extract_image_rank_size * sizeof(size_t));
  workspace_size_list_.push_back(extract_image_rank_size * sizeof(size_t));
  return static_cast<int>(KRET_OK);
}

using KernelRunFunc = ExtractImagePatchesKernelMod::KernelRunFunc;
// int the python api description, input data type is number but CalExtractImagePatchesNHWC only support four type.
const std::vector<std::pair<KernelAttr, KernelRunFunc>> &ExtractImagePatchesKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, KernelRunFunc>> func_list = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &ExtractImagePatchesKernelMod::LaunchKernel<double>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &ExtractImagePatchesKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &ExtractImagePatchesKernelMod::LaunchKernel<half>},
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
     &ExtractImagePatchesKernelMod::LaunchKernel<int8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
     &ExtractImagePatchesKernelMod::LaunchKernel<int16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
     &ExtractImagePatchesKernelMod::LaunchKernel<int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
     &ExtractImagePatchesKernelMod::LaunchKernel<int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
     &ExtractImagePatchesKernelMod::LaunchKernel<uint8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
     &ExtractImagePatchesKernelMod::LaunchKernel<uint16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
     &ExtractImagePatchesKernelMod::LaunchKernel<uint32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
     &ExtractImagePatchesKernelMod::LaunchKernel<uint64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
     &ExtractImagePatchesKernelMod::LaunchKernel<Complex<float>>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
     &ExtractImagePatchesKernelMod::LaunchKernel<Complex<double>>},
    {KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
     &ExtractImagePatchesKernelMod::LaunchKernel<bool>}};
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, ExtractImagePatches, ExtractImagePatchesKernelMod);
}  // namespace kernel
}  // namespace mindspore
