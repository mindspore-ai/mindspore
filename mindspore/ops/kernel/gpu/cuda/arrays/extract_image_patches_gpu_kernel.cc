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

#include "kernel/gpu/cuda/arrays/extract_image_patches_gpu_kernel.h"

#include <complex>
#include <functional>
#include "kernel/gpu/cuda_impl/cuda_ops/complex.h"

namespace mindspore {
namespace kernel {
template <typename T>
using Complex = mindspore::utils::Complex<T>;
constexpr size_t extract_image_rank_size = 2;
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

bool ExtractImagePatchesKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &outputs) {
  ResetResource();

  if (!MatchKernelFunc(kernel_name_, inputs, outputs)) {
    return false;
  }
  return true;
}

int ExtractImagePatchesKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                         const std::vector<KernelTensor *> &outputs) {
  if (int ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  ResetResource();
  input_shape_ = inputs[0]->GetShapeVector();
  auto output_shape = outputs[0]->GetShapeVector();
  is_null_input_ =
    CHECK_SHAPE_NULL(input_shape_, kernel_name_, "input") || CHECK_SHAPE_NULL(output_shape, kernel_name_, "output");
  if (is_null_input_) {
    return true;
  }

  input_size_ = SizeOf(input_shape_);
  output_size_ = SizeOf(output_shape);

  // transposed NHWC shape
  t_output_shape_ = {output_shape[kIndex0], output_shape[kIndex2], output_shape[kIndex3], output_shape[kIndex1]};
  // transposed NHWC shape
  std::vector<int64_t> t_input_shape = {input_shape_[kIndex0], input_shape_[kIndex2], input_shape_[kIndex3],
                                        input_shape_[kIndex1]};
  int64_t input_depth = t_input_shape[kIndex3];
  input_col_size_ = t_input_shape[kIndex2];
  input_row_size_ = t_input_shape[kIndex1];

  // get attr
  auto ksizes = inputs[kIndex1]->GetValueWithCheck<std::vector<int64_t>>();
  auto strides = inputs[kIndex2]->GetValueWithCheck<std::vector<int64_t>>();
  auto rates = inputs[kIndex3]->GetValueWithCheck<std::vector<int64_t>>();
  mindspore::PadMode padding = static_cast<mindspore::PadMode>(inputs[kIndex4]->GetValueWithCheck<int64_t>());

  // After arg_handle: to_kernel_sizes, the ksizes tuple will return (input[2], input[3]).
  // strides and rates are the same as ksizes.
  ksize_row_ = ksizes[kIndex0];
  ksize_col_ = ksizes[kIndex1];
  stride_row_ = strides[kIndex0];
  stride_col_ = strides[kIndex1];
  rate_row_ = rates[kIndex0];
  rate_col_ = rates[kIndex1];
  MS_EXCEPTION_IF_ZERO("stride row", stride_row_);
  MS_EXCEPTION_IF_ZERO("stride col", stride_col_);
  patch_rows_eff_ = ksize_row_ + (ksize_row_ - 1) * (rate_row_ - 1);
  patch_cols_eff_ = ksize_col_ + (ksize_col_ - 1) * (rate_col_ - 1);
  if (ksizes.size() != extract_image_rank_size || strides.size() != extract_image_rank_size ||
      rates.size() != extract_image_rank_size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the size of 'ksizes', 'strides' and 'rates' must be 2, but got the size of 'ksizes': "
                      << ksizes.size() << ", the size of 'strides': " << strides.size()
                      << ", the size of 'rates': " << rates.size();
  }
  if (padding == PadMode::VALID) {
    output_rows_ = std::ceil((input_row_size_ - patch_rows_eff_ + 1.f) / static_cast<float>(stride_row_));
    output_cols_ = std::ceil((input_col_size_ - patch_cols_eff_ + 1.f) / static_cast<float>(stride_col_));
  } else if (padding == PadMode::SAME) {
    output_rows_ = std::ceil(input_row_size_ / static_cast<float>(stride_row_));
    output_cols_ = std::ceil(input_col_size_ / static_cast<float>(stride_col_));
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the ' padding' must be 'VALID' or 'SAME', but got " << padding;
  }

  constexpr int64_t zero_value = 0;
  row_padding_top_ =
    std::max(zero_value, ((output_rows_ - 1) * stride_row_ + patch_rows_eff_ - input_row_size_) / kMidDividend);
  col_padding_left_ =
    std::max(zero_value, ((output_cols_ - 1) * stride_col_ + patch_cols_eff_ - input_col_size_) / kMidDividend);

  row_stride_ = ksize_col_;
  patch_stride_ = row_stride_ * ksize_row_ * input_depth;
  other_stride_ = patch_stride_ * output_rows_ * output_cols_;
  col_input_stride_ = input_depth;
  row_input_stride_ = input_depth * input_col_size_;
  patch_input_stride_ = input_depth * input_col_size_ * input_row_size_;
  output_depth_ = input_depth;
  MS_EXCEPTION_IF_ZERO("other stride", other_stride_);
  need_batch_ = (output_size_ - 1) / other_stride_;

  size_t type_size = GetTypeByte(TypeIdToType(inputs[0]->dtype_id()));
  workspace_size_list_.push_back(input_size_ * type_size);
  workspace_size_list_.push_back(output_size_ * type_size);
  return static_cast<int>(KRET_OK);
}

#define EXTRACT_IMAGE_PATCHES_REGISTER(INPUTX, T)      \
  KernelAttr()                                         \
    .AddInputAttr(INPUTX)                              \
    .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)  \
    .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)  \
    .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)  \
    .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64) \
    .AddOutputAttr(INPUTX),                            \
    &ExtractImagePatchesKernelMod::LaunchKernel<T>

using KernelRunFunc = ExtractImagePatchesKernelMod::KernelRunFunc;
// int the python api description, input data type is number but CalExtractImagePatchesNHWC only support four type.
const std::vector<std::pair<KernelAttr, KernelRunFunc>> &ExtractImagePatchesKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, KernelRunFunc>> func_list = {
    {EXTRACT_IMAGE_PATCHES_REGISTER(kNumberTypeBool, bool)},
    {EXTRACT_IMAGE_PATCHES_REGISTER(kNumberTypeFloat16, half)},
    {EXTRACT_IMAGE_PATCHES_REGISTER(kNumberTypeFloat32, float)},
    {EXTRACT_IMAGE_PATCHES_REGISTER(kNumberTypeFloat64, double)},
    {EXTRACT_IMAGE_PATCHES_REGISTER(kNumberTypeInt8, int8_t)},
    {EXTRACT_IMAGE_PATCHES_REGISTER(kNumberTypeInt16, int16_t)},
    {EXTRACT_IMAGE_PATCHES_REGISTER(kNumberTypeInt32, int32_t)},
    {EXTRACT_IMAGE_PATCHES_REGISTER(kNumberTypeInt64, int64_t)},
    {EXTRACT_IMAGE_PATCHES_REGISTER(kNumberTypeUInt8, uint8_t)},
    {EXTRACT_IMAGE_PATCHES_REGISTER(kNumberTypeUInt16, uint16_t)},
    {EXTRACT_IMAGE_PATCHES_REGISTER(kNumberTypeUInt32, uint32_t)},
    {EXTRACT_IMAGE_PATCHES_REGISTER(kNumberTypeUInt64, uint64_t)},
    {EXTRACT_IMAGE_PATCHES_REGISTER(kNumberTypeComplex64, Complex<float>)},
    {EXTRACT_IMAGE_PATCHES_REGISTER(kNumberTypeComplex128, Complex<double>)},
  };
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, ExtractImagePatches, ExtractImagePatchesKernelMod);
}  // namespace kernel
}  // namespace mindspore
