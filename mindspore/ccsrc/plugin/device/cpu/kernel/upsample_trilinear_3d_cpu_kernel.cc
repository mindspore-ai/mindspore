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

#include "plugin/device/cpu/kernel/upsample_trilinear_3d_cpu_kernel.h"
#include <string>
#include <utility>
#include "kernel/kernel_get_value.h"
#include "kernel/ops_utils.h"
#include "ops/upsample_trilinear_3d.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kUpsampleTrilinear3DInputsNum = 2;
constexpr size_t kUpsampleTrilinear3DOutputNum = 1;
constexpr int64_t kGrainSize = 32768;
const double kValueZero = 0.;
}  // namespace
template <typename S>
void UpsampleTrilinear3DCpuKernelMod::ComputeWeightsAndIndices(
  UpsampleTrilinear3DCpuKernelMod::WeightsAndIndices<S> *const wi, const S scale, const int64_t out_idx,
  const int64_t input_size, const int64_t output_size, const int64_t stride) const {
  (void)ComputeSourceIndexAndLambda<S>(&(wi->id0), &(wi->id1), &(wi->lambda0), &(wi->lambda1), scale, out_idx,
                                       input_size, output_size, align_corners_);
  wi->Step(stride);
}

template <typename S>
void UpsampleTrilinear3DCpuKernelMod::ComputeHelper(UpsampleTrilinear3DCpuKernelMod::WeightsAndIndices<S> *const helper,
                                                    const S scale, const int64_t input_size, const int64_t output_size,
                                                    const int64_t stride) const {
  auto loop = [&](int64_t begin, int64_t end) {
    for (int64_t out_idx = begin; out_idx < end; ++out_idx) {
      (void)ComputeWeightsAndIndices<S>(helper + out_idx, scale, out_idx, input_size, output_size, stride);
    }
  };
  float block_size = 64.0;
  ParallelLaunch(loop, static_cast<size_t>(output_size), block_size);
}

bool UpsampleTrilinear3DCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                           const std::vector<KernelTensorPtr> &inputs,
                                           const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  auto kernel_ptr = std::dynamic_pointer_cast<ops::UpsampleTrilinear3D>(base_operator);
  MS_EXCEPTION_IF_NULL(kernel_ptr);
  align_corners_ = kernel_ptr->get_align_corners();
  auto x = inputs.at(kIndex0);
  MS_EXCEPTION_IF_NULL(x);
  x_type_ = x->GetDtype();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int UpsampleTrilinear3DCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                            const std::vector<KernelTensorPtr> &inputs,
                                            const std::vector<KernelTensorPtr> &outputs,
                                            const std::map<uint32_t, tensor::TensorPtr> &) {
  if (auto ret = NativeCpuKernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  // shape
  x_shape_ = inputs.at(kIndex0)->GetShapeVector();
  y_shape_ = outputs.at(kIndex0)->GetShapeVector();
  // workspace
  size_t unit_size = sizeof(WeightsAndIndices<float>);
  if (x_type_ == kNumberTypeFloat64) {
    unit_size = sizeof(WeightsAndIndices<double>);
  }
  workspace_size_list_.push_back(unit_size * LongToSize(y_shape_[kIndex2]));
  workspace_size_list_.push_back(unit_size * LongToSize(y_shape_[kIndex3]));
  workspace_size_list_.push_back(unit_size * LongToSize(y_shape_[kIndex4]));
  // none_list
  MS_EXCEPTION_IF_NULL(base_operator);
  none_list_ = GetValue<std::vector<int64_t>>(base_operator->GetAttr(kAttrNoneList));
  if (none_list_.size() != kIndex1) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', only one of output_size or scales should be specified.";
  }
  if (none_list_[kIndex0] == static_cast<int64_t>(kIndex2)) {
    scales_ = std::vector<double>(kIndex3, kValueZero);
  } else {
    if (!TryGetFloatValue(inputs, kIndex1, kernel_name_, &scales_, false)) {
      MS_LOG(EXCEPTION) << "For " << kernel_name_ << " can't get scales input! ";
    }
  }
  return KRET_OK;
}

template <typename T, typename S>
bool UpsampleTrilinear3DCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                   const std::vector<kernel::AddressPtr> &workspace,
                                                   const std::vector<kernel::AddressPtr> &outputs) {
  auto x_ptr = GetDeviceAddress<T>(inputs, kIndex0);
  MS_EXCEPTION_IF_NULL(x_ptr);
  auto y_ptr = GetDeviceAddress<T>(outputs, kIndex0);
  MS_EXCEPTION_IF_NULL(y_ptr);

  // treat batch and channels as one dimension
  int64_t channels = x_shape_[kIndex0] * x_shape_[kIndex1];
  int64_t input_depth = x_shape_[kIndex2];
  int64_t input_height = x_shape_[kIndex3];
  int64_t input_width = x_shape_[kIndex4];
  int64_t input_slice_size = input_depth * input_height * input_width;

  int64_t output_depth = y_shape_[kIndex2];
  int64_t output_height = y_shape_[kIndex3];
  int64_t output_width = y_shape_[kIndex4];

  if (input_depth == output_depth && input_height == output_height && input_width == output_width) {
    auto cpy_ret = memcpy_s(y_ptr, outputs[kIndex0]->size, x_ptr, outputs[kIndex0]->size);
    if (cpy_ret != EOK) {
      MS_EXCEPTION(MemoryError) << "For " << kernel_name_ << ", memcpy_s to output failed.";
    }
    return true;
  }

  const S depth_scale = AreaPixelComputeScale<S>(input_depth, output_depth, align_corners_, scales_[kIndex0]);
  const S height_scale = AreaPixelComputeScale<S>(input_height, output_height, align_corners_, scales_[kIndex1]);
  const S width_scale = AreaPixelComputeScale<S>(input_width, output_width, align_corners_, scales_[kIndex2]);

  WeightsAndIndices<S> *const d_helper = GetDeviceAddress<WeightsAndIndices<S>>(workspace, kIndex0);
  MS_EXCEPTION_IF_NULL(d_helper);
  WeightsAndIndices<S> *const h_helper = GetDeviceAddress<WeightsAndIndices<S>>(workspace, kIndex1);
  MS_EXCEPTION_IF_NULL(h_helper);
  WeightsAndIndices<S> *const w_helper = GetDeviceAddress<WeightsAndIndices<S>>(workspace, kIndex2);
  MS_EXCEPTION_IF_NULL(w_helper);
  (void)ComputeHelper<S>(d_helper, depth_scale, input_depth, output_depth, input_height * input_width);
  (void)ComputeHelper<S>(h_helper, height_scale, input_height, output_height, input_width);
  (void)ComputeHelper<S>(w_helper, width_scale, input_width, output_width, 1);

  auto get_value = [=](int64_t idx) -> S { return static_cast<S>(x_ptr[idx]); };
  auto task = [&](int64_t begin, int64_t end) {
    int64_t n = 0;
    int64_t od = 0;
    int64_t oh = 0;
    (void)DataIndexInit(&begin, &n, &channels, &od, &output_depth, &oh, &output_height);

    for (int64_t i = begin; i < end; ++i) {
      int64_t id0(0);
      int64_t id1(0);
      int64_t ih0(0);
      int64_t ih1(0);
      int64_t iw0(0);
      int64_t iw1(0);
      S d0lambda(0);
      S d1lambda(0);
      S h0lambda(0);
      S h1lambda(0);
      S w0lambda(0);
      S w1lambda(0);
      d_helper[od](&id0, &id1, &d0lambda, &d1lambda);
      h_helper[oh](&ih0, &ih1, &h0lambda, &h1lambda);
      int64_t src_offset = n * input_slice_size;
      std::array<int64_t, 4> indices = {src_offset + id0 + ih0, src_offset + id0 + ih1, src_offset + id1 + ih0,
                                        src_offset + id1 + ih1};
      std::array<S, 4> weights = {d0lambda * h0lambda, d0lambda * h1lambda, d1lambda * h0lambda, d1lambda * h1lambda};
      int64_t dst_offset = i * output_width;
      for (int64_t ow = 0; ow < output_width; ++ow) {
        w_helper[ow](&iw0, &iw1, &w0lambda, &w1lambda);
        // weights
        S w000 = weights[0] * w0lambda;
        S w001 = weights[0] * w1lambda;
        S w010 = weights[1] * w0lambda;
        S w011 = weights[1] * w1lambda;
        S w100 = weights[2] * w0lambda;
        S w101 = weights[2] * w1lambda;
        S w110 = weights[3] * w0lambda;
        S w111 = weights[3] * w1lambda;
        // indices
        int64_t i000 = indices[0] + iw0;
        int64_t i001 = indices[0] + iw1;
        int64_t i010 = indices[1] + iw0;
        int64_t i011 = indices[1] + iw1;
        int64_t i100 = indices[2] + iw0;
        int64_t i101 = indices[2] + iw1;
        int64_t i110 = indices[3] + iw0;
        int64_t i111 = indices[3] + iw1;
        // get result
        y_ptr[dst_offset + ow] = static_cast<T>(
          w000 * get_value(i000) + w001 * get_value(i001) + w010 * get_value(i010) + w011 * get_value(i011) +
          w100 * get_value(i100) + w101 * get_value(i101) + w110 * get_value(i110) + w111 * get_value(i111));
      }

      (void)DataIndexStep(&n, &channels, &od, &output_depth, &oh, &output_height);
    }
  };
  float block_size = static_cast<float>(kGrainSize) / output_width / 8;
  ParallelLaunch(task, static_cast<size_t>(channels * output_depth * output_height), block_size);

  return true;
}

#define UpsampleTrilinear3D_CPU_KERNEL_REG(M_S, M_T, S, T)             \
  KernelAttr().AddInputAttr(M_S).AddInputAttr(M_T).AddOutputAttr(M_S), \
    &UpsampleTrilinear3DCpuKernelMod::LaunchKernel<S, T>

std::vector<std::pair<KernelAttr, UpsampleTrilinear3DCpuKernelMod::KernelRunFunc>>
  UpsampleTrilinear3DCpuKernelMod::func_list_ = {
    {UpsampleTrilinear3D_CPU_KERNEL_REG(kNumberTypeFloat16, kNumberTypeInt32, float16, float)},
    {UpsampleTrilinear3D_CPU_KERNEL_REG(kNumberTypeFloat32, kNumberTypeInt32, float, float)},
    {UpsampleTrilinear3D_CPU_KERNEL_REG(kNumberTypeFloat64, kNumberTypeInt32, double, double)},
    {UpsampleTrilinear3D_CPU_KERNEL_REG(kNumberTypeFloat16, kNumberTypeInt64, float16, float)},
    {UpsampleTrilinear3D_CPU_KERNEL_REG(kNumberTypeFloat32, kNumberTypeInt64, float, float)},
    {UpsampleTrilinear3D_CPU_KERNEL_REG(kNumberTypeFloat64, kNumberTypeInt64, double, double)},
    {UpsampleTrilinear3D_CPU_KERNEL_REG(kNumberTypeFloat16, kNumberTypeFloat32, float16, float)},
    {UpsampleTrilinear3D_CPU_KERNEL_REG(kNumberTypeFloat32, kNumberTypeFloat32, float, float)},
    {UpsampleTrilinear3D_CPU_KERNEL_REG(kNumberTypeFloat64, kNumberTypeFloat32, double, double)},
};

std::vector<KernelAttr> UpsampleTrilinear3DCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, UpsampleTrilinear3DCpuKernelMod::KernelRunFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, UpsampleTrilinear3D, UpsampleTrilinear3DCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
