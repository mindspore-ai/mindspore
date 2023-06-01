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

#include "plugin/device/cpu/kernel/upsample_trilinear_3d_grad_cpu_kernel.h"
#include <string>
#include "kernel/common_utils.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
const double kValueZero = 0.;
constexpr size_t kUpsampleTrilinear3DGradInputsNum = 3;
constexpr size_t kUpsampleTrilinear3DGradOutputNum = 1;
// GRAIN_SIZE for Parallel
constexpr size_t kGrainSize = 32768;
}  // namespace
template <typename S>
void UpsampleTrilinear3DGradCpuKernelMod::ComputeWeightsAndIndices(
  UpsampleTrilinear3DGradCpuKernelMod::WeightsAndIndices<S> *const wi, S scale, int64_t out_idx, int64_t input_size,
  int64_t output_size, int64_t stride) {
  (void)ComputeSourceIndexAndLambda<S>(&(wi->id0), &(wi->id1), &(wi->lambda0), &(wi->lambda1), scale, out_idx,
                                       input_size, output_size, align_corners_);
  wi->Step(stride);
}

template <typename S>
void UpsampleTrilinear3DGradCpuKernelMod::ComputeHelper(
  UpsampleTrilinear3DGradCpuKernelMod::WeightsAndIndices<S> *const helper, S scale, int64_t input_size,
  int64_t output_size, int64_t stride) {
  auto loop = [&](int64_t begin, int64_t end) {
    for (int64_t out_idx = begin; out_idx < end; ++out_idx) {
      (void)ComputeWeightsAndIndices<S>(helper + out_idx, scale, out_idx, input_size, output_size, stride);
    }
  };
  float block_size = 64.0;
  ParallelLaunch(loop, static_cast<size_t>(output_size), block_size);
}

bool UpsampleTrilinear3DGradCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                               const std::vector<KernelTensorPtr> &inputs,
                                               const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  x_type_ = inputs[kIndex0]->GetDtype();
  auto kernel_ptr = std::make_shared<ops::UpsampleTrilinear3DGrad>(base_operator->GetPrim());
  align_corners_ = kernel_ptr->get_align_corners();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int UpsampleTrilinear3DGradCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                                const std::vector<KernelTensorPtr> &inputs,
                                                const std::vector<KernelTensorPtr> &outputs,
                                                const std::map<uint32_t, tensor::TensorPtr> &) {
  if (auto ret = NativeCpuKernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  // shape
  output_shape_ = inputs.at(kIndex0)->GetDeviceShapeAdaptively();
  input_shape_ = outputs.at(kIndex0)->GetDeviceShapeAdaptively();
  // workspace
  size_t unit_size = sizeof(WeightsAndIndices<float>);
  if (x_type_ == kNumberTypeFloat64) {
    unit_size = sizeof(WeightsAndIndices<double>);
  }
  workspace_size_list_.push_back(unit_size * LongToSize(output_shape_[kIndex2]));
  workspace_size_list_.push_back(unit_size * LongToSize(output_shape_[kIndex3]));
  workspace_size_list_.push_back(unit_size * LongToSize(output_shape_[kIndex4]));
  // none_list
  none_list_ = GetValue<std::vector<int64_t>>(base_operator->GetAttr(kAttrNoneList));
  if (none_list_.size() != kIndex1) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', only one of output_size or scales should be specified.";
  }
  return KRET_OK;
}

template <typename T, typename S>
bool UpsampleTrilinear3DGradCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                       const std::vector<kernel::AddressPtr> &workspace,
                                                       const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kUpsampleTrilinear3DGradInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kUpsampleTrilinear3DGradOutputNum, kernel_name_);
  // fetch scales
  if (none_list_[kIndex0] == static_cast<int64_t>(kIndex3)) {
    scales_ = std::vector<double>(kIndex3, kValueZero);
  } else {
    scales_.clear();
    auto scales_ptr = GetDeviceAddress<float>(inputs, kIndex2);
    for (size_t i = 0; i < kIndex3; ++i) {
      scales_.push_back(static_cast<double>(scales_ptr[i]));
    }
  }
  // the input grad of backward process is the output of forward process
  auto grad_output_ptr = reinterpret_cast<T *>(inputs[kIndex0]->addr);
  const int64_t total = CPUKernelUtils::CalcElementNum(input_shape_);
  S *grad_input_ptr = nullptr;
  bool is_fp16 = std::is_same<T, float16>::value;
  // define for fp16
  std::vector<S> grad_input_copy(1, 0);
  if (is_fp16) {
    grad_input_copy.resize(total, 0);
    grad_input_ptr = grad_input_copy.data();
  } else {
    grad_input_ptr = reinterpret_cast<S *>(outputs[kIndex0]->addr);
    int ret = memset_s(outputs[kIndex0]->addr, outputs[kIndex0]->size, 0, outputs[kIndex0]->size);
    if (ret != EOK) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memset_s error. Error no: " << ret;
    }
  }
  // treat nbatch and channels as one dimension
  int64_t channels = input_shape_[kIndex0] * input_shape_[kIndex1];
  int64_t input_depth = input_shape_[kIndex2];
  int64_t input_height = input_shape_[kIndex3];
  int64_t input_width = input_shape_[kIndex4];

  int64_t output_depth = output_shape_[kIndex2];
  int64_t output_height = output_shape_[kIndex3];
  int64_t output_width = output_shape_[kIndex4];

  int64_t output_slice_size = output_depth * output_height * output_width;
  int64_t input_slice_size = input_depth * input_height * input_width;
  MS_EXCEPTION_IF_CHECK_FAIL(channels != 0 && output_depth != 0 && output_height != 0 && output_width != 0,
                             "Invalid output shape.");
  const S depth_scale = AreaPixelComputeScale<S>(input_depth, output_depth, align_corners_, scales_[kIndex0]);
  const S height_scale = AreaPixelComputeScale<S>(input_height, output_height, align_corners_, scales_[kIndex1]);
  const S width_scale = AreaPixelComputeScale<S>(input_width, output_width, align_corners_, scales_[kIndex2]);

  WeightsAndIndices<S> *const d_helper = reinterpret_cast<WeightsAndIndices<S> *>(workspace[kIndex0]->addr);
  WeightsAndIndices<S> *const h_helper = reinterpret_cast<WeightsAndIndices<S> *>(workspace[kIndex1]->addr);
  WeightsAndIndices<S> *const w_helper = reinterpret_cast<WeightsAndIndices<S> *>(workspace[kIndex2]->addr);
  (void)ComputeHelper<S>(d_helper, depth_scale, input_depth, output_depth, input_height * input_width);
  (void)ComputeHelper<S>(h_helper, height_scale, input_height, output_height, input_width);
  (void)ComputeHelper<S>(w_helper, width_scale, input_width, output_width, 1);

  auto loop3d = [&](int64_t begin, int64_t end) {
    auto input_index = [=](int64_t c_idx, int64_t d_idx, int64_t h_idx, int64_t w_idx) {
      return c_idx * input_slice_size + d_idx + h_idx + w_idx;
    };

    int64_t id0{0}, id1{0}, ih0{0}, ih1{0}, iw0{0}, iw1{0};
    S d0lambda{0}, d1lambda{0}, h0lambda{0}, h1lambda{0}, w0lambda{0}, w1lambda{0};

    for (int64_t c_idx = begin; c_idx < end; ++c_idx) {
      for (int64_t od = 0; od < output_depth; ++od) {
        d_helper[od](&id0, &id1, &d0lambda, &d1lambda);

        for (int64_t oh = 0; oh < output_height; ++oh) {
          h_helper[oh](&ih0, &ih1, &h0lambda, &h1lambda);

          for (int64_t ow = 0; ow < output_width; ++ow) {
            w_helper[ow](&iw0, &iw1, &w0lambda, &w1lambda);

            auto grad_output_value = static_cast<S>(
              grad_output_ptr[c_idx * output_slice_size + od * output_height * output_width + oh * output_width + ow]);
            S w000 = d0lambda * h0lambda * w0lambda;
            S w001 = d0lambda * h0lambda * w1lambda;
            S w010 = d0lambda * h1lambda * w0lambda;
            S w011 = d0lambda * h1lambda * w1lambda;
            S w100 = d1lambda * h0lambda * w0lambda;
            S w101 = d1lambda * h0lambda * w1lambda;
            S w110 = d1lambda * h1lambda * w0lambda;
            S w111 = d1lambda * h1lambda * w1lambda;
            grad_input_ptr[input_index(c_idx, id0, ih0, iw0)] += w000 * grad_output_value;
            grad_input_ptr[input_index(c_idx, id0, ih0, iw1)] += w001 * grad_output_value;
            grad_input_ptr[input_index(c_idx, id0, ih1, iw0)] += w010 * grad_output_value;
            grad_input_ptr[input_index(c_idx, id0, ih1, iw1)] += w011 * grad_output_value;
            grad_input_ptr[input_index(c_idx, id1, ih0, iw0)] += w100 * grad_output_value;
            grad_input_ptr[input_index(c_idx, id1, ih0, iw1)] += w101 * grad_output_value;
            grad_input_ptr[input_index(c_idx, id1, ih1, iw0)] += w110 * grad_output_value;
            grad_input_ptr[input_index(c_idx, id1, ih1, iw1)] += w111 * grad_output_value;
          }
        }
      }
    }
  };

  float block_size = static_cast<float>(kGrainSize) / output_slice_size / 8;
  ParallelLaunch(loop3d, static_cast<size_t>(channels), block_size);
  // memcopy and cast for fp16
  if (is_fp16) {
    T *real_input_ptr = reinterpret_cast<T *>(outputs[kIndex0]->addr);
    auto task_fp16 = [&](int64_t begin, int64_t end) {
      for (int64_t idx = begin; idx < end; ++idx) {
        real_input_ptr[idx] = static_cast<T>(grad_input_ptr[idx]);
      }
    };
    block_size = kGrainSize;
    ParallelLaunch(task_fp16, static_cast<size_t>(total), block_size);
  }
  return true;
}

#define UpsampleTrilinear3D_GRAD_CPU_KERNEL_INT32_REG(M_S, M_T, T, S)                                 \
  KernelAttr().AddInputAttr(M_S).AddInputAttr(kNumberTypeInt32).AddInputAttr(M_T).AddOutputAttr(M_S), \
    &UpsampleTrilinear3DGradCpuKernelMod::LaunchKernel<T, S>
#define UpsampleTrilinear3D_GRAD_CPU_KERNEL_INT64_REG(M_S, M_T, T, S)                                 \
  KernelAttr().AddInputAttr(M_S).AddInputAttr(kNumberTypeInt64).AddInputAttr(M_T).AddOutputAttr(M_S), \
    &UpsampleTrilinear3DGradCpuKernelMod::LaunchKernel<T, S>

std::vector<std::pair<KernelAttr, UpsampleTrilinear3DGradCpuKernelMod::KernelRunFunc>>
  UpsampleTrilinear3DGradCpuKernelMod::func_list_ = {
    {UpsampleTrilinear3D_GRAD_CPU_KERNEL_INT32_REG(kNumberTypeFloat16, kNumberTypeInt32, float16, float)},
    {UpsampleTrilinear3D_GRAD_CPU_KERNEL_INT32_REG(kNumberTypeFloat32, kNumberTypeInt32, float, float)},
    {UpsampleTrilinear3D_GRAD_CPU_KERNEL_INT32_REG(kNumberTypeFloat64, kNumberTypeInt32, double, double)},
    {UpsampleTrilinear3D_GRAD_CPU_KERNEL_INT32_REG(kNumberTypeFloat16, kNumberTypeInt64, float16, float)},
    {UpsampleTrilinear3D_GRAD_CPU_KERNEL_INT32_REG(kNumberTypeFloat32, kNumberTypeInt64, float, float)},
    {UpsampleTrilinear3D_GRAD_CPU_KERNEL_INT32_REG(kNumberTypeFloat64, kNumberTypeInt64, double, double)},
    {UpsampleTrilinear3D_GRAD_CPU_KERNEL_INT32_REG(kNumberTypeFloat16, kNumberTypeFloat32, float16, float)},
    {UpsampleTrilinear3D_GRAD_CPU_KERNEL_INT32_REG(kNumberTypeFloat32, kNumberTypeFloat32, float, float)},
    {UpsampleTrilinear3D_GRAD_CPU_KERNEL_INT32_REG(kNumberTypeFloat64, kNumberTypeFloat32, double, double)},
    {UpsampleTrilinear3D_GRAD_CPU_KERNEL_INT64_REG(kNumberTypeFloat16, kNumberTypeInt32, float16, float)},
    {UpsampleTrilinear3D_GRAD_CPU_KERNEL_INT64_REG(kNumberTypeFloat32, kNumberTypeInt32, float, float)},
    {UpsampleTrilinear3D_GRAD_CPU_KERNEL_INT64_REG(kNumberTypeFloat64, kNumberTypeInt32, double, double)},
    {UpsampleTrilinear3D_GRAD_CPU_KERNEL_INT64_REG(kNumberTypeFloat16, kNumberTypeInt64, float16, float)},
    {UpsampleTrilinear3D_GRAD_CPU_KERNEL_INT64_REG(kNumberTypeFloat32, kNumberTypeInt64, float, float)},
    {UpsampleTrilinear3D_GRAD_CPU_KERNEL_INT64_REG(kNumberTypeFloat64, kNumberTypeInt64, double, double)},
    {UpsampleTrilinear3D_GRAD_CPU_KERNEL_INT64_REG(kNumberTypeFloat16, kNumberTypeFloat32, float16, float)},
    {UpsampleTrilinear3D_GRAD_CPU_KERNEL_INT64_REG(kNumberTypeFloat32, kNumberTypeFloat32, float, float)},
    {UpsampleTrilinear3D_GRAD_CPU_KERNEL_INT64_REG(kNumberTypeFloat64, kNumberTypeFloat32, double, double)}};

std::vector<KernelAttr> UpsampleTrilinear3DGradCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, UpsampleTrilinear3DGradCpuKernelMod::KernelRunFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, UpsampleTrilinear3DGrad, UpsampleTrilinear3DGradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
