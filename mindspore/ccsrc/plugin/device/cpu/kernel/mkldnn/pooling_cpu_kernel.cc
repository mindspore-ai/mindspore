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

#include "plugin/device/cpu/kernel/mkldnn/pooling_cpu_kernel.h"

#include <string>
#include <functional>
#include "utils/ms_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kPoolingInputsNum = 1;
constexpr size_t kPoolingOutputsNum = 1;
}  // namespace

bool PoolingCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                               const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  PrimitivePtr prim = base_operator->GetPrim();
  MS_EXCEPTION_IF_NULL(prim);
  kernel_name_ = prim->name();
  if (prim->HasAttr(CEIL_MODE)) {
    ValuePtr ceil_mode = prim->GetAttr(CEIL_MODE);
    ceil_mode_ = (ceil_mode->isa<BoolImm>() && GetValue<bool>(ceil_mode)) ||
                 (ceil_mode->isa<Int64Imm>() && GetValue<int64_t>(ceil_mode) == 1);
  }
  if (kernel_name_ == kAvgPoolOpName || kernel_name_ == kAvgPool3DOpName) {
    algorithm_ = dnnl::algorithm::pooling_avg;
    if (prim->HasAttr(COUNT_INCLUDE_PAD) && GetValue<bool>(prim->GetAttr(COUNT_INCLUDE_PAD))) {
      algorithm_ = dnnl::algorithm::pooling_avg_include_padding;
    }
    if (prim->HasAttr(DIVISOR_OVERRIDE) && GetValue<int64_t>(prim->GetAttr(DIVISOR_OVERRIDE)) != 0) {
      divisor_override_ = GetValue<int64_t>(prim->GetAttr(DIVISOR_OVERRIDE));
    }
  }
  format_ = GetValue<std::string>(prim->GetAttr(FORMAT));
  pad_mode_ = GetValue<std::string>(prim->GetAttr(PAD_MODE));
  kernel_include_nc_ = GetValue<std::vector<int64_t>>(prim->GetAttr(KERNEL_SIZE));
  strides_include_nc_ = GetValue<std::vector<int64_t>>(prim->GetAttr(STRIDES));
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kPoolingInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kPoolingOutputsNum, kernel_name_);
  return true;
}

int PoolingCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs,
                                const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  auto src_shape = inputs[0]->GetDeviceShapeAdaptively();
  dst_shape_ = outputs[0]->GetDeviceShapeAdaptively();
  const size_t src_dim = src_shape.size();
  if (src_dim != SHAPE_4D && src_dim != SHAPE_5D) {
    MS_LOG(ERROR) << "Pooling only supports 4D/5D input, but got " << src_dim << "D!";
    return KRET_RESIZE_FAILED;
  }
  const dnnl::memory::desc src_desc = GetDefaultMemDesc(src_shape);
  const dnnl::memory::desc dst_desc = GetDefaultMemDesc(dst_shape_);
  if (src_dim == SHAPE_4D && format_ != NCHW) {
    MS_LOG(ERROR) << kernel_name_ << " only supports 4D input with NCHW format, but got format " << format_;
    return KRET_RESIZE_FAILED;
  }
  if (src_dim == SHAPE_5D && format_ != NCDHW) {
    MS_LOG(ERROR) << kernel_name_ << " only supports 5D input with NCDHW format, but got format " << format_;
    return KRET_RESIZE_FAILED;
  }
  if (kernel_include_nc_.size() != src_dim) {
    MS_LOG(ERROR) << kernel_name_ << " requires kernel_size must be " << src_dim << "D, but got "
                  << kernel_include_nc_.size() << "D!";
    return KRET_RESIZE_FAILED;
  }
  if (strides_include_nc_.size() != src_dim) {
    MS_LOG(ERROR) << kernel_name_ << " requires strides must be " << src_dim << "D, but got "
                  << strides_include_nc_.size() << "D!";
    return KRET_RESIZE_FAILED;
  }
  const dnnl::memory::dims kernel(kernel_include_nc_.begin() + NC_LEN, kernel_include_nc_.end());
  const dnnl::memory::dims strides(strides_include_nc_.begin() + NC_LEN, strides_include_nc_.end());
  const dnnl::memory::dims dilation(kernel.size(), kPoolingDilation);
  dnnl::memory::dims padding_l;
  dnnl::memory::dims padding_r;
  kernel_ = kernel;
  PaddingInfo padding_info{pad_mode_,  kernel_,    strides,           dilation,
                           &padding_l, &padding_r, &padding_invalid_, ceil_mode_};
  GetPadding(base_operator, src_shape, padding_info);

  const auto desc = CreateDesc<dnnl::pooling_forward::desc>(dnnl::prop_kind::forward_inference, algorithm_, src_desc,
                                                            dst_desc, strides, kernel, padding_l, padding_r);
  const auto prim_desc = CreateDesc<dnnl::pooling_forward::primitive_desc>(desc, engine_);
  primitive_ = CreatePrimitive<dnnl::pooling_forward>(prim_desc);
  AddArgument(DNNL_ARG_SRC, src_desc);
  AddArgument(DNNL_ARG_DST, dst_desc);

  base_operator_ = base_operator;
  inputs_ = inputs;
  outputs_ = outputs;
  inputs_on_host_ = inputsOnHost;
  return KRET_OK;
}

void PoolingCpuKernelMod::EliminateInvalidPadding(float *dst) {
  if (dst_shape_.size() < SHAPE_5D || kernel_.size() + NC_LEN < SHAPE_5D ||
      padding_invalid_.size() + NC_LEN < SHAPE_5D) {
    MS_LOG(ERROR) << "The dst_shape must be 5D, the kernel and the padding_invalid must be 3D!";
  }
  const auto d_max = LongToSize(dst_shape_[D_INDEX] - 1);
  const auto h_max = LongToSize(dst_shape_[H_INDEX] - 1);
  const auto w_max = LongToSize(dst_shape_[W_INDEX] - 1);
  const size_t d_index = D_INDEX - NC_LEN;
  const size_t h_index = H_INDEX - NC_LEN;
  const size_t w_index = W_INDEX - NC_LEN;
  const int64_t valid_d = kernel_[d_index] - padding_invalid_[d_index];
  const int64_t valid_h = kernel_[h_index] - padding_invalid_[h_index];
  const int64_t valid_w = kernel_[w_index] - padding_invalid_[w_index];
  const std::vector<int64_t> valid_kernel_array{kernel_[d_index] * kernel_[h_index] * kernel_[w_index],
                                                kernel_[d_index] * kernel_[h_index] * valid_w,
                                                kernel_[d_index] * valid_h * kernel_[w_index],
                                                kernel_[d_index] * valid_h * valid_w,
                                                valid_d * kernel_[h_index] * kernel_[w_index],
                                                valid_d * kernel_[h_index] * valid_w,
                                                valid_d * valid_h * kernel_[w_index],
                                                valid_d * valid_h * valid_w};
  const int base = 2;
  const int64_t kernel_size = std::accumulate(kernel_.begin(), kernel_.end(), int64_t(1), std::multiplies<int64_t>());
  CTask task = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      for (size_t d = 0; d <= d_max; d++) {
        for (size_t h = 0; h <= h_max; h++) {
          for (size_t w = 0; w <= w_max; w++) {
            const char d_bound = d == d_max ? '1' : '0';
            const char h_bound = h == h_max ? '1' : '0';
            const char w_bound = w == w_max ? '1' : '0';
            const std::string bin{d_bound, h_bound, w_bound};
            const int kernel_index = std::stoi(bin, nullptr, base);
            const int64_t valid_kernel_size = valid_kernel_array[kernel_index];
            if (valid_kernel_size != kernel_size) {
              const size_t index =
                static_cast<size_t>(i * dst_shape_[D_INDEX] * dst_shape_[H_INDEX] * dst_shape_[W_INDEX] +
                                    d * dst_shape_[H_INDEX] * dst_shape_[W_INDEX] + h * dst_shape_[W_INDEX] + w);
              dst[index] = dst[index] * LongToFloat(kernel_size) / LongToFloat(valid_kernel_size);
            }
          }
        }
      }
    }
  };
  ParallelLaunchAutoSearch(task, static_cast<size_t>(dst_shape_[N_INDEX] * dst_shape_[C_INDEX]), this,
                           &parallel_search_info_);
}

void PoolingCpuKernelMod::ReComputeDivisor(float *dst) {
  const int64_t kernel_size = std::accumulate(kernel_.begin(), kernel_.end(), int64_t(1), std::multiplies<int64_t>());
  const size_t size = std::accumulate(dst_shape_.begin(), dst_shape_.end(), size_t(1), std::multiplies<size_t>());
  CTask task = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      dst[i] = dst[i] * LongToFloat(kernel_size) / LongToFloat(divisor_override_);
    }
  };
  ParallelLaunchAutoSearch(task, size, this, &parallel_search_info_);
}

std::vector<KernelAttr> PoolingCpuKernelMod::GetOpSupport() {
  static std::map<std::string, std::vector<KernelAttr>> support_list_map = {
    {kMaxPoolOpName, {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32)}},
    {kMaxPool3DOpName, {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32)}},
    {kAvgPoolOpName, {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32)}},
    {kAvgPool3DOpName, {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32)}}};

  auto iter = support_list_map.find(kernel_type_);
  if (iter == support_list_map.end()) {
    MS_LOG(EXCEPTION) << "Does not support " << kernel_type_ << "!";
  }
  return iter->second;
}

bool PoolingCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &,
                                 const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kPoolingInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kPoolingOutputsNum, kernel_name_);

  // From CPUKernelExecutor::LaunchKernel
  if (!Init(base_operator_, inputs_, outputs_)) {
    MS_LOG(ERROR) << "Re-init PoolingCpuKernelMod while launching failed";
    return false;
  }
  auto resize_ret = Resize(base_operator_, inputs_, outputs_, inputs_on_host_);
  if (resize_ret != KRET_OK) {
    MS_LOG(ERROR) << "Resize PoolingCpuKernelMod while launching failed: " << resize_ret;
    return false;
  }

  SetArgumentHandle(DNNL_ARG_SRC, inputs[0]->addr);
  SetArgumentHandle(DNNL_ARG_DST, outputs[0]->addr);
  ExecutePrimitive();

  float *dst = reinterpret_cast<float *>(outputs[0]->addr);
  if (divisor_override_ != 0) {
    ReComputeDivisor(dst);
    return true;
  }

  bool has_invalid_padding =
    std::any_of(padding_invalid_.begin(), padding_invalid_.end(), [](const int64_t &padding) { return padding != 0; });
  if (algorithm_ == dnnl::algorithm::pooling_avg_include_padding && has_invalid_padding) {
    EliminateInvalidPadding(dst);
  }
  return true;
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, MaxPool3D,
                                 []() { return std::make_shared<PoolingCpuKernelMod>(kMaxPool3DOpName); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, MaxPool,
                                 []() { return std::make_shared<PoolingCpuKernelMod>(kMaxPoolOpName); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, AvgPool,
                                 []() { return std::make_shared<PoolingCpuKernelMod>(kAvgPoolOpName); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, AvgPool3D,
                                 []() { return std::make_shared<PoolingCpuKernelMod>(kAvgPool3DOpName); });
}  // namespace kernel
}  // namespace mindspore
