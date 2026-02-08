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

#include "kernel/gpu/cuda/arrays/data_format_dim_map_gpu_kernel.h"
#include <algorithm>
#include <functional>
#include <memory>
#include "mindspore/ops/infer/data_format_dim_map.h"

namespace mindspore::kernel {
constexpr auto kDataFormatDimMap = "DataFormatDimMap";
constexpr const size_t kDataFormatDimMapInputsNum = 1;
constexpr const size_t kDataFormatDimMapOutputsNum = 1;
constexpr const size_t kDimMapNum = 4;
const std::vector<int32_t> kDimMapSameFormat = {0, 1, 2, 3};
const std::vector<int32_t> kDimMapNHWC2NCHW = {0, 3, 1, 2};
const std::vector<int32_t> kDimMapNCHW2NHWC = {0, 2, 3, 1};

template <typename T>
bool DataFormatDimMapGpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                                const std::vector<kernel::KernelTensor *> &workspace,
                                                const std::vector<kernel::KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kDataFormatDimMapInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kDataFormatDimMapOutputsNum, kernel_name_);
  T *input_addr = GetDeviceAddress<T>(inputs, kIndex0);
  T *output_addr = GetDeviceAddress<T>(outputs, kIndex0);
  auto *d_dim_map = GetDeviceAddress<int32_t>(workspace, kIndex0);

  // code block for sync dim_map
  {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(d_dim_map, dim_map_.data(), kDimMapNum * sizeof(int32_t), cudaMemcpyHostToDevice,
                      reinterpret_cast<cudaStream_t>(cuda_stream_)),
      "cudaMemcpy failed in DataFormatDimMapGpuKernelMod::Launch.");
  }

  auto status = DataFormatDimMap(static_cast<size_t>(input_elements_), input_addr, output_addr, d_dim_map,
                                 reinterpret_cast<cudaStream_t>(cuda_stream_));
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

using dataFormatPair = std::pair<KernelAttr, DataFormatDimMapGpuKernelMod::KernelRunFunc>;
const std::vector<dataFormatPair> &DataFormatDimMapGpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, DataFormatDimMapGpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
     &DataFormatDimMapGpuKernelMod::LaunchKernel<int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
     &DataFormatDimMapGpuKernelMod::LaunchKernel<int64_t>},
  };
  return func_list;
}

bool DataFormatDimMapGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &outputs) {
  src_format_ = GetValue<std::string>(primitive_->GetAttr("src_format"));
  dst_format_ = GetValue<std::string>(primitive_->GetAttr("dst_format"));
  if (inputs.size() != kDataFormatDimMapInputsNum || outputs.size() != kDataFormatDimMapOutputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input and output size must be " << kDataFormatDimMapInputsNum
                  << " and " << kDataFormatDimMapOutputsNum << ", but got " << inputs.size() << " and "
                  << outputs.size();
    return false;
  }

  if (!MatchKernelFunc(kernel_name_, inputs, outputs)) {
    return false;
  }

  if (src_format_ == dst_format_) {
    dim_map_ = kDimMapSameFormat;
  } else if (src_format_ == "NHWC" && dst_format_ == "NCHW") {
    dim_map_ = kDimMapNHWC2NCHW;
  } else if (src_format_ == "NCHW" && dst_format_ == "NHWC") {
    dim_map_ = kDimMapNCHW2NHWC;
  } else {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', src_format and dst_format must be 'NCHW' or 'NHWC' "
                  << ", but got src_format " << src_format_ << " dst_format " << dst_format_;
    return false;
  }

  return true;
}

int DataFormatDimMapGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                         const std::vector<KernelTensor *> &outputs) {
  int ret = KRET_OK;
  if ((ret = KernelMod::Resize(inputs, outputs)) != 0) {
    return ret;
  }
  input_shape_ = std::vector<size_t>(inputs.at(kIndex0)->GetDeviceShapeVector().begin(),
                                     inputs.at(kIndex0)->GetDeviceShapeVector().end());
  output_shape_ = std::vector<size_t>(outputs.at(kIndex0)->GetDeviceShapeVector().begin(),
                                      outputs.at(kIndex0)->GetDeviceShapeVector().end());
  auto in_shape_size = input_shape_.size();
  if (in_shape_size > max_dims_) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dimension of output should be less than or equal to max_dims 7, but got "
                      << in_shape_size << ".";
    return KRET_RESIZE_FAILED;
  }
  auto output_shape_size = output_shape_.size();
  if (in_shape_size != output_shape_size) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input shape size should be the same as output shape size, but got"
                  << " input shape size " << in_shape_size << " output shape size" << output_shape_size;
    return KRET_RESIZE_FAILED;
  }
  // A Code Block For setting input and output shape.
  input_elements_ = std::accumulate(input_shape_.begin(), input_shape_.end(), size_t(1), std::multiplies<size_t>());
  is_null_input_ = (input_elements_ == 0);
  // The number of dim map
  size_t workspace_size = kDimMapNum * sizeof(int32_t);
  workspace_size_list_.emplace_back(workspace_size);
  return KRET_OK;
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, DataFormatDimMap,
                                 []() { return std::make_shared<DataFormatDimMapGpuKernelMod>(kDataFormatDimMap); });
}  // namespace mindspore::kernel
