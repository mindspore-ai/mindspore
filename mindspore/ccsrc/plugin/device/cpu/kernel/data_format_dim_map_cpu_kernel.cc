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

#include "plugin/device/cpu/kernel/data_format_dim_map_cpu_kernel.h"
#include <algorithm>
#include <functional>
#include "mindspore/core/ops/data_format_dim_map.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore::kernel {
constexpr auto kDataFormatDimMap = "DataFormatDimMap";
constexpr const size_t kDataFormatDimMapInputsNum = 1;
constexpr const size_t kDataFormatDimMapOutputsNum = 1;
constexpr const int32_t kInvalidOutput = -1;
constexpr const int32_t kNumberFour = 4;
const std::vector<int32_t> kDimMapSameFormat = {0, 1, 2, 3};
const std::vector<int32_t> kDimMapNHWC2NCHW = {0, 3, 1, 2};
const std::vector<int32_t> kDimMapNCHW2NHWC = {0, 2, 3, 1};

template <typename T>
bool DataFormatDimMapCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                const std::vector<AddressPtr> &,
                                                const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kDataFormatDimMapInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kDataFormatDimMapOutputsNum, kernel_name_);
  T *input = reinterpret_cast<T *>(inputs[kIndex0]->addr);
  MS_ERROR_IF_NULL_W_RET_VAL(input, false);
  T *output = reinterpret_cast<T *>(outputs[kIndex0]->addr);
  MS_ERROR_IF_NULL_W_RET_VAL(output, false);

  const size_t lens = outputs[0]->size > 0 ? static_cast<size_t>(outputs[0]->size / sizeof(T)) : 1;
  T number_four = static_cast<T>(kNumberFour);
  auto task = [this, &input, &output, number_four](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      output[i] =
        static_cast<T>(this->dim_map_[static_cast<size_t>((input[i] % number_four + number_four) % number_four)]);
    }
  };
  ParallelLaunchAutoSearch(task, lens, this, &parallel_search_info_);
  return true;
}

using dataFormatPair = std::pair<KernelAttr, DataFormatDimMapCpuKernelMod::KernelRunFunc>;
const std::vector<dataFormatPair> &DataFormatDimMapCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, DataFormatDimMapCpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
     &DataFormatDimMapCpuKernelMod::LaunchKernel<int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
     &DataFormatDimMapCpuKernelMod::LaunchKernel<int64_t>},
  };
  return func_list;
}

bool DataFormatDimMapCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                        const std::vector<KernelTensorPtr> &inputs,
                                        const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::DataFormatDimMap>(base_operator);
  MS_ERROR_IF_NULL_W_RET_VAL(kernel_ptr, false);

  kernel_name_ = kernel_ptr->name();
  src_format_ = kernel_ptr->get_src_format();
  dst_format_ = kernel_ptr->get_dst_format();
  if (inputs.size() != kDataFormatDimMapInputsNum || outputs.size() != kDataFormatDimMapOutputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input and output size must be " << kDataFormatDimMapInputsNum
                  << " and " << kDataFormatDimMapOutputsNum << ", but got " << inputs.size() << " and "
                  << outputs.size();
    return false;
  }

  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
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

int DataFormatDimMapCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                         const std::vector<KernelTensorPtr> &inputs,
                                         const std::vector<KernelTensorPtr> &outputs,
                                         const std::map<uint32_t, tensor::TensorPtr> &) {
  int ret = KRET_OK;
  if ((ret = KernelMod::Resize(base_operator, inputs, outputs)) != 0) {
    return ret;
  }
  std::vector<int64_t> input_shape = inputs[0]->GetShapeVector();
  std::vector<int64_t> output_shape = outputs[0]->GetShapeVector();
  auto in_shape_size = input_shape.size();
  if (in_shape_size > max_dims_) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the dimension of output should be less than or equal to max_dims 7, but got " << in_shape_size
                  << ".";
    return KRET_RESIZE_FAILED;
  }
  auto output_shape_size = output_shape.size();
  if (in_shape_size != output_shape_size) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input shape size should be the same as output shape size, but got"
                  << " input shape size " << in_shape_size << " output shape size" << output_shape_size;
    return KRET_RESIZE_FAILED;
  }
  return KRET_OK;
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, DataFormatDimMap,
                                 []() { return std::make_shared<DataFormatDimMapCpuKernelMod>(kDataFormatDimMap); });
}  // namespace mindspore::kernel
