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
#include "utils/ms_utils.h"

namespace mindspore::kernel {
constexpr auto kDataFormatDimMap = "DataFormatDimMap";
constexpr const size_t kDataFormatDimMapInputsNum = 1;
constexpr const size_t kDataFormatDimMapOutputsNum = 1;
constexpr const int32_t kInvalidOutput = -1;
const std::unordered_map<int32_t, int32_t> kDimMapSameFormat = {{-4, 0}, {-3, 1}, {-2, 2}, {-1, 3},
                                                                {0, 0},  {1, 1},  {2, 2},  {3, 3}};
const std::unordered_map<int32_t, int32_t> kDimMapNHWC2NCHW = {{-4, 0}, {-3, 2}, {-2, 3}, {-1, 1},
                                                               {0, 0},  {1, 2},  {2, 3},  {3, 1}};
const std::unordered_map<int32_t, int32_t> kDimMapNCHW2NHWC = {{-4, 0}, {-3, 3}, {-2, 1}, {-1, 2},
                                                               {0, 0},  {1, 3},  {2, 1},  {3, 2}};

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
  auto task = [this, &input, &output](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      const auto res_pair = this->dim_map_.find(input[i]);
      if (this->dim_map_.find(input[i]) == this->dim_map_.end()) {
        this->invalid_value_ = static_cast<int32_t>(input[i]);
        this->value_valid_ = false;
        output[i] = static_cast<T>(kInvalidOutput);
      } else {
        output[i] = res_pair->second;
      }
    }
  };
  ParallelLaunchAutoSearch(task, lens, this, &parallel_search_info_);

  if (value_valid_ == false) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input must be in [-4, 4) but got one input " << invalid_value_;
    return false;
  }
  return true;
}

using dataFormatPair = std::pair<KernelAttr, DataFormatDimMapCpuKernelMod::KernelRunFunc>;
const std::vector<dataFormatPair> &DataFormatDimMapCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, DataFormatDimMapCpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
     &DataFormatDimMapCpuKernelMod::LaunchKernel<int32_t>},
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
    MS_LOG(ERROR) << kernel_name_ << " reinit failed.";
    return ret;
  }
  std::vector<int64_t> input_shape = inputs[0]->GetShapeVector();
  std::vector<int64_t> output_shape = outputs[0]->GetShapeVector();
  auto in_shape_size = input_shape.size();
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
