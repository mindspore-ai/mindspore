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

#include "plugin/device/gpu/kernel/nn/data_format_vec_permute_gpu_kernel.h"

namespace mindspore {
namespace kernel {
const std::vector<int32_t> kDataSameFormat = {0, 1, 2, 3};
const std::vector<int32_t> kDataNHWC2NCHW = {0, 3, 1, 2};
const std::vector<int32_t> kDataNCHW2NHWC = {0, 2, 3, 1};
constexpr const size_t k1DElementNum = 4;

bool DataFormatVecPermuteGpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                            const std::vector<KernelTensorPtr> &inputs,
                                            const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr_ = std::dynamic_pointer_cast<ops::DataFormatVecPermute>(base_operator);
  kernel_name_ = kernel_ptr_->name();
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
    return false;
  }
  constexpr int INPUT_NUM = 1;
  constexpr int OUTPUT_NUM = 1;
  if (inputs.size() != INPUT_NUM || outputs.size() != OUTPUT_NUM) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input and output must be " << INPUT_NUM << " and " << OUTPUT_NUM
                  << ", but got " << inputs.size() << " and " << outputs.size();
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the kernel type should be in [int32, int64], but got: " << kernel_attr << ".";
    return false;
  }
  src_format = kernel_ptr_->get_src_format();
  dst_format = kernel_ptr_->get_dst_format();
  if (src_format == dst_format) {
    data_map_ = kDataSameFormat;
  } else if (src_format == "NHWC" && dst_format == "NCHW") {
    data_map_ = kDataNHWC2NCHW;
  } else if (src_format == "NCHW" && dst_format == "NHWC") {
    data_map_ = kDataNCHW2NHWC;
  } else {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', src_format and dst_format must be 'NCHW' or 'NHWC' "
                  << ", but got src_format " << src_format << " dst_format " << dst_format;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  unit_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex0).first);
  return true;
}

int DataFormatVecPermuteGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                             const std::vector<KernelTensorPtr> &inputs,
                                             const std::vector<KernelTensorPtr> &outputs,
                                             const std::map<uint32_t, tensor::TensorPtr> &) {
  for (const auto &input : inputs) {
    // If any input shape contains -1, means input shape is dynamic, so just return do nothing.
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }
  ResetResource();
  std::vector<int64_t> input_shape_ = std::vector<int64_t>(inputs.at(kIndex0)->GetDeviceShapeAdaptively().begin(),
                                                           inputs.at(kIndex0)->GetDeviceShapeAdaptively().end());
  std::vector<int64_t> output_shape_ = std::vector<int64_t>(outputs.at(kIndex0)->GetDeviceShapeAdaptively().begin(),
                                                            outputs.at(kIndex0)->GetDeviceShapeAdaptively().end());
  std::vector<int64_t> shape1 = {4};
  std::vector<int64_t> shape2 = {4, 2};
  if (input_shape_ != shape1 && input_shape_ != shape2) {
    MS_EXCEPTION(ValueError) << "For " << kernel_name_ << ", input shape must be (4, ) or (4, 2), but got "
                             << input_shape_ << ".";
  }
  auto in_shape_size = input_shape_.size();
  auto output_shape_size = output_shape_.size();
  if (in_shape_size != output_shape_size) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input shape size should be the same as output shape size, but got"
                  << " input shape size " << in_shape_size << " output shape size" << output_shape_size;
    return KRET_RESIZE_FAILED;
  }
  output_elements_ = std::accumulate(output_shape_.begin(), output_shape_.end(), 1, std::multiplies<int64_t>());
  if (output_elements_ == 0) {
    is_null_input_ = true;
  }
  InitSizeLists();
  return KRET_OK;
}

template <typename T>
bool DataFormatVecPermuteGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                    const std::vector<AddressPtr> &workspace,
                                                    const std::vector<AddressPtr> &outputs) {
  T *input = GetDeviceAddress<T>(inputs, 0);
  T *output = GetDeviceAddress<T>(outputs, 0);
  auto *index = GetDeviceAddress<int32_t>(workspace, kIndex0);

  // code block for sync dim_map
  {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(index, data_map_.data(), kDataFormatNum * sizeof(int32_t), cudaMemcpyHostToDevice,
                      reinterpret_cast<cudaStream_t>(cuda_stream_)),
      "cudaMemcpy failed in DataFormatVecPermuteGpuKernelMod::LaunchKernel.");
  }

  if (output_elements_ == k1DElementNum) {
    CalDataFormatVecPermute1D(output_elements_, input, output, index, device_id_,
                              reinterpret_cast<cudaStream_t>(cuda_stream_));
  } else {
    CalDataFormatVecPermute2D(output_elements_, input, output, index, device_id_,
                              reinterpret_cast<cudaStream_t>(cuda_stream_));
  }
  return true;
}

std::vector<std::pair<KernelAttr, DataFormatVecPermuteGpuKernelMod::DataFormatVecPermuteFunc>>
  DataFormatVecPermuteGpuKernelMod::func_list_ = {
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
     &DataFormatVecPermuteGpuKernelMod::LaunchKernel<int>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
     &DataFormatVecPermuteGpuKernelMod::LaunchKernel<int64_t>}};

std::vector<KernelAttr> DataFormatVecPermuteGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, DataFormatVecPermuteFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, DataFormatVecPermute, DataFormatVecPermuteGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
