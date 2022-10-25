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

#include "plugin/device/gpu/kernel/arrays/slice_gpu_kernel.h"
namespace mindspore {
namespace kernel {
namespace {
template <typename T, typename S = int64_t>
std::unique_ptr<cukernel::GpuKernelHelperBase> CreateSliceKernelPtr(const std::string &kernel_name,
                                                                    const uint32_t &device_id) {
  return std::make_unique<cukernel::SliceHelperGpuKernel<T, S>>(kernel_name, device_id);
}
using SlicePtrCreatorFunc =
  std::function<std::unique_ptr<cukernel::GpuKernelHelperBase>(const std::string &, const uint32_t &)>;

const std::vector<std::pair<KernelAttr, SlicePtrCreatorFunc>> kernel_attr = {{KernelAttr()
                                                                                .AddInputAttr(kNumberTypeFloat64)
                                                                                .AddInputAttr(kNumberTypeInt64)
                                                                                .AddInputAttr(kNumberTypeInt64)
                                                                                .AddOutputAttr(kNumberTypeFloat64),
                                                                              CreateSliceKernelPtr<double, int64_t>},
                                                                             {KernelAttr()
                                                                                .AddInputAttr(kNumberTypeFloat32)
                                                                                .AddInputAttr(kNumberTypeInt64)
                                                                                .AddInputAttr(kNumberTypeInt64)
                                                                                .AddOutputAttr(kNumberTypeFloat32),
                                                                              CreateSliceKernelPtr<float, int64_t>},
                                                                             {KernelAttr()
                                                                                .AddInputAttr(kNumberTypeFloat16)
                                                                                .AddInputAttr(kNumberTypeInt64)
                                                                                .AddInputAttr(kNumberTypeInt64)
                                                                                .AddOutputAttr(kNumberTypeFloat16),
                                                                              CreateSliceKernelPtr<half, int64_t>},
                                                                             {KernelAttr()
                                                                                .AddInputAttr(kNumberTypeInt64)
                                                                                .AddInputAttr(kNumberTypeInt64)
                                                                                .AddInputAttr(kNumberTypeInt64)
                                                                                .AddOutputAttr(kNumberTypeInt64),
                                                                              CreateSliceKernelPtr<int64_t, int64_t>},
                                                                             {KernelAttr()
                                                                                .AddInputAttr(kNumberTypeInt32)
                                                                                .AddInputAttr(kNumberTypeInt64)
                                                                                .AddInputAttr(kNumberTypeInt64)
                                                                                .AddOutputAttr(kNumberTypeInt32),
                                                                              CreateSliceKernelPtr<int32_t, int64_t>},
                                                                             {KernelAttr()
                                                                                .AddInputAttr(kNumberTypeInt16)
                                                                                .AddInputAttr(kNumberTypeInt64)
                                                                                .AddInputAttr(kNumberTypeInt64)
                                                                                .AddOutputAttr(kNumberTypeInt16),
                                                                              CreateSliceKernelPtr<int16_t, int64_t>},
                                                                             {KernelAttr()
                                                                                .AddInputAttr(kNumberTypeInt8)
                                                                                .AddInputAttr(kNumberTypeInt64)
                                                                                .AddInputAttr(kNumberTypeInt64)
                                                                                .AddOutputAttr(kNumberTypeInt8),
                                                                              CreateSliceKernelPtr<char, int64_t>},
                                                                             {KernelAttr()
                                                                                .AddInputAttr(kNumberTypeUInt64)
                                                                                .AddInputAttr(kNumberTypeInt64)
                                                                                .AddInputAttr(kNumberTypeInt64)
                                                                                .AddOutputAttr(kNumberTypeUInt64),
                                                                              CreateSliceKernelPtr<uint64_t, int64_t>},
                                                                             {KernelAttr()
                                                                                .AddInputAttr(kNumberTypeUInt32)
                                                                                .AddInputAttr(kNumberTypeInt64)
                                                                                .AddInputAttr(kNumberTypeInt64)
                                                                                .AddOutputAttr(kNumberTypeUInt32),
                                                                              CreateSliceKernelPtr<uint32_t, int64_t>},
                                                                             {KernelAttr()
                                                                                .AddInputAttr(kNumberTypeUInt16)
                                                                                .AddInputAttr(kNumberTypeInt64)
                                                                                .AddInputAttr(kNumberTypeInt64)
                                                                                .AddOutputAttr(kNumberTypeUInt16),
                                                                              CreateSliceKernelPtr<uint16_t, int64_t>},
                                                                             {KernelAttr()
                                                                                .AddInputAttr(kNumberTypeUInt8)
                                                                                .AddInputAttr(kNumberTypeInt64)
                                                                                .AddInputAttr(kNumberTypeInt64)
                                                                                .AddOutputAttr(kNumberTypeUInt8),
                                                                              CreateSliceKernelPtr<uchar, int64_t>},
                                                                             {KernelAttr()
                                                                                .AddInputAttr(kNumberTypeBool)
                                                                                .AddInputAttr(kNumberTypeInt64)
                                                                                .AddInputAttr(kNumberTypeInt64)
                                                                                .AddOutputAttr(kNumberTypeBool),
                                                                              CreateSliceKernelPtr<bool, int64_t>},
                                                                             {KernelAttr()
                                                                                .AddInputAttr(kNumberTypeFloat64)
                                                                                .AddInputAttr(kNumberTypeInt32)
                                                                                .AddInputAttr(kNumberTypeInt32)
                                                                                .AddOutputAttr(kNumberTypeFloat64),
                                                                              CreateSliceKernelPtr<double, int32_t>},
                                                                             {KernelAttr()
                                                                                .AddInputAttr(kNumberTypeFloat32)
                                                                                .AddInputAttr(kNumberTypeInt32)
                                                                                .AddInputAttr(kNumberTypeInt32)
                                                                                .AddOutputAttr(kNumberTypeFloat32),
                                                                              CreateSliceKernelPtr<float, int32_t>},
                                                                             {KernelAttr()
                                                                                .AddInputAttr(kNumberTypeFloat16)
                                                                                .AddInputAttr(kNumberTypeInt32)
                                                                                .AddInputAttr(kNumberTypeInt32)
                                                                                .AddOutputAttr(kNumberTypeFloat16),
                                                                              CreateSliceKernelPtr<half, int32_t>},
                                                                             {KernelAttr()
                                                                                .AddInputAttr(kNumberTypeInt64)
                                                                                .AddInputAttr(kNumberTypeInt32)
                                                                                .AddInputAttr(kNumberTypeInt32)
                                                                                .AddOutputAttr(kNumberTypeInt64),
                                                                              CreateSliceKernelPtr<int64_t, int32_t>},
                                                                             {KernelAttr()
                                                                                .AddInputAttr(kNumberTypeInt32)
                                                                                .AddInputAttr(kNumberTypeInt32)
                                                                                .AddInputAttr(kNumberTypeInt32)
                                                                                .AddOutputAttr(kNumberTypeInt32),
                                                                              CreateSliceKernelPtr<int32_t, int32_t>},
                                                                             {KernelAttr()
                                                                                .AddInputAttr(kNumberTypeInt16)
                                                                                .AddInputAttr(kNumberTypeInt32)
                                                                                .AddInputAttr(kNumberTypeInt32)
                                                                                .AddOutputAttr(kNumberTypeInt16),
                                                                              CreateSliceKernelPtr<int16_t, int32_t>},
                                                                             {KernelAttr()
                                                                                .AddInputAttr(kNumberTypeInt8)
                                                                                .AddInputAttr(kNumberTypeInt32)
                                                                                .AddInputAttr(kNumberTypeInt32)
                                                                                .AddOutputAttr(kNumberTypeInt8),
                                                                              CreateSliceKernelPtr<char, int32_t>},
                                                                             {KernelAttr()
                                                                                .AddInputAttr(kNumberTypeUInt64)
                                                                                .AddInputAttr(kNumberTypeInt32)
                                                                                .AddInputAttr(kNumberTypeInt32)
                                                                                .AddOutputAttr(kNumberTypeUInt64),
                                                                              CreateSliceKernelPtr<uint64_t, int32_t>},
                                                                             {KernelAttr()
                                                                                .AddInputAttr(kNumberTypeUInt32)
                                                                                .AddInputAttr(kNumberTypeInt32)
                                                                                .AddInputAttr(kNumberTypeInt32)
                                                                                .AddOutputAttr(kNumberTypeUInt32),
                                                                              CreateSliceKernelPtr<uint32_t, int32_t>},
                                                                             {KernelAttr()
                                                                                .AddInputAttr(kNumberTypeUInt16)
                                                                                .AddInputAttr(kNumberTypeInt32)
                                                                                .AddInputAttr(kNumberTypeInt32)
                                                                                .AddOutputAttr(kNumberTypeUInt16),
                                                                              CreateSliceKernelPtr<uint16_t, int32_t>},
                                                                             {KernelAttr()
                                                                                .AddInputAttr(kNumberTypeUInt8)
                                                                                .AddInputAttr(kNumberTypeInt32)
                                                                                .AddInputAttr(kNumberTypeInt32)
                                                                                .AddOutputAttr(kNumberTypeUInt8),
                                                                              CreateSliceKernelPtr<uchar, int32_t>},
                                                                             {KernelAttr()
                                                                                .AddInputAttr(kNumberTypeBool)
                                                                                .AddInputAttr(kNumberTypeInt32)
                                                                                .AddInputAttr(kNumberTypeInt32)
                                                                                .AddOutputAttr(kNumberTypeBool),
                                                                              CreateSliceKernelPtr<bool, int32_t>}};
}  // namespace

bool SliceGpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                               const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  std::vector<void *> input_ptrs = ConvertPtrs(inputs);
  std::vector<void *> work_ptrs = ConvertPtrs(workspace);
  std::vector<void *> output_ptrs = ConvertPtrs(outputs);
  if (is_dynamic_attr_ && !get_dynamic_attr_value_) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', fail to get value of the dynamic attr!";
  }
  if (helper_ptr_->Process(input_ptrs, output_ptrs, work_ptrs, stream_ptr) != 0) {
    return false;
  }
  return true;
}

bool SliceGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                             const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::Slice>(base_operator);
  kernel_name_ = kernel_ptr->name();
  auto tensor_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(tensor_attr, GetOpSupport());
  if (!is_match) {
    return false;
  }
  helper_ptr_ = std::move(kernel_attr[index].second(kernel_name_, device_id_));
  (void)CheckParam(inputs, outputs);
  if (!is_dynamic_attr_) {
    size_ = kernel_ptr->get_size();
    begin_ = kernel_ptr->get_begin();
    ProccessAttr(inputs);
  }
  return true;
}

int SliceGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                              const std::vector<KernelTensorPtr> &outputs,
                              const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (is_dynamic_attr_) {
    if (GetDynamicAttrIntValue(inputs, kBeginIndex_, inputsOnHost, kernel_name_, &begin_) &&
        GetDynamicAttrIntValue(inputs, kSizeIndex_, inputsOnHost, kernel_name_, &size_)) {
      get_dynamic_attr_value_ = true;
      ProccessAttr(inputs);
    }
  }
  helper_ptr_->SetKernelParam(attr_ptr_);
  std::vector<std::vector<int64_t>> input_shapes;
  std::vector<std::vector<int64_t>> output_shapes;
  std::transform(inputs.begin(), inputs.end(), std::back_inserter(input_shapes),
                 [](const KernelTensorPtr &input) { return input->GetDeviceShapeAdaptively(); });
  std::vector<int64_t> out_shape = outputs[0]->GetDeviceShapeAdaptively();
  output_shapes.emplace_back(out_shape);
  if (helper_ptr_->CalMemSize(input_shapes, output_shapes) == -1) {
    return KRET_RESIZE_FAILED;
  }
  input_size_list_ = helper_ptr_->GetInputSizeList();
  output_size_list_ = helper_ptr_->GetOutputSizeList();
  workspace_size_list_ = helper_ptr_->GetWorkSizeList();
  return 0;
}

std::vector<KernelAttr> SliceGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(kernel_attr.begin(), kernel_attr.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, SlicePtrCreatorFunc> &item) { return item.first; });
  return support_list;
}

void SliceGpuKernelMod::CheckParam(const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs) {
  size_t input_num = inputs.size();
  constexpr size_t kDynamicSliceInputNum = 3;
  if (input_num != 1 && input_num != kDynamicSliceInputNum) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs must be 1 or " << kDynamicSliceInputNum
                      << ", but got " << input_num;
  }
  if (input_num == kDynamicSliceInputNum) {
    is_dynamic_attr_ = true;
  }
  size_t output_num = outputs.size();
  if (output_num != 1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of outputs must be 1, but got " << output_num;
  }
  auto input_shape = inputs[0]->GetShapeVector();
  const size_t kInputNumUpperLimit = 7;
  if (input_shape.size() > kInputNumUpperLimit) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input cannot be greater than 7, but got "
                      << input_shape.size();
  }
  if (input_shape.size() == 0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input cannot be equal to 0, but got "
                      << input_shape.size();
  }
}

void SliceGpuKernelMod::ProccessAttr(const std::vector<KernelTensorPtr> &inputs) {
  auto input_shape = inputs[0]->GetShapeVector();
  if (size_.size() != input_shape.size() || begin_.size() != input_shape.size()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dimension of size, begin and input_x must be the same, but got the dimension "
                      << "of size: " << size_.size() << ", the dimension of begin: " << begin_.size()
                      << ", the dimension of input_x: " << input_shape.size();
  }
  for (size_t i = 0; i < input_shape.size(); i++) {
    if (size_[i] == -1) {
      size_[i] = input_shape[i] - begin_[i];
    }
    if (input_shape[i] > 0 && size_[i] <= 0) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the element of 'size' must be greater than 0, but got "
                        << "size[" << i << "]: " << size_[i];
    }
  }
  // transpose begin and size for NHWC data
  constexpr auto kIdx2 = 2;
  constexpr auto kIdx3 = 3;
  constexpr auto kIdx4 = 4;
  auto data_format = inputs[0]->GetFormat();
  if (data_format == mindspore::Format::NHWC) {
    std::swap(begin_[1], begin_[kIdx3]);
    std::swap(begin_[1], begin_[kIdx2]);
    std::swap(size_[1], size_[kIdx3]);
    std::swap(size_[1], size_[kIdx2]);
  } else if (data_format == mindspore::Format::NDHWC) {
    std::swap(begin_[1], begin_[kIdx4]);
    std::swap(begin_[1], begin_[kIdx3]);
    std::swap(begin_[1], begin_[kIdx2]);
    std::swap(size_[1], size_[kIdx4]);
    std::swap(size_[1], size_[kIdx3]);
    std::swap(size_[1], size_[kIdx2]);
  }
  attr_ptr_->size = size_;
  attr_ptr_->begin = begin_;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Slice, SliceGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
