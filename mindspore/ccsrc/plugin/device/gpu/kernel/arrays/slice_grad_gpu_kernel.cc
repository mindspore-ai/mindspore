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

#include "plugin/device/gpu/kernel/arrays/slice_grad_gpu_kernel.h"
#include <memory>
#include <functional>

namespace mindspore {
namespace kernel {
namespace {
void ShapeNdToMd(const ShapeVector &src, ShapeVector *dst, size_t nd_maximum_size) {
  if (src.size() > nd_maximum_size) {
    MS_LOG(ERROR) << src.size() << "-D data is not supported!";
    return;
  }

  for (size_t i = nd_maximum_size; i > 0; --i) {
    dst->push_back(src.size() < i ? 1 : src[src.size() - i]);
  }
}
}  // namespace

using SliceGradPtrCreatorFunc =
  std::function<std::unique_ptr<cukernel::GpuKernelHelperBase>(const std::string &, const uint32_t &)>;

template <typename T, typename S = int64_t>
std::unique_ptr<cukernel::GpuKernelHelperBase> CreateSliceKernelPtr(const std::string &kernel_name,
                                                                    const uint32_t &device_id) {
  return std::make_unique<cukernel::SliceGradHelperGpuKernel<T, S>>(kernel_name, device_id);
}

std::vector<std::pair<KernelAttr, SliceGradPtrCreatorFunc>> kernel_attr = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat64),
   CreateSliceKernelPtr<double>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32),
   CreateSliceKernelPtr<float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat16),
   CreateSliceKernelPtr<half>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt32),
   CreateSliceKernelPtr<int>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt16)
     .AddInputAttr(kNumberTypeInt16)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt16),
   CreateSliceKernelPtr<int16_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt8)
     .AddInputAttr(kNumberTypeUInt8)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeUInt8),
   CreateSliceKernelPtr<uchar>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeBool)
     .AddInputAttr(kNumberTypeBool)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeBool),
   CreateSliceKernelPtr<bool>},
};

std::vector<KernelAttr> SliceGradGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(kernel_attr.begin(), kernel_attr.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, SliceGradPtrCreatorFunc> &item) { return item.first; });
  return support_list;
}

SliceGradGpuKernelMod::SliceGradGpuKernelMod() : kernel_name_("SliceGrad") {
  attr_ptr_ = std::make_shared<cukernel::SliceGradAttr>();
}

bool SliceGradGpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                   const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  std::vector<void *> input_ptrs = ConvertPtrs(inputs);
  std::vector<void *> work_ptrs = ConvertPtrs(workspace);
  std::vector<void *> output_ptrs = ConvertPtrs(outputs);
  return helper_ptr_->Process(input_ptrs, output_ptrs, work_ptrs, stream_ptr) == 0;
}

bool SliceGradGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::SliceGrad>(base_operator);
  kernel_name_ = kernel_ptr->name();
  auto tensor_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(tensor_attr, GetOpSupport());
  if (!is_match) {
    return false;
  }
  helper_ptr_ = std::move(kernel_attr[index].second(kernel_name_, device_id_));
  (void)CheckParam(inputs, outputs);

  return true;
}

int SliceGradGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs,
                                  const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (GetDynamicAttrIntValue(inputs, kBeginIndex_, inputsOnHost, kernel_name_, &begin_) &&
      GetDynamicAttrIntValue(inputs, kSizeIndex_, inputsOnHost, kernel_name_, &size_)) {
    ProccessAttr(inputs);
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
  return KRET_OK;
}

void SliceGradGpuKernelMod::ProccessAttr(const std::vector<KernelTensorPtr> &inputs) {
  auto input_shape = inputs[1]->GetShapeVector();
  auto data_format = inputs[1]->GetFormat();
  auto dy_shape = inputs[0]->GetShapeVector();
  if (dy_shape.size() <= kSliceGradDefaultInputShapeSize) {
    ShapeNdToMd(dy_shape, &dy_shape_, kDim4);
    CalcBeginAndSize(data_format, kSliceGradDefaultInputShapeSize);
  } else {
    ShapeNdToMd(dy_shape, &dy_shape_, kDim7);
    CalcBeginAndSize(data_format, kSliceGradMaxInputShapeSize);
  }
  if (input_shape.size() <= kSliceGradDefaultInputShapeSize) {
    ShapeNdToMd(input_shape, &input_shape_, kDim4);
  } else {
    ShapeNdToMd(input_shape, &input_shape_, kDim7);
  }
  attr_ptr_->size = size_;
  attr_ptr_->begin = begin_;
  attr_ptr_->input_shape = input_shape_;
  int64_t output_num = std::accumulate(dy_shape_.begin(), dy_shape_.end(), 1, std::multiplies<int64_t>());
  attr_ptr_->output_num = output_num;
}

void SliceGradGpuKernelMod::CalcBeginAndSize(const mindspore::Format &data_format, size_t dim) {
  for (auto i = begin_.size(); i < dim; i++) {
    (void)begin_.insert(begin_.begin(), 0);
  }
  for (auto i = size_.size(); i < dim; i++) {
    (void)size_.insert(size_.begin(), 1);
  }
  if (dim == kSliceGradDefaultInputShapeSize && data_format == mindspore::Format::NHWC) {
    std::swap(begin_[1], begin_[3]);
    std::swap(begin_[1], begin_[2]);
    std::swap(size_[1], size_[3]);
    std::swap(size_[1], size_[2]);
  }
  for (size_t i = 0; i != begin_.size(); ++i) {
    if (i < input_shape_.size() && begin_[i] < 0) {
      begin_[i] = begin_[i] + input_shape_[i];
    }
  }
  for (size_t i = 0; i != size_.size(); ++i) {
    if (i < input_shape_.size() && size_[i] < 0) {
      size_[i] = (size_[i] + input_shape_[i]) > 0 ? (size_[i] + input_shape_[i]) : 0;
    }
  }
}

void SliceGradGpuKernelMod::CheckParam(const std::vector<KernelTensorPtr> &inputs,
                                       const std::vector<KernelTensorPtr> &outputs) {
  size_t output_num = outputs.size();
  if (output_num != 1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of outputs must be 1, but got " << output_num;
  }
  auto input_shape = inputs[0]->GetShapeVector();
  if (input_shape.size() > kSliceGradMaxInputShapeSize) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input cannot be greater than 7, but got "
                      << input_shape.size();
  }
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, SliceGrad, SliceGradGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
