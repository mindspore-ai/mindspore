/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "kernel/gpu/cuda/arrays/crop_and_resize_grad_image_gpu_kernel.h"
#include <functional>
namespace mindspore {
namespace kernel {
namespace {
template <typename T, typename G>
std::unique_ptr<cukernel::GpuKernelHelperBase> CreateCropAndResizeGradImageKernelPtr(const std::string &kernel_name,
                                                                                     const uint32_t &device_id) {
  return std::make_unique<cukernel::CropAndResizeGradImageHelperGpuKernel<T, G>>(kernel_name, device_id);
}
using CropAndResizeGradImagePtrCreatorFunc =
  std::function<std::unique_ptr<cukernel::GpuKernelHelperBase>(const std::string &, const uint32_t &)>;

const std::vector<std::pair<KernelAttr, CropAndResizeGradImagePtrCreatorFunc>> kernel_attr = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeFloat16),
   CreateCropAndResizeGradImageKernelPtr<float, half>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeFloat32),
   CreateCropAndResizeGradImageKernelPtr<float, float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeFloat64),
   CreateCropAndResizeGradImageKernelPtr<float, double>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeFloat16),
   CreateCropAndResizeGradImageKernelPtr<double, half>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeFloat32),
   CreateCropAndResizeGradImageKernelPtr<double, float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeFloat64),
   CreateCropAndResizeGradImageKernelPtr<double, double>},
};
}  // namespace

bool CropAndResizeGradImageGpuKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                                const std::vector<KernelTensor *> &workspace,
                                                const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  std::vector<void *> input_ptrs = ConvertPtrs(inputs);
  std::vector<void *> work_ptrs = ConvertPtrs(workspace);
  std::vector<void *> output_ptrs = ConvertPtrs(outputs);
  if (helper_ptr_->Process(input_ptrs, output_ptrs, work_ptrs, stream_ptr) != 0) {
    return false;
  }
  return true;
}

bool CropAndResizeGradImageGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                              const std::vector<KernelTensor *> &outputs) {
  auto tensor_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(tensor_attr, GetOpSupport());
  if (!is_match) {
    return false;
  }
  std::string method = GetValue<std::string>(primitive_->GetAttr("method"));
  if (method == "bilinear") {
    attr_ptr_->method_ = ResizeMethod::LINEAR;
  } else if (method == "nearest") {
    attr_ptr_->method_ = ResizeMethod::NEAREST;
  } else {  //  bilinear-v2
    attr_ptr_->method_ = ResizeMethod::CUBIC;
  }
  helper_ptr_ = std::move(kernel_attr[index].second(kernel_name_, device_id_));
  helper_ptr_->SetKernelParam(attr_ptr_);
  Resize(inputs, outputs);
  return true;
}

int CropAndResizeGradImageGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                               const std::vector<KernelTensor *> &outputs) {
  for (const auto &input : inputs) {
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }
  for (const auto &output : outputs) {
    auto output_shape = output->GetShapeVector();
    if (!IsValidShape(output_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }
  std::vector<std::vector<int64_t>> input_shapes;
  std::vector<std::vector<int64_t>> output_shapes;
  std::vector<int64_t> grads_shape_ = inputs[kGrads]->GetShapeVector();
  std::vector<int64_t> boxes_shape_ = inputs[kBoxes]->GetShapeVector();
  std::vector<int64_t> box_in_shape_ = inputs[kBoxIndex]->GetShapeVector();
  std::vector<int64_t> image_size_shape_ = inputs[kImageSize]->GetShapeVector();
  std::vector<int64_t> output_shape_ = outputs[0]->GetShapeVector();
  input_shapes.emplace_back(grads_shape_);
  input_shapes.emplace_back(boxes_shape_);
  input_shapes.emplace_back(box_in_shape_);
  input_shapes.emplace_back(image_size_shape_);
  output_shapes.emplace_back(output_shape_);
  if (helper_ptr_->CalMemSize(input_shapes, output_shapes) == -1) {
    return KRET_RESIZE_FAILED;
  }
  output_size_list_ = helper_ptr_->GetOutputSizeList();
  workspace_size_list_ = helper_ptr_->GetWorkSizeList();
  return KRET_OK;
}

std::vector<KernelAttr> CropAndResizeGradImageGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    kernel_attr.begin(), kernel_attr.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, CropAndResizeGradImagePtrCreatorFunc> &item) { return item.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, CropAndResizeGradImage, CropAndResizeGradImageGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
