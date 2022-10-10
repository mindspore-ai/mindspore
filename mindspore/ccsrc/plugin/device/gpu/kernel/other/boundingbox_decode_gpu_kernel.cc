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

#include "plugin/device/gpu/kernel/other/boundingbox_decode_gpu_kernel.h"

namespace mindspore {
namespace kernel {
bool BoundingBoxDecodeGpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                         const std::vector<KernelTensorPtr> &inputs,
                                         const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->GetPrim()->name();
  constexpr size_t input_num = 2;
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), input_num, kernel_name_);

  const size_t coordinate_size = 4;
  auto means = base_operator->GetAttr("means");
  if (means->isa<api::ValueSequence>()) {
    means_ = api::GetValue<std::vector<float>>(means);
  } else if (means->isa<api::FloatImm>()) {
    float mean = api::GetValue<float>(means);
    for (size_t i = 0; i < coordinate_size; i++) {
      (void)means_.emplace_back(mean);
    }
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the input 'means' must be a tuple or a list, and dtype must be float, but got is not.";
  }

  auto stds = base_operator->GetAttr("stds");
  if (stds->isa<api::ValueSequence>()) {
    stds_ = api::GetValue<std::vector<float>>(stds);
  } else if (stds->isa<api::FloatImm>()) {
    float std = api::GetValue<float>(stds);
    for (size_t i = 0; i < coordinate_size; i++) {
      (void)stds_.emplace_back(std);
    }
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the input 'stds' must be a tuple or a list, and dtype must be float, but got is not.";
  }

  if (means_.size() < coordinate_size || stds_.size() < coordinate_size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the length of input 'means' and 'stds' must be at least 4, "
                         "but got the length of 'means': "
                      << means_.size() << ", and the length of 'stds': " << stds_.size();
  }

  auto max_shape = base_operator->GetAttr("max_shape");
  std::vector<int64_t> max_shape_me = api::GetValue<std::vector<int64_t>>(max_shape);
  (void)std::transform(max_shape_me.begin(), max_shape_me.end(), std::back_inserter(max_shape_),
                       [](const int64_t &value) { return LongToInt(value); });
  auto wh_ratio_clip = base_operator->GetAttr("wh_ratio_clip");
  wh_ratio_clip_ = api::GetValue<float>(wh_ratio_clip);

  if (max_shape_.size() < kMinMaxShapeSize) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the length of 'max_shape' must be at least 2, but got: " << max_shape_.size();
  }

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int BoundingBoxDecodeGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                          const std::vector<KernelTensorPtr> &inputs,
                                          const std::vector<KernelTensorPtr> &outputs,
                                          const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }
  return KRET_OK;
}

template <typename T>
bool BoundingBoxDecodeGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                 const std::vector<AddressPtr> &workspace,
                                                 const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  T *rois_addr = GetDeviceAddress<T>(inputs, 0);
  T *deltas_addr = GetDeviceAddress<T>(inputs, 1);
  T *bboxes_addr = GetDeviceAddress<T>(outputs, 0);

  if (inputs[0]->size != inputs[1]->size) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', rois box size must equal with deltas box size: " << inputs[1]->size
                  << ", but got " << inputs[0]->size;
    return false;
  }

  const size_t coordinate = 4;
  const size_t block_size = inputs[0]->size / sizeof(T);
  if ((block_size % coordinate) != 0) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << ", the size of the box should be a multiple of 4.";
    return false;
  }
  BoundingBoxDecode(block_size / coordinate, rois_addr, deltas_addr, bboxes_addr, means_[0], means_[1], means_[2],
                    means_[3], stds_[0], stds_[1], stds_[2], stds_[3], max_shape_[0], max_shape_[1], wh_ratio_clip_,
                    reinterpret_cast<cudaStream_t>(stream_ptr));
  return true;
}

std::vector<std::pair<KernelAttr, BoundingBoxDecodeGpuKernelMod::BoundingBoxDecodeLaunchFunc>>
  BoundingBoxDecodeGpuKernelMod::func_list_ = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &BoundingBoxDecodeGpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &BoundingBoxDecodeGpuKernelMod::LaunchKernel<half>}};

std::vector<KernelAttr> BoundingBoxDecodeGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, BoundingBoxDecodeGpuKernelMod::BoundingBoxDecodeLaunchFunc> &pair) {
      return pair.first;
    });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, BoundingBoxDecode, BoundingBoxDecodeGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
