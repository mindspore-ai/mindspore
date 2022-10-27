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

#include "plugin/device/gpu/kernel/nn/mirror_pad_gpu_kernel.h"
#include "ops/mirror_pad.h"

namespace mindspore {
namespace kernel {
bool MirrorPadGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::MirrorPad>(base_operator);
  if (kernel_ptr == nullptr) {
    MS_LOG(EXCEPTION) << "cast ExtractVolumePatches ops failed!";
  }
  kernel_name_ = kernel_ptr->name();
  size_t input_num = inputs.size();
  if (input_num != kInputNum) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs must be 2, but got " << input_num;
  }
  size_t output_num = outputs.size();
  if (output_num != 1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of outputs must be 1, but got " << output_num;
  }
  in_type_size_ = GetTypeByte(TypeIdToType(inputs[0]->GetDtype()));
  out_type_size_ = GetTypeByte(TypeIdToType(outputs[0]->GetDtype()));
  string mode = kernel_ptr->get_mode();
  if (mode == "REFLECT") {
    mode_ = 0;  // reflected mirroring
  } else {
    mode_ = 1;  // symmetric mirroring
  }

  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }
  return true;
}

int MirrorPadGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs,
                                  const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  input_size_list_.clear();
  output_size_list_.clear();
  auto input_shape = inputs[0]->GetShapeVector();
  auto padding_shape = inputs[1]->GetShapeVector();
  auto output_shape = outputs[0]->GetShapeVector();
  if (!IsValidShape(input_shape) || !IsValidShape(padding_shape) || !IsValidShape(output_shape)) {
    return static_cast<int>(KRET_UNKNOWN_SHAPE);
  }

  if (input_shape.size() < kDimMin || input_shape.size() > kDimMax) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input_x must be in [2, 4], but "
                      << "got the " << input_shape.size();
  }

  // shape adjustment -> from 2d/3d to 4d to standardize
  if (input_shape.size() == kDimNeedPadBatch) {
    auto it = input_shape.begin();
    (void)input_shape.insert(it, 1);  // batch padding
  } else if (input_shape.size() == kDimNeedPadBatchAndChannel) {
    auto it = input_shape.begin();
    const size_t pos = 2;
    (void)input_shape.insert(it, pos, 1);  // channel padding
  }

  input_size_ = 1;
  for (auto in_shape : input_shape) {
    input_size_ *= LongToSizeClipNeg(in_shape);
    input_shape_.push_back(LongToInt(in_shape));
  }
  num_input_ = input_size_;
  input_size_ *= in_type_size_;

  num_paddings_ = LongToSizeClipNeg(padding_shape[0]);
  input_size_ += IntToSize(kSymmetricCoef) * num_paddings_ * sizeof(int64_t);

  if (output_shape.size() < kOutputDimLowerLimit) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of output cannot be less than 2, but "
                      << "got the " << output_shape.size();
  }
  output_size_ = out_type_size_;
  for (auto x : output_shape) {
    output_size_ *= LongToSizeClipNeg(x);
    output_shape_.push_back(LongToInt(x));
  }
  input_size_list_.push_back(num_input_ * in_type_size_);
  input_size_list_.push_back(kSymmetricCoef * num_paddings_ * sizeof(int64_t));  // for 64 bit int defined in API
  output_size_list_.push_back(output_size_);
  return static_cast<int>(KRET_OK);
}

using KernelRunFunc = MirrorPadGpuKernelMod::KernelRunFunc;
// int the python api description, input data type is number but CalExtractImagePatchesNHWC only support four type.
const std::vector<std::pair<KernelAttr, KernelRunFunc>> &MirrorPadGpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, KernelRunFunc>> func_list = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),
     &MirrorPadGpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat16),
     &MirrorPadGpuKernelMod::LaunchKernel<half>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
     &MirrorPadGpuKernelMod::LaunchKernel<int>},
  };
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, MirrorPad, MirrorPadGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
