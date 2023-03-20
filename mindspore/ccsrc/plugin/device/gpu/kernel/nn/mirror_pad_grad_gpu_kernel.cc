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

#include "plugin/device/gpu/kernel/nn/mirror_pad_grad_gpu_kernel.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kInputNum = 2;
constexpr size_t kOutputNum = 1;
constexpr size_t kIndex1st = 1;
constexpr size_t kIndex2nd = 2;
constexpr size_t kIndex3rd = 3;
constexpr size_t kDimNeedPadBatch = 3;
constexpr size_t kDimNeedPadBatchAndChannel = 2;
constexpr size_t kInputXDimLowerLimit = 4;
constexpr size_t kOutputDimLowerLimit = 2;
constexpr int kSymmetricCoef = 2;
constexpr size_t kIndexForMaxWidth = 3;
constexpr size_t kIndexForMaxHeight = 2;
constexpr size_t kMaxIndexOffset = 2;
constexpr size_t kLongSizeCoeff = 2;
constexpr size_t kChannelPaddingCoeff = 2;
}  // namespace

template <typename T>
bool MirrorPadGradGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                             const std::vector<AddressPtr> &workspace,
                                             const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  T *input = GetDeviceAddress<T>(inputs, 0);
  int64_t *paddings = GetDeviceAddress<int64_t>(inputs, 1);
  T *interim = GetDeviceAddress<T>(workspace, 0);
  T *output = GetDeviceAddress<T>(outputs, 0);

  size_t dx_size = output_size_ / sizeof(T);
  size_t interim_dy_size = workspace_size_ / sizeof(T);
  CalMirrorPadGrad(dx_size, interim_dy_size, input, interim, output_shape_[0], output_shape_[kIndex1st],
                   output_shape_[kIndex2nd], output_shape_[kIndex3rd], input_shape_[kIndex2nd], input_shape_[kIndex3rd],
                   num_paddings_, paddings, mode_, output, reinterpret_cast<cudaStream_t>(stream_ptr));
  return true;
}

bool MirrorPadGradGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                     const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  auto prim = base_operator->GetPrim();
  MS_EXCEPTION_IF_NULL(prim);
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputNum, kernel_name_);
  auto kernel_ptr = std::dynamic_pointer_cast<ops::MirrorPadGrad>(base_operator);
  std::string mode = kernel_ptr->get_mode();
  if (mode == "REFLECT") {
    mode_ = 0;  // reflected mirroring
  } else if (mode == "SYMMETRIC") {
    mode_ = 1;  // symmetric mirroring
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'mode' should be 'REFLECT' or 'SYMMETRIC', but got "
                      << mode;
  }
  input_type_size_ = abstract::TypeIdSize(inputs.at(kIndex0)->GetDtype());
  padding_type_size_ = abstract::TypeIdSize(inputs.at(kIndex1)->GetDtype());
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

void MirrorPadGradGpuKernelMod::CalculateWorkspace(const ShapeVector &input_shape,
                                                   const std::vector<size_t> &output_shape) {
  workspace_size_ = input_type_size_;
  for (int i = 0; i < SizeToInt(kOutputDimLowerLimit); i++) {
    workspace_size_ *= output_shape[i];                        // BATCH, CHANNEL -> Output size
    workspace_size_ *= input_shape[i + kOutputDimLowerLimit];  // WIDTH, HEIGHT -> Input Size
  }
  workspace_size_list_.push_back(workspace_size_);
}

int MirrorPadGradGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs,
                                      const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  auto ret = KRET_OK;
  workspace_size_list_.clear();
  input_size_list_.clear();
  output_size_list_.clear();

  auto input_shape = inputs.at(kIndex0)->GetShapeVector();
  if (!IsValidShape(input_shape)) {
    ret = KRET_UNKNOWN_SHAPE;
    input_size_list_.push_back(input_type_size_);
  } else {
    // shape adjustment -> from 2d/3d to 4d to standardize
    if (input_shape.size() == kDimNeedPadBatch) {
      auto it = input_shape.begin();
      (void)input_shape.insert(it, 1);  // batch padding
    } else if (input_shape.size() == kDimNeedPadBatchAndChannel) {
      auto it = input_shape.begin();
      (void)input_shape.insert(it, kChannelPaddingCoeff, 1);  // channel padding
    }
    if (input_shape.size() < kInputXDimLowerLimit) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input_x cannot be less than 4, but "
                        << "got the " << input_shape.size();
    }
    input_size_ = input_type_size_;
    input_shape_.clear();
    for (auto in_shape : input_shape) {
      input_size_ *= LongToSizeClipNeg(in_shape);
      input_shape_.push_back(LongToInt(in_shape));
    }
    num_input_ = input_size_;
    input_size_list_.push_back(num_input_ * input_type_size_);
  }

  auto padding_shape = inputs.at(kIndex1)->GetShapeVector();
  if (!IsValidShape(padding_shape)) {
    ret = (ret == KRET_OK ? KRET_UNKNOWN_OUT_SHAPE : ret);
    input_size_list_.push_back(padding_type_size_);
  } else {
    // account for paddings in input size -> passed as int64_ts
    num_paddings_ = LongToSizeClipNeg(padding_shape[0]);
    input_size_ += (IntToSize(kSymmetricCoef) * num_paddings_ * padding_type_size_);
    input_size_list_.push_back(kLongSizeCoeff * num_paddings_ * padding_type_size_);  // for 64 bit int defined in API
  }

  auto shape_signed = outputs.at(kIndex0)->GetShapeVector();
  auto output_shape = Convert2SizeTClipNeg(shape_signed);
  if (!IsValidShape(shape_signed)) {
    ret = (ret == KRET_OK ? KRET_UNKNOWN_OUT_SHAPE : ret);
    output_size_list_.push_back(input_type_size_);
  } else {
    if (output_shape.size() == kDimNeedPadBatch) {
      auto it = output_shape.begin();
      (void)output_shape.insert(it, 1);  // batch padding
    } else if (output_shape.size() == kDimNeedPadBatchAndChannel) {
      auto it = output_shape.begin();
      (void)output_shape.insert(it, kChannelPaddingCoeff, 1);  // channel padding
    }
    if (output_shape.size() < kOutputDimLowerLimit) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of output cannot be less than 2, but "
                        << "got the " << output_shape.size();
    }
    output_size_ = input_type_size_;
    output_shape_.clear();
    for (auto x : output_shape) {
      output_size_ *= x;
      output_shape_.push_back(SizeToInt(x));
    }
    output_size_list_.push_back(output_size_);
  }

  // calc workspace size
  // store dy values with accumulation across batch and channel only
  if (ret == KRET_OK) {
    CalculateWorkspace(input_shape, output_shape);
  }
  return static_cast<int>(ret);
}

#define REG_MIRROR_PAD_GRAD_GPU_KERNEL(TypeId1, TypeId2, Type1)                      \
  {                                                                                  \
    KernelAttr().AddInputAttr(TypeId1).AddInputAttr(TypeId2).AddOutputAttr(TypeId1), \
      &MirrorPadGradGpuKernelMod::LaunchKernel<Type1>                                \
  }

std::vector<std::pair<KernelAttr, MirrorPadGradGpuKernelMod::MirrorPadGradLaunchFunc>>
  MirrorPadGradGpuKernelMod::func_list_ = {
    REG_MIRROR_PAD_GRAD_GPU_KERNEL(kNumberTypeFloat64, kNumberTypeInt64, double),
    REG_MIRROR_PAD_GRAD_GPU_KERNEL(kNumberTypeFloat32, kNumberTypeInt64, float),
    REG_MIRROR_PAD_GRAD_GPU_KERNEL(kNumberTypeFloat16, kNumberTypeInt64, half),
    REG_MIRROR_PAD_GRAD_GPU_KERNEL(kNumberTypeInt64, kNumberTypeInt64, int64_t),
    REG_MIRROR_PAD_GRAD_GPU_KERNEL(kNumberTypeInt32, kNumberTypeInt64, int32_t),
    REG_MIRROR_PAD_GRAD_GPU_KERNEL(kNumberTypeInt16, kNumberTypeInt64, int16_t),
    REG_MIRROR_PAD_GRAD_GPU_KERNEL(kNumberTypeInt8, kNumberTypeInt64, int8_t),
    REG_MIRROR_PAD_GRAD_GPU_KERNEL(kNumberTypeUInt64, kNumberTypeInt64, uint64_t),
    REG_MIRROR_PAD_GRAD_GPU_KERNEL(kNumberTypeUInt32, kNumberTypeInt64, uint32_t),
    REG_MIRROR_PAD_GRAD_GPU_KERNEL(kNumberTypeUInt16, kNumberTypeInt64, uint16_t),
    REG_MIRROR_PAD_GRAD_GPU_KERNEL(kNumberTypeUInt8, kNumberTypeInt64, uint8_t),
    REG_MIRROR_PAD_GRAD_GPU_KERNEL(kNumberTypeBool, kNumberTypeInt64, bool),
    REG_MIRROR_PAD_GRAD_GPU_KERNEL(kNumberTypeComplex64, kNumberTypeInt64, utils::Complex<float>),
    REG_MIRROR_PAD_GRAD_GPU_KERNEL(kNumberTypeComplex128, kNumberTypeInt64, utils::Complex<double>),

    REG_MIRROR_PAD_GRAD_GPU_KERNEL(kNumberTypeFloat64, kNumberTypeInt32, double),
    REG_MIRROR_PAD_GRAD_GPU_KERNEL(kNumberTypeFloat32, kNumberTypeInt32, float),
    REG_MIRROR_PAD_GRAD_GPU_KERNEL(kNumberTypeFloat16, kNumberTypeInt32, half),
    REG_MIRROR_PAD_GRAD_GPU_KERNEL(kNumberTypeInt64, kNumberTypeInt32, int64_t),
    REG_MIRROR_PAD_GRAD_GPU_KERNEL(kNumberTypeInt32, kNumberTypeInt32, int32_t),
    REG_MIRROR_PAD_GRAD_GPU_KERNEL(kNumberTypeInt16, kNumberTypeInt32, int16_t),
    REG_MIRROR_PAD_GRAD_GPU_KERNEL(kNumberTypeInt8, kNumberTypeInt32, int8_t),
    REG_MIRROR_PAD_GRAD_GPU_KERNEL(kNumberTypeUInt64, kNumberTypeInt32, uint64_t),
    REG_MIRROR_PAD_GRAD_GPU_KERNEL(kNumberTypeUInt32, kNumberTypeInt32, uint32_t),
    REG_MIRROR_PAD_GRAD_GPU_KERNEL(kNumberTypeUInt16, kNumberTypeInt32, uint16_t),
    REG_MIRROR_PAD_GRAD_GPU_KERNEL(kNumberTypeUInt8, kNumberTypeInt32, uint8_t),
    REG_MIRROR_PAD_GRAD_GPU_KERNEL(kNumberTypeBool, kNumberTypeInt32, bool),
    REG_MIRROR_PAD_GRAD_GPU_KERNEL(kNumberTypeComplex64, kNumberTypeInt32, utils::Complex<float>),
    REG_MIRROR_PAD_GRAD_GPU_KERNEL(kNumberTypeComplex128, kNumberTypeInt32, utils::Complex<double>),
};

std::vector<KernelAttr> MirrorPadGradGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, MirrorPadGradGpuKernelMod::MirrorPadGradLaunchFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, MirrorPadGrad, MirrorPadGradGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
