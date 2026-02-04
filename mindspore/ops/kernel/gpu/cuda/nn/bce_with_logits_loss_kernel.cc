/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "kernel/gpu/cuda/nn/bce_with_logits_loss_kernel.h"
#include <algorithm>
#include "kernel/gpu/cuda_impl/cuda_ops/bce_with_logits_loss_impl.cuh"
#include "kernel/gpu/cuda_impl/cuda_ops/fill_v2_impl.cuh"
#include "kernel/gpu/cuda_impl/cuda_ops/reduce_impl.cuh"
#include "mindapi/base/types.h"

namespace mindspore {
namespace kernel {

const std::map<Reduction, ReduceType_t> kReduceTypeMap = {{Reduction::MEAN, ReduceMean},
                                                          {Reduction::REDUCTION_SUM, ReduceSum}};

bool BCEWithLogitsLossKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &outputs) {
  constexpr size_t input_num = 5;
  constexpr size_t output_num = 1;

  CHECK_KERNEL_INPUTS_NUM(inputs.size(), input_num, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), output_num, kernel_name_);
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  auto input_data_type = inputs[kIndex0]->dtype_id();
  type_id_size_ = abstract::TypeIdSize(input_data_type);
  return true;
}

int BCEWithLogitsLossKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  input_shape_ = inputs[kIndex0]->GetShapeVector();
  input_size_ = SizeOf(input_shape_);
  // extra space for holding extra array shape of input, for broadcasted
  // weight and pos_weight
  workspace_size_list_.push_back(input_size_ * type_id_size_);

  // weight shape
  auto weight_type = inputs[kIndex2]->GetType();
  MS_EXCEPTION_IF_NULL(weight_type);
  if (weight_type->isa<TypeNone>()) {
    // ones_like(input) in launch
    weight_shape_ = input_shape_;
    weight_workspace_index_ = workspace_size_list_.size();
    workspace_size_list_.push_back(type_id_size_);
    workspace_size_list_.push_back(input_size_ * type_id_size_);
  } else {
    weight_shape_ = inputs[kIndex2]->GetShapeVector();
  }
  weight_size_ = SizeOf(weight_shape_);
  weight_need_broadcast_ = NeedBroadcast(&weight_shape_, input_shape_);

  // pos_weight shape
  auto pos_weight_type = inputs[kIndex3]->GetType();
  MS_EXCEPTION_IF_NULL(pos_weight_type);
  if (pos_weight_type->isa<TypeNone>()) {
    pos_weight_shape_ = input_shape_;
    pos_weight_workspace_index_ = workspace_size_list_.size();
    workspace_size_list_.push_back(type_id_size_);
    workspace_size_list_.push_back(input_size_ * type_id_size_);
  } else {
    pos_weight_shape_ = inputs[kIndex3]->GetShapeVector();
  }
  pos_weight_size_ = SizeOf(pos_weight_shape_);
  pos_weight_need_broadcast_ = NeedBroadcast(&pos_weight_shape_, input_shape_);

  auto reduction = static_cast<Reduction>(inputs[kIndex4]->GetValueWithCheck<int64_t>());
  if (reduction == Reduction::MEAN || reduction == Reduction::REDUCTION_SUM) {
    output_tmp_index_ = workspace_size_list_.size();
    workspace_size_list_.push_back(input_size_ * type_id_size_);
    reduce_workspace_index_ = workspace_size_list_.size();
    workspace_size_list_.push_back(input_size_ * type_id_size_);
  }

  return KRET_OK;
}

template <typename T>
bool BCEWithLogitsLossKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                              const std::vector<KernelTensor *> &workspace,
                                              const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  T *predict = GetDeviceAddress<T>(inputs, kIndex0);
  T *target = GetDeviceAddress<T>(inputs, kIndex1);
  T *weight = nullptr;
  T *pos_weight = nullptr;
  auto weight_type = inputs[kIndex2]->GetType();
  MS_EXCEPTION_IF_NULL(weight_type);
  if (weight_type->isa<TypeNone>()) {
    // ones_like
    T *dev_value = GetDeviceAddress<T>(workspace, weight_workspace_index_);
    weight = GetDeviceAddress<T>(workspace, weight_workspace_index_ + 1);

    if constexpr (std::is_same<T, half>::value) {
      float16 host_value = float16(1);
      CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
        cudaMemcpyAsync(dev_value, &host_value, sizeof(float16), cudaMemcpyHostToDevice,
                        reinterpret_cast<cudaStream_t>(stream_ptr)),
        "Memcpy slice data from host to device failed.");
    } else {
      T host_value = T(1);
      CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpyAsync(dev_value, &host_value, sizeof(T), cudaMemcpyHostToDevice,
                                                         reinterpret_cast<cudaStream_t>(stream_ptr)),
                                         "Memcpy slice data from host to device failed.");
    }
    auto status = FillV2(workspace[weight_workspace_index_ + 1]->size(), dev_value, weight, device_id_,
                         reinterpret_cast<cudaStream_t>(stream_ptr));
    CHECK_CUDA_STATUS(status, kernel_name_);
  } else {
    weight = GetDeviceAddress<T>(inputs, kIndex2);
  }

  auto pos_weight_type = inputs[kIndex3]->GetType();
  MS_EXCEPTION_IF_NULL(pos_weight_type);
  if (pos_weight_type->isa<TypeNone>()) {
    // one_like
    T *dev_value = GetDeviceAddress<T>(workspace, pos_weight_workspace_index_);
    pos_weight = GetDeviceAddress<T>(workspace, pos_weight_workspace_index_ + 1);

    if constexpr (std::is_same<T, half>::value) {
      float16 host_value = float16(1);
      CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
        cudaMemcpyAsync(dev_value, &host_value, sizeof(float16), cudaMemcpyHostToDevice,
                        reinterpret_cast<cudaStream_t>(stream_ptr)),
        "Memcpy slice data from host to device failed.");
    } else {
      T host_value = T(1);
      CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpyAsync(dev_value, &host_value, sizeof(T), cudaMemcpyHostToDevice,
                                                         reinterpret_cast<cudaStream_t>(stream_ptr)),
                                         "Memcpy slice data from host to device failed.");
    }

    auto status = FillV2(workspace[pos_weight_workspace_index_ + 1]->size(), dev_value, pos_weight, device_id_,
                         reinterpret_cast<cudaStream_t>(stream_ptr));
    CHECK_CUDA_STATUS(status, kernel_name_);
  } else {
    pos_weight = GetDeviceAddress<T>(inputs, kIndex3);
  }

  T *shape_broadcasted = GetDeviceAddress<T>(workspace, kIndex0);
  T *output = GetDeviceAddress<T>(outputs, kIndex0);

  auto reduction = static_cast<Reduction>(inputs[kIndex4]->GetValueWithCheck<int64_t>());
  if (reduction == Reduction::NONE) {
    auto status =
      CalBCEWithLogitsLoss(input_size_, predict, target, input_shape_, input_shape_.size(), weight, weight_shape_,
                           weight_need_broadcast_, pos_weight, pos_weight_shape_, pos_weight_need_broadcast_,
                           shape_broadcasted, output, reinterpret_cast<cudaStream_t>(stream_ptr));
    CHECK_CUDA_STATUS(status, kernel_name_);
    return true;
  }

  if (reduction == Reduction::MEAN || reduction == Reduction::REDUCTION_SUM) {
    T *output_tmp = GetDeviceAddress<T>(workspace, output_tmp_index_);
    auto status =
      CalBCEWithLogitsLoss(input_size_, predict, target, input_shape_, input_shape_.size(), weight, weight_shape_,
                           weight_need_broadcast_, pos_weight, pos_weight_shape_, pos_weight_need_broadcast_,
                           shape_broadcasted, output_tmp, reinterpret_cast<cudaStream_t>(stream_ptr));
    CHECK_CUDA_STATUS(status, kernel_name_);

    // reduce
    auto reduce_type = kReduceTypeMap.find(reduction)->second;
    std::vector<size_t> input_reshape = {input_size_};
    auto reduce_tmp = GetDeviceAddress<T>(workspace, reduce_workspace_index_);

    status = ArrayReduce(output_tmp, input_reshape, true, reduce_type, reduce_tmp, output,
                         reinterpret_cast<cudaStream_t>(stream_ptr));
    CHECK_CUDA_STATUS(status, kernel_name_);
    return true;
  }

  MS_LOG(EXCEPTION) << "For BCEWithLogits, the value of reduction is invalid.";
  return false;
}

std::vector<std::pair<KernelAttr, BCEWithLogitsLossKernelMod::BCEWithLogitsLossLaunchFunc>>
  BCEWithLogitsLossKernelMod::func_list_ = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddOptionalInputAttr(kNumberTypeFloat16)
       .AddOptionalInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat16),
     &BCEWithLogitsLossKernelMod::LaunchKernel<half>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOptionalInputAttr(kNumberTypeFloat32)
       .AddOptionalInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat32),
     &BCEWithLogitsLossKernelMod::LaunchKernel<float>},
};

std::vector<KernelAttr> BCEWithLogitsLossKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, BCEWithLogitsLossKernelMod::BCEWithLogitsLossLaunchFunc> &pair) {
                         return pair.first;
                       });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, BCEWithLogitsLoss, BCEWithLogitsLossKernelMod);
}  // namespace kernel
}  // namespace mindspore
