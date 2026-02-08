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

#include "kernel/gpu/cuda/random/random_choice_with_mask_gpu_kernel.h"
#include "mindspore/ops/infer/random_choice_with_mask.h"
#include "include/runtime/hardware_abstract/kernel_base/philox_random.h"

namespace mindspore {
namespace kernel {
bool RandomChoiceWithMaskGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                            const std::vector<KernelTensor *> &outputs) {
  constexpr size_t input_num = 1;
  constexpr size_t output_num = 2;

  CHECK_KERNEL_INPUTS_NUM(inputs.size(), input_num, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), output_num, kernel_name_);
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  // init seed_
  uint64_t seed = static_cast<uint64_t>(GetValue<int64_t>(primitive_->GetAttr("seed")));
  uint64_t seed2 = static_cast<uint64_t>(GetValue<int64_t>(primitive_->GetAttr("seed2")));
  seed_ = random::GetSeed(seed, seed2);
  count_ = static_cast<int32_t>(GetValue<int64_t>(primitive_->GetAttr("count")));
  if (primitive_->HasAttr("batch_rank")) {
    batch_rank_ = static_cast<int>(GetValue<int64_t>(primitive_->GetAttr("batch_rank")));
  }
  return true;
}

int RandomChoiceWithMaskGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                             const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  auto input_shape_with_batch = inputs[kIndex0]->GetShapeVector();
  input_shape_size_ = input_shape_with_batch.size() - batch_rank_;
  input_shape_5D_.clear();
  // convert size_t to int
  batch_size_ = 1;
  for (size_t i = 0; i < batch_rank_; i++) {
    batch_size_ *= input_shape_with_batch[i];
  }
  ShapeVector input_shape_without_batch;
  for (auto i = batch_rank_; i < input_shape_size_ + batch_rank_; i++) {
    input_shape_5D_.push_back(input_shape_with_batch[i]);
    input_shape_without_batch.push_back(input_shape_with_batch[i]);
  }
  // convert shape to 5D
  while (input_shape_5D_.size() != MAX_DIMENSION) {
    (void)input_shape_5D_.insert(input_shape_5D_.begin(), 1);
  }

  // init memory
  input_size_ = 1;
  input_size_ *= SizeOf(input_shape_without_batch);
  // upper ceiling for input for ceil_power2
  if (count_ > kSmallK || input_shape_size_ > 1) {
    ceil_power2_ = RcwmRoundUpPower2(input_size_);
  }
  InitWorkSpaceSizeLists();
  return KRET_OK;
}

template <typename T, typename S>
bool RandomChoiceWithMaskGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                                    const std::vector<KernelTensor *> &workspaces,
                                                    const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  T *input = GetDeviceAddress<T>(inputs, 0);
  S *output_index = GetDeviceAddress<S>(outputs, 0);
  T *output_mask = GetDeviceAddress<T>(outputs, 1);
  for (size_t i = 0; i < batch_size_; i++) {
    input += i * input_size_;
    output_index += i * count_ * input_shape_size_;
    output_mask += i * count_;
    cudaError_t status = cudaErrorNotReady;
    if (count_ > kSmallK || input_shape_size_ > 1) {
      S *index_buff = GetDeviceAddress<S>(workspaces, 0);
      S *mask_buff = GetDeviceAddress<S>(workspaces, 1);
      S *rank_buff = GetDeviceAddress<S>(workspaces, 2);
      S *Tnum_buff = GetDeviceAddress<S>(workspaces, 3);
      S *tmp_buff = GetDeviceAddress<S>(workspaces, 4);
      void *States = GetDeviceAddress<void *>(workspaces, 5);
      curandState *devStates = reinterpret_cast<curandState *>(States);
      status =
        CalRandomChoiceWithMask(input_size_, input_shape_size_, input_shape_5D_[kIndex0], input_shape_5D_[kIndex1],
                                input_shape_5D_[kIndex2], input_shape_5D_[kIndex3], input_shape_5D_[kIndex4], seed_,
                                seed_offset_, count_, input, output_index, output_mask, index_buff, mask_buff,
                                rank_buff, Tnum_buff, tmp_buff, devStates, reinterpret_cast<cudaStream_t>(stream_ptr));
    } else {
      status = CalRandomChoiceWithMaskSmall<float, S, T>(input_size_, seed_, seed_offset_, count_, input, output_index,
                                                         output_mask, reinterpret_cast<cudaStream_t>(stream_ptr));
    }
    CHECK_CUDA_STATUS(status, kernel_name_);
  }
  seed_offset_ += 1;
  return true;
}

std::vector<std::pair<KernelAttr, RandomChoiceWithMaskGpuKernelMod::RandomChoiceWithMaskLaunchFunc>>
  RandomChoiceWithMaskGpuKernelMod::func_list_ = {
    {KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeBool),
     &RandomChoiceWithMaskGpuKernelMod::LaunchKernel<bool, int>},
};

std::vector<KernelAttr> RandomChoiceWithMaskGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, RandomChoiceWithMaskGpuKernelMod::RandomChoiceWithMaskLaunchFunc> &pair) {
      return pair.first;
    });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, RandomChoiceWithMask, RandomChoiceWithMaskGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
