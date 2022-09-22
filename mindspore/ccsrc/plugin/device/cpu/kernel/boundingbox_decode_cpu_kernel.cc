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

#include "plugin/device/cpu/kernel/boundingbox_decode_cpu_kernel.h"
#include <utility>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
const size_t kInputRank = 2;
const size_t kLastDim = 4;
}  // namespace

bool BoundingBoxDecodeCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                         const std::vector<KernelTensorPtr> &inputs,
                                         const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
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

  if (max_shape_.size() < MIN_MAX_SHAPE_SIZE) {
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

int BoundingBoxDecodeCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                          const std::vector<KernelTensorPtr> &inputs,
                                          const std::vector<KernelTensorPtr> &outputs,
                                          const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }

  auto anchor_box_shape = LongVecToSizeVec(inputs[kIndex0]->GetShapeVector());
  auto deltas_shape = LongVecToSizeVec(inputs[kIndex1]->GetShapeVector());

  auto it_x = std::find_if(anchor_box_shape.begin(), anchor_box_shape.end(), [](int64_t sh) { return sh <= 0; });
  if (it_x != anchor_box_shape.end()) {
    return KRET_UNKNOWN_SHAPE;
  }

  size_t anchor_box_rank = anchor_box_shape.size();
  size_t deltas_rank = deltas_shape.size();

  if (anchor_box_rank != kInputRank) {
    MS_LOG(ERROR) << "The rank of anchor box must be 2, but got " << anchor_box_rank;
    return KRET_RESIZE_FAILED;
  }

  if (deltas_rank != kInputRank) {
    MS_LOG(ERROR) << "The rank of deltas must be 2, but got " << deltas_rank;
    return KRET_RESIZE_FAILED;
  }

  if (anchor_box_shape[1] != kLastDim) {
    MS_LOG(ERROR) << "The shape of anchor box must be (n, 4), but got the second dimension of " << anchor_box_shape[1];
    return KRET_RESIZE_FAILED;
  }

  if (deltas_shape[1] != kLastDim) {
    MS_LOG(ERROR) << "The shape of deltas must be (n, 4), but got the second dimension of " << deltas_shape[1];
    return KRET_RESIZE_FAILED;
  }

  return KRET_OK;
}

template <typename T>
bool BoundingBoxDecodeCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                                 const std::vector<AddressPtr> &outputs) {
  auto anchor_box = reinterpret_cast<T *>(inputs[0]->addr);
  auto deltas = reinterpret_cast<T *>(inputs[1]->addr);
  auto bboxes = reinterpret_cast<T *>(outputs[0]->addr);

  T ms1 = static_cast<T>(max_shape_[0]);
  T ms2 = static_cast<T>(max_shape_[1]);

  if (inputs[0]->size != inputs[1]->size) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the dtype of inputs 'anchor_box' and 'deltas' must be the same, "
                     "but got the memory size of 'anchor_box': "
                  << inputs[0]->size << " and 'deltas': " << inputs[1]->size;
    return false;
  }

  const size_t coordinate = 4;
  const size_t block_size = inputs[0]->size / sizeof(T);
  if ((block_size % coordinate) != 0) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the memory size of input 'anchor_box' must be a multiple of 4, "
                     "but got the memory size of 'anchor_box': "
                  << inputs[0]->size;
    return false;
  }

  size_t elem_num = block_size / coordinate;
  auto task = [this, &deltas, &anchor_box, &bboxes, ms1, ms2](size_t start, size_t end) {
    constexpr size_t X_INDEX = 0;
    constexpr size_t Y_INDEX = 1;
    constexpr size_t W_INDEX = 2;
    constexpr size_t H_INDEX = 3;
    const T ZERO = static_cast<T>(0);
    const T HALF = static_cast<T>(0.5);
    const T ONE = static_cast<T>(1);
    for (size_t i = start; i < end; i++) {
      const size_t left_x = i * 4;
      const size_t left_y = i * 4 + 1;
      const size_t right_x = i * 4 + 2;
      const size_t right_y = i * 4 + 3;

      T dx = deltas[left_x] * static_cast<T>(stds_[X_INDEX]) + static_cast<T>(means_[X_INDEX]);
      T dy = deltas[left_y] * static_cast<T>(stds_[Y_INDEX]) + static_cast<T>(means_[Y_INDEX]);
      T dw = deltas[right_x] * static_cast<T>(stds_[W_INDEX]) + static_cast<T>(means_[W_INDEX]);
      T dh = deltas[right_y] * static_cast<T>(stds_[H_INDEX]) + static_cast<T>(means_[H_INDEX]);

      T max_ratio = static_cast<T>(abs(log(wh_ratio_clip_)));

      dw = dw > max_ratio ? max_ratio : (dw < (-max_ratio) ? (-max_ratio) : dw);
      dh = dh > max_ratio ? max_ratio : (dh < (-max_ratio) ? (-max_ratio) : dh);

      T px = (anchor_box[left_x] + anchor_box[right_x]) * HALF;
      T py = (anchor_box[left_y] + anchor_box[right_y]) * HALF;
      T pw = anchor_box[right_x] - anchor_box[left_x] + ONE;
      T ph = anchor_box[right_y] - anchor_box[left_y] + ONE;

      T gx = px + pw * dx;
      T gy = py + ph * dy;
      T gw = pw * exp(dw);
      T gh = ph * exp(dh);

      T x1 = gx - gw * HALF + HALF;
      T y1 = gy - gh * HALF + HALF;
      T x2 = gx + gw * HALF - HALF;
      T y2 = gy + gh * HALF - HALF;

      x1 = x1 > ms2 ? ms2 : (x1 < ZERO ? ZERO : x1);
      y1 = y1 > ms1 ? ms1 : (y1 < ZERO ? ZERO : y1);
      x2 = x2 > ms2 ? ms2 : (x2 < ZERO ? ZERO : x2);
      y2 = y2 > ms1 ? ms1 : (y2 < ZERO ? ZERO : y2);

      bboxes[left_x] = x1;
      bboxes[left_y] = y1;
      bboxes[right_x] = x2;
      bboxes[right_y] = y2;
    }
  };
  ParallelLaunchAutoSearch(task, elem_num, this, &parallel_search_info_);
  return true;
}

std::vector<std::pair<KernelAttr, BoundingBoxDecodeCpuKernelMod::BoundingBoxDecodeFunc>>
  BoundingBoxDecodeCpuKernelMod::func_list_ = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &BoundingBoxDecodeCpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &BoundingBoxDecodeCpuKernelMod::LaunchKernel<float16>}};

std::vector<KernelAttr> BoundingBoxDecodeCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, BoundingBoxDecodeFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, BoundingBoxDecode, BoundingBoxDecodeCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
