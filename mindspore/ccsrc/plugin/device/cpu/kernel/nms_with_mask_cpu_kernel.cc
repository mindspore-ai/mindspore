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

#include "plugin/device/cpu/kernel/nms_with_mask_cpu_kernel.h"
#include <algorithm>
#include <limits>
#include <map>
#include <utility>
#include <vector>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kIndexDataBuff = 0;
constexpr size_t kIndexIndexBuff = 1;
constexpr size_t kIndexRowMask = 2;
constexpr size_t kIndexOutput = 0;
constexpr size_t kIndexSelIdx = 1;
constexpr size_t kIndexSelBoxes = 2;

uint32_t NmsRoundUpPower2(int v) {
  constexpr uint32_t ONE = 1, TWO = 2, FOUR = 4, EIGHT = 8, SIXTEEN = 16;
  v--;
  uint32_t value = IntToUint(v);
  value |= value >> ONE;
  value |= value >> TWO;
  value |= value >> FOUR;
  value |= value >> EIGHT;
  value |= value >> SIXTEEN;
  value++;
  return value;
}

template <typename T>
void Swap(T *lhs, T *rhs) {
  T tmp = lhs[0];
  lhs[0] = rhs[0];
  rhs[0] = tmp;
}
}  // namespace

bool NMSWithMaskCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  if (inputs.size() != INPUT_NUM) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the number of inputs must be 1, but got " << inputs.size()
                  << "input(s).";
  }
  if (outputs.size() != OUTPUT_NUM) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the number of outputs must be 3, but got " << outputs.size()
                  << "output(s).";
  }
  iou_value_ = GetValue<float>(base_operator->GetAttr(kAttrIouThreshold));
  if (auto ret = MatchKernelFunc(base_operator, inputs, outputs); !ret) {
    return ret;
  }
  return true;
}

int NMSWithMaskCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs,
                                    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }
  auto in_shape = inputs[kIndex0]->GetShapeVector();
  num_input_ = LongToInt(in_shape[0]);  //  Get N values in  [N, 5] data.
  ceil_power_2_ = static_cast<size_t>(NmsRoundUpPower2(num_input_));

  workspace_size_list_.push_back(ceil_power_2_ * abstract::TypeIdSize(inputs[kIndex0]->GetDtype()));  //  data buff
  workspace_size_list_.push_back(ceil_power_2_ * sizeof(int));                                        //  index buff
  workspace_size_list_.push_back(IntToSize(num_input_ * num_input_) * sizeof(bool));                  //  mask list
  return KRET_OK;
}

// Sorting function based on BitonicSort from TopK kernel
template <typename T>
void NMSWithMaskCpuKernelMod::NmsBitonicSortByKeyKernel(const int inner, const size_t ceil_power2, const T *input,
                                                        T *data_buff, int *index_buff, int box_size) {
  auto task1 = [this, &data_buff, &index_buff, &input, inner, box_size](int start, int end) {
    for (int i = start; i < end; i++) {
      data_buff[i] = (i < inner) ? input[(i * box_size) + 4] : std::numeric_limits<T>::max();
      index_buff[i] = i;
    }
  };
  ParallelLaunchAutoSearch(task1, ceil_power2, this, &parallel_search_info_);

  for (size_t i = 2; i <= ceil_power2; i <<= 1) {
    for (size_t j = (i >> 1); j > 0; j >>= 1) {
      auto task2 = [i, j, &data_buff, &index_buff](size_t start, size_t end) {
        for (size_t tid = start; tid < end; tid++) {
          size_t tid_comp = tid ^ j;
          if (tid_comp > tid) {
            if ((tid & i) == 0) {
              if (data_buff[tid] > data_buff[tid_comp]) {
                Swap(&data_buff[tid], &data_buff[tid_comp]);
                Swap(&index_buff[tid], &index_buff[tid_comp]);
              }
            } else {
              if (data_buff[tid] < data_buff[tid_comp]) {
                Swap(&data_buff[tid], &data_buff[tid_comp]);
                Swap(&index_buff[tid], &index_buff[tid_comp]);
              }
            }
          }
        }
      };
      ParallelLaunchAutoSearch(task2, ceil_power2, this, &parallel_search_info_);
    }
  }
}

// Initialize per row mask array to all true
void NMSWithMaskCpuKernelMod::MaskInit(size_t numSq, bool *row_mask) {
  auto task = [this, &row_mask](int start, int end) {
    for (int mat_pos = start; mat_pos < end; mat_pos++) {
      row_mask[mat_pos] = true;
    }
  };
  ParallelLaunchAutoSearch(task, numSq, this, &parallel_search_info_);
}

// copy data from input to output array sorted by indices returned from bitonic sort
// flips boxes if asked to,  default - false -> if (x1/y1 > x2/y2)
template <typename T>
void NMSWithMaskCpuKernelMod::PopulateOutput(const T *data_in, T *data_out, const int *index_buff, const int num,
                                             int box_size, bool flip_mode) {
  auto task = [this, &index_buff, &data_in, &data_out, flip_mode, num, box_size](int start, int end) {
    for (int box_num = start; box_num < end; box_num++) {
      int correct_index = index_buff[(num - 1) - box_num];  // flip the array around
      int correct_arr_start = correct_index * box_size;
      int current_arr_start = box_num * box_size;
      if (flip_mode) {  // flip boxes
        // check x
        if (data_in[correct_arr_start + X0] > data_in[correct_arr_start + X1]) {
          data_out[current_arr_start + X0] = data_in[correct_arr_start + X1];
          data_out[current_arr_start + X1] = data_in[correct_arr_start + X0];
        } else {
          data_out[current_arr_start + X0] = data_in[correct_arr_start + X0];
          data_out[current_arr_start + X1] = data_in[correct_arr_start + X1];
        }
        // check y
        if (data_in[correct_arr_start + Y0] > data_in[correct_arr_start + Y1]) {
          data_out[current_arr_start + Y0] = data_in[correct_arr_start + Y1];
          data_out[current_arr_start + Y1] = data_in[correct_arr_start + Y0];
        } else {
          data_out[current_arr_start + Y0] = data_in[correct_arr_start + Y0];
          data_out[current_arr_start + Y1] = data_in[correct_arr_start + Y1];
        }
        data_out[current_arr_start + SCORE] = data_in[correct_arr_start + SCORE];
      } else {  // default behaviour, don't flip
        for (int x = 0; x < box_size; x++) {
          data_out[current_arr_start + x] = data_in[correct_arr_start + x];
        }
      }
    }
  };
  ParallelLaunchAutoSearch(task, IntToSize(num), this, &parallel_search_info_);
}

// populated return mask (init to all true) and return index array
void NMSWithMaskCpuKernelMod::Preprocess(const int num, int *sel_idx, bool *sel_boxes) {
  auto task = [this, &sel_idx, &sel_boxes](int start, int end) {
    for (int box_num = start; box_num < end; box_num++) {
      sel_idx[box_num] = box_num;
      sel_boxes[box_num] = true;
    }
  };
  ParallelLaunchAutoSearch(task, IntToSize(num), this, &parallel_search_info_);
}

template <typename T>
bool NMSWithMaskCpuKernelMod::IouDecision(const T *output, int box_A_start, int box_B_start, float IOU_value) const {
  constexpr int X1_OFFSET = 0;
  constexpr int Y1_OFFSET = 1;
  constexpr int X2_OFFSET = 2;
  constexpr int Y2_OFFSET = 3;
  T x_1 = std::max(output[box_A_start + X1_OFFSET], output[box_B_start + X1_OFFSET]);
  T y_1 = std::max(output[box_A_start + Y1_OFFSET], output[box_B_start + Y1_OFFSET]);
  T x_2 = std::min(output[box_A_start + X2_OFFSET], output[box_B_start + X2_OFFSET]);
  T y_2 = std::min(output[box_A_start + Y2_OFFSET], output[box_B_start + Y2_OFFSET]);
  T width = std::max(x_2 - x_1, T(0));  // in case of no overlap
  T height = std::max(y_2 - y_1, T(0));

  T area1 = (output[box_A_start + X2_OFFSET] - output[box_A_start + X1_OFFSET]) *
            (output[box_A_start + Y2_OFFSET] - output[box_A_start + Y1_OFFSET]);
  T area2 = (output[box_B_start + X2_OFFSET] - output[box_B_start + X1_OFFSET]) *
            (output[box_B_start + Y2_OFFSET] - output[box_B_start + Y1_OFFSET]);

  T combined_area = area1 + area2;
  return !(((width * height) / (combined_area - (width * height))) > static_cast<T>(IOU_value));
}

// Run parallel NMS pass
// Every position in the row_mask array is updated wit correct IOU decision after being init to all True
template <typename T>
void NMSWithMaskCpuKernelMod::NmsPass(const int num, const float IOU_value, const T *output, int box_size,
                                      bool *row_mask) {
  auto task = [this, &row_mask, &output, num, box_size, IOU_value](int start, int end) {
    for (int mask_index = start; mask_index < end; mask_index++) {
      int box_i = mask_index / num;                // row in 2d row_mask array
      int box_j = mask_index % num;                // col in 2d row_mask array
      if (box_j > box_i) {                         // skip when box_j index lower/equal to box_i - will remain true
        int box_i_start_index = box_i * box_size;  // adjust starting indices
        int box_j_start_index = box_j * box_size;
        row_mask[mask_index] = IouDecision<T>(output, box_i_start_index, box_j_start_index, IOU_value);
      }
    }
  };
  ParallelLaunchAutoSearch(task, IntToSize(num * num), this, &parallel_search_info_);
}

// Reduce pass runs on 1 block to allow thread sync
void NMSWithMaskCpuKernelMod::ReducePass(const int num, bool *sel_boxes, const bool *row_mask) {
  // loop over every box in order of high to low confidence score
  for (int i = 0; i < num - 1; ++i) {
    if (!sel_boxes[i]) {
      continue;
    }
    // every thread handles a different set of boxes (per all boxes in order)
    auto task = [this, &sel_boxes, &row_mask, i, num](int start, int end) {
      for (int j = start; j < end; j++) {
        sel_boxes[j] = sel_boxes[j] && row_mask[i * num + j];
      }
    };
    ParallelLaunchAutoSearch(task, IntToSize(num), this, &parallel_search_info_);
  }
}

template <typename T>
bool NMSWithMaskCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                           const std::vector<kernel::AddressPtr> &workspace,
                                           const std::vector<kernel::AddressPtr> &outputs) {
  auto input = reinterpret_cast<T *>(inputs[0]->addr);
  auto data_buff = reinterpret_cast<T *>(workspace[kIndexDataBuff]->addr);
  auto index_buff = reinterpret_cast<int *>(workspace[kIndexIndexBuff]->addr);
  auto row_mask = reinterpret_cast<bool *>(workspace[kIndexRowMask]->addr);
  auto output = reinterpret_cast<T *>(outputs[kIndexOutput]->addr);
  auto sel_idx = reinterpret_cast<int *>(outputs[kIndexSelIdx]->addr);
  auto sel_boxes = reinterpret_cast<bool *>(outputs[kIndexSelBoxes]->addr);

  NmsBitonicSortByKeyKernel<T>(num_input_, ceil_power_2_, input, data_buff, index_buff, box_size_);
  size_t total_val = IntToSize(num_input_ * num_input_);
  MaskInit(total_val, row_mask);
  PopulateOutput<T>(input, output, index_buff, num_input_, box_size_, false);
  Preprocess(num_input_, sel_idx, sel_boxes);
  NmsPass<T>(num_input_, iou_value_, output, box_size_, row_mask);
  ReducePass(num_input_, sel_boxes, row_mask);
  return true;
}

const std::vector<std::pair<KernelAttr, NMSWithMaskCpuKernelMod::KernelRunFunc>> &NMSWithMaskCpuKernelMod::GetFuncList()
  const {
  static const std::vector<std::pair<KernelAttr, NMSWithMaskCpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeBool),
     &NMSWithMaskCpuKernelMod::LaunchKernel<float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeBool),
     &NMSWithMaskCpuKernelMod::LaunchKernel<float16>}};
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, NMSWithMask, NMSWithMaskCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
