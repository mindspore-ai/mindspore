/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/triplet_margin_loss_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
void TripletMarginLossCPUKernelMod::InitKernel(const CNodePtr &kernel_node) {
  CheckParam(kernel_node);
  constexpr int kZero = 0;
  constexpr int kOne = 1;
  constexpr int kTwo = 2;
  constexpr int kThree = 3;
  constexpr int kParallel = 28;
  constexpr int kParallelunit = 1024;
  p = common::AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "p");
  swap = common::AnfAlgo::GetNodeAttr<bool>(kernel_node, "swap");
  eps = common::AnfAlgo::GetNodeAttr<float>(kernel_node, "eps");
  reduction = common::AnfAlgo::GetNodeAttr<string>(kernel_node, "reduction");
  dtype_0 = AnfAlgo::GetInputDeviceDataType(kernel_node, kZero);
  dtype_1 = AnfAlgo::GetInputDeviceDataType(kernel_node, kOne);
  dtype_2 = AnfAlgo::GetInputDeviceDataType(kernel_node, kTwo);
  dtype_3 = AnfAlgo::GetInputDeviceDataType(kernel_node, kThree);
  x_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, kZero);
  positive_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, kOne);
  negative_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, kTwo);
  if (AnfAlgo::IsShapesDynamic({x_shape, positive_shape, negative_shape})) {
    return;
  }
  kParallelDataNum = kParallel * kParallelunit;
  auto broadcast_shape_x_and_positive = CPUKernelUtils::GetBroadcastShape(x_shape, positive_shape);
  broadcast_shape = CPUKernelUtils::GetBroadcastShape(broadcast_shape_x_and_positive, negative_shape);
  size_t dim_x = x_shape.size();
  size_t dim_positive = positive_shape.size();
  size_t dim_negative = negative_shape.size();
  size_t max_size = std::max(std::max(dim_x, dim_positive), dim_negative);
  x_reshape_vector = x_shape;
  positive_reshape_vector = positive_shape;
  negative_reshape_vector = negative_shape;
  std::reverse(x_reshape_vector.begin(), x_reshape_vector.end());
  std::reverse(positive_reshape_vector.begin(), positive_reshape_vector.end());
  std::reverse(negative_reshape_vector.begin(), negative_reshape_vector.end());
  if (dim_x < max_size) x_reshape_vector.resize(max_size, kOne);
  if (dim_positive < max_size) positive_reshape_vector.resize(max_size, kOne);
  if (dim_negative < max_size) negative_reshape_vector.resize(max_size, kOne);
  std::reverse(x_reshape_vector.begin(), x_reshape_vector.end());
  std::reverse(positive_reshape_vector.begin(), positive_reshape_vector.end());
  std::reverse(negative_reshape_vector.begin(), negative_reshape_vector.end());
  numelements = LongToSize(SizeOf(broadcast_shape));

  data_num = (numelements) / LongToSize(broadcast_shape[1]);
  data_num_each_batch = (numelements) / LongToSize(broadcast_shape[0]);
  index = data_num / LongToSize(broadcast_shape[0]);
  batch_size = LongToSize(broadcast_shape[0]);
  once_compute_size = LongToSize(broadcast_shape[1]);
  broadcast = false;
  if (x_shape != positive_shape || x_shape != negative_shape || positive_shape != negative_shape) {
    broadcast = true;
  }
}

bool TripletMarginLossCPUKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                           const std::vector<kernel::AddressPtr> &,
                                           const std::vector<kernel::AddressPtr> &outputs) {
  switch (dtype_0) {
    case kNumberTypeFloat16:
      TripletMarginLossCompute_realtype<float>(inputs, outputs);
      break;
    case kNumberTypeFloat32:
      TripletMarginLossCompute_realtype<float>(inputs, outputs);
      break;
    case kNumberTypeFloat64:
      TripletMarginLossCompute_realtype<double>(inputs, outputs);
      break;
    case kNumberTypeInt8:
      TripletMarginLossCompute_realtype<int8_t>(inputs, outputs);
      break;
    case kNumberTypeInt16:
      TripletMarginLossCompute_realtype<int16_t>(inputs, outputs);
      break;
    case kNumberTypeInt32:
      TripletMarginLossCompute_realtype<int32_t>(inputs, outputs);
      break;
    case kNumberTypeInt64:
      TripletMarginLossCompute_realtype<int64_t>(inputs, outputs);
      break;
    case kNumberTypeUInt8:
      TripletMarginLossCompute_realtype<uint8_t>(inputs, outputs);
      break;
    case kNumberTypeUInt16:
      TripletMarginLossCompute_realtype<uint16_t>(inputs, outputs);
      break;
    case kNumberTypeUInt32:
      TripletMarginLossCompute_realtype<uint32_t>(inputs, outputs);
      break;
    case kNumberTypeUInt64:
      TripletMarginLossCompute_realtype<uint64_t>(inputs, outputs);
      break;
    case kNumberTypeComplex64:
      TripletMarginLossCompute_complextype<std::complex<float>>(inputs, outputs);
      break;
    case kNumberTypeComplex128:
      TripletMarginLossCompute_complextype<std::complex<double>>(inputs, outputs);
      break;
    default:
      MS_EXCEPTION(TypeError) << "For TripletMarginLoss, the data type of x and positive and negative is "
                              << TypeIdLabel(dtype_0) << " which is not supported.";
  }
  return true;
}

template <typename T>
void TripletMarginLossCPUKernelMod::TripletMarginLossCompute_realtype(const std::vector<kernel::AddressPtr> &inputs,
                                                                      const std::vector<kernel::AddressPtr> &outputs) {
  auto out_data = reinterpret_cast<float *>(outputs[0]->addr);
  Eigen::Array<float, Eigen::Dynamic, 1> out(data_num, 1);
  float *output_reduction_none_data = reinterpret_cast<float *>(out.data());
  auto task_nobroadcast = [&](size_t start, size_t end) {
    TripletMarginLossCPUKernelMod::realtype_nobroadcast_task<T>(start, end, output_reduction_none_data, inputs,
                                                                outputs);
  };
  auto task_broadcast = [&](size_t start, size_t end) {
    TripletMarginLossCPUKernelMod::realtype_broadcast_task<T>(start, end, output_reduction_none_data, inputs, outputs);
  };
  if (broadcast == true) {
    if (numelements * sizeof(T) > kParallelDataNum) {
      CPUKernelUtils::ParallelFor(task_broadcast, batch_size);
    } else {
      TripletMarginLossCPUKernelMod::realtype_broadcast_compute<T>(output_reduction_none_data, inputs, outputs);
    }
    if (reduction == NONE) {
      for (size_t i = 0; i < data_num; i++) {
        *(out_data + i) = *(output_reduction_none_data + i);
      }
    }
    if (reduction == MEAN) {
      *(out_data) = (out.mean());
    }
    if (reduction == SUM) {
      *(out_data) = (out.sum());
    }
    return;
  }
  if (numelements * sizeof(T) > kParallelDataNum) {
    CPUKernelUtils::ParallelFor(task_nobroadcast, batch_size);
  } else {
    TripletMarginLossCPUKernelMod::realtype_nobroadcast_compute<T>(output_reduction_none_data, inputs, outputs);
  }
  if (reduction == NONE) {
    for (size_t i = 0; i < data_num; i++) {
      *(out_data + i) = *(output_reduction_none_data + i);
    }
  }
  if (reduction == MEAN) {
    *(out_data) = (out.mean());
  }
  if (reduction == SUM) {
    *(out_data) = (out.sum());
  }
  return;
}

template <typename T>
void TripletMarginLossCPUKernelMod::TripletMarginLossCompute_complextype(
  const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs) {
  auto out_data = reinterpret_cast<float *>(outputs[0]->addr);
  Eigen::Array<float, Eigen::Dynamic, 1> out(data_num, 1);
  float *output_reduction_none_data = reinterpret_cast<float *>(out.data());
  auto task_nobroadcast = [&](size_t start, size_t end) {
    TripletMarginLossCPUKernelMod::complextype_nobroadcast_task<T>(start, end, output_reduction_none_data, inputs,
                                                                   outputs);
  };
  auto task_broadcast = [&](size_t start, size_t end) {
    TripletMarginLossCPUKernelMod::complextype_broadcast_task<T>(start, end, output_reduction_none_data, inputs,
                                                                 outputs);
  };
  if (broadcast == true) {
    if (numelements * sizeof(T) > kParallelDataNum) {
      CPUKernelUtils::ParallelFor(task_broadcast, batch_size);
    } else {
      TripletMarginLossCPUKernelMod::complextype_broadcast_compute<T>(output_reduction_none_data, inputs, outputs);
    }
    if (reduction == NONE) {
      for (size_t i = 0; i < data_num; i++) {
        *(out_data + i) = *(output_reduction_none_data + i);
      }
    }
    if (reduction == MEAN) {
      *(out_data) = (out.mean());
    }
    if (reduction == SUM) {
      *(out_data) = (out.sum());
    }
    return;
  }
  if (numelements * sizeof(T) > kParallelDataNum) {
    CPUKernelUtils::ParallelFor(task_nobroadcast, batch_size);
  } else {
    TripletMarginLossCPUKernelMod::complextype_nobroadcast_compute<T>(output_reduction_none_data, inputs, outputs);
  }
  if (reduction == NONE) {
    for (size_t i = 0; i < data_num; i++) {
      *(out_data + i) = *(output_reduction_none_data + i);
    }
  }
  if (reduction == MEAN) {
    *(out_data) = (out.mean());
  }
  if (reduction == SUM) {
    *(out_data) = (out.sum());
  }
  return;
}

template <typename T>
void TripletMarginLossCPUKernelMod::realtype_nobroadcast_task(size_t start, size_t end,
                                                              float *output_reduction_none_data,
                                                              const std::vector<kernel::AddressPtr> &inputs,
                                                              const std::vector<kernel::AddressPtr> &outputs) {
  auto x_addr = reinterpret_cast<T *>(inputs[0]->addr);
  auto positive_addr = reinterpret_cast<T *>(inputs[1]->addr);
  auto negative_addr = reinterpret_cast<T *>(inputs[2]->addr);
  auto margin = *(reinterpret_cast<float *>(inputs[3]->addr));
  const size_t kNoBroadcastValue = 1;
  start *= data_num_each_batch;
  end *= data_num_each_batch;
  Eigen::Array<float, Eigen::Dynamic, 1> calculate_positive(once_compute_size, kNoBroadcastValue);
  Eigen::Array<float, Eigen::Dynamic, 1> calculate_negative(once_compute_size, kNoBroadcastValue);
  Eigen::Array<float, Eigen::Dynamic, 1> calculate_swap(once_compute_size, kNoBroadcastValue);
  float *positive_data = reinterpret_cast<float *>(calculate_positive.data());
  float *negative_data = reinterpret_cast<float *>(calculate_negative.data());
  float *swap_data = reinterpret_cast<float *>(calculate_swap.data());
  size_t once_compute_thread_size = end - start;
  float positive_distance;
  float negative_distance;
  float swap_distance;
  float temp1;
  float temp2;
  float temp3;
  for (size_t n = 0; n < once_compute_thread_size / data_num_each_batch; n++) {
    size_t i = start / data_num_each_batch;
    for (size_t j = 0; j < index; j++) {
      for (size_t m = 0; m < once_compute_size; m++) {
        *(positive_data + m) = (eps);
        *(negative_data + m) = (eps);
        if (swap == true) {
          *(swap_data + m) = (eps);
        }
      }
      for (size_t k = 0; k < once_compute_size; k++) {
        *(positive_data + k) += static_cast<float>(*(x_addr + start + j + k * index)) -
                                static_cast<float>(*(positive_addr + start + j + k * index));
        *(negative_data + k) += static_cast<float>(*(x_addr + start + j + k * index)) -
                                static_cast<float>(*(negative_addr + start + j + k * index));
        if (swap == true) {
          *(swap_data + k) += static_cast<float>(*(positive_addr + start + j + k * index)) -
                              static_cast<float>(*(negative_addr + start + j + k * index));
        }
      }
      calculate_positive = (calculate_positive).abs();
      calculate_negative = (calculate_negative).abs();
      for (size_t a = 0; a < once_compute_size; a++) {
        temp1 = *(positive_data + a);
        temp2 = *(negative_data + a);
        for (int l = 1; l < p; l++) {
          *(positive_data + a) = *(positive_data + a) * temp1;
          *(negative_data + a) = *(negative_data + a) * temp2;
        }
      }
      positive_distance =
        static_cast<float>(std::pow(static_cast<double>(calculate_positive.sum()), (1 / static_cast<float>(p))));
      negative_distance =
        static_cast<float>(std::pow(static_cast<double>(calculate_negative.sum()), (1 / static_cast<float>(p))));
      if (swap == true) {
        calculate_swap = ((calculate_swap)).abs();
        for (size_t a = 0; a < once_compute_size; a++) {
          temp3 = *(swap_data + a);
          for (int l = 1; l < p; l++) {
            *(swap_data + a) = *(swap_data + a) * temp3;
          }
        }
        swap_distance =
          static_cast<float>(std::pow(static_cast<double>(calculate_swap.sum()), (1 / static_cast<float>(p))));
        negative_distance = (negative_distance < swap_distance) ? negative_distance : swap_distance;
      }
      *(output_reduction_none_data + index * i + j) =
        (positive_distance + margin - negative_distance > 0) ? (positive_distance + margin - negative_distance) : 0;
    }
    start += data_num_each_batch;
  }
}

template <typename T>
void TripletMarginLossCPUKernelMod::realtype_broadcast_task(size_t start, size_t end, float *output_reduction_none_data,
                                                            const std::vector<kernel::AddressPtr> &inputs,
                                                            const std::vector<kernel::AddressPtr> &outputs) {
  auto x_addr = reinterpret_cast<T *>(inputs[0]->addr);
  auto positive_addr = reinterpret_cast<T *>(inputs[1]->addr);
  auto negative_addr = reinterpret_cast<T *>(inputs[2]->addr);
  auto margin = *(reinterpret_cast<float *>(inputs[3]->addr));
  start *= data_num_each_batch;
  end *= data_num_each_batch;
  float positive_distance;
  float negative_distance;
  float swap_distance;
  float temp1;
  float temp2;
  size_t once_compute_thread_size = end - start;
  BroadcastIterator broad_base_iter_1(x_shape, positive_shape, broadcast_shape);
  BroadcastIterator broad_base_iter_2(negative_shape, positive_shape, broadcast_shape);
  std::vector<T> x_broadcast(numelements);
  std::vector<T> positive_broadcast(numelements);
  std::vector<T> negative_broadcast(numelements);
  std::vector<float> calculate_positive(once_compute_size);
  std::vector<float> calculate_negative(once_compute_size);
  std::vector<float> calculate_swap(once_compute_size);
  broad_base_iter_1.SetPos(0);
  broad_base_iter_2.SetPos(0);
  for (size_t t = 0; t < numelements; t++) {
    x_broadcast[t] = x_addr[broad_base_iter_1.GetInputPosA()];
    positive_broadcast[t] = positive_addr[broad_base_iter_1.GetInputPosB()];
    negative_broadcast[t] = negative_addr[broad_base_iter_2.GetInputPosA()];
    broad_base_iter_1.GenNextPos();
    broad_base_iter_2.GenNextPos();
  }
  for (size_t n = 0; n < (once_compute_thread_size) / data_num_each_batch; n++) {
    size_t i = start / data_num_each_batch;
    for (size_t j = 0; j < index; j++) {
      float calc_1_sum = 0;
      float calc_2_sum = 0;
      float calc_swap_sum = 0;
      for (size_t k = 0; k < once_compute_size; k++) {
        calculate_positive[k] = abs(static_cast<float>(x_broadcast[start + j + k * index]) -
                                    static_cast<float>(positive_broadcast[start + j + k * index]) + eps);
        calculate_negative[k] = abs(static_cast<float>(x_broadcast[start + j + k * index]) -
                                    static_cast<float>(negative_broadcast[start + j + k * index]) + eps);
        temp1 = calculate_positive[k];
        temp2 = calculate_negative[k];
        for (int l = 1; l < p; l++) {
          calculate_positive[k] = calculate_positive[k] * temp1;
          calculate_negative[k] = calculate_negative[k] * temp2;
        }
        calc_1_sum += calculate_positive[k];
        calc_2_sum += calculate_negative[k];
        TripletMarginLossCPUKernelMod::realtype_swap<T>(start, positive_broadcast, negative_broadcast, calculate_swap,
                                                        j, k, calc_swap_sum, inputs, outputs);
      }
      positive_distance = static_cast<float>(std::pow(static_cast<double>(calc_1_sum), (1 / static_cast<float>(p))));
      if (x_reshape_vector[1] == 1 && positive_reshape_vector[1] == 1 && broadcast_shape[1] != 1) {
        positive_distance /= static_cast<float>(std::pow(broadcast_shape[1], (1 / static_cast<float>(p))));
      }
      negative_distance = static_cast<float>(std::pow(static_cast<double>(calc_2_sum), (1 / static_cast<float>(p))));
      if (x_reshape_vector[1] == 1 && negative_reshape_vector[1] == 1 && broadcast_shape[1] != 1) {
        negative_distance /= static_cast<float>(std::pow(broadcast_shape[1], (1 / static_cast<float>(p))));
      }
      if (swap == true) {
        swap_distance = static_cast<float>(std::pow(static_cast<double>(calc_swap_sum), (1 / static_cast<float>(p))));
        if (positive_reshape_vector[1] == 1 && negative_reshape_vector[1] == 1 && broadcast_shape[1] != 1) {
          swap_distance /= static_cast<float>(std::pow(broadcast_shape[1], (1 / static_cast<float>(p))));
        }
        negative_distance = (negative_distance < swap_distance) ? negative_distance : swap_distance;
      }
      *(output_reduction_none_data + index * i + j) =
        (positive_distance + margin - negative_distance > 0) ? (positive_distance + margin - negative_distance) : 0;
    }
    start += data_num_each_batch;
  }
}

template <typename T>
void TripletMarginLossCPUKernelMod::realtype_broadcast_compute(float *output_reduction_none_data,
                                                               const std::vector<kernel::AddressPtr> &inputs,
                                                               const std::vector<kernel::AddressPtr> &outputs) {
  auto x_addr = reinterpret_cast<T *>(inputs[0]->addr);
  auto positive_addr = reinterpret_cast<T *>(inputs[1]->addr);
  auto negative_addr = reinterpret_cast<T *>(inputs[2]->addr);
  auto margin = *(reinterpret_cast<float *>(inputs[3]->addr));
  float positive_distance;
  float negative_distance;
  float swap_distance;
  float temp1;
  float temp2;
  BroadcastIterator broad_base_iter_1(x_shape, positive_shape, broadcast_shape);
  BroadcastIterator broad_base_iter_2(negative_shape, positive_shape, broadcast_shape);
  std::vector<T> x_broadcast(numelements);
  std::vector<T> positive_broadcast(numelements);
  std::vector<T> negative_broadcast(numelements);
  std::vector<float> calculate_positive(once_compute_size);
  std::vector<float> calculate_negative(once_compute_size);
  std::vector<float> calculate_swap(once_compute_size);
  broad_base_iter_1.SetPos(0);
  broad_base_iter_2.SetPos(0);
  for (size_t t = 0; t < numelements; t++) {
    x_broadcast[t] = x_addr[broad_base_iter_1.GetInputPosA()];
    positive_broadcast[t] = positive_addr[broad_base_iter_1.GetInputPosB()];
    negative_broadcast[t] = negative_addr[broad_base_iter_2.GetInputPosA()];
    broad_base_iter_1.GenNextPos();
    broad_base_iter_2.GenNextPos();
  }
  for (size_t i = 0; i < batch_size; i++) {
    for (size_t j = 0; j < index; j++) {
      float calc_1_sum = 0;
      float calc_2_sum = 0;
      float calc_swap_sum = 0;
      for (size_t k = 0; k < once_compute_size; k++) {
        calculate_positive[k] =
          abs(static_cast<float>(x_broadcast[i * data_num_each_batch + j + k * index]) -
              static_cast<float>(positive_broadcast[i * data_num_each_batch + j + k * index] + eps));
        calculate_negative[k] =
          abs(static_cast<float>(x_broadcast[i * data_num_each_batch + j + k * index]) -
              static_cast<float>(negative_broadcast[i * data_num_each_batch + j + k * index] + eps));
        temp1 = calculate_positive[k];
        temp2 = calculate_negative[k];
        for (int l = 1; l < p; l++) {
          calculate_positive[k] = calculate_positive[k] * temp1;
          calculate_negative[k] = calculate_negative[k] * temp2;
        }
        calc_1_sum += calculate_positive[k];
        calc_2_sum += calculate_negative[k];
        TripletMarginLossCPUKernelMod::realtype_swap<T>(i * data_num_each_batch, positive_broadcast, negative_broadcast,
                                                        calculate_swap, j, k, calc_swap_sum, inputs, outputs);
      }
      positive_distance = static_cast<float>(std::pow(static_cast<double>(calc_1_sum), (1 / static_cast<float>(p))));
      if (x_reshape_vector[1] == 1 && positive_reshape_vector[1] == 1 && broadcast_shape[1] != 1) {
        positive_distance /= static_cast<float>(std::pow(broadcast_shape[1], (1 / static_cast<float>(p))));
      }
      negative_distance = static_cast<float>(std::pow(static_cast<double>(calc_2_sum), (1 / static_cast<float>(p))));
      if (x_reshape_vector[1] == 1 && negative_reshape_vector[1] == 1 && broadcast_shape[1] != 1) {
        negative_distance /= static_cast<float>(std::pow(broadcast_shape[1], (1 / static_cast<float>(p))));
      }
      if (swap == true) {
        swap_distance = static_cast<float>(std::pow(static_cast<double>(calc_swap_sum), (1 / static_cast<float>(p))));
        if (positive_reshape_vector[1] == 1 && negative_reshape_vector[1] == 1 && broadcast_shape[1] != 1) {
          swap_distance /= static_cast<float>(std::pow(broadcast_shape[1], (1 / static_cast<float>(p))));
        }
        negative_distance = (negative_distance < swap_distance) ? negative_distance : swap_distance;
      }
      *(output_reduction_none_data + index * i + j) =
        (positive_distance + margin - negative_distance > 0) ? (positive_distance + margin - negative_distance) : 0;
    }
  }
}

template <typename T>
void TripletMarginLossCPUKernelMod::realtype_nobroadcast_compute(float *output_reduction_none_data,
                                                                 const std::vector<kernel::AddressPtr> &inputs,
                                                                 const std::vector<kernel::AddressPtr> &outputs) {
  auto x_addr = reinterpret_cast<T *>(inputs[0]->addr);
  auto positive_addr = reinterpret_cast<T *>(inputs[1]->addr);
  auto negative_addr = reinterpret_cast<T *>(inputs[2]->addr);
  auto margin = *(reinterpret_cast<float *>(inputs[3]->addr));
  const size_t kNoBroadcastValue = 1;
  Eigen::Array<float, Eigen::Dynamic, 1> calculate_positive(once_compute_size, kNoBroadcastValue);
  Eigen::Array<float, Eigen::Dynamic, 1> calculate_negative(once_compute_size, kNoBroadcastValue);
  Eigen::Array<float, Eigen::Dynamic, 1> calculate_swap(once_compute_size, kNoBroadcastValue);
  float *positive_data = reinterpret_cast<float *>(calculate_positive.data());
  float *negative_data = reinterpret_cast<float *>(calculate_negative.data());
  float *swap_data = reinterpret_cast<float *>(calculate_swap.data());
  float positive_distance;
  float negative_distance;
  float swap_distance;
  float temp1;
  float temp2;
  float temp3;
  for (size_t i = 0; i < batch_size; i++) {
    for (size_t j = 0; j < index; j++) {
      for (size_t m = 0; m < once_compute_size; m++) {
        *(positive_data + m) = (eps);
        *(negative_data + m) = (eps);
        if (swap == true) {
          *(swap_data + m) = (eps);
        }
      }
      for (size_t k = 0; k < once_compute_size; k++) {
        *(positive_data + k) += static_cast<float>(*(x_addr + i * data_num_each_batch + j + k * index)) -
                                static_cast<float>(*(positive_addr + i * data_num_each_batch + j + k * index));
        *(negative_data + k) += static_cast<float>(*(x_addr + i * data_num_each_batch + j + k * index)) -
                                static_cast<float>(*(negative_addr + i * data_num_each_batch + j + k * index));
        if (swap == true) {
          *(swap_data + k) += static_cast<float>(*(positive_addr + i * data_num_each_batch + j + k * index)) -
                              static_cast<float>(*(negative_addr + i * data_num_each_batch + j + k * index));
        }
      }
      calculate_positive = (calculate_positive).abs();
      calculate_negative = (calculate_negative).abs();
      for (size_t n = 0; n < once_compute_size; n++) {
        temp1 = *(positive_data + n);
        temp2 = *(negative_data + n);
        for (int l = 1; l < p; l++) {
          *(positive_data + n) = *(positive_data + n) * temp1;
          *(negative_data + n) = *(negative_data + n) * temp2;
        }
      }
      positive_distance =
        static_cast<float>(std::pow(static_cast<double>(calculate_positive.sum()), (1 / static_cast<float>(p))));
      negative_distance =
        static_cast<float>(std::pow(static_cast<double>(calculate_negative.sum()), (1 / static_cast<float>(p))));
      if (swap == true) {
        calculate_swap = ((calculate_swap)).abs();
        for (size_t n = 0; n < once_compute_size; n++) {
          temp3 = *(swap_data + n);
          for (int l = 1; l < p; l++) {
            *(swap_data + n) = *(swap_data + n) * temp3;
          }
        }
        swap_distance =
          static_cast<float>(std::pow(static_cast<double>(calculate_swap.sum()), (1 / static_cast<float>(p))));
        negative_distance = (negative_distance < swap_distance) ? negative_distance : swap_distance;
      }
      *(output_reduction_none_data + index * i + j) =
        (positive_distance + margin - negative_distance > 0) ? (positive_distance + margin - negative_distance) : 0;
    }
  }
}

template <typename T>
void TripletMarginLossCPUKernelMod::complextype_nobroadcast_task(size_t start, size_t end,
                                                                 float *output_reduction_none_data,
                                                                 const std::vector<kernel::AddressPtr> &inputs,
                                                                 const std::vector<kernel::AddressPtr> &outputs) {
  auto x_addr = reinterpret_cast<T *>(inputs[0]->addr);
  auto positive_addr = reinterpret_cast<T *>(inputs[1]->addr);
  auto negative_addr = reinterpret_cast<T *>(inputs[2]->addr);
  auto margin = *(reinterpret_cast<float *>(inputs[3]->addr));
  const size_t kNoBroadcastValue = 1;
  start *= data_num_each_batch;
  end *= data_num_each_batch;
  Eigen::Array<T, Eigen::Dynamic, 1> calculate_positive(once_compute_size, kNoBroadcastValue);
  Eigen::Array<T, Eigen::Dynamic, 1> calculate_negative(once_compute_size, kNoBroadcastValue);
  Eigen::Array<T, Eigen::Dynamic, 1> calculate_swap(once_compute_size, kNoBroadcastValue);
  T *positive_data = reinterpret_cast<T *>(calculate_positive.data());
  T *negative_data = reinterpret_cast<T *>(calculate_negative.data());
  T *swap_data = reinterpret_cast<T *>(calculate_swap.data());
  size_t once_compute_thread_size = end - start;
  float positive_distance;
  float negative_distance;
  float swap_distance;
  for (size_t n = 0; n < (once_compute_thread_size) / data_num_each_batch; n++) {
    size_t i = start / data_num_each_batch;
    for (size_t j = 0; j < index; j++) {
      for (size_t m = 0; m < once_compute_size; m++) {
        *(positive_data + m) = (eps);
        *(negative_data + m) = (eps);
        if (swap == true) {
          *(swap_data + m) = (eps);
        }
      }
      for (size_t k = 0; k < once_compute_size; k++) {
        *(positive_data + k) += (*(x_addr + start + j + k * index)) - (*(positive_addr + start + j + k * index));
        *(negative_data + k) += (*(x_addr + start + j + k * index)) - (*(negative_addr + start + j + k * index));
        if (swap == true) {
          *(swap_data + k) += (*(positive_addr + start + j + k * index)) - (*(negative_addr + start + j + k * index));
        }
      }
      auto calculate_positive_float =
        (calculate_positive * (calculate_positive.matrix().conjugate().array())).real().sqrt();
      auto calculate_negative_float =
        (calculate_negative * (calculate_negative.matrix().conjugate().array())).real().sqrt();
      positive_distance =
        static_cast<float>(std::pow(calculate_positive_float.pow(p).sum(), 1 / static_cast<float>(p)));
      negative_distance =
        static_cast<float>(std::pow(calculate_negative_float.pow(p).sum(), 1 / static_cast<float>(p)));
      if (swap == true) {
        auto calculate_swap_float = (calculate_swap * (calculate_swap.matrix().conjugate().array())).real().sqrt();
        swap_distance = static_cast<float>(std::pow(calculate_swap_float.pow(p).sum(), 1 / static_cast<float>(p)));
        negative_distance = (negative_distance < swap_distance) ? negative_distance : swap_distance;
      }
      *(output_reduction_none_data + index * i + j) =
        (positive_distance + margin - negative_distance > 0) ? (positive_distance + margin - negative_distance) : 0;
    }
    start += data_num_each_batch;
  }
}

template <typename T>
void TripletMarginLossCPUKernelMod::complextype_broadcast_task(size_t start, size_t end,
                                                               float *output_reduction_none_data,
                                                               const std::vector<kernel::AddressPtr> &inputs,
                                                               const std::vector<kernel::AddressPtr> &outputs) {
  auto x_addr = reinterpret_cast<T *>(inputs[0]->addr);
  auto positive_addr = reinterpret_cast<T *>(inputs[1]->addr);
  auto negative_addr = reinterpret_cast<T *>(inputs[2]->addr);
  auto margin = *(reinterpret_cast<float *>(inputs[3]->addr));
  start *= data_num_each_batch;
  end *= data_num_each_batch;
  float positive_distance;
  float negative_distance;
  float swap_distance;
  float temp1;
  float temp2;
  size_t once_compute_thread_size = end - start;
  BroadcastIterator broad_base_iter_1(x_shape, positive_shape, broadcast_shape);
  BroadcastIterator broad_base_iter_2(negative_shape, positive_shape, broadcast_shape);
  std::vector<T> x_broadcast(numelements);
  std::vector<T> positive_broadcast(numelements);
  std::vector<T> negative_broadcast(numelements);
  std::vector<T> calculate_positive(once_compute_size);
  std::vector<T> calculate_negative(once_compute_size);
  std::vector<T> calculate_swap(once_compute_size);
  broad_base_iter_1.SetPos(0);
  broad_base_iter_2.SetPos(0);
  for (size_t t = 0; t < numelements; t++) {
    x_broadcast[t] = x_addr[broad_base_iter_1.GetInputPosA()];
    positive_broadcast[t] = positive_addr[broad_base_iter_1.GetInputPosB()];
    negative_broadcast[t] = negative_addr[broad_base_iter_2.GetInputPosA()];
    broad_base_iter_1.GenNextPos();
    broad_base_iter_2.GenNextPos();
  }
  for (size_t n = 0; n < (once_compute_thread_size) / data_num_each_batch; n++) {
    size_t i = start / data_num_each_batch;
    for (size_t j = 0; j < index; j++) {
      float calc_1_sum = 0;
      float calc_2_sum = 0;
      float calc_swap_sum = 0;
      for (size_t k = 0; k < once_compute_size; k++) {
        calculate_positive[k] =
          x_broadcast[start + j + k * index] - positive_broadcast[start + j + k * index] + static_cast<T>(eps);
        calculate_negative[k] =
          x_broadcast[start + j + k * index] - negative_broadcast[start + j + k * index] + static_cast<T>(eps);
        float calculate_positive_float =
          static_cast<float>(sqrt((calculate_positive[k].real() * calculate_positive[k].real() +
                                   calculate_positive[k].imag() * calculate_positive[k].imag())));
        float calculate_negative_float =
          static_cast<float>(sqrt((calculate_negative[k].real() * calculate_negative[k].real() +
                                   calculate_negative[k].imag() * calculate_negative[k].imag())));
        temp1 = calculate_positive_float;
        temp2 = calculate_negative_float;
        for (int l = 1; l < p; l++) {
          calculate_positive_float = calculate_positive_float * temp1;
          calculate_negative_float = calculate_negative_float * temp2;
        }
        calc_1_sum += calculate_positive_float;
        calc_2_sum += calculate_negative_float;
        TripletMarginLossCPUKernelMod::complextype_swap<T>(start, positive_broadcast, negative_broadcast,
                                                           calculate_swap, j, k, calc_swap_sum, inputs, outputs);
      }
      positive_distance = static_cast<float>(std::pow(static_cast<double>(calc_1_sum), (1 / static_cast<float>(p))));
      if (x_reshape_vector[1] == 1 && positive_reshape_vector[1] == 1 && broadcast_shape[1] != 1) {
        positive_distance /= static_cast<float>(std::pow(broadcast_shape[1], (1 / static_cast<float>(p))));
      }
      negative_distance = static_cast<float>(std::pow(static_cast<double>(calc_2_sum), (1 / static_cast<float>(p))));
      if (x_reshape_vector[1] == 1 && negative_reshape_vector[1] == 1 && broadcast_shape[1] != 1) {
        negative_distance /= static_cast<float>(std::pow(broadcast_shape[1], (1 / static_cast<float>(p))));
      }
      if (swap == true) {
        swap_distance = static_cast<float>(std::pow(static_cast<double>(calc_swap_sum), (1 / static_cast<float>(p))));
        if (positive_reshape_vector[1] == 1 && negative_reshape_vector[1] == 1 && broadcast_shape[1] != 1) {
          swap_distance /= static_cast<float>(std::pow(broadcast_shape[1], (1 / static_cast<float>(p))));
        }
        negative_distance = (negative_distance < swap_distance) ? negative_distance : swap_distance;
      }
      *(output_reduction_none_data + index * i + j) =
        (positive_distance + margin - negative_distance > 0) ? (positive_distance + margin - negative_distance) : 0;
    }
    start += data_num_each_batch;
  }
}

template <typename T>
void TripletMarginLossCPUKernelMod::complextype_broadcast_compute(float *output_reduction_none_data,
                                                                  const std::vector<kernel::AddressPtr> &inputs,
                                                                  const std::vector<kernel::AddressPtr> &outputs) {
  auto x_addr = reinterpret_cast<T *>(inputs[0]->addr);
  auto positive_addr = reinterpret_cast<T *>(inputs[1]->addr);
  auto negative_addr = reinterpret_cast<T *>(inputs[2]->addr);
  auto margin = *(reinterpret_cast<float *>(inputs[3]->addr));
  float positive_distance;
  float negative_distance;
  float swap_distance;
  float temp1;
  float temp2;
  BroadcastIterator broad_base_iter_1(x_shape, positive_shape, broadcast_shape);
  BroadcastIterator broad_base_iter_2(negative_shape, positive_shape, broadcast_shape);
  std::vector<T> x_broadcast(numelements);
  std::vector<T> positive_broadcast(numelements);
  std::vector<T> negative_broadcast(numelements);
  std::vector<T> calculate_positive(once_compute_size);
  std::vector<T> calculate_negative(once_compute_size);
  std::vector<T> calculate_swap(once_compute_size);
  broad_base_iter_1.SetPos(0);
  broad_base_iter_2.SetPos(0);
  for (size_t t = 0; t < numelements; t++) {
    x_broadcast[t] = x_addr[broad_base_iter_1.GetInputPosA()];
    positive_broadcast[t] = positive_addr[broad_base_iter_1.GetInputPosB()];
    negative_broadcast[t] = negative_addr[broad_base_iter_2.GetInputPosA()];
    broad_base_iter_1.GenNextPos();
    broad_base_iter_2.GenNextPos();
  }
  for (size_t i = 0; i < batch_size; i++) {
    for (size_t j = 0; j < index; j++) {
      float calc_1_sum = 0;
      float calc_2_sum = 0;
      float calc_swap_sum = 0;
      for (size_t k = 0; k < once_compute_size; k++) {
        calculate_positive[k] = x_broadcast[i * data_num_each_batch + j + k * index] -
                                positive_broadcast[i * data_num_each_batch + j + k * index] + static_cast<T>(eps);
        calculate_negative[k] = x_broadcast[i * data_num_each_batch + j + k * index] -
                                negative_broadcast[i * data_num_each_batch + j + k * index] + static_cast<T>(eps);
        float calculate_positive_float =
          static_cast<float>(sqrt((calculate_positive[k].real() * calculate_positive[k].real() +
                                   calculate_positive[k].imag() * calculate_positive[k].imag())));
        float calculate_negative_float =
          static_cast<float>(sqrt((calculate_negative[k].real() * calculate_negative[k].real() +
                                   calculate_negative[k].imag() * calculate_negative[k].imag())));
        temp1 = calculate_positive_float;
        temp2 = calculate_negative_float;
        for (int l = 1; l < p; l++) {
          calculate_positive_float = calculate_positive_float * temp1;
          calculate_negative_float = calculate_negative_float * temp2;
        }
        calc_1_sum += calculate_positive_float;
        calc_2_sum += calculate_negative_float;
        TripletMarginLossCPUKernelMod::complextype_swap<T>(i * data_num_each_batch, positive_broadcast,
                                                           negative_broadcast, calculate_swap, j, k, calc_swap_sum,
                                                           inputs, outputs);
      }
      positive_distance = static_cast<float>(std::pow(static_cast<double>(calc_1_sum), (1 / static_cast<float>(p))));
      if (x_reshape_vector[1] == 1 && positive_reshape_vector[1] == 1 && broadcast_shape[1] != 1) {
        positive_distance /= static_cast<float>(std::pow(broadcast_shape[1], (1 / static_cast<float>(p))));
      }
      negative_distance = static_cast<float>(std::pow(static_cast<double>(calc_2_sum), (1 / static_cast<float>(p))));
      if (x_reshape_vector[1] == 1 && negative_reshape_vector[1] == 1 && broadcast_shape[1] != 1) {
        negative_distance /= static_cast<float>(std::pow(broadcast_shape[1], (1 / static_cast<float>(p))));
      }
      if (swap == true) {
        swap_distance = static_cast<float>(std::pow(static_cast<double>(calc_swap_sum), (1 / static_cast<float>(p))));
        if (positive_reshape_vector[1] == 1 && negative_reshape_vector[1] == 1 && broadcast_shape[1] != 1) {
          swap_distance /= static_cast<float>(std::pow(broadcast_shape[1], (1 / static_cast<float>(p))));
        }
        negative_distance = (negative_distance < swap_distance) ? negative_distance : swap_distance;
      }
      *(output_reduction_none_data + index * i + j) =
        (positive_distance + margin - negative_distance > 0) ? (positive_distance + margin - negative_distance) : 0;
    }
  }
}

template <typename T>
void TripletMarginLossCPUKernelMod::complextype_nobroadcast_compute(float *output_reduction_none_data,
                                                                    const std::vector<kernel::AddressPtr> &inputs,
                                                                    const std::vector<kernel::AddressPtr> &outputs) {
  auto x_addr = reinterpret_cast<T *>(inputs[0]->addr);
  auto positive_addr = reinterpret_cast<T *>(inputs[1]->addr);
  auto negative_addr = reinterpret_cast<T *>(inputs[2]->addr);
  auto margin = *(reinterpret_cast<float *>(inputs[3]->addr));
  const size_t kNoBroadcastValue = 1;
  Eigen::Array<T, Eigen::Dynamic, 1> calculate_positive(once_compute_size, kNoBroadcastValue);
  Eigen::Array<T, Eigen::Dynamic, 1> calculate_negative(once_compute_size, kNoBroadcastValue);
  Eigen::Array<T, Eigen::Dynamic, 1> calculate_swap(once_compute_size, kNoBroadcastValue);
  T *positive_data = reinterpret_cast<T *>(calculate_positive.data());
  T *negative_data = reinterpret_cast<T *>(calculate_negative.data());
  T *swap_data = reinterpret_cast<T *>(calculate_swap.data());
  float positive_distance;
  float negative_distance;
  float swap_distance;
  for (size_t i = 0; i < batch_size; i++) {
    for (size_t j = 0; j < index; j++) {
      for (size_t m = 0; m < once_compute_size; m++) {
        *(positive_data + m) = (eps);
        *(negative_data + m) = (eps);
        if (swap == true) {
          *(swap_data + m) = (eps);
        }
      }
      for (size_t k = 0; k < once_compute_size; k++) {
        *(positive_data + k) += (*(x_addr + i * data_num_each_batch + j + k * index)) -
                                (*(positive_addr + i * data_num_each_batch + j + k * index));
        *(negative_data + k) += (*(x_addr + i * data_num_each_batch + j + k * index)) -
                                (*(negative_addr + i * data_num_each_batch + j + k * index));
        if (swap == true) {
          *(swap_data + k) += (*(positive_addr + i * data_num_each_batch + j + k * index)) -
                              (*(negative_addr + i * data_num_each_batch + j + k * index));
        }
      }
      auto calculate_positive_float =
        (calculate_positive * (calculate_positive.matrix().conjugate().array())).real().sqrt();
      auto calculate_negative_float =
        (calculate_negative * (calculate_negative.matrix().conjugate().array())).real().sqrt();
      positive_distance =
        static_cast<float>(std::pow(calculate_positive_float.pow(p).sum(), 1 / static_cast<float>(p)));
      negative_distance =
        static_cast<float>(std::pow(calculate_negative_float.pow(p).sum(), 1 / static_cast<float>(p)));
      if (swap == true) {
        auto calculate_swap_float = (calculate_swap * (calculate_swap.matrix().conjugate().array())).real().sqrt();
        swap_distance = static_cast<float>(std::pow(calculate_swap_float.pow(p).sum(), 1 / static_cast<float>(p)));
        negative_distance = (negative_distance < swap_distance) ? negative_distance : swap_distance;
      }
      *(output_reduction_none_data + index * i + j) =
        (positive_distance + margin - negative_distance > 0) ? (positive_distance + margin - negative_distance) : 0;
    }
  }
}

template <typename T>
void TripletMarginLossCPUKernelMod::realtype_swap(size_t start, std::vector<T> &positive_broadcast,
                                                  std::vector<T> &negative_broadcast,
                                                  std::vector<float> &calculate_swap, size_t j, size_t k,
                                                  float &calc_swap_sum, const std::vector<kernel::AddressPtr> &inputs,
                                                  const std::vector<kernel::AddressPtr> &outputs) {
  if (swap == true) {
    calculate_swap[k] = abs(static_cast<float>(positive_broadcast[start + j + k * index]) -
                            static_cast<float>(negative_broadcast[start + j + k * index]) + eps);
    float temp3 = calculate_swap[k];
    for (int l = 1; l < p; l++) {
      calculate_swap[k] = calculate_swap[k] * temp3;
    }
    calc_swap_sum += calculate_swap[k];
  }
}

template <typename T>
void TripletMarginLossCPUKernelMod::complextype_swap(size_t start, std::vector<T> &positive_broadcast,
                                                     std::vector<T> &negative_broadcast, std::vector<T> &calculate_swap,
                                                     size_t j, size_t k, float &calc_swap_sum,
                                                     const std::vector<kernel::AddressPtr> &inputs,
                                                     const std::vector<kernel::AddressPtr> &outputs) {
  if (swap == true) {
    calculate_swap[k] =
      positive_broadcast[start + j + k * index] - negative_broadcast[start + j + k * index] + static_cast<T>(eps);
    float calculate_swap_float = static_cast<float>(sqrt(
      (calculate_swap[k].real() * calculate_swap[k].real() + calculate_swap[k].imag() * calculate_swap[k].imag())));
    float temp3 = calculate_swap_float;
    for (int l = 1; l < p; l++) {
      calculate_swap_float = calculate_swap_float * temp3;
    }
    calc_swap_sum += calculate_swap_float;
  }
}

void TripletMarginLossCPUKernelMod::CheckParam(const CNodePtr &kernel_node) {
  input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  constexpr int kOne = 1;
  constexpr int kfour = 4;
  if (input_num != kfour) {
    MS_LOG(EXCEPTION) << "Input number is " << input_num << ", but TripletMarginLossCPUKernelMod needs 4 inputs.";
  }
  output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
  if (output_num != kOne) {
    MS_LOG(EXCEPTION) << "Output number is " << output_num << ", but TripletMarginLossCPUKernelMod needs 1 output.";
  }
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, TripletMarginLoss, TripletMarginLossCPUKernelMod);
}  // namespace kernel
}  // namespace mindspore
