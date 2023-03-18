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
#include "mindspore/core/ops/triplet_margin_loss.h"

namespace mindspore {
namespace kernel {
bool TripletMarginLossCPUKernelMod::Init(const BaseOperatorPtr &base_operator,
                                         const std::vector<KernelTensorPtr> &inputs,
                                         const std::vector<KernelTensorPtr> &outputs) {
  MS_ERROR_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputNumber, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputNumber, kernel_name_);
  auto op_prim = std::dynamic_pointer_cast<ops::TripletMarginLoss>(base_operator);
  MS_ERROR_IF_NULL(op_prim);
  p_ = op_prim->get_p();
  swap_ = op_prim->get_swap();
  eps_ = op_prim->get_eps();
  reduction_ = op_prim->get_reduction();
  return true;
}

int TripletMarginLossCPUKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                          const std::vector<KernelTensorPtr> &inputs,
                                          const std::vector<KernelTensorPtr> &outputs,
                                          const std::map<uint32_t, tensor::TensorPtr> &) {
  auto ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  dtype_0_ = inputs[kIndex0]->GetDtype();
  x_shape_ = inputs[kIndex0]->GetDeviceShapeAdaptively();
  positive_shape_ = inputs[kIndex1]->GetDeviceShapeAdaptively();
  negative_shape_ = inputs[kIndex2]->GetDeviceShapeAdaptively();
  auto broadcast_shape_x_and_positive = CPUKernelUtils::GetBroadcastShape(x_shape_, positive_shape_);
  broadcast_shape_ = CPUKernelUtils::GetBroadcastShape(broadcast_shape_x_and_positive, negative_shape_);
  size_t dim_x = x_shape_.size();
  size_t dim_positive = positive_shape_.size();
  size_t dim_negative = negative_shape_.size();
  size_t max_size = std::max(std::max(dim_x, dim_positive), dim_negative);
  x_reshape_vector_ = x_shape_;
  positive_reshape_vector_ = positive_shape_;
  negative_reshape_vector_ = negative_shape_;
  std::reverse(x_reshape_vector_.begin(), x_reshape_vector_.end());
  std::reverse(positive_reshape_vector_.begin(), positive_reshape_vector_.end());
  std::reverse(negative_reshape_vector_.begin(), negative_reshape_vector_.end());
  if (dim_x < max_size) x_reshape_vector_.resize(max_size, kNumber1);
  if (dim_positive < max_size) positive_reshape_vector_.resize(max_size, kNumber1);
  if (dim_negative < max_size) negative_reshape_vector_.resize(max_size, kNumber1);
  std::reverse(x_reshape_vector_.begin(), x_reshape_vector_.end());
  std::reverse(positive_reshape_vector_.begin(), positive_reshape_vector_.end());
  std::reverse(negative_reshape_vector_.begin(), negative_reshape_vector_.end());
  numelements_ = LongToSize(SizeOf(broadcast_shape_));

  data_num_ = (numelements_) / LongToSize(broadcast_shape_[1]);
  data_num_each_batch_ = (numelements_) / LongToSize(broadcast_shape_[0]);
  index_ = data_num_ / LongToSize(broadcast_shape_[0]);
  batch_size_ = LongToSize(broadcast_shape_[0]);
  once_compute_size_ = LongToSize(broadcast_shape_[1]);
  if (x_shape_ != positive_shape_ || x_shape_ != negative_shape_ || positive_shape_ != negative_shape_) {
    broadcast_ = true;
  }
  return KRET_OK;
}

bool TripletMarginLossCPUKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                           const std::vector<kernel::AddressPtr> &,
                                           const std::vector<kernel::AddressPtr> &outputs) {
  switch (dtype_0_) {
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
                              << TypeIdLabel(dtype_0_) << " which is not supported.";
  }
  return true;
}

template <typename T>
void TripletMarginLossCPUKernelMod::TripletMarginLossCompute_realtype(const std::vector<kernel::AddressPtr> &inputs,
                                                                      const std::vector<kernel::AddressPtr> &outputs) {
  auto out_data = reinterpret_cast<float *>(outputs[0]->addr);
  Eigen::Array<float, Eigen::Dynamic, 1> out(data_num_, 1);
  float *output_reduction_none_data = reinterpret_cast<float *>(out.data());
  auto task_nobroadcast = [&](size_t start, size_t end) {
    TripletMarginLossCPUKernelMod::realtype_nobroadcast_task<T>(start, end, output_reduction_none_data, inputs,
                                                                outputs);
  };
  auto task_broadcast = [&](size_t start, size_t end) {
    TripletMarginLossCPUKernelMod::realtype_broadcast_task<T>(start, end, output_reduction_none_data, inputs, outputs);
  };
  if (broadcast_ == true) {
    if (numelements_ * sizeof(T) > kParallelDataNum_) {
      CPUKernelUtils::ParallelFor(task_broadcast, batch_size_);
    } else {
      TripletMarginLossCPUKernelMod::realtype_broadcast_compute<T>(output_reduction_none_data, inputs, outputs);
    }
    if (reduction_ == NONE) {
      for (size_t i = 0; i < data_num_; i++) {
        *(out_data + i) = *(output_reduction_none_data + i);
      }
    }
    if (reduction_ == MEAN) {
      *(out_data) = (out.mean());
    }
    if (reduction_ == SUM) {
      *(out_data) = (out.sum());
    }
    return;
  }
  if (numelements_ * sizeof(T) > kParallelDataNum_) {
    CPUKernelUtils::ParallelFor(task_nobroadcast, batch_size_);
  } else {
    TripletMarginLossCPUKernelMod::realtype_nobroadcast_compute<T>(output_reduction_none_data, inputs, outputs);
  }
  if (reduction_ == NONE) {
    for (size_t i = 0; i < data_num_; i++) {
      *(out_data + i) = *(output_reduction_none_data + i);
    }
  }
  if (reduction_ == MEAN) {
    *(out_data) = (out.mean());
  }
  if (reduction_ == SUM) {
    *(out_data) = (out.sum());
  }
  return;
}

template <typename T>
void TripletMarginLossCPUKernelMod::TripletMarginLossCompute_complextype(
  const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs) {
  auto out_data = reinterpret_cast<float *>(outputs[0]->addr);
  Eigen::Array<float, Eigen::Dynamic, 1> out(data_num_, 1);
  float *output_reduction_none_data = reinterpret_cast<float *>(out.data());
  auto task_nobroadcast = [&](size_t start, size_t end) {
    TripletMarginLossCPUKernelMod::complextype_nobroadcast_task<T>(start, end, output_reduction_none_data, inputs,
                                                                   outputs);
  };
  auto task_broadcast = [&](size_t start, size_t end) {
    TripletMarginLossCPUKernelMod::complextype_broadcast_task<T>(start, end, output_reduction_none_data, inputs,
                                                                 outputs);
  };
  if (broadcast_ == true) {
    if (numelements_ * sizeof(T) > kParallelDataNum_) {
      CPUKernelUtils::ParallelFor(task_broadcast, batch_size_);
    } else {
      TripletMarginLossCPUKernelMod::complextype_broadcast_compute<T>(output_reduction_none_data, inputs, outputs);
    }
    if (reduction_ == NONE) {
      for (size_t i = 0; i < data_num_; i++) {
        *(out_data + i) = *(output_reduction_none_data + i);
      }
    }
    if (reduction_ == MEAN) {
      *(out_data) = (out.mean());
    }
    if (reduction_ == SUM) {
      *(out_data) = (out.sum());
    }
    return;
  }
  if (numelements_ * sizeof(T) > kParallelDataNum_) {
    CPUKernelUtils::ParallelFor(task_nobroadcast, batch_size_);
  } else {
    TripletMarginLossCPUKernelMod::complextype_nobroadcast_compute<T>(output_reduction_none_data, inputs, outputs);
  }
  if (reduction_ == NONE) {
    for (size_t i = 0; i < data_num_; i++) {
      *(out_data + i) = *(output_reduction_none_data + i);
    }
  }
  if (reduction_ == MEAN) {
    *(out_data) = (out.mean());
  }
  if (reduction_ == SUM) {
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
  start *= data_num_each_batch_;
  end *= data_num_each_batch_;
  Eigen::Array<float, Eigen::Dynamic, 1> calculate_positive(once_compute_size_, kNoBroadcastValue);
  Eigen::Array<float, Eigen::Dynamic, 1> calculate_negative(once_compute_size_, kNoBroadcastValue);
  Eigen::Array<float, Eigen::Dynamic, 1> calculate_swap(once_compute_size_, kNoBroadcastValue);
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
  for (size_t n = 0; n < once_compute_thread_size / data_num_each_batch_; n++) {
    size_t i = start / data_num_each_batch_;
    for (size_t j = 0; j < index_; j++) {
      for (size_t m = 0; m < once_compute_size_; m++) {
        *(positive_data + m) = (eps_);
        *(negative_data + m) = (eps_);
        if (swap_ == true) {
          *(swap_data + m) = (eps_);
        }
      }
      for (size_t k = 0; k < once_compute_size_; k++) {
        *(positive_data + k) += static_cast<float>(*(x_addr + start + j + k * index_)) -
                                static_cast<float>(*(positive_addr + start + j + k * index_));
        *(negative_data + k) += static_cast<float>(*(x_addr + start + j + k * index_)) -
                                static_cast<float>(*(negative_addr + start + j + k * index_));
        if (swap_ == true) {
          *(swap_data + k) += static_cast<float>(*(positive_addr + start + j + k * index_)) -
                              static_cast<float>(*(negative_addr + start + j + k * index_));
        }
      }
      calculate_positive = (calculate_positive).abs();
      calculate_negative = (calculate_negative).abs();
      for (size_t a = 0; a < once_compute_size_; a++) {
        temp1 = *(positive_data + a);
        temp2 = *(negative_data + a);
        for (int l = 1; l < p_; l++) {
          *(positive_data + a) = *(positive_data + a) * temp1;
          *(negative_data + a) = *(negative_data + a) * temp2;
        }
      }
      positive_distance =
        static_cast<float>(std::pow(static_cast<double>(calculate_positive.sum()), (1 / static_cast<float>(p_))));
      negative_distance =
        static_cast<float>(std::pow(static_cast<double>(calculate_negative.sum()), (1 / static_cast<float>(p_))));
      if (swap_ == true) {
        calculate_swap = ((calculate_swap)).abs();
        for (size_t a = 0; a < once_compute_size_; a++) {
          temp3 = *(swap_data + a);
          for (int l = 1; l < p_; l++) {
            *(swap_data + a) = *(swap_data + a) * temp3;
          }
        }
        swap_distance =
          static_cast<float>(std::pow(static_cast<double>(calculate_swap.sum()), (1 / static_cast<float>(p_))));
        negative_distance = (negative_distance < swap_distance) ? negative_distance : swap_distance;
      }
      *(output_reduction_none_data + index_ * i + j) =
        (positive_distance + margin - negative_distance > 0) ? (positive_distance + margin - negative_distance) : 0;
    }
    start += data_num_each_batch_;
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
  start *= data_num_each_batch_;
  end *= data_num_each_batch_;
  float positive_distance;
  float negative_distance;
  float swap_distance;
  float temp1;
  float temp2;
  size_t once_compute_thread_size = end - start;
  BroadcastIterator broad_base_iter_1(x_shape_, positive_shape_, broadcast_shape_);
  BroadcastIterator broad_base_iter_2(negative_shape_, positive_shape_, broadcast_shape_);
  std::vector<T> x_broadcast(numelements_);
  std::vector<T> positive_broadcast(numelements_);
  std::vector<T> negative_broadcast(numelements_);
  std::vector<float> calculate_positive(once_compute_size_);
  std::vector<float> calculate_negative(once_compute_size_);
  std::vector<float> calculate_swap(once_compute_size_);
  broad_base_iter_1.SetPos(0);
  broad_base_iter_2.SetPos(0);
  for (size_t t = 0; t < numelements_; t++) {
    x_broadcast[t] = x_addr[broad_base_iter_1.GetInputPosA()];
    positive_broadcast[t] = positive_addr[broad_base_iter_1.GetInputPosB()];
    negative_broadcast[t] = negative_addr[broad_base_iter_2.GetInputPosA()];
    broad_base_iter_1.GenNextPos();
    broad_base_iter_2.GenNextPos();
  }
  for (size_t n = 0; n < (once_compute_thread_size) / data_num_each_batch_; n++) {
    size_t i = start / data_num_each_batch_;
    for (size_t j = 0; j < index_; j++) {
      float calc_1_sum = 0;
      float calc_2_sum = 0;
      float calc_swap_sum = 0;
      for (size_t k = 0; k < once_compute_size_; k++) {
        calculate_positive[k] = abs(static_cast<float>(x_broadcast[start + j + k * index_]) -
                                    static_cast<float>(positive_broadcast[start + j + k * index_]) + eps_);
        calculate_negative[k] = abs(static_cast<float>(x_broadcast[start + j + k * index_]) -
                                    static_cast<float>(negative_broadcast[start + j + k * index_]) + eps_);
        temp1 = calculate_positive[k];
        temp2 = calculate_negative[k];
        for (int l = 1; l < p_; l++) {
          calculate_positive[k] = calculate_positive[k] * temp1;
          calculate_negative[k] = calculate_negative[k] * temp2;
        }
        calc_1_sum += calculate_positive[k];
        calc_2_sum += calculate_negative[k];
        TripletMarginLossCPUKernelMod::realtype_swap<T>(start, positive_broadcast, negative_broadcast, calculate_swap,
                                                        j, k, calc_swap_sum, inputs, outputs);
      }
      positive_distance = static_cast<float>(std::pow(static_cast<double>(calc_1_sum), (1 / static_cast<float>(p_))));
      if (x_reshape_vector_[1] == 1 && positive_reshape_vector_[1] == 1 && broadcast_shape_[1] != 1) {
        positive_distance /= static_cast<float>(std::pow(broadcast_shape_[1], (1 / static_cast<float>(p_))));
      }
      negative_distance = static_cast<float>(std::pow(static_cast<double>(calc_2_sum), (1 / static_cast<float>(p_))));
      if (x_reshape_vector_[1] == 1 && negative_reshape_vector_[1] == 1 && broadcast_shape_[1] != 1) {
        negative_distance /= static_cast<float>(std::pow(broadcast_shape_[1], (1 / static_cast<float>(p_))));
      }
      if (swap_ == true) {
        swap_distance = static_cast<float>(std::pow(static_cast<double>(calc_swap_sum), (1 / static_cast<float>(p_))));
        if (positive_reshape_vector_[1] == 1 && negative_reshape_vector_[1] == 1 && broadcast_shape_[1] != 1) {
          swap_distance /= static_cast<float>(std::pow(broadcast_shape_[1], (1 / static_cast<float>(p_))));
        }
        negative_distance = (negative_distance < swap_distance) ? negative_distance : swap_distance;
      }
      *(output_reduction_none_data + index_ * i + j) =
        (positive_distance + margin - negative_distance > 0) ? (positive_distance + margin - negative_distance) : 0;
    }
    start += data_num_each_batch_;
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
  BroadcastIterator broad_base_iter_1(x_shape_, positive_shape_, broadcast_shape_);
  BroadcastIterator broad_base_iter_2(negative_shape_, positive_shape_, broadcast_shape_);
  std::vector<T> x_broadcast(numelements_);
  std::vector<T> positive_broadcast(numelements_);
  std::vector<T> negative_broadcast(numelements_);
  std::vector<float> calculate_positive(once_compute_size_);
  std::vector<float> calculate_negative(once_compute_size_);
  std::vector<float> calculate_swap(once_compute_size_);
  broad_base_iter_1.SetPos(0);
  broad_base_iter_2.SetPos(0);
  for (size_t t = 0; t < numelements_; t++) {
    x_broadcast[t] = x_addr[broad_base_iter_1.GetInputPosA()];
    positive_broadcast[t] = positive_addr[broad_base_iter_1.GetInputPosB()];
    negative_broadcast[t] = negative_addr[broad_base_iter_2.GetInputPosA()];
    broad_base_iter_1.GenNextPos();
    broad_base_iter_2.GenNextPos();
  }
  for (size_t i = 0; i < batch_size_; i++) {
    for (size_t j = 0; j < index_; j++) {
      float calc_1_sum = 0;
      float calc_2_sum = 0;
      float calc_swap_sum = 0;
      for (size_t k = 0; k < once_compute_size_; k++) {
        calculate_positive[k] =
          abs(static_cast<float>(x_broadcast[i * data_num_each_batch_ + j + k * index_]) -
              static_cast<float>(positive_broadcast[i * data_num_each_batch_ + j + k * index_] + eps_));
        calculate_negative[k] =
          abs(static_cast<float>(x_broadcast[i * data_num_each_batch_ + j + k * index_]) -
              static_cast<float>(negative_broadcast[i * data_num_each_batch_ + j + k * index_] + eps_));
        temp1 = calculate_positive[k];
        temp2 = calculate_negative[k];
        for (int l = 1; l < p_; l++) {
          calculate_positive[k] = calculate_positive[k] * temp1;
          calculate_negative[k] = calculate_negative[k] * temp2;
        }
        calc_1_sum += calculate_positive[k];
        calc_2_sum += calculate_negative[k];
        TripletMarginLossCPUKernelMod::realtype_swap<T>(i * data_num_each_batch_, positive_broadcast,
                                                        negative_broadcast, calculate_swap, j, k, calc_swap_sum, inputs,
                                                        outputs);
      }
      positive_distance = static_cast<float>(std::pow(static_cast<double>(calc_1_sum), (1 / static_cast<float>(p_))));
      if (x_reshape_vector_[1] == 1 && positive_reshape_vector_[1] == 1 && broadcast_shape_[1] != 1) {
        positive_distance /= static_cast<float>(std::pow(broadcast_shape_[1], (1 / static_cast<float>(p_))));
      }
      negative_distance = static_cast<float>(std::pow(static_cast<double>(calc_2_sum), (1 / static_cast<float>(p_))));
      if (x_reshape_vector_[1] == 1 && negative_reshape_vector_[1] == 1 && broadcast_shape_[1] != 1) {
        negative_distance /= static_cast<float>(std::pow(broadcast_shape_[1], (1 / static_cast<float>(p_))));
      }
      if (swap_ == true) {
        swap_distance = static_cast<float>(std::pow(static_cast<double>(calc_swap_sum), (1 / static_cast<float>(p_))));
        if (positive_reshape_vector_[1] == 1 && negative_reshape_vector_[1] == 1 && broadcast_shape_[1] != 1) {
          swap_distance /= static_cast<float>(std::pow(broadcast_shape_[1], (1 / static_cast<float>(p_))));
        }
        negative_distance = (negative_distance < swap_distance) ? negative_distance : swap_distance;
      }
      *(output_reduction_none_data + index_ * i + j) =
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
  Eigen::Array<float, Eigen::Dynamic, 1> calculate_positive(once_compute_size_, kNoBroadcastValue);
  Eigen::Array<float, Eigen::Dynamic, 1> calculate_negative(once_compute_size_, kNoBroadcastValue);
  Eigen::Array<float, Eigen::Dynamic, 1> calculate_swap(once_compute_size_, kNoBroadcastValue);
  float *positive_data = reinterpret_cast<float *>(calculate_positive.data());
  float *negative_data = reinterpret_cast<float *>(calculate_negative.data());
  float *swap_data = reinterpret_cast<float *>(calculate_swap.data());
  float positive_distance;
  float negative_distance;
  float swap_distance;
  float temp1;
  float temp2;
  float temp3;
  for (size_t i = 0; i < batch_size_; i++) {
    for (size_t j = 0; j < index_; j++) {
      for (size_t m = 0; m < once_compute_size_; m++) {
        *(positive_data + m) = (eps_);
        *(negative_data + m) = (eps_);
        if (swap_ == true) {
          *(swap_data + m) = (eps_);
        }
      }
      for (size_t k = 0; k < once_compute_size_; k++) {
        *(positive_data + k) += static_cast<float>(*(x_addr + i * data_num_each_batch_ + j + k * index_)) -
                                static_cast<float>(*(positive_addr + i * data_num_each_batch_ + j + k * index_));
        *(negative_data + k) += static_cast<float>(*(x_addr + i * data_num_each_batch_ + j + k * index_)) -
                                static_cast<float>(*(negative_addr + i * data_num_each_batch_ + j + k * index_));
        if (swap_ == true) {
          *(swap_data + k) += static_cast<float>(*(positive_addr + i * data_num_each_batch_ + j + k * index_)) -
                              static_cast<float>(*(negative_addr + i * data_num_each_batch_ + j + k * index_));
        }
      }
      calculate_positive = (calculate_positive).abs();
      calculate_negative = (calculate_negative).abs();
      for (size_t n = 0; n < once_compute_size_; n++) {
        temp1 = *(positive_data + n);
        temp2 = *(negative_data + n);
        for (int l = 1; l < p_; l++) {
          *(positive_data + n) = *(positive_data + n) * temp1;
          *(negative_data + n) = *(negative_data + n) * temp2;
        }
      }
      positive_distance =
        static_cast<float>(std::pow(static_cast<double>(calculate_positive.sum()), (1 / static_cast<float>(p_))));
      negative_distance =
        static_cast<float>(std::pow(static_cast<double>(calculate_negative.sum()), (1 / static_cast<float>(p_))));
      if (swap_ == true) {
        calculate_swap = ((calculate_swap)).abs();
        for (size_t n = 0; n < once_compute_size_; n++) {
          temp3 = *(swap_data + n);
          for (int l = 1; l < p_; l++) {
            *(swap_data + n) = *(swap_data + n) * temp3;
          }
        }
        swap_distance =
          static_cast<float>(std::pow(static_cast<double>(calculate_swap.sum()), (1 / static_cast<float>(p_))));
        negative_distance = (negative_distance < swap_distance) ? negative_distance : swap_distance;
      }
      *(output_reduction_none_data + index_ * i + j) =
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
  start *= data_num_each_batch_;
  end *= data_num_each_batch_;
  Eigen::Array<T, Eigen::Dynamic, 1> calculate_positive(once_compute_size_, kNoBroadcastValue);
  Eigen::Array<T, Eigen::Dynamic, 1> calculate_negative(once_compute_size_, kNoBroadcastValue);
  Eigen::Array<T, Eigen::Dynamic, 1> calculate_swap(once_compute_size_, kNoBroadcastValue);
  T *positive_data = reinterpret_cast<T *>(calculate_positive.data());
  T *negative_data = reinterpret_cast<T *>(calculate_negative.data());
  T *swap_data = reinterpret_cast<T *>(calculate_swap.data());
  size_t once_compute_thread_size = end - start;
  float positive_distance;
  float negative_distance;
  float swap_distance;
  for (size_t n = 0; n < (once_compute_thread_size) / data_num_each_batch_; n++) {
    size_t i = start / data_num_each_batch_;
    for (size_t j = 0; j < index_; j++) {
      for (size_t m = 0; m < once_compute_size_; m++) {
        *(positive_data + m) = (eps_);
        *(negative_data + m) = (eps_);
        if (swap_ == true) {
          *(swap_data + m) = (eps_);
        }
      }
      for (size_t k = 0; k < once_compute_size_; k++) {
        *(positive_data + k) += (*(x_addr + start + j + k * index_)) - (*(positive_addr + start + j + k * index_));
        *(negative_data + k) += (*(x_addr + start + j + k * index_)) - (*(negative_addr + start + j + k * index_));
        if (swap_ == true) {
          *(swap_data + k) += (*(positive_addr + start + j + k * index_)) - (*(negative_addr + start + j + k * index_));
        }
      }
      auto calculate_positive_float =
        (calculate_positive * (calculate_positive.matrix().conjugate().array())).real().sqrt();
      auto calculate_negative_float =
        (calculate_negative * (calculate_negative.matrix().conjugate().array())).real().sqrt();
      positive_distance =
        static_cast<float>(std::pow(calculate_positive_float.pow(p_).sum(), 1 / static_cast<float>(p_)));
      negative_distance =
        static_cast<float>(std::pow(calculate_negative_float.pow(p_).sum(), 1 / static_cast<float>(p_)));
      if (swap_ == true) {
        auto calculate_swap_float = (calculate_swap * (calculate_swap.matrix().conjugate().array())).real().sqrt();
        swap_distance = static_cast<float>(std::pow(calculate_swap_float.pow(p_).sum(), 1 / static_cast<float>(p_)));
        negative_distance = (negative_distance < swap_distance) ? negative_distance : swap_distance;
      }
      *(output_reduction_none_data + index_ * i + j) =
        (positive_distance + margin - negative_distance > 0) ? (positive_distance + margin - negative_distance) : 0;
    }
    start += data_num_each_batch_;
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
  start *= data_num_each_batch_;
  end *= data_num_each_batch_;
  float positive_distance;
  float negative_distance;
  float swap_distance;
  float temp1;
  float temp2;
  size_t once_compute_thread_size = end - start;
  BroadcastIterator broad_base_iter_1(x_shape_, positive_shape_, broadcast_shape_);
  BroadcastIterator broad_base_iter_2(negative_shape_, positive_shape_, broadcast_shape_);
  std::vector<T> x_broadcast(numelements_);
  std::vector<T> positive_broadcast(numelements_);
  std::vector<T> negative_broadcast(numelements_);
  std::vector<T> calculate_positive(once_compute_size_);
  std::vector<T> calculate_negative(once_compute_size_);
  std::vector<T> calculate_swap(once_compute_size_);
  broad_base_iter_1.SetPos(0);
  broad_base_iter_2.SetPos(0);
  for (size_t t = 0; t < numelements_; t++) {
    x_broadcast[t] = x_addr[broad_base_iter_1.GetInputPosA()];
    positive_broadcast[t] = positive_addr[broad_base_iter_1.GetInputPosB()];
    negative_broadcast[t] = negative_addr[broad_base_iter_2.GetInputPosA()];
    broad_base_iter_1.GenNextPos();
    broad_base_iter_2.GenNextPos();
  }
  for (size_t n = 0; n < (once_compute_thread_size) / data_num_each_batch_; n++) {
    size_t i = start / data_num_each_batch_;
    for (size_t j = 0; j < index_; j++) {
      float calc_1_sum = 0;
      float calc_2_sum = 0;
      float calc_swap_sum = 0;
      for (size_t k = 0; k < once_compute_size_; k++) {
        calculate_positive[k] =
          x_broadcast[start + j + k * index_] - positive_broadcast[start + j + k * index_] + static_cast<T>(eps_);
        calculate_negative[k] =
          x_broadcast[start + j + k * index_] - negative_broadcast[start + j + k * index_] + static_cast<T>(eps_);
        float calculate_positive_float =
          static_cast<float>(sqrt((calculate_positive[k].real() * calculate_positive[k].real() +
                                   calculate_positive[k].imag() * calculate_positive[k].imag())));
        float calculate_negative_float =
          static_cast<float>(sqrt((calculate_negative[k].real() * calculate_negative[k].real() +
                                   calculate_negative[k].imag() * calculate_negative[k].imag())));
        temp1 = calculate_positive_float;
        temp2 = calculate_negative_float;
        for (int l = 1; l < p_; l++) {
          calculate_positive_float = calculate_positive_float * temp1;
          calculate_negative_float = calculate_negative_float * temp2;
        }
        calc_1_sum += calculate_positive_float;
        calc_2_sum += calculate_negative_float;
        TripletMarginLossCPUKernelMod::complextype_swap<T>(start, positive_broadcast, negative_broadcast,
                                                           calculate_swap, j, k, calc_swap_sum, inputs, outputs);
      }
      positive_distance = static_cast<float>(std::pow(static_cast<double>(calc_1_sum), (1 / static_cast<float>(p_))));
      if (x_reshape_vector_[1] == 1 && positive_reshape_vector_[1] == 1 && broadcast_shape_[1] != 1) {
        positive_distance /= static_cast<float>(std::pow(broadcast_shape_[1], (1 / static_cast<float>(p_))));
      }
      negative_distance = static_cast<float>(std::pow(static_cast<double>(calc_2_sum), (1 / static_cast<float>(p_))));
      if (x_reshape_vector_[1] == 1 && negative_reshape_vector_[1] == 1 && broadcast_shape_[1] != 1) {
        negative_distance /= static_cast<float>(std::pow(broadcast_shape_[1], (1 / static_cast<float>(p_))));
      }
      if (swap_ == true) {
        swap_distance = static_cast<float>(std::pow(static_cast<double>(calc_swap_sum), (1 / static_cast<float>(p_))));
        if (positive_reshape_vector_[1] == 1 && negative_reshape_vector_[1] == 1 && broadcast_shape_[1] != 1) {
          swap_distance /= static_cast<float>(std::pow(broadcast_shape_[1], (1 / static_cast<float>(p_))));
        }
        negative_distance = (negative_distance < swap_distance) ? negative_distance : swap_distance;
      }
      *(output_reduction_none_data + index_ * i + j) =
        (positive_distance + margin - negative_distance > 0) ? (positive_distance + margin - negative_distance) : 0;
    }
    start += data_num_each_batch_;
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
  BroadcastIterator broad_base_iter_1(x_shape_, positive_shape_, broadcast_shape_);
  BroadcastIterator broad_base_iter_2(negative_shape_, positive_shape_, broadcast_shape_);
  std::vector<T> x_broadcast(numelements_);
  std::vector<T> positive_broadcast(numelements_);
  std::vector<T> negative_broadcast(numelements_);
  std::vector<T> calculate_positive(once_compute_size_);
  std::vector<T> calculate_negative(once_compute_size_);
  std::vector<T> calculate_swap(once_compute_size_);
  broad_base_iter_1.SetPos(0);
  broad_base_iter_2.SetPos(0);
  for (size_t t = 0; t < numelements_; t++) {
    x_broadcast[t] = x_addr[broad_base_iter_1.GetInputPosA()];
    positive_broadcast[t] = positive_addr[broad_base_iter_1.GetInputPosB()];
    negative_broadcast[t] = negative_addr[broad_base_iter_2.GetInputPosA()];
    broad_base_iter_1.GenNextPos();
    broad_base_iter_2.GenNextPos();
  }
  for (size_t i = 0; i < batch_size_; i++) {
    for (size_t j = 0; j < index_; j++) {
      float calc_1_sum = 0;
      float calc_2_sum = 0;
      float calc_swap_sum = 0;
      for (size_t k = 0; k < once_compute_size_; k++) {
        calculate_positive[k] = x_broadcast[i * data_num_each_batch_ + j + k * index_] -
                                positive_broadcast[i * data_num_each_batch_ + j + k * index_] + static_cast<T>(eps_);
        calculate_negative[k] = x_broadcast[i * data_num_each_batch_ + j + k * index_] -
                                negative_broadcast[i * data_num_each_batch_ + j + k * index_] + static_cast<T>(eps_);
        float calculate_positive_float =
          static_cast<float>(sqrt((calculate_positive[k].real() * calculate_positive[k].real() +
                                   calculate_positive[k].imag() * calculate_positive[k].imag())));
        float calculate_negative_float =
          static_cast<float>(sqrt((calculate_negative[k].real() * calculate_negative[k].real() +
                                   calculate_negative[k].imag() * calculate_negative[k].imag())));
        temp1 = calculate_positive_float;
        temp2 = calculate_negative_float;
        for (int l = 1; l < p_; l++) {
          calculate_positive_float = calculate_positive_float * temp1;
          calculate_negative_float = calculate_negative_float * temp2;
        }
        calc_1_sum += calculate_positive_float;
        calc_2_sum += calculate_negative_float;
        TripletMarginLossCPUKernelMod::complextype_swap<T>(i * data_num_each_batch_, positive_broadcast,
                                                           negative_broadcast, calculate_swap, j, k, calc_swap_sum,
                                                           inputs, outputs);
      }
      positive_distance = static_cast<float>(std::pow(static_cast<double>(calc_1_sum), (1 / static_cast<float>(p_))));
      if (x_reshape_vector_[1] == 1 && positive_reshape_vector_[1] == 1 && broadcast_shape_[1] != 1) {
        positive_distance /= static_cast<float>(std::pow(broadcast_shape_[1], (1 / static_cast<float>(p_))));
      }
      negative_distance = static_cast<float>(std::pow(static_cast<double>(calc_2_sum), (1 / static_cast<float>(p_))));
      if (x_reshape_vector_[1] == 1 && negative_reshape_vector_[1] == 1 && broadcast_shape_[1] != 1) {
        negative_distance /= static_cast<float>(std::pow(broadcast_shape_[1], (1 / static_cast<float>(p_))));
      }
      if (swap_ == true) {
        swap_distance = static_cast<float>(std::pow(static_cast<double>(calc_swap_sum), (1 / static_cast<float>(p_))));
        if (positive_reshape_vector_[1] == 1 && negative_reshape_vector_[1] == 1 && broadcast_shape_[1] != 1) {
          swap_distance /= static_cast<float>(std::pow(broadcast_shape_[1], (1 / static_cast<float>(p_))));
        }
        negative_distance = (negative_distance < swap_distance) ? negative_distance : swap_distance;
      }
      *(output_reduction_none_data + index_ * i + j) =
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
  Eigen::Array<T, Eigen::Dynamic, 1> calculate_positive(once_compute_size_, kNoBroadcastValue);
  Eigen::Array<T, Eigen::Dynamic, 1> calculate_negative(once_compute_size_, kNoBroadcastValue);
  Eigen::Array<T, Eigen::Dynamic, 1> calculate_swap(once_compute_size_, kNoBroadcastValue);
  T *positive_data = reinterpret_cast<T *>(calculate_positive.data());
  T *negative_data = reinterpret_cast<T *>(calculate_negative.data());
  T *swap_data = reinterpret_cast<T *>(calculate_swap.data());
  float positive_distance;
  float negative_distance;
  float swap_distance;
  for (size_t i = 0; i < batch_size_; i++) {
    for (size_t j = 0; j < index_; j++) {
      for (size_t m = 0; m < once_compute_size_; m++) {
        *(positive_data + m) = (eps_);
        *(negative_data + m) = (eps_);
        if (swap_ == true) {
          *(swap_data + m) = (eps_);
        }
      }
      for (size_t k = 0; k < once_compute_size_; k++) {
        *(positive_data + k) += (*(x_addr + i * data_num_each_batch_ + j + k * index_)) -
                                (*(positive_addr + i * data_num_each_batch_ + j + k * index_));
        *(negative_data + k) += (*(x_addr + i * data_num_each_batch_ + j + k * index_)) -
                                (*(negative_addr + i * data_num_each_batch_ + j + k * index_));
        if (swap_ == true) {
          *(swap_data + k) += (*(positive_addr + i * data_num_each_batch_ + j + k * index_)) -
                              (*(negative_addr + i * data_num_each_batch_ + j + k * index_));
        }
      }
      auto calculate_positive_float =
        (calculate_positive * (calculate_positive.matrix().conjugate().array())).real().sqrt();
      auto calculate_negative_float =
        (calculate_negative * (calculate_negative.matrix().conjugate().array())).real().sqrt();
      positive_distance =
        static_cast<float>(std::pow(calculate_positive_float.pow(p_).sum(), 1 / static_cast<float>(p_)));
      negative_distance =
        static_cast<float>(std::pow(calculate_negative_float.pow(p_).sum(), 1 / static_cast<float>(p_)));
      if (swap_ == true) {
        auto calculate_swap_float = (calculate_swap * (calculate_swap.matrix().conjugate().array())).real().sqrt();
        swap_distance = static_cast<float>(std::pow(calculate_swap_float.pow(p_).sum(), 1 / static_cast<float>(p_)));
        negative_distance = (negative_distance < swap_distance) ? negative_distance : swap_distance;
      }
      *(output_reduction_none_data + index_ * i + j) =
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
  if (swap_ == true) {
    calculate_swap[k] = abs(static_cast<float>(positive_broadcast[start + j + k * index_]) -
                            static_cast<float>(negative_broadcast[start + j + k * index_]) + eps_);
    float temp3 = calculate_swap[k];
    for (int l = 1; l < p_; l++) {
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
  if (swap_ == true) {
    calculate_swap[k] =
      positive_broadcast[start + j + k * index_] - negative_broadcast[start + j + k * index_] + static_cast<T>(eps_);
    float calculate_swap_float = static_cast<float>(sqrt(
      (calculate_swap[k].real() * calculate_swap[k].real() + calculate_swap[k].imag() * calculate_swap[k].imag())));
    float temp3 = calculate_swap_float;
    for (int l = 1; l < p_; l++) {
      calculate_swap_float = calculate_swap_float * temp3;
    }
    calc_swap_sum += calculate_swap_float;
  }
}
}  // namespace kernel
}  // namespace mindspore
