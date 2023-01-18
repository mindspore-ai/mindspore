/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "triplet_margin_loss.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>

#include "cpu_kernel_utils.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "utils/broadcast_iterator.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kInputNum = 4;
const uint32_t kOutputNum = 1;
const int64_t kNoBroadcastValue = 1;
const char *kTripletMarginLoss = "TripletMarginLoss";
// when input data size is more than kParallelDataNum, use Parallel func
const int64_t kParallelDataNum = 28 * 1024;
const int64_t kParallelDataNumMid = 56 * 1024;
}  // namespace

namespace aicpu {
uint32_t TripletMarginLossCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      " TripletMarginLoss check input and output number failed.");
  auto data_type_x = static_cast<DataType>(ctx.Input(0)->GetDataType());
  auto data_type_positive = static_cast<DataType>(ctx.Input(1)->GetDataType());
  auto data_type_negative = static_cast<DataType>(ctx.Input(2)->GetDataType());
  if (data_type_x != data_type_negative || data_type_positive != data_type_negative ||
      data_type_x != data_type_positive) {
    KERNEL_LOG_ERROR(
      "[%s] Data type of inputs requires to be the same, but got data type "
      "[%s] and "
      "[%s], type[%s].",
      ctx.GetOpType().c_str(), DTypeStr(data_type_x).c_str(), DTypeStr(data_type_positive).c_str(),
      DTypeStr(data_type_negative).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  AttrValue *Attr_p = ctx.GetAttr("p");
  int p_value = (Attr_p == nullptr) ? 2 : Attr_p->GetInt();
  float margin_value = *(reinterpret_cast<float *>(ctx.Input(3)->GetData()));
  AttrValue *Attr_eps = ctx.GetAttr("eps");
  float eps_value = (Attr_eps == nullptr) ? 1e-6 : Attr_eps->GetFloat();
  AttrValue *Attr_swap = ctx.GetAttr("swap");
  bool swap_value = (Attr_swap == nullptr) ? false : Attr_swap->GetBool();
  AttrValue *Attr_red = ctx.GetAttr("reduction");
  std::string reduction_value = (Attr_red == nullptr) ? "mean" : Attr_red->GetString();
  Tensor *input_x = (ctx.Input(0));
  Tensor *input_positive = (ctx.Input(1));
  Tensor *input_negative = (ctx.Input(2));
  const std::vector<int64_t> &shape_x = input_x->GetTensorShape()->GetDimSizes();
  const std::vector<int64_t> &shape_positive = input_positive->GetTensorShape()->GetDimSizes();
  const std::vector<int64_t> &shape_negative = input_negative->GetTensorShape()->GetDimSizes();
  std::vector<int64_t> broadcast_shape;
  std::vector<int64_t> broadcast_shape_x_and_positive;
  (void)GetBroadcastShape(shape_x, shape_positive, broadcast_shape_x_and_positive);
  (void)GetBroadcastShape(broadcast_shape_x_and_positive, shape_negative, broadcast_shape);
  int64_t num_elements = 1;
  for (size_t i = 0; i < broadcast_shape.size(); i++) {
    num_elements *= broadcast_shape[i];
  }
  int64_t data_num_output_reduction_none = (num_elements) / (broadcast_shape[1]);
  int64_t data_num_each_batch_input = (num_elements) / (broadcast_shape[0]);
  int64_t data_num_each_batch_output_reduction_none = data_num_output_reduction_none / (broadcast_shape[0]);
  int64_t batch_size = broadcast_shape[0];
  int64_t once_compute_size = broadcast_shape[1];
  bool broadcast = false;
  std::vector<int64_t> x_reshape_vector = shape_x;
  std::vector<int64_t> positive_reshape_vector = shape_positive;
  std::vector<int64_t> negative_reshape_vector = shape_negative;
  if (shape_x != shape_positive || shape_x != shape_negative || shape_positive != shape_negative) {
    broadcast = true;
    std::reverse(x_reshape_vector.begin(), x_reshape_vector.end());
    std::reverse(positive_reshape_vector.begin(), positive_reshape_vector.end());
    std::reverse(negative_reshape_vector.begin(), negative_reshape_vector.end());
    int64_t dim_num_x = input_x->GetTensorShape()->GetDims();
    int64_t dim_num_positive = input_positive->GetTensorShape()->GetDims();
    int64_t dim_num_negative = input_negative->GetTensorShape()->GetDims();
    auto dims = std::max(dim_num_x, std::max(dim_num_positive, dim_num_negative));
    if (dim_num_x < dims) x_reshape_vector.resize(dims, kNoBroadcastValue);
    if (dim_num_positive < dims) positive_reshape_vector.resize(dims, kNoBroadcastValue);
    if (dim_num_negative < dims) negative_reshape_vector.resize(dims, kNoBroadcastValue);
    std::reverse(x_reshape_vector.begin(), x_reshape_vector.end());
    std::reverse(positive_reshape_vector.begin(), positive_reshape_vector.end());
    std::reverse(negative_reshape_vector.begin(), negative_reshape_vector.end());
  }
  switch (data_type_x) {
    case DT_FLOAT16:
      return TripletMarginLossComputeRealTypeFloat16<Eigen::half>(
        ctx, p_value, margin_value, eps_value, swap_value, reduction_value, num_elements,
        data_num_output_reduction_none, data_num_each_batch_input, data_num_each_batch_output_reduction_none,
        batch_size, once_compute_size, broadcast, x_reshape_vector, positive_reshape_vector, negative_reshape_vector);
    case DT_FLOAT:
      return TripletMarginLossComputeRealType<float>(
        ctx, p_value, margin_value, eps_value, swap_value, reduction_value, num_elements,
        data_num_output_reduction_none, data_num_each_batch_input, data_num_each_batch_output_reduction_none,
        batch_size, once_compute_size, broadcast, x_reshape_vector, positive_reshape_vector, negative_reshape_vector);
    case DT_DOUBLE:
      return TripletMarginLossComputeRealType<double>(
        ctx, p_value, margin_value, eps_value, swap_value, reduction_value, num_elements,
        data_num_output_reduction_none, data_num_each_batch_input, data_num_each_batch_output_reduction_none,
        batch_size, once_compute_size, broadcast, x_reshape_vector, positive_reshape_vector, negative_reshape_vector);
    case DT_INT8:
      return TripletMarginLossComputeRealType<int8_t>(
        ctx, p_value, margin_value, eps_value, swap_value, reduction_value, num_elements,
        data_num_output_reduction_none, data_num_each_batch_input, data_num_each_batch_output_reduction_none,
        batch_size, once_compute_size, broadcast, x_reshape_vector, positive_reshape_vector, negative_reshape_vector);
    case DT_INT16:
      return TripletMarginLossComputeRealType<int16_t>(
        ctx, p_value, margin_value, eps_value, swap_value, reduction_value, num_elements,
        data_num_output_reduction_none, data_num_each_batch_input, data_num_each_batch_output_reduction_none,
        batch_size, once_compute_size, broadcast, x_reshape_vector, positive_reshape_vector, negative_reshape_vector);
    case DT_INT32:
      return TripletMarginLossComputeRealType<int32_t>(
        ctx, p_value, margin_value, eps_value, swap_value, reduction_value, num_elements,
        data_num_output_reduction_none, data_num_each_batch_input, data_num_each_batch_output_reduction_none,
        batch_size, once_compute_size, broadcast, x_reshape_vector, positive_reshape_vector, negative_reshape_vector);
    case DT_INT64:
      return TripletMarginLossComputeRealType<int64_t>(
        ctx, p_value, margin_value, eps_value, swap_value, reduction_value, num_elements,
        data_num_output_reduction_none, data_num_each_batch_input, data_num_each_batch_output_reduction_none,
        batch_size, once_compute_size, broadcast, x_reshape_vector, positive_reshape_vector, negative_reshape_vector);
    case DT_UINT8:
      return TripletMarginLossComputeRealType<uint8_t>(
        ctx, p_value, margin_value, eps_value, swap_value, reduction_value, num_elements,
        data_num_output_reduction_none, data_num_each_batch_input, data_num_each_batch_output_reduction_none,
        batch_size, once_compute_size, broadcast, x_reshape_vector, positive_reshape_vector, negative_reshape_vector);
    case DT_UINT16:
      return TripletMarginLossComputeRealType<uint16_t>(
        ctx, p_value, margin_value, eps_value, swap_value, reduction_value, num_elements,
        data_num_output_reduction_none, data_num_each_batch_input, data_num_each_batch_output_reduction_none,
        batch_size, once_compute_size, broadcast, x_reshape_vector, positive_reshape_vector, negative_reshape_vector);
    case DT_UINT32:
      return TripletMarginLossComputeRealType<uint32_t>(
        ctx, p_value, margin_value, eps_value, swap_value, reduction_value, num_elements,
        data_num_output_reduction_none, data_num_each_batch_input, data_num_each_batch_output_reduction_none,
        batch_size, once_compute_size, broadcast, x_reshape_vector, positive_reshape_vector, negative_reshape_vector);
    case DT_UINT64:
      return TripletMarginLossComputeRealType<uint64_t>(
        ctx, p_value, margin_value, eps_value, swap_value, reduction_value, num_elements,
        data_num_output_reduction_none, data_num_each_batch_input, data_num_each_batch_output_reduction_none,
        batch_size, once_compute_size, broadcast, x_reshape_vector, positive_reshape_vector, negative_reshape_vector);
    case DT_COMPLEX128:
      return TripletMarginLossComputeComplexType<std::complex<double>>(
        ctx, p_value, margin_value, eps_value, swap_value, reduction_value, num_elements,
        data_num_output_reduction_none, data_num_each_batch_input, data_num_each_batch_output_reduction_none,
        batch_size, once_compute_size, broadcast, x_reshape_vector, positive_reshape_vector, negative_reshape_vector);
    case DT_COMPLEX64:
      return TripletMarginLossComputeComplexType<std::complex<float>>(
        ctx, p_value, margin_value, eps_value, swap_value, reduction_value, num_elements,
        data_num_output_reduction_none, data_num_each_batch_input, data_num_each_batch_output_reduction_none,
        batch_size, once_compute_size, broadcast, x_reshape_vector, positive_reshape_vector, negative_reshape_vector);
    default:
      KERNEL_LOG_ERROR("[%s] Data type of input is not supported, input data type is [%s].", ctx.GetOpType().c_str(),
                       DTypeStr(data_type_x).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t TripletMarginLossCpuKernel::TripletMarginLossComputeRealType(
  CpuKernelContext &ctx, int p_value, float margin_value, float eps_value, bool swap_value, std::string reduction_value,
  int64_t num_elements, int64_t data_num_output_reduction_none, int64_t data_num_each_batch_input,
  int64_t data_num_each_batch_output_reduction_none, int64_t batch_size, int64_t once_compute_size, bool broadcast,
  std::vector<int64_t> x_reshape_vector, std::vector<int64_t> positive_reshape_vector,
  std::vector<int64_t> negative_reshape_vector) {
  constexpr int ADULT_AGE = 4;
  Tensor *input_x = (ctx.Input(0));
  Tensor *input_positive = (ctx.Input(1));
  Tensor *input_negative = (ctx.Input(2));
  Tensor *output = (ctx.Output(0));
  const std::vector<int64_t> &shape_x = input_x->GetTensorShape()->GetDimSizes();
  const std::vector<int64_t> &shape_positive = input_positive->GetTensorShape()->GetDimSizes();
  const std::vector<int64_t> &shape_negative = input_negative->GetTensorShape()->GetDimSizes();
  T *x_data = reinterpret_cast<T *>(input_x->GetData());
  T *positive_data = reinterpret_cast<T *>(input_positive->GetData());
  T *negative_data = reinterpret_cast<T *>(input_negative->GetData());
  std::vector<int64_t> broadcast_shape;
  std::vector<int64_t> broadcast_shape_x_and_positive;
  (void)GetBroadcastShape(shape_x, shape_positive, broadcast_shape_x_and_positive);
  (void)GetBroadcastShape(broadcast_shape_x_and_positive, shape_negative, broadcast_shape);
  std::vector<T> x_broadcast_tensor;
  std::vector<T> positive_broadcast_tensor;
  std::vector<T> negative_broadcast_tensor;
  if (broadcast == true) {
    auto shape_x1 = shape_x;
    auto shape_x2 = shape_x;
    auto shape_positive1 = shape_positive;
    auto shape_negative1 = shape_negative;
    auto broadcast_shape1 = broadcast_shape;
    auto broadcast_shape2 = broadcast_shape;
    BroadcastIterator iter1(shape_x1, shape_positive1, broadcast_shape1);
    BroadcastIterator iter2(shape_x2, shape_negative1, broadcast_shape2);
    iter1.SetPos(0);
    iter2.SetPos(0);
    for (int64_t i = 0; i < num_elements; i++) {
      x_broadcast_tensor.push_back(x_data[iter1.GetInputPosA()]);
      positive_broadcast_tensor.push_back(positive_data[iter1.GetInputPosB()]);
      negative_broadcast_tensor.push_back(negative_data[iter2.GetInputPosB()]);
      iter1.GenNextPos();
      iter2.GenNextPos();
    }
    x_data = x_broadcast_tensor.data();
    positive_data = positive_broadcast_tensor.data();
    negative_data = negative_broadcast_tensor.data();
  }
  auto output_data = reinterpret_cast<float *>(output->GetData());
  Eigen::Array<float, Eigen::Dynamic, 1> output_reduction_none(data_num_output_reduction_none, 1);
  float *output_reduction_none_data = reinterpret_cast<float *>(output_reduction_none.data());
  auto shard_triplet_margin_loss = [&](int64_t start, int64_t end) {
    Eigen::Array<float, Eigen::Dynamic, 1> calculate_positive_distance(once_compute_size, 1);
    Eigen::Array<float, Eigen::Dynamic, 1> calculate_negative_distance(once_compute_size, 1);
    Eigen::Array<float, Eigen::Dynamic, 1> calculate_swap_distance(once_compute_size, 1);
    float *calculate_positive_distance_data = reinterpret_cast<float *>(calculate_positive_distance.data());
    float *calculate_negative_distance_data = reinterpret_cast<float *>(calculate_negative_distance.data());
    float *calculate_swap_distance_data = reinterpret_cast<float *>(calculate_swap_distance.data());
    int64_t once_compute_thread_size = (end - start);
    float positive_distance;
    float negative_distance;
    float swap_distance;
    float temp1;
    float temp2;
    float temp3;
    if (data_num_each_batch_input == 0) {
      KERNEL_LOG_ERROR("data_num_each_batch_input could not be 0.");
    }
    for (int64_t n = 0; n < once_compute_thread_size / data_num_each_batch_input; n++) {
      int64_t i = start / data_num_each_batch_input;
      for (int64_t j = 0; j < data_num_each_batch_output_reduction_none; j++) {
        for (int64_t k = 0; k < once_compute_size; k++) {
          *(calculate_positive_distance_data + k) =
            eps_value +
            static_cast<float>(
              *(x_data + i * data_num_each_batch_input + j + k * data_num_each_batch_output_reduction_none)) -
            static_cast<float>(
              *(positive_data + i * data_num_each_batch_input + j + k * data_num_each_batch_output_reduction_none));
          *(calculate_negative_distance_data + k) =
            eps_value +
            static_cast<float>(
              *(x_data + i * data_num_each_batch_input + j + k * data_num_each_batch_output_reduction_none)) -
            static_cast<float>(
              *(negative_data + i * data_num_each_batch_input + j + k * data_num_each_batch_output_reduction_none));
          if (swap_value == true) {
            *(calculate_swap_distance_data + k) =
              eps_value +
              static_cast<float>(
                *(positive_data + i * data_num_each_batch_input + j + k * data_num_each_batch_output_reduction_none)) -
              static_cast<float>(
                *(negative_data + i * data_num_each_batch_input + j + k * data_num_each_batch_output_reduction_none));
          }
        }
        calculate_positive_distance = (calculate_positive_distance).abs();
        calculate_negative_distance = (calculate_negative_distance).abs();
        for (int64_t n = 0; n < once_compute_size; n++) {
          temp1 = *(calculate_positive_distance_data + n);
          temp2 = *(calculate_negative_distance_data + n);
          for (int64_t l = 1; l < p_value; l++) {
            *(calculate_positive_distance_data + n) = *(calculate_positive_distance_data + n) * temp1;
            *(calculate_negative_distance_data + n) = *(calculate_negative_distance_data + n) * temp2;
          }
        }
        positive_distance =
          std::pow(static_cast<double>(calculate_positive_distance.sum()), (1 / static_cast<float>(p_value)));
        negative_distance =
          std::pow(static_cast<double>(calculate_negative_distance.sum()), (1 / static_cast<float>(p_value)));
        if (broadcast == true) {
          if (x_reshape_vector[1] == 1 && positive_reshape_vector[1] == 1 && broadcast_shape[1] != 1) {
            positive_distance /= std::pow(broadcast_shape[1], (1 / static_cast<float>(p_value)));
          }
          if (x_reshape_vector[1] == 1 && negative_reshape_vector[1] == 1 && broadcast_shape[1] != 1) {
            negative_distance /= std::pow(broadcast_shape[1], (1 / static_cast<float>(p_value)));
          }
        }
        if (swap_value == true) {
          calculate_swap_distance = ((calculate_swap_distance)).abs();
          for (int64_t n = 0; n < once_compute_size; n++) {
            temp3 = *(calculate_swap_distance_data + n);
            for (int64_t l = 1; l < p_value; l++) {
              *(calculate_swap_distance_data + n) = *(calculate_swap_distance_data + n) * temp3;
            }
          }
          swap_distance =
            std::pow(static_cast<double>(calculate_swap_distance.sum()), (1 / static_cast<float>(p_value)));
          if (broadcast == true) {
            if (positive_reshape_vector[1] == 1 && negative_reshape_vector[1] == 1 && broadcast_shape[1] != 1) {
              swap_distance /= std::pow(broadcast_shape[1], (1 / static_cast<float>(p_value)));
            }
          }
          negative_distance = (negative_distance < swap_distance) ? negative_distance : swap_distance;
        }
        *(output_reduction_none_data + data_num_each_batch_output_reduction_none * i + j) =
          (positive_distance + margin_value - negative_distance > 0)
            ? (positive_distance + margin_value - negative_distance)
            : 0;
      }
      start += data_num_each_batch_input;
    }
  };
  if (num_elements * sizeof(T) > kParallelDataNum) {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
    if (num_elements * sizeof(T) <= kParallelDataNumMid) {
      max_core_num = std::min(max_core_num, 4U);  // up to 4 cpu cores
    }
    if (max_core_num == 0) {
      KERNEL_LOG_ERROR("max_core_num could not be 0.");
    }
    CpuKernelUtils::ParallelFor(ctx, num_elements,
                                data_num_each_batch_input * ADULT_AGE * (batch_size / max_core_num + 1),
                                shard_triplet_margin_loss);
  } else {
    Eigen::Array<float, Eigen::Dynamic, 1> calculate_positive_distance(once_compute_size, 1);
    Eigen::Array<float, Eigen::Dynamic, 1> calculate_negative_distance(once_compute_size, 1);
    Eigen::Array<float, Eigen::Dynamic, 1> calculate_swap_distance(once_compute_size, 1);
    float *calculate_positive_distance_data = reinterpret_cast<float *>(calculate_positive_distance.data());
    float *calculate_negative_distance_data = reinterpret_cast<float *>(calculate_negative_distance.data());
    float *calculate_swap_distance_data = reinterpret_cast<float *>(calculate_swap_distance.data());
    float positive_distance;
    float negative_distance;
    float swap_distance;
    float temp1;
    float temp2;
    float temp3;
    for (int64_t i = 0; i < batch_size; i++) {
      for (int64_t j = 0; j < data_num_each_batch_output_reduction_none; j++) {
        for (int64_t k = 0; k < once_compute_size; k++) {
          *(calculate_positive_distance_data + k) =
            eps_value +
            static_cast<float>(
              *(x_data + i * data_num_each_batch_input + j + k * data_num_each_batch_output_reduction_none)) -
            static_cast<float>(
              *(positive_data + i * data_num_each_batch_input + j + k * data_num_each_batch_output_reduction_none));
          *(calculate_negative_distance_data + k) =
            eps_value +
            static_cast<float>(
              *(x_data + i * data_num_each_batch_input + j + k * data_num_each_batch_output_reduction_none)) -
            static_cast<float>(
              *(negative_data + i * data_num_each_batch_input + j + k * data_num_each_batch_output_reduction_none));
          if (swap_value == true) {
            *(calculate_swap_distance_data + k) =
              eps_value +
              static_cast<float>(
                *(positive_data + i * data_num_each_batch_input + j + k * data_num_each_batch_output_reduction_none)) -
              static_cast<float>(
                *(negative_data + i * data_num_each_batch_input + j + k * data_num_each_batch_output_reduction_none));
          }
        }
        calculate_positive_distance = (calculate_positive_distance).abs();
        calculate_negative_distance = (calculate_negative_distance).abs();
        for (int64_t n = 0; n < once_compute_size; n++) {
          temp1 = *(calculate_positive_distance_data + n);
          temp2 = *(calculate_negative_distance_data + n);
          for (int64_t l = 1; l < p_value; l++) {
            *(calculate_positive_distance_data + n) = *(calculate_positive_distance_data + n) * temp1;
            *(calculate_negative_distance_data + n) = *(calculate_negative_distance_data + n) * temp2;
          }
        }
        positive_distance =
          std::pow(static_cast<double>(calculate_positive_distance.sum()), (1 / static_cast<float>(p_value)));
        negative_distance =
          std::pow(static_cast<double>(calculate_negative_distance.sum()), (1 / static_cast<float>(p_value)));
        if (broadcast == true) {
          if (x_reshape_vector[1] == 1 && positive_reshape_vector[1] == 1 && broadcast_shape[1] != 1) {
            positive_distance /= std::pow(broadcast_shape[1], (1 / static_cast<float>(p_value)));
          }
          if (x_reshape_vector[1] == 1 && negative_reshape_vector[1] == 1 && broadcast_shape[1] != 1) {
            negative_distance /= std::pow(broadcast_shape[1], (1 / static_cast<float>(p_value)));
          }
        }
        if (swap_value == true) {
          calculate_swap_distance = ((calculate_swap_distance)).abs();
          for (int64_t n = 0; n < once_compute_size; n++) {
            temp3 = *(calculate_swap_distance_data + n);
            for (int64_t l = 1; l < p_value; l++) {
              *(calculate_swap_distance_data + n) = *(calculate_swap_distance_data + n) * temp3;
            }
          }
          swap_distance =
            std::pow(static_cast<double>(calculate_swap_distance.sum()), (1 / static_cast<float>(p_value)));
          if (broadcast == true) {
            if (positive_reshape_vector[1] == 1 && negative_reshape_vector[1] == 1 && broadcast_shape[1] != 1) {
              swap_distance /= std::pow(broadcast_shape[1], (1 / static_cast<float>(p_value)));
            }
          }
          negative_distance = (negative_distance < swap_distance) ? negative_distance : swap_distance;
        }
        *(output_reduction_none_data + data_num_each_batch_output_reduction_none * i + j) =
          (positive_distance + margin_value - negative_distance > 0)
            ? (positive_distance + margin_value - negative_distance)
            : 0;
      }
    }
  }
  if (reduction_value == "none") {
    for (int64_t i = 0; i < data_num_output_reduction_none; i++) {
      *(output_data + i) = *(output_reduction_none_data + i);
    }
  }
  if (reduction_value == "mean") {
    *(output_data) = (output_reduction_none.mean());
  }
  if (reduction_value == "sum") {
    *(output_data) = (output_reduction_none.sum());
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t TripletMarginLossCpuKernel::TripletMarginLossComputeComplexType(
  CpuKernelContext &ctx, int p_value, float margin_value, float eps_value, bool swap_value, std::string reduction_value,
  int64_t num_elements, int64_t data_num_output_reduction_none, int64_t data_num_each_batch_input,
  int64_t data_num_each_batch_output_reduction_none, int64_t batch_size, int64_t once_compute_size, bool broadcast,
  std::vector<int64_t> x_reshape_vector, std::vector<int64_t> positive_reshape_vector,
  std::vector<int64_t> negative_reshape_vector) {
  constexpr int ADULT_AGE = 4;
  Tensor *input_x = (ctx.Input(0));
  Tensor *input_positive = (ctx.Input(1));
  Tensor *input_negative = (ctx.Input(2));
  Tensor *output = (ctx.Output(0));
  const std::vector<int64_t> &shape_x = input_x->GetTensorShape()->GetDimSizes();
  const std::vector<int64_t> &shape_positive = input_positive->GetTensorShape()->GetDimSizes();
  const std::vector<int64_t> &shape_negative = input_negative->GetTensorShape()->GetDimSizes();
  T *x_data = reinterpret_cast<T *>(input_x->GetData());
  T *positive_data = reinterpret_cast<T *>(input_positive->GetData());
  T *negative_data = reinterpret_cast<T *>(input_negative->GetData());
  std::vector<int64_t> broadcast_shape;
  std::vector<int64_t> broadcast_shape_x_and_positive;
  (void)GetBroadcastShape(shape_x, shape_positive, broadcast_shape_x_and_positive);
  (void)GetBroadcastShape(broadcast_shape_x_and_positive, shape_negative, broadcast_shape);
  std::vector<T> x_broadcast_tensor;
  std::vector<T> positive_broadcast_tensor;
  std::vector<T> negative_broadcast_tensor;
  if (broadcast == true) {
    auto shape_x1 = shape_x;
    auto shape_x2 = shape_x;
    auto shape_positive1 = shape_positive;
    auto shape_negative1 = shape_negative;
    auto broadcast_shape1 = broadcast_shape;
    auto broadcast_shape2 = broadcast_shape;
    BroadcastIterator iter1(shape_x1, shape_positive1, broadcast_shape1);
    BroadcastIterator iter2(shape_x2, shape_negative1, broadcast_shape2);
    iter1.SetPos(0);
    iter2.SetPos(0);
    for (int64_t i = 0; i < num_elements; i++) {
      x_broadcast_tensor.push_back(x_data[iter1.GetInputPosA()]);
      positive_broadcast_tensor.push_back(positive_data[iter1.GetInputPosB()]);
      negative_broadcast_tensor.push_back(negative_data[iter2.GetInputPosB()]);
      iter1.GenNextPos();
      iter2.GenNextPos();
    }
    x_data = x_broadcast_tensor.data();
    positive_data = positive_broadcast_tensor.data();
    negative_data = negative_broadcast_tensor.data();
  }
  auto output_data = reinterpret_cast<float *>(output->GetData());
  Eigen::Array<float, Eigen::Dynamic, 1> output_reduction_none(data_num_output_reduction_none, 1);
  float *output_reduction_none_data = reinterpret_cast<float *>(output_reduction_none.data());
  auto shard_triplet_margin_loss = [&](int64_t start, int64_t end) {
    Eigen::Array<T, Eigen::Dynamic, 1> calculate_positive_distance(once_compute_size, 1);
    Eigen::Array<T, Eigen::Dynamic, 1> calculate_negative_distance(once_compute_size, 1);
    Eigen::Array<T, Eigen::Dynamic, 1> calculate_swap_distance(once_compute_size, 1);
    T *calculate_positive_distance_data = reinterpret_cast<T *>(calculate_positive_distance.data());
    T *calculate_negative_distance_data = reinterpret_cast<T *>(calculate_negative_distance.data());
    T *calculate_swap_distance_data = reinterpret_cast<T *>(calculate_swap_distance.data());
    int64_t once_compute_thread_size = end - start;
    float positive_distance;
    float negative_distance;
    float swap_distance;
    if (data_num_each_batch_input == 0) {
      KERNEL_LOG_ERROR("data_num_each_batch_input could not be 0.");
    }
    for (int64_t n = 0; n < (once_compute_thread_size) / data_num_each_batch_input; n++) {
      int64_t i = start / data_num_each_batch_input;
      for (int64_t j = 0; j < data_num_each_batch_output_reduction_none; j++) {
        for (int64_t k = 0; k < once_compute_size; k++) {
          *(calculate_positive_distance_data + k) =
            static_cast<T>(eps_value) +
            (*(x_data + i * data_num_each_batch_input + j + k * data_num_each_batch_output_reduction_none)) -
            (*(positive_data + i * data_num_each_batch_input + j + k * data_num_each_batch_output_reduction_none));
          *(calculate_negative_distance_data + k) =
            static_cast<T>(eps_value) +
            (*(x_data + i * data_num_each_batch_input + j + k * data_num_each_batch_output_reduction_none)) -
            (*(negative_data + i * data_num_each_batch_input + j + k * data_num_each_batch_output_reduction_none));
          if (swap_value == true) {
            *(calculate_swap_distance_data + k) =
              static_cast<T>(eps_value) +
              (*(positive_data + i * data_num_each_batch_input + j + k * data_num_each_batch_output_reduction_none)) -
              (*(negative_data + i * data_num_each_batch_input + j + k * data_num_each_batch_output_reduction_none));
          }
        }
        auto calculate_positive_distance_float =
          (calculate_positive_distance * (calculate_positive_distance.matrix().conjugate().array())).real().sqrt();
        auto calculate_negative_distance_float =
          (calculate_negative_distance * (calculate_negative_distance.matrix().conjugate().array())).real().sqrt();
        positive_distance =
          std::pow(calculate_positive_distance_float.pow(p_value).sum(), 1 / static_cast<float>(p_value));
        negative_distance =
          std::pow(calculate_negative_distance_float.pow(p_value).sum(), 1 / static_cast<float>(p_value));
        if (broadcast == true) {
          if (x_reshape_vector[1] == 1 && positive_reshape_vector[1] == 1 && broadcast_shape[1] != 1) {
            positive_distance /= std::pow(broadcast_shape[1], (1 / static_cast<float>(p_value)));
          }
          if (x_reshape_vector[1] == 1 && negative_reshape_vector[1] == 1 && broadcast_shape[1] != 1) {
            negative_distance /= std::pow(broadcast_shape[1], (1 / static_cast<float>(p_value)));
          }
        }
        if (swap_value == true) {
          auto calculate_swap_distance_float =
            (calculate_swap_distance * (calculate_swap_distance.matrix().conjugate().array())).real().sqrt();
          swap_distance = std::pow(calculate_swap_distance_float.pow(p_value).sum(), 1 / static_cast<float>(p_value));
          if (broadcast == true) {
            if (positive_reshape_vector[1] == 1 && negative_reshape_vector[1] == 1 && broadcast_shape[1] != 1) {
              swap_distance /= std::pow(broadcast_shape[1], (1 / static_cast<float>(p_value)));
            }
          }
          negative_distance = (negative_distance < swap_distance) ? negative_distance : swap_distance;
        }
        *(output_reduction_none_data + data_num_each_batch_output_reduction_none * i + j) =
          (positive_distance + margin_value - negative_distance > 0)
            ? (positive_distance + margin_value - negative_distance)
            : 0;
      }
      start += data_num_each_batch_input;
    }
  };
  if (num_elements * sizeof(T) > kParallelDataNum) {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
    if (num_elements * sizeof(T) <= kParallelDataNumMid) {
      max_core_num = std::min(max_core_num, 4U);  // up to 4 cpu cores
    }
    if (max_core_num == 0) {
      KERNEL_LOG_ERROR("max_core_num could not be 0.");
    }
    CpuKernelUtils::ParallelFor(ctx, num_elements,
                                data_num_each_batch_input * ADULT_AGE * (batch_size / max_core_num + 1),
                                shard_triplet_margin_loss);
  } else {
    Eigen::Array<T, Eigen::Dynamic, 1> calculate_positive_distance(once_compute_size, 1);
    Eigen::Array<T, Eigen::Dynamic, 1> calculate_negative_distance(once_compute_size, 1);
    Eigen::Array<T, Eigen::Dynamic, 1> calculate_swap_distance(once_compute_size, 1);
    T *calculate_positive_distance_data = reinterpret_cast<T *>(calculate_positive_distance.data());
    T *calculate_negative_distance_data = reinterpret_cast<T *>(calculate_negative_distance.data());
    T *calculate_swap_distance_data = reinterpret_cast<T *>(calculate_swap_distance.data());
    for (int64_t i = 0; i < batch_size; i++) {
      for (int64_t j = 0; j < data_num_each_batch_output_reduction_none; j++) {
        for (int64_t k = 0; k < once_compute_size; k++) {
          *(calculate_positive_distance_data + k) =
            static_cast<T>(eps_value) +
            (*(x_data + i * data_num_each_batch_input + j + k * data_num_each_batch_output_reduction_none)) -
            (*(positive_data + i * data_num_each_batch_input + j + k * data_num_each_batch_output_reduction_none));
          *(calculate_negative_distance_data + k) =
            static_cast<T>(eps_value) +
            (*(x_data + i * data_num_each_batch_input + j + k * data_num_each_batch_output_reduction_none)) -
            (*(negative_data + i * data_num_each_batch_input + j + k * data_num_each_batch_output_reduction_none));
          if (swap_value == true) {
            *(calculate_swap_distance_data + k) =
              static_cast<T>(eps_value) +
              (*(positive_data + i * data_num_each_batch_input + j + k * data_num_each_batch_output_reduction_none)) -
              (*(negative_data + i * data_num_each_batch_input + j + k * data_num_each_batch_output_reduction_none));
          }
        }
        float positive_distance;
        float negative_distance;
        float swap_distance;
        auto calculate_positive_distance_float =
          (calculate_positive_distance * (calculate_positive_distance.matrix().conjugate().array())).real().sqrt();
        auto calculate_negative_distance_float =
          (calculate_negative_distance * (calculate_negative_distance.matrix().conjugate().array())).real().sqrt();
        positive_distance =
          std::pow(calculate_positive_distance_float.pow(p_value).sum(), 1 / static_cast<float>(p_value));
        negative_distance =
          std::pow(calculate_negative_distance_float.pow(p_value).sum(), 1 / static_cast<float>(p_value));
        if (broadcast == true) {
          if (x_reshape_vector[1] == 1 && positive_reshape_vector[1] == 1 && broadcast_shape[1] != 1) {
            positive_distance /= std::pow(broadcast_shape[1], (1 / static_cast<float>(p_value)));
          }
          if (x_reshape_vector[1] == 1 && negative_reshape_vector[1] == 1 && broadcast_shape[1] != 1) {
            negative_distance /= std::pow(broadcast_shape[1], (1 / static_cast<float>(p_value)));
          }
        }
        if (swap_value == true) {
          auto calculate_swap_distance_float =
            (calculate_swap_distance * (calculate_swap_distance.matrix().conjugate().array())).real().sqrt();
          swap_distance = std::pow(calculate_swap_distance_float.pow(p_value).sum(), 1 / static_cast<float>(p_value));
          if (broadcast == true) {
            if (positive_reshape_vector[1] == 1 && negative_reshape_vector[1] == 1 && broadcast_shape[1] != 1) {
              swap_distance /= std::pow(broadcast_shape[1], (1 / static_cast<float>(p_value)));
            }
          }
          negative_distance = (negative_distance < swap_distance) ? negative_distance : swap_distance;
        }
        *(output_reduction_none_data + data_num_each_batch_output_reduction_none * i + j) =
          (positive_distance + margin_value - negative_distance > 0)
            ? positive_distance + margin_value - negative_distance
            : 0;
      }
    }
  }
  if (reduction_value == "none") {
    for (int64_t i = 0; i < data_num_output_reduction_none; i++) {
      *(output_data + i) = *(output_reduction_none_data + i);
    }
  }
  if (reduction_value == "mean") {
    *(output_data) = (output_reduction_none.mean());
  }
  if (reduction_value == "sum") {
    *(output_data) = (output_reduction_none.sum());
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t TripletMarginLossCpuKernel::TripletMarginLossComputeRealTypeFloat16(
  CpuKernelContext &ctx, int p_value, float margin_value, float eps_value, bool swap_value, std::string reduction_value,
  int64_t num_elements, int64_t data_num_output_reduction_none, int64_t data_num_each_batch_input,
  int64_t data_num_each_batch_output_reduction_none, int64_t batch_size, int64_t once_compute_size, bool broadcast,
  std::vector<int64_t> x_reshape_vector, std::vector<int64_t> positive_reshape_vector,
  std::vector<int64_t> negative_reshape_vector) {
  constexpr int ADULT_AGE = 4;
  Tensor *input_x = (ctx.Input(0));
  Tensor *input_positive = (ctx.Input(1));
  Tensor *input_negative = (ctx.Input(2));
  Tensor *output = (ctx.Output(0));
  const std::vector<int64_t> &shape_x = input_x->GetTensorShape()->GetDimSizes();
  const std::vector<int64_t> &shape_positive = input_positive->GetTensorShape()->GetDimSizes();
  const std::vector<int64_t> &shape_negative = input_negative->GetTensorShape()->GetDimSizes();
  T *x_data = reinterpret_cast<T *>(input_x->GetData());
  T *positive_data = reinterpret_cast<T *>(input_positive->GetData());
  T *negative_data = reinterpret_cast<T *>(input_negative->GetData());
  std::vector<int64_t> broadcast_shape;
  std::vector<int64_t> broadcast_shape_x_and_positive;
  (void)GetBroadcastShape(shape_x, shape_positive, broadcast_shape_x_and_positive);
  (void)GetBroadcastShape(broadcast_shape_x_and_positive, shape_negative, broadcast_shape);
  std::vector<T> x_broadcast_tensor;
  std::vector<T> positive_broadcast_tensor;
  std::vector<T> negative_broadcast_tensor;
  if (broadcast == true) {
    auto shape_x1 = shape_x;
    auto shape_x2 = shape_x;
    auto shape_positive1 = shape_positive;
    auto shape_negative1 = shape_negative;
    auto broadcast_shape1 = broadcast_shape;
    auto broadcast_shape2 = broadcast_shape;
    BroadcastIterator iter1(shape_x1, shape_positive1, broadcast_shape1);
    BroadcastIterator iter2(shape_x2, shape_negative1, broadcast_shape2);
    iter1.SetPos(0);
    iter2.SetPos(0);
    for (int64_t i = 0; i < num_elements; i++) {
      x_broadcast_tensor.push_back(x_data[iter1.GetInputPosA()]);
      positive_broadcast_tensor.push_back(positive_data[iter1.GetInputPosB()]);
      negative_broadcast_tensor.push_back(negative_data[iter2.GetInputPosB()]);
      iter1.GenNextPos();
      iter2.GenNextPos();
    }
    x_data = x_broadcast_tensor.data();
    positive_data = positive_broadcast_tensor.data();
    negative_data = negative_broadcast_tensor.data();
  }
  auto output_data = reinterpret_cast<T *>(output->GetData());
  Eigen::Array<float, Eigen::Dynamic, 1> output_reduction_none(data_num_output_reduction_none, 1);
  float *output_reduction_none_data = reinterpret_cast<float *>(output_reduction_none.data());
  auto shard_triplet_margin_loss = [&](int64_t start, int64_t end) {
    Eigen::Array<float, Eigen::Dynamic, 1> calculate_positive_distance(once_compute_size, 1);
    Eigen::Array<float, Eigen::Dynamic, 1> calculate_negative_distance(once_compute_size, 1);
    Eigen::Array<float, Eigen::Dynamic, 1> calculate_swap_distance(once_compute_size, 1);
    float *calculate_positive_distance_data = reinterpret_cast<float *>(calculate_positive_distance.data());
    float *calculate_negative_distance_data = reinterpret_cast<float *>(calculate_negative_distance.data());
    float *calculate_swap_distance_data = reinterpret_cast<float *>(calculate_swap_distance.data());
    int64_t once_compute_thread_size = end - start;
    float positive_distance;
    float negative_distance;
    float swap_distance;
    float temp1;
    float temp2;
    float temp3;
    if (data_num_each_batch_input == 0) {
      KERNEL_LOG_ERROR("data_num_each_batch_input could not be 0.");
    }
    for (int64_t n = 0; n < (once_compute_thread_size) / data_num_each_batch_input; n++) {
      int64_t i = start / data_num_each_batch_input;
      for (int64_t j = 0; j < data_num_each_batch_output_reduction_none; j++) {
        for (int64_t k = 0; k < once_compute_size; k++) {
          *(calculate_positive_distance_data + k) =
            eps_value + (static_cast<float>(*(x_data + i * data_num_each_batch_input + j +
                                              k * data_num_each_batch_output_reduction_none)) -
                         static_cast<float>(*(positive_data + i * data_num_each_batch_input + j +
                                              k * data_num_each_batch_output_reduction_none)));
          *(calculate_negative_distance_data + k) =
            eps_value + (static_cast<float>(*(x_data + i * data_num_each_batch_input + j +
                                              k * data_num_each_batch_output_reduction_none)) -
                         static_cast<float>(*(negative_data + i * data_num_each_batch_input + j +
                                              k * data_num_each_batch_output_reduction_none)));
          if (swap_value == true) {
            *(calculate_swap_distance_data + k) =
              eps_value + (static_cast<float>(*(positive_data + i * data_num_each_batch_input + j +
                                                k * data_num_each_batch_output_reduction_none)) -
                           static_cast<float>(*(negative_data + i * data_num_each_batch_input + j +
                                                k * data_num_each_batch_output_reduction_none)));
          }
        }
        calculate_positive_distance = (calculate_positive_distance).abs();
        calculate_negative_distance = (calculate_negative_distance).abs();
        for (int64_t n = 0; n < once_compute_size; n++) {
          temp1 = *(calculate_positive_distance_data + n);
          temp2 = *(calculate_negative_distance_data + n);
          for (int64_t l = 1; l < p_value; l++) {
            *(calculate_positive_distance_data + n) = *(calculate_positive_distance_data + n) * temp1;
            *(calculate_negative_distance_data + n) = *(calculate_negative_distance_data + n) * temp2;
          }
        }
        positive_distance = static_cast<float>(
          std::pow(static_cast<double>(calculate_positive_distance.sum()), (1 / static_cast<float>(p_value))));
        negative_distance = static_cast<float>(
          std::pow(static_cast<double>(calculate_negative_distance.sum()), (1 / static_cast<float>(p_value))));
        if (x_reshape_vector[1] == 1 && positive_reshape_vector[1] == 1 && broadcast_shape[1] != 1) {
          positive_distance /= std::pow(broadcast_shape[1], (1 / static_cast<float>(p_value)));
        }
        if (x_reshape_vector[1] == 1 && negative_reshape_vector[1] == 1 && broadcast_shape[1] != 1) {
          negative_distance /= std::pow(broadcast_shape[1], (1 / static_cast<float>(p_value)));
        }
        if (swap_value == true) {
          calculate_swap_distance = ((calculate_swap_distance)).abs();
          for (int64_t n = 0; n < once_compute_size; n++) {
            temp3 = *(calculate_swap_distance_data + n);
            for (int64_t l = 1; l < p_value; l++) {
              *(calculate_swap_distance_data + n) = *(calculate_swap_distance_data + n) * temp3;
            }
          }
          swap_distance = static_cast<float>(
            std::pow(static_cast<double>(calculate_swap_distance.sum()), (1 / static_cast<float>(p_value))));
          if (positive_reshape_vector[1] == 1 && negative_reshape_vector[1] == 1 && broadcast_shape[1] != 1) {
            swap_distance /= std::pow(broadcast_shape[1], (1 / static_cast<float>(p_value)));
          }
          negative_distance = (negative_distance < swap_distance) ? negative_distance : swap_distance;
        }
        *(output_reduction_none_data + data_num_each_batch_output_reduction_none * i + j) =
          (positive_distance + margin_value - negative_distance > static_cast<float>(0))
            ? ((positive_distance + margin_value - negative_distance))
            : static_cast<float>(0);
      }
      start += data_num_each_batch_input;
    }
  };
  if (num_elements * sizeof(T) > kParallelDataNum) {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
    if (num_elements * sizeof(T) <= kParallelDataNumMid) {
      max_core_num = std::min(max_core_num, 4U);  // up to 4 cpu cores
    }
    if (max_core_num == 0) {
      KERNEL_LOG_ERROR("max_core_num could not be 0.");
    }
    CpuKernelUtils::ParallelFor(ctx, num_elements,
                                data_num_each_batch_input * ADULT_AGE * (batch_size / max_core_num + 1),
                                shard_triplet_margin_loss);
  } else {
    Eigen::Array<float, Eigen::Dynamic, 1> calculate_positive_distance(once_compute_size, 1);
    Eigen::Array<float, Eigen::Dynamic, 1> calculate_negative_distance(once_compute_size, 1);
    Eigen::Array<float, Eigen::Dynamic, 1> calculate_swap_distance(once_compute_size, 1);
    float *calculate_positive_distance_data = reinterpret_cast<float *>(calculate_positive_distance.data());
    float *calculate_negative_distance_data = reinterpret_cast<float *>(calculate_negative_distance.data());
    float *calculate_swap_distance_data = reinterpret_cast<float *>(calculate_swap_distance.data());
    for (int64_t i = 0; i < batch_size; i++) {
      for (int64_t j = 0; j < data_num_each_batch_output_reduction_none; j++) {
        float positive_distance;
        float negative_distance;
        float swap_distance;
        for (int64_t k = 0; k < once_compute_size; k++) {
          *(calculate_positive_distance_data + k) =
            eps_value + (static_cast<float>(*(x_data + i * data_num_each_batch_input + j +
                                              k * data_num_each_batch_output_reduction_none)) -
                         static_cast<float>(*(positive_data + i * data_num_each_batch_input + j +
                                              k * data_num_each_batch_output_reduction_none)));
          *(calculate_negative_distance_data + k) =
            eps_value + (static_cast<float>(*(x_data + i * data_num_each_batch_input + j +
                                              k * data_num_each_batch_output_reduction_none)) -
                         static_cast<float>(*(negative_data + i * data_num_each_batch_input + j +
                                              k * data_num_each_batch_output_reduction_none)));
          if (swap_value == true) {
            *(calculate_swap_distance_data + k) =
              eps_value + (static_cast<float>(*(positive_data + i * data_num_each_batch_input + j +
                                                k * data_num_each_batch_output_reduction_none)) -
                           static_cast<float>(*(negative_data + i * data_num_each_batch_input + j +
                                                k * data_num_each_batch_output_reduction_none)));
          }
        }
        calculate_positive_distance = (calculate_positive_distance).abs();
        calculate_negative_distance = (calculate_negative_distance).abs();
        float temp1;
        float temp2;
        float temp3;
        for (int64_t n = 0; n < once_compute_size; n++) {
          temp1 = *(calculate_positive_distance_data + n);
          temp2 = *(calculate_negative_distance_data + n);
          for (int64_t l = 1; l < p_value; l++) {
            *(calculate_positive_distance_data + n) = *(calculate_positive_distance_data + n) * temp1;
            *(calculate_negative_distance_data + n) = *(calculate_negative_distance_data + n) * temp2;
          }
        }
        positive_distance = static_cast<float>(
          std::pow(static_cast<double>(calculate_positive_distance.sum()), (1 / static_cast<float>(p_value))));
        negative_distance = static_cast<float>(
          std::pow(static_cast<double>(calculate_negative_distance.sum()), (1 / static_cast<float>(p_value))));
        if (broadcast == true) {
          if (x_reshape_vector[1] == 1 && positive_reshape_vector[1] == 1 && broadcast_shape[1] != 1) {
            positive_distance /= std::pow(broadcast_shape[1], (1 / static_cast<float>(p_value)));
          }
          if (x_reshape_vector[1] == 1 && negative_reshape_vector[1] == 1 && broadcast_shape[1] != 1) {
            negative_distance /= std::pow(broadcast_shape[1], (1 / static_cast<float>(p_value)));
          }
        }
        if (swap_value == true) {
          calculate_swap_distance = ((calculate_swap_distance)).abs();
          for (int64_t n = 0; n < once_compute_size; n++) {
            temp3 = *(calculate_swap_distance_data + n);
            for (int64_t l = 1; l < p_value; l++) {
              *(calculate_swap_distance_data + n) = *(calculate_swap_distance_data + n) * temp3;
            }
          }
          swap_distance = static_cast<float>(
            std::pow(static_cast<double>(calculate_swap_distance.sum()), (1 / static_cast<float>(p_value))));
          if (broadcast == true) {
            if (positive_reshape_vector[1] == 1 && negative_reshape_vector[1] == 1 && broadcast_shape[1] != 1) {
              swap_distance /= std::pow(broadcast_shape[1], (1 / static_cast<float>(p_value)));
            }
          }
          negative_distance = (negative_distance < swap_distance) ? negative_distance : swap_distance;
        }
        *(output_reduction_none_data + data_num_each_batch_output_reduction_none * i + j) =
          (positive_distance + margin_value - negative_distance > static_cast<float>(0))
            ? ((positive_distance + margin_value - negative_distance))
            : static_cast<float>(0);
      }
    }
  }

  if (reduction_value == "none") {
    for (int64_t i = 0; i < data_num_output_reduction_none; i++) {
      *(output_data + i) = static_cast<T>(*(output_reduction_none_data + i));
    }
  }
  if (reduction_value == "mean") {
    *(output_data) = static_cast<T>(output_reduction_none.mean());
  }
  if (reduction_value == "sum") {
    *(output_data) = static_cast<T>(output_reduction_none.sum());
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kTripletMarginLoss, TripletMarginLossCpuKernel);
}  // namespace aicpu
