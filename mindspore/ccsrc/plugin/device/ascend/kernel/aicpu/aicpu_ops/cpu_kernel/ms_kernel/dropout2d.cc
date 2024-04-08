/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "dropout2d.h"

#include <securec.h>
#include <random>
#include <chrono>
#include <algorithm>

#include "cpu_types.h"
#include "inc/kernel_log.h"
#include "context/common/status.h"
#include "unsupported/Eigen/CXX11/Tensor"

namespace {
const char *kDropout2d = "Dropout2D";

template <typename T>
T CalcOutput(T input, float p) {
  return input / (1 - p);
}

template <>
Eigen::half CalcOutput(Eigen::half input, float p) {
  return input / static_cast<Eigen::half>(1 - p);
}
}  // namespace

namespace aicpu {
template <typename T>
static uint32_t CalDropout2d(CpuKernelContext &ctx, float p, std::vector<void *> &io_addrs_,
                             std::vector<int64_t> &shape) {
  // inputs
  T *input = reinterpret_cast<T *>(io_addrs_[0]);
  // outputs
  T *output = reinterpret_cast<T *>(io_addrs_[1]);
  bool *mask = reinterpret_cast<bool *>(io_addrs_[2]);

  int64_t N = shape[0];
  int64_t C = shape[1];
  int64_t H = shape[2];
  int64_t W = shape[3];

  size_t channel_num = N * C;
  size_t channel_size = H * W;
  size_t data_num = channel_num * channel_size;
  if (INT64_MAX / channel_num < channel_size) {
    CUST_KERNEL_LOG_ERROR(ctx, "channel_num is out of range!");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  if (p > 1 || p < 0) {
    CUST_KERNEL_LOG_ERROR(ctx, "dropout probability must be between 0 and 1, but got %f", p);
    return KERNEL_STATUS_PARAM_INVALID;
  } else if (p == 0) {
    auto ret = memcpy_s(output, data_num * sizeof(T), input, data_num * sizeof(T));
    CUST_KERNEL_CHECK_FALSE(ctx, (ret == EOK), KERNEL_STATUS_PARAM_INVALID, "Dropout2d memcpy_s failed.");
    std::fill(&mask[0], &mask[data_num], true);
    return KERNEL_STATUS_OK;
  } else if (p == 1) {
    auto ret = memset_s(output, data_num * sizeof(T), 0x00, data_num * sizeof(T));
    CUST_KERNEL_CHECK_FALSE(ctx, (ret == EOK), KERNEL_STATUS_PARAM_INVALID, "Dropout2d memset_s failed.");
    std::fill(&mask[0], &mask[data_num], false);
    return KERNEL_STATUS_OK;
  }

  long seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::mt19937 g(seed);
  std::bernoulli_distribution b(p);

  for (int i = 0; i < static_cast<int>(channel_num); ++i) {
    bool drop = b(g);
    if (drop) {
      auto ret = memset_s(output + channel_size * i, channel_size * sizeof(T), 0x00, channel_size * sizeof(T));
      CUST_KERNEL_CHECK_FALSE(ctx, (ret == EOK), KERNEL_STATUS_PARAM_INVALID, "Dropout2d memset_s failed.");
      std::fill(&mask[channel_size * i], &mask[channel_size * (i + 1)], false);
    } else {
      for (int j = 0; j < static_cast<int>(channel_size); ++j) {
        size_t id = channel_size * i + j;
        output[id] = CalcOutput<T>(input[id], p);
        mask[id] = true;
      }
    }
  }
  return KERNEL_STATUS_OK;
}

uint32_t Dropout2dCpuKernel::Compute(CpuKernelContext &ctx) {
  uint32_t res = GetInputAndCheck(ctx);
  if (res != KERNEL_STATUS_OK) {
    return res;
  }

  static std::map<int,
                  std::function<uint32_t(CpuKernelContext &, float, std::vector<void *> &, std::vector<int64_t> &)>>
    calls = {
      {DT_FLOAT16, CalDropout2d<Eigen::half>}, {DT_FLOAT, CalDropout2d<float>},     {DT_DOUBLE, CalDropout2d<double>},
      {DT_INT8, CalDropout2d<int8_t>},         {DT_INT16, CalDropout2d<int16_t>},   {DT_INT32, CalDropout2d<int32_t>},
      {DT_INT64, CalDropout2d<int64_t>},       {DT_UINT8, CalDropout2d<uint8_t>},   {DT_UINT16, CalDropout2d<uint16_t>},
      {DT_UINT32, CalDropout2d<uint32_t>},     {DT_UINT64, CalDropout2d<uint64_t>}, {DT_BOOL, CalDropout2d<bool>}};

  auto iter = calls.find(input_dtype_);
  if (iter == calls.end()) {
    CUST_KERNEL_LOG_ERROR(ctx, "Dropout2d op don't support index tensor types: %s", typeid(input_dtype_).name());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  return iter->second(ctx, p_, io_addrs_, input_shape_);
}

uint32_t Dropout2dCpuKernel::GetInputAndCheck(CpuKernelContext &ctx) {
  AttrValue *keep_prob = ctx.GetAttr("keep_prob");
  CUST_KERNEL_CHECK_NULLPTR(ctx, keep_prob, KERNEL_STATUS_PARAM_INVALID, "get attr:keep_prob failed.");
  p_ = 1 - keep_prob->GetFloat();

  // get input_tensor
  Tensor *input_tensor = ctx.Input(0);
  if (input_tensor == nullptr) {
    CUST_KERNEL_LOG_ERROR(ctx, "Dropout2dCpuKernel::get input:0 failed");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  input_dtype_ = static_cast<DataType>(input_tensor->GetDataType());

  std::shared_ptr<TensorShape> input_shape = input_tensor->GetTensorShape();
  CUST_KERNEL_CHECK_FALSE(ctx, (input_shape->GetDims() == 4), KERNEL_STATUS_PARAM_INVALID,
                          "Dropout2d input tensor must be 4-D.");
  input_shape_ = input_shape->GetDimSizes();

  Tensor *output_tensor = ctx.Output(0);
  io_addrs_.push_back(reinterpret_cast<void *>(input_tensor->GetData()));
  io_addrs_.push_back(reinterpret_cast<void *>(output_tensor->GetData()));
  io_addrs_.push_back(reinterpret_cast<void *>(ctx.Output(1)->GetData()));
  CUST_KERNEL_CHECK_FALSE(ctx, (io_addrs_.size() == 3), KERNEL_STATUS_PARAM_INVALID,
                          "The size of io_addrs_ must be 3.");
  return KERNEL_STATUS_OK;
}

REGISTER_MS_CPU_KERNEL(kDropout2d, Dropout2dCpuKernel);
}  // namespace aicpu
