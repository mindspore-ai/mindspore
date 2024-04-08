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
#include "ms_kernel/dropout3d.h"
#include <random>
#include <securec.h>
#include <Eigen/Core>
#include "utils/kernel_util.h"

namespace {
constexpr uint32_t kDropout3DInputNum = 1;
constexpr uint32_t kDropout3DOutputNum = 2;
const char *kDropout3D = "Dropout3D";
}  // namespace

namespace aicpu {
template <typename T>
static uint32_t CalDropout3d(float p, CpuKernelContext &ctx, std::vector<int64_t> &shape) {
  // inputs
  T *input = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  // outputs
  T *output = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  bool *mask = reinterpret_cast<bool *>(ctx.Output(1)->GetData());

  int64_t N = shape[0];
  int64_t C = shape[1];
  int64_t D = shape[2];
  int64_t H = shape[3];
  int64_t W = shape[4];

  size_t channel_num = N * C;
  size_t channel_size = D * H * W;
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
    CUST_KERNEL_CHECK_FALSE(ctx, (ret == EOK), KERNEL_STATUS_PARAM_INVALID, "Dropout3d memcpy_s failed.");
    std::fill(&mask[0], &mask[data_num], true);
    return KERNEL_STATUS_OK;
  } else if (p == 1) {
    auto ret = memset_s(output, data_num * sizeof(T), 0x00, data_num * sizeof(T));
    CUST_KERNEL_CHECK_FALSE(ctx, (ret == EOK), KERNEL_STATUS_PARAM_INVALID, "Dropout3d memset_s failed.");
    std::fill(&mask[0], &mask[data_num], false);
    return KERNEL_STATUS_OK;
  }

  long seed = std::random_device()();
  std::mt19937 g(seed);
  std::bernoulli_distribution b(p);

  std::function<void(size_t)> dropout_compute;
  if constexpr (std::is_same_v<T, Eigen::half>) {
    dropout_compute = [output, input, p](size_t id) { output[id] = input[id] / static_cast<T>(1 - p); };
  } else {
    dropout_compute = [output, input, p](size_t id) { output[id] = input[id] / (1 - p); };
  }

  for (int i = 0; i < static_cast<int>(channel_num); ++i) {
    bool drop = b(g);
    if (drop) {
      auto ret = memset_s(output + channel_size * i, channel_size * sizeof(T), 0x00, channel_size * sizeof(T));
      CUST_KERNEL_CHECK_FALSE(ctx, (ret == EOK), KERNEL_STATUS_PARAM_INVALID, "Dropout3d memset_s failed.");
      std::fill(&mask[channel_size * i], &mask[channel_size * (i + 1)], false);
    } else {
      for (int j = 0; j < static_cast<int>(channel_size); ++j) {
        size_t id = channel_size * i + j;
        dropout_compute(id);
        mask[id] = true;
      }
    }
  }
  return KERNEL_STATUS_OK;
}

uint32_t Dropout3DCpuKernel::Compute(CpuKernelContext &ctx) {
  uint32_t res = GetInputAndCheck(ctx);
  if (res != KERNEL_STATUS_OK) {
    return res;
  }

  std::map<DataType, std::function<uint32_t(float, CpuKernelContext &, std::vector<int64_t> &)>> calls;
  calls[DT_FLOAT16] = CalDropout3d<Eigen::half>;
  calls[DT_FLOAT] = CalDropout3d<float>;
  calls[DT_DOUBLE] = CalDropout3d<double>;
  calls[DT_INT8] = CalDropout3d<int8_t>;
  calls[DT_INT16] = CalDropout3d<int16_t>;
  calls[DT_INT32] = CalDropout3d<int32_t>;
  calls[DT_INT64] = CalDropout3d<int64_t>;
  calls[DT_UINT8] = CalDropout3d<uint8_t>;
  calls[DT_UINT16] = CalDropout3d<uint16_t>;
  calls[DT_UINT32] = CalDropout3d<uint32_t>;
  calls[DT_UINT64] = CalDropout3d<uint64_t>;
  calls[DT_BOOL] = CalDropout3d<bool>;

  auto iter = calls.find(input_dtype_);
  if (iter == calls.end()) {
    CUST_KERNEL_LOG_ERROR(ctx, "Dropout3d op don't support index tensor types: %s", typeid(input_dtype_).name());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  return iter->second(p_, ctx, input_shape_);
}

uint32_t Dropout3DCpuKernel::GetInputAndCheck(CpuKernelContext &ctx) {
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kDropout3DInputNum, kDropout3DOutputNum), "%s check failed.",
                           ctx.GetOpType().c_str());
  AttrValue *keep_prob = ctx.GetAttr("keep_prob");
  CUST_KERNEL_CHECK_NULLPTR(ctx, keep_prob, KERNEL_STATUS_PARAM_INVALID, "get attr:keep_prob failed.");
  p_ = 1 - keep_prob->GetFloat();

  // get input_tensor
  Tensor *input_tensor = ctx.Input(0);
  input_dtype_ = static_cast<DataType>(input_tensor->GetDataType());

  std::shared_ptr<TensorShape> input_shape = input_tensor->GetTensorShape();
  CUST_KERNEL_CHECK_FALSE(ctx, (input_shape->GetDims() == 5), KERNEL_STATUS_PARAM_INVALID,
                          "Dropout3d input tensor must be 5-D.");
  input_shape_ = input_shape->GetDimSizes();
  return KERNEL_STATUS_OK;
}
REGISTER_MS_CPU_KERNEL(kDropout3D, Dropout3DCpuKernel);
}  // namespace aicpu
