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
#include "segment_mean.h"

#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kInputNum = 2;
const uint32_t kOutputNum = 1;
const int64_t kParallelDataNum = 2 * 1024;
const char *kSegmentMean = "SegmentMean";

#define SEGMENTMEAN_COMPUTE_CASE(DTYPE, TYPE1, TYPE2, CTX)    \
  case (DTYPE): {                                             \
    uint32_t result = SegmentMeanCompute<TYPE1, TYPE2>(CTX);  \
    if (result != KERNEL_STATUS_OK) {                         \
      KERNEL_LOG_ERROR("SegmentMean kernel compute failed."); \
      return result;                                          \
    }                                                         \
    break;                                                    \
  }

#define SEGMENTMEAN_COMPUTE_CASE_Complex(DTYPE, TYPE1, TYPE2, CTX)   \
  case (DTYPE): {                                                    \
    uint32_t result = SegmentMeanCompute_Complex<TYPE1, TYPE2>(CTX); \
    if (result != KERNEL_STATUS_OK) {                                \
      KERNEL_LOG_ERROR("SegmentMean kernel compute failed.");        \
      return result;                                                 \
    }                                                                \
    break;                                                           \
  }

#define SEGMENTMEAN_COMPUTE_CASE_ALL(TYPE, CTX)                                                                       \
  SEGMENTMEAN_COMPUTE_CASE_Complex(DT_COMPLEX64, std::complex<float>, TYPE, CTX)                                      \
    SEGMENTMEAN_COMPUTE_CASE_Complex(DT_COMPLEX128, std::complex<double>, TYPE, CTX)                                  \
      SEGMENTMEAN_COMPUTE_CASE(DT_INT8, int8_t, TYPE, CTX) SEGMENTMEAN_COMPUTE_CASE(DT_INT16, int16_t, TYPE, CTX)     \
        SEGMENTMEAN_COMPUTE_CASE(DT_INT32, int32_t, TYPE, CTX) SEGMENTMEAN_COMPUTE_CASE(DT_INT64, int64_t, TYPE, CTX) \
          SEGMENTMEAN_COMPUTE_CASE(DT_UINT8, uint8_t, TYPE, CTX)                                                      \
            SEGMENTMEAN_COMPUTE_CASE(DT_UINT16, uint16_t, TYPE, CTX)                                                  \
              SEGMENTMEAN_COMPUTE_CASE(DT_UINT32, uint32_t, TYPE, CTX)                                                \
                SEGMENTMEAN_COMPUTE_CASE(DT_UINT64, uint64_t, TYPE, CTX)                                              \
                  SEGMENTMEAN_COMPUTE_CASE(DT_FLOAT16, Eigen::half, TYPE, CTX)                                        \
                    SEGMENTMEAN_COMPUTE_CASE(DT_FLOAT, float, TYPE, CTX)                                              \
                      SEGMENTMEAN_COMPUTE_CASE(DT_DOUBLE, double, TYPE, CTX)
}  // namespace

namespace aicpu {
template <typename T>
T ComplexDiv(T sum, int64_t num) {
  T res;
  auto real = sum.real();
  auto imag = sum.imag();
  res.real(real / num);
  res.imag(imag / num);
  return res;
}

uint32_t SegmentMeanCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "SegmentMean check input and output number failed.");
  Tensor *input_data = ctx.Input(0);
  KERNEL_CHECK_NULLPTR(input_data->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input[0] failed.")
  Tensor *segment_ids_data = ctx.Input(1);
  KERNEL_CHECK_NULLPTR(segment_ids_data->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input[1] failed.")
  Tensor *output_data = ctx.Output(0);
  KERNEL_CHECK_NULLPTR(output_data->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get output[0] failed.");
  auto data_type = ctx.Input(0)->GetDataType();
  auto segment_ids_type = ctx.Input(1)->GetDataType();
  switch (segment_ids_type) {
    case DT_INT32: {
      switch (data_type) {
        SEGMENTMEAN_COMPUTE_CASE_ALL(int32_t, ctx)
        default:
          KERNEL_LOG_ERROR("Input[0] data type[%s] not supported.", DTypeStr(data_type).c_str());
          return KERNEL_STATUS_PARAM_INVALID;
      }
      break;
    }
    case DT_INT64: {
      switch (data_type) {
        SEGMENTMEAN_COMPUTE_CASE_ALL(int64_t, ctx)
        default:
          KERNEL_LOG_ERROR("Input[0] data type[%s] not supported.", DTypeStr(data_type).c_str());
          return KERNEL_STATUS_PARAM_INVALID;
      }
      break;
    }
    default: {
      KERNEL_LOG_ERROR("Input[1] data type[%s] not supported.", DTypeStr(segment_ids_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }
  return KERNEL_STATUS_OK;
}

template <typename T1, typename T2>
uint32_t SegmentMeanCpuKernel::SegmentMeanCompute(CpuKernelContext &ctx) {
  Tensor *input_data = ctx.Input(0);
  auto input_data_addr = reinterpret_cast<T1 *>(input_data->GetData());
  int64_t input_data_num = input_data->NumElements();
  Tensor *segment_ids_data = ctx.Input(1);
  auto segment_ids_len = segment_ids_data->NumElements();
  auto segment_ids_data_addr = reinterpret_cast<T2 *>(segment_ids_data->GetData());
  int64_t segment_ids_data_num = segment_ids_data->NumElements();
  Tensor *output_data = ctx.Output(0);
  auto output_data_addr = reinterpret_cast<T1 *>(output_data->GetData());
  auto output_data_shape_sizes = input_data->GetTensorShape()->GetDimSizes();
  output_data_shape_sizes[0] = segment_ids_data_addr[segment_ids_len - 1] + 1;
  output_data->GetTensorShape()->SetDimSizes(output_data_shape_sizes);
  int64_t output_data_num = output_data->NumElements();
  for (int64_t i = 0; i < output_data_num; i++) {
    output_data_addr[i] = static_cast<T1>(0);
  }
  std::vector<int64_t> segments;
  if (segment_ids_data_num != (input_data->GetTensorShape()->GetDimSize(0))) {
    KERNEL_LOG_ERROR("The amount of data for input[1] must be equal to the first dimension of input[0].");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (segment_ids_data_addr[0] < 0) {
    KERNEL_LOG_ERROR("Input[1] must be nonnegative data.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  int64_t seg_tmp = 1;
  for (int64_t i = 0; i < segment_ids_data_num - 1; i++) {
    if (segment_ids_data_addr[i] > segment_ids_data_addr[i + 1]) {
      KERNEL_LOG_ERROR("Input[1] must be an ascending ordered sequence.");
      return KERNEL_STATUS_PARAM_INVALID;
    }
    if (segment_ids_data_addr[i] == segment_ids_data_addr[i + 1]) {
      seg_tmp++;
    } else {
      segments.push_back(seg_tmp);
      seg_tmp = 1;
    }
    if (i == segment_ids_data_num - 2) {
      segments.push_back(seg_tmp);
    }
  }
  const int64_t num_compare_per = input_data_num / (input_data->GetTensorShape()->GetDimSize(0));
  const int64_t num_segments = segments.size();
  if (num_segments < kParallelDataNum) {
    for (int64_t i = 0; i < num_segments; i++) {
      int64_t count = segments[i];
      int64_t count_no = 0;
      for (int64_t j = 0; j < i; j++) {
        count_no += segments[j];
      }
      int64_t input_addr_base = count_no * num_compare_per;
      if (num_compare_per < kParallelDataNum) {
        for (int64_t j = 0; j < num_compare_per; j++) {
          int64_t mean_init_addr = input_addr_base + j;
          T1 sum_value = input_data_addr[mean_init_addr];
          for (int64_t k = 1; k < count; k++) {
            int cmp_addr = mean_init_addr + k * num_compare_per;
            sum_value += input_data_addr[cmp_addr];
          }
          output_data_addr[segment_ids_data_addr[count_no] * num_compare_per + j] = sum_value / count;
        }
      } else {
        uint32_t min_core_num = 1;
        int64_t mean_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
        if (mean_core_num > num_compare_per) {
          mean_core_num = num_compare_per;
        }
        auto shard_compute = [&](size_t start, size_t end) {
          for (size_t j = start; j < end; j++) {
            int64_t mean_init_addr = input_addr_base + j;
            T1 sum_value = input_data_addr[mean_init_addr];
            for (int64_t k = 1; k < count; k++) {
              int cmp_addr = mean_init_addr + k * num_compare_per;
              sum_value += input_data_addr[cmp_addr];
            }
            output_data_addr[segment_ids_data_addr[count_no] * num_compare_per + j] = sum_value / count;
          }
        };
        KERNEL_HANDLE_ERROR(
          CpuKernelUtils::ParallelFor(ctx, num_compare_per, num_compare_per / mean_core_num, shard_compute),
          "SegmentMean Compute failed.");
      }
    }
  } else {
    uint32_t min_core_num_seg = 1;
    int64_t mean_core_num_seg = std::max(min_core_num_seg, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
    if (mean_core_num_seg > num_segments) {
      mean_core_num_seg = num_segments;
    }
    auto shard_compute_seg = [&](size_t start_seg, size_t end_seg) {
      for (size_t i = start_seg; i < end_seg; i++) {
        int64_t count = segments[i];
        int64_t count_no = 0;
        for (size_t j = 0; j < i; j++) {
          count_no += segments[j];
        }
        int64_t input_addr_base = count_no * num_compare_per;
        for (int64_t j = 0; j < num_compare_per; j++) {
          int64_t mean_init_addr = input_addr_base + j;
          T1 sum_value = input_data_addr[mean_init_addr];
          for (int64_t k = 1; k < count; k++) {
            int cmp_addr = mean_init_addr + k * num_compare_per;
            sum_value += input_data_addr[cmp_addr];
          }
          output_data_addr[segment_ids_data_addr[count_no] * num_compare_per + j] = sum_value / count;
        }
      }
    };
    KERNEL_HANDLE_ERROR(
      CpuKernelUtils::ParallelFor(ctx, num_segments, num_segments / mean_core_num_seg, shard_compute_seg),
      "SegmentMean Compute failed.");
  }
  return KERNEL_STATUS_OK;
}

template <typename T1, typename T2>
uint32_t SegmentMeanCpuKernel::SegmentMeanCompute_Complex(CpuKernelContext &ctx) {
  Tensor *input_data = ctx.Input(0);
  auto input_data_addr = reinterpret_cast<T1 *>(input_data->GetData());
  int64_t input_data_num = input_data->NumElements();
  Tensor *segment_ids_data = ctx.Input(1);
  auto segment_ids_len = segment_ids_data->NumElements();
  auto segment_ids_data_addr = reinterpret_cast<T2 *>(segment_ids_data->GetData());
  int64_t segment_ids_data_num = segment_ids_data->NumElements();
  Tensor *output_data = ctx.Output(0);
  auto output_data_addr = reinterpret_cast<T1 *>(output_data->GetData());
  auto output_data_shape_sizes = input_data->GetTensorShape()->GetDimSizes();
  output_data_shape_sizes[0] = segment_ids_data_addr[segment_ids_len - 1] + 1;
  output_data->GetTensorShape()->SetDimSizes(output_data_shape_sizes);
  int64_t output_data_num = output_data->NumElements();
  for (int64_t i = 0; i < output_data_num; i++) {
    output_data_addr[i] = static_cast<T1>(0);
  }
  std::vector<int64_t> segments;
  if (segment_ids_data_num != (input_data->GetTensorShape()->GetDimSize(0))) {
    KERNEL_LOG_ERROR("The amount of data for input[1] must be equal to the first dimension of input[0].");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (segment_ids_data_addr[0] < 0) {
    KERNEL_LOG_ERROR("Input[1] must be nonnegative data.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  int64_t seg_tmp = 1;
  for (int64_t i = 0; i < segment_ids_data_num - 1; i++) {
    if (segment_ids_data_addr[i] > segment_ids_data_addr[i + 1]) {
      KERNEL_LOG_ERROR("Input[1] must be an ascending ordered sequence.");
      return KERNEL_STATUS_PARAM_INVALID;
    }
    if (segment_ids_data_addr[i] == segment_ids_data_addr[i + 1]) {
      seg_tmp++;
    } else {
      segments.push_back(seg_tmp);
      seg_tmp = 1;
    }
    if (i == segment_ids_data_num - 2) {
      segments.push_back(seg_tmp);
    }
  }
  const int64_t num_compare_per = input_data_num / (input_data->GetTensorShape()->GetDimSize(0));
  const int64_t num_segments = segments.size();
  if (num_segments < kParallelDataNum) {
    for (int64_t i = 0; i < num_segments; i++) {
      int64_t count = segments[i];
      int64_t count_no = 0;
      for (int64_t j = 0; j < i; j++) {
        count_no += segments[j];
      }
      int64_t input_addr_base = count_no * num_compare_per;
      if (num_compare_per < kParallelDataNum) {
        for (int64_t j = 0; j < num_compare_per; j++) {
          int64_t mean_init_addr = input_addr_base + j;
          T1 sum_value = input_data_addr[mean_init_addr];
          for (int64_t k = 1; k < count; k++) {
            int cmp_addr = mean_init_addr + k * num_compare_per;
            sum_value += input_data_addr[cmp_addr];
          }
          output_data_addr[segment_ids_data_addr[count_no] * num_compare_per + j] = ComplexDiv(sum_value, count);
        }
      } else {
        uint32_t min_core_num = 1;
        int64_t mean_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
        if (mean_core_num > num_compare_per) {
          mean_core_num = num_compare_per;
        }
        auto shard_compute = [&](size_t start, size_t end) {
          for (size_t j = start; j < end; j++) {
            int64_t mean_init_addr = input_addr_base + j;
            T1 sum_value = input_data_addr[mean_init_addr];
            for (int64_t k = 1; k < count; k++) {
              int cmp_addr = mean_init_addr + k * num_compare_per;
              sum_value += input_data_addr[cmp_addr];
            }
            output_data_addr[segment_ids_data_addr[count_no] * num_compare_per + j] = ComplexDiv(sum_value, count);
          }
        };
        KERNEL_HANDLE_ERROR(
          CpuKernelUtils::ParallelFor(ctx, num_compare_per, num_compare_per / mean_core_num, shard_compute),
          "SegmentMean Compute failed.");
      }
    }
  } else {
    uint32_t min_core_num_seg = 1;
    int64_t mean_core_num_seg = std::max(min_core_num_seg, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
    if (mean_core_num_seg > num_segments) {
      mean_core_num_seg = num_segments;
    }
    auto shard_compute_seg = [&](size_t start_seg, size_t end_seg) {
      for (size_t i = start_seg; i < end_seg; i++) {
        int64_t count = segments[i];
        int64_t count_no = 0;
        for (size_t j = 0; j < i; j++) {
          count_no += segments[j];
        }
        int64_t input_addr_base = count_no * num_compare_per;
        for (int64_t j = 0; j < num_compare_per; j++) {
          int64_t mean_init_addr = input_addr_base + j;
          T1 sum_value = input_data_addr[mean_init_addr];
          for (int64_t k = 1; k < count; k++) {
            int cmp_addr = mean_init_addr + k * num_compare_per;
            sum_value += input_data_addr[cmp_addr];
          }
          output_data_addr[segment_ids_data_addr[count_no] * num_compare_per + j] = ComplexDiv(sum_value, count);
        }
      }
    };
    KERNEL_HANDLE_ERROR(
      CpuKernelUtils::ParallelFor(ctx, num_segments, num_segments / mean_core_num_seg, shard_compute_seg),
      "SegmentMean Compute failed.");
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kSegmentMean, SegmentMeanCpuKernel);
}  // namespace aicpu