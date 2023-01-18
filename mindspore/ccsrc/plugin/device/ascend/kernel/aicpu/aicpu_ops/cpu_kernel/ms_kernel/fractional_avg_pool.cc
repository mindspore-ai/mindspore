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

#include "fractional_avg_pool.h"

#include "Eigen/Dense"
#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const char *kFractionalAvgPool = "FractionalAvgPool";
const uint32_t k_InputNum = 1;
const uint32_t k_OutputNum = 3;
const int64_t kParallelDataNum = 1024 * 1024;
constexpr uint32_t tensor_in_and_out_dims = 4;
}  // namespace

namespace aicpu {
uint32_t FractionalAvgPoolCpuKernel::FractionalAvgPoolParamCheck(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, k_InputNum, k_OutputNum),
                      "FractionalAvgPool Check input and output number failed.");
  Tensor *input = ctx.Input(0);
  if (ctx.Input(0)->GetDataType() != ctx.Output(0)->GetDataType()) {
    KERNEL_LOG_ERROR("The data type of the output [%s] need be the same as the input [%s]",
                     DTypeStr(ctx.Output(0)->GetDataType()).c_str(), DTypeStr(ctx.Input(0)->GetDataType()).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  auto input_shape = input->GetTensorShape();
  int32_t input_dims = input_shape->GetDims();
  for (int32_t i = 0; i < input_dims; i++) {
    KERNEL_CHECK_FALSE((input_shape->GetDimSize(i) > 0), KERNEL_STATUS_PARAM_INVALID,
                       "FractionalAvgPool: expected input to have non-empty spatial "
                       "dimensions, "
                       "but input has sizes [%d] with dimension [%d] being empty.",
                       input_dims, i);
  }
  KERNEL_CHECK_FALSE((input_dims == tensor_in_and_out_dims), KERNEL_STATUS_PARAM_INVALID,
                     "tensor_in must be 4-dimensional.");
  AttrValue *pooling_ratio = ctx.GetAttr("pooling_ratio");
  KERNEL_CHECK_NULLPTR(pooling_ratio, KERNEL_STATUS_PARAM_INVALID, "[%s] get attr:pooling_ratio failed.",
                       kFractionalAvgPool);
  int32_t pooling_ratio_size = pooling_ratio->ListFloatSize();
  KERNEL_CHECK_FALSE((pooling_ratio_size == tensor_in_and_out_dims), KERNEL_STATUS_PARAM_INVALID,
                     "pooling_ratio field must specify 4 dimensions.");
  std::vector<float> pooling_ratio_data = ctx.GetAttr("pooling_ratio")->GetListFloat();
  KERNEL_CHECK_FALSE((pooling_ratio_data[0] == 1.0 && pooling_ratio_data[3] == 1.0), KERNEL_STATUS_PARAM_INVALID,
                     "FractionalAvgPool is not yet supported on the batch nor channel "
                     "dimension.The first and last elements of pooling ratio must be 1.0.");
  return KERNEL_STATUS_OK;
}

static std::vector<int64_t> GeneratePoolingSequencePseudoRandom(int input_length, int output_length, int seed) {
  // generate a random number u which is in (0,1)
  std::vector<int64_t> cum_seq(output_length + 1, 0);
  std::vector<int64_t> diff(output_length, 0);
  double alpha = static_cast<double>(input_length) / output_length;
  int k = input_length / output_length;
  double u_max1 = (k + 2) / alpha - 1;
  double u_max2 = (input_length + 1 - k) / alpha - (output_length - 1);
  double max_u = std::min(u_max1, u_max2);
  std::default_random_engine random(seed);
  std::uniform_real_distribution<double> dis2(0.0, 1.0);
  const double u = dis2(random) * max_u;
  cum_seq[0] = 1;
  cum_seq[output_length] = input_length + 1;
  for (int i = 1; i < output_length; ++i) {
    cum_seq[i] = static_cast<int>(ceil(alpha * (i + u)));
  }
  for (int i = 0; i < output_length; ++i) {
    diff[i] = cum_seq[i + 1] - cum_seq[i];
  }
  return diff;
}

static std::vector<int64_t> GeneratePoolingSequenceRandom(int input_length, int output_length, int seed) {
  int k = input_length / output_length;
  int num_random_spot = input_length % output_length;
  std::vector<int64_t> diff(output_length, k);
  for (int i = 0; i < num_random_spot; ++i) {
    diff[i] += 1;
  }
  std::srand(seed);
  random_shuffle(diff.begin(), diff.end());
  return diff;
}

std::vector<int64_t> GeneratePoolingSequence(int input_length, int output_length, bool pseudo_random, int seed) {
  std::vector<int64_t> diff;
  if (input_length % output_length == 0) {
    diff = std::vector<int64_t>(output_length, input_length / output_length);
  }
  if (pseudo_random) {
    diff = GeneratePoolingSequencePseudoRandom(input_length, output_length, seed);
  } else {
    diff = GeneratePoolingSequenceRandom(input_length, output_length, seed);
  }
  int k = input_length / output_length;
  for (int i = 0; i < output_length; i++) {
    if (diff[i] < k || diff[i] > k + 1) {
      KERNEL_LOG_ERROR("FractionalAvgPool kernel GeneratePoolingSequence diff[%d] is error");
    }
  }
  std::vector<int64_t> cum_seq(output_length + 1, 0);
  for (size_t i = 1; i < cum_seq.size(); ++i) {
    cum_seq[i] = cum_seq[i - 1] + diff[i - 1];
  }
  return cum_seq;
}

template <typename T>
uint32_t FractionalAvgPoolCpuKernel::DoCompute(CpuKernelContext &ctx) {
  Tensor *input = ctx.Input(0);
  Tensor *output = ctx.Output(0);
  Tensor *row_pooling_sequence = ctx.Output(1);
  Tensor *col_pooling_sequence = ctx.Output(2);
  std::vector<float> pooling_ratio = ctx.GetAttr("pooling_ratio")->GetListFloat();
  AttrValue *pseudo_random_ = ctx.GetAttr("pseudo_random");
  bool pseudo_random = (pseudo_random_ == nullptr) ? false : (pseudo_random_->GetBool());
  AttrValue *overlapping_ = ctx.GetAttr("overlapping");
  bool overlapping = (overlapping_ == nullptr) ? false : (overlapping_->GetBool());
  AttrValue *deterministic_ = ctx.GetAttr("deterministic");
  bool deterministic = (deterministic_ == nullptr) ? false : (deterministic_->GetBool());
  AttrValue *seed_ = ctx.GetAttr("seed");
  int seed = (seed_ == nullptr) ? 0 : (seed_->GetInt());
  AttrValue *seed2_ = ctx.GetAttr("seed2");
  int seed2 = (seed2_ == nullptr) ? 0 : (seed2_->GetInt());
  auto input_shape = input->GetTensorShape();
  std::vector<int> input_size(tensor_in_and_out_dims);
  std::vector<int> output_size(tensor_in_and_out_dims);
  for (uint32_t i = 0; i < tensor_in_and_out_dims; ++i) {
    input_size[i] = input_shape->GetDimSize(i);
  }
  for (uint32_t i = 0; i < tensor_in_and_out_dims; ++i) {
    output_size[i] = static_cast<int>(std::floor(input_size[i] / pooling_ratio[i]));
    KERNEL_CHECK_FALSE((output_size[i] > 0), KERNEL_STATUS_PARAM_INVALID,
                       "FractionalAvgPool kernel outputsize[%d] cannot be 0");
  }
  auto input_data = static_cast<T *>(ctx.Input(0)->GetData());
  auto output_data = static_cast<T *>(output->GetData());
  auto output_height_seq_tensor = static_cast<int64_t *>(row_pooling_sequence->GetData());
  auto output_width_seq_tensor = static_cast<int64_t *>(col_pooling_sequence->GetData());
  std::random_device rd;
  std::mt19937 generator(rd());
  if (deterministic) {
    // If both seeds are not set when deterministic is true, force set seeds.
    if ((seed == 0) && (seed2 == 0)) {
      seed = generator();
      seed2 = generator();
    }
  } else {
    KERNEL_CHECK_FALSE(((seed == 0) && (seed2 == 0)), KERNEL_STATUS_PARAM_INVALID,
                       "Both seed and seed2 should be 0 if deterministic is false.");
  }
  if (seed == 0 && seed2 != 0) {
    seed = seed2;
  }
  // Generate pooling sequence.
  std::vector<int64_t> height_cum_seq;
  std::vector<int64_t> width_cum_seq;
  height_cum_seq = GeneratePoolingSequence(input_size[1], output_size[1], pseudo_random, seed);
  width_cum_seq = GeneratePoolingSequence(input_size[2], output_size[2], pseudo_random, seed);
  for (uint32_t i = 0; i < height_cum_seq.size(); ++i) {
    *(output_height_seq_tensor + i) = height_cum_seq[i];
  }
  for (uint32_t i = 0; i < width_cum_seq.size(); ++i) {
    *(output_width_seq_tensor + i) = width_cum_seq[i];
  }
  const int64_t height_max = input_size[1] - 1;
  const int64_t width_max = input_size[2] - 1;
  const int64_t depth_max = input_size[3] - 1;
  uint64_t data_num = input->NumElements();
  /**
   * For both input and output,
   * 0: batch
   * 1: height / row
   * 2: width / col
   * 3: depth / channel
   */
  if (data_num < kParallelDataNum) {
    for (int64_t b = 0; b < input_size[0]; ++b) {
      // height sequence.
      for (uint32_t hs = 0; hs < height_cum_seq.size() - 1; ++hs) {
        // height start and end.
        const int64_t height_start = height_cum_seq[hs];
        int64_t height_end = overlapping ? height_cum_seq[hs + 1] : height_cum_seq[hs + 1] - 1;
        height_end = std::min(height_end, height_max);
        // width sequence.
        for (uint32_t ws = 0; ws < width_cum_seq.size() - 1; ++ws) {
          for (int64_t c = 0; c <= depth_max; ++c) {
            const int64_t out_offset = ((b * output_size[1] + hs) * output_size[2] + ws) * output_size[3] + c;
            // Initializes the output tensor with 0.
            T sum = static_cast<T>(0);
            T avg = static_cast<T>(0);
            int count = 0;
            // width start and end.
            const int64_t width_start = width_cum_seq[ws];
            int64_t width_end = overlapping ? width_cum_seq[ws + 1] : width_cum_seq[ws + 1] - 1;
            width_end = std::min(width_end, width_max);
            for (int64_t h = height_start; h <= height_end; ++h) {
              for (int64_t w = width_start; w <= width_end; ++w) {
                const int64_t in_offset = ((b * input_size[1] + h) * input_size[2] + w) * output_size[3] + c;
                sum += input_data[in_offset];
                count++;
              }
            }
            avg = sum / static_cast<T>(count);
            *(output_data + out_offset) = avg;
          }
        }
      }
    }
  } else {
    uint64_t height_cum_len = height_cum_seq.size() - 1;
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx));
    if (max_core_num > height_cum_len) {
      max_core_num = height_cum_len;
    }
    for (int64_t b = 0; b < input_size[0]; ++b) {
      // height sequence.
      auto sharder_fractionalavgpool_index = [&](size_t start, size_t end) {
        for (uint32_t hs = start; hs < end; ++hs) {
          // height start and end.
          const int64_t height_start = height_cum_seq[hs];
          int64_t height_end = overlapping ? height_cum_seq[hs + 1] : height_cum_seq[hs + 1] - 1;
          height_end = std::min(height_end, height_max);
          // width sequence.
          for (uint32_t ws = 0; ws < width_cum_seq.size() - 1; ++ws) {
            for (int64_t c = 0; c <= depth_max; ++c) {
              const int64_t out_offset = ((b * output_size[1] + hs) * output_size[2] + ws) * output_size[3] + c;
              // Initializes the output tensor with 0.
              T sum = static_cast<T>(0);
              T avg = static_cast<T>(0);
              int count = 0;
              // width start and end.
              const int64_t width_start = width_cum_seq[ws];
              int64_t width_end = overlapping ? width_cum_seq[ws + 1] : width_cum_seq[ws + 1] - 1;
              width_end = std::min(width_end, width_max);
              for (int64_t h = height_start; h <= height_end; ++h) {
                for (int64_t w = width_start; w <= width_end; ++w) {
                  const int64_t in_offset = ((b * input_size[1] + h) * input_size[2] + w) * output_size[3] + c;
                  sum += input_data[in_offset];
                  count++;
                }
              }
              avg = sum / static_cast<T>(count);
              *(output_data + out_offset) = avg;
            }
          }
        }
      };
      KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, height_cum_len, height_cum_len / max_core_num,
                                                      sharder_fractionalavgpool_index),
                          "FractionalAvgPool Index Compute failed");
    }
  }
  return KERNEL_STATUS_OK;
}

uint32_t FractionalAvgPoolCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(FractionalAvgPoolParamCheck(ctx), "Check FractionalAvgPool params failed.");
  Tensor *input = ctx.Input(0);
  auto data_type = input->GetDataType();
  switch (data_type) {
    case DT_FLOAT:
      return DoCompute<float>(ctx);
    case DT_DOUBLE:
      return DoCompute<double>(ctx);
    case DT_INT32:
      return DoCompute<int32_t>(ctx);
    case DT_INT64:
      return DoCompute<int64_t>(ctx);
    default:
      KERNEL_LOG_ERROR("FractionalAvgPool kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kFractionalAvgPool, FractionalAvgPoolCpuKernel);
}  // namespace aicpu
