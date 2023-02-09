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
#include "multinomial.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iterator>
#include <limits>
#include <random>
#include <string>

#include "Eigen/Core"
#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

using namespace std;

namespace {
const char *kMultinomial = "Multinomial";
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 2;
constexpr int64_t kParallelDataNums = 40 * 1024;
constexpr int64_t kNumPerThread = 2048;
using RNG_Engine = std::mt19937;

// override isfinite for half
bool isfinite(Eigen::half &data) { return Eigen::half_impl::isfinite(data); }

}  // namespace

namespace aicpu {

template <typename T_in, typename T_out>
uint32_t Generate(Tensor *&input_0, Tensor *&input_1, Tensor *&input_count, Tensor *&input_state, Tensor *&output,
                  CpuKernelContext &ctx) {
  int64_t num_classes = input_0->GetTensorShape()->GetDimSize(1);
  int64_t batch_size = input_0->GetTensorShape()->GetDimSize(0);
  int32_t num_samples = *(reinterpret_cast<int32_t *>(input_1->GetData()));
  // count the execution times of the op
  uint64_t count = *(reinterpret_cast<uint64_t *>(input_count->GetData()));
  // seed of the op, passed between executions, which make op stateful
  int64_t state = *(reinterpret_cast<int64_t *>(input_state->GetData()));

  // setup seed
  int64_t final_seed = 0;
  auto attr_seed = ctx.GetAttr("seed");
  auto attr_seed2 = ctx.GetAttr("seed2");
  if (count != 0) {
    final_seed = state;
  } else if (attr_seed2 != nullptr && attr_seed2->GetInt() != 0) {
    final_seed = attr_seed2->GetInt();
  } else if (attr_seed != nullptr && attr_seed->GetInt() != 0) {
    final_seed = attr_seed->GetInt();
  } else {
    std::random_device r;
    final_seed = r();
  }
  // setup random engine
  RNG_Engine rng;
  rng.seed(final_seed);
  auto count_ptr = reinterpret_cast<uint64_t *>(input_count->GetData());
  ++count_ptr[0];
  if (count_ptr[0] == 0) {
    ++count_ptr[0];
  }
  auto state_ptr = reinterpret_cast<int64_t *>(input_state->GetData());
  state_ptr[0] = rng();

  auto input_0_data = reinterpret_cast<T_in *>(input_0->GetData());
  auto output_data = reinterpret_cast<T_out *>(output->GetData());
  auto total_num = output->NumElements();

  if (total_num < kParallelDataNums) {
    auto cur_out = output_data;

    for (int i = 0; i < batch_size; i++) {
      double *cumulative_distribution_function = new double[num_classes];

      double running_total = 0;
      auto row_start = input_0_data + i * num_classes;

      // calculate cdf (Cumulative Distribution Function)
      for (int64_t j = 0; j < num_classes; ++j) {
        if (isfinite(*(row_start + j))) {
          running_total += row_start[j];
        }
        cumulative_distribution_function[j] = running_total;
      }

      for (int j = 0; j < num_samples; j++) {
        std::uniform_real_distribution<double> Unifrom_01(0, 1);
        double rand = Unifrom_01(rng);
        auto found_iter = std::upper_bound(cumulative_distribution_function,
                                           cumulative_distribution_function + num_classes, rand * running_total);

        *cur_out =
          static_cast<T_out>(std::min(num_classes - 1, std::distance(cumulative_distribution_function, found_iter)));
        ++cur_out;
      }
      if (cumulative_distribution_function != nullptr) {
        delete[] cumulative_distribution_function;
      }
    }
  } else {
    double *rand_list = new double[total_num];
    auto shard = [&](size_t start_outer, size_t end_outer) {
      double *cumulative_distribution_function = new double[num_classes];
      RNG_Engine rng_outer = rng;
      rng_outer.discard(start_outer * num_samples);
      double running_total = 0;

      auto row_start = input_0_data + start_outer * num_classes;

      // calculate cdf (Cumulative Distribution Function)
      for (int64_t j = 0; j < num_classes; ++j) {
        if (isfinite(*(row_start + j))) {
          running_total += row_start[j];
        }
        cumulative_distribution_function[j] = running_total;
      }

      auto cur_output = output_data + start_outer * num_samples;
      auto shard_inner = [&](int start_inner, int end_inner) {
        RNG_Engine rng_inner = rng_outer;
        rng_inner.discard(start_inner * kNumPerThread);
        // Generate each sample.
        auto j_start = start_inner * kNumPerThread;
        auto j_end = j_start + kNumPerThread > num_samples ? num_samples : j_start + kNumPerThread;

        for (int j = j_start; j < j_end; j++) {
          std::uniform_real_distribution<double> Unifrom_01(0, 1);
          double rand = Unifrom_01(rng_inner);
          rand_list[start_outer * num_samples + j] = rand;
          auto found_iter = std::upper_bound(cumulative_distribution_function,
                                             cumulative_distribution_function + num_classes, rand * running_total);

          *(cur_output + j) =
            static_cast<T_out>(std::min(num_classes - 1, std::distance(cumulative_distribution_function, found_iter)));
        }
      };
      CpuKernelUtils::ParallelFor(ctx, ceil((double)num_samples / kNumPerThread), 1, shard_inner);
      if (cumulative_distribution_function != nullptr) {
        delete[] cumulative_distribution_function;
      }
    };
    CpuKernelUtils::ParallelFor(ctx, batch_size, 1, shard);

    delete[] rand_list;
  }

  return KERNEL_STATUS_OK;
}

// inspired by cast.cc
void MultinomialCpuKernel::SetMap() {
  calls_[DT_FLOAT16][DT_INT32] = Generate<Eigen::half, int32_t>;
  calls_[DT_FLOAT][DT_INT32] = Generate<float, int32_t>;
  calls_[DT_DOUBLE][DT_INT32] = Generate<double, int32_t>;
  calls_[DT_FLOAT16][DT_INT64] = Generate<Eigen::half, int64_t>;
  calls_[DT_FLOAT][DT_INT64] = Generate<float, int64_t>;
  calls_[DT_DOUBLE][DT_INT64] = Generate<double, int64_t>;
}

uint32_t MultinomialCpuKernel::Compute(CpuKernelContext &ctx) {
  // check input output size
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "Multinomial check input and output number failed.");
  Tensor *input_0 = ctx.Input(kFirstInputIndex);
  Tensor *input_1 = ctx.Input(kSecondInputIndex);
  Tensor *input_count = ctx.Input(kThirdInputIndex);
  Tensor *input_state = ctx.Input(kFourthInputIndex);
  Tensor *output = ctx.Output(kFirstOutputIndex);

  // check input datatype
  DataType input0_datatype = input_0->GetDataType();
  KERNEL_CHECK_FALSE((input0_datatype == DT_FLOAT16 || input0_datatype == DT_DOUBLE || input0_datatype == DT_FLOAT),
                     KERNEL_STATUS_PARAM_INVALID,
                     "Input[0] data type must DT_FLOAT16 or DT_FLOAT or DT_DOUBLE,"
                     "but got data type[%s].",
                     DTypeStr(input0_datatype).c_str());
  DataType input1_datatype = input_1->GetDataType();
  KERNEL_CHECK_FALSE((input1_datatype == DT_INT32), KERNEL_STATUS_PARAM_INVALID,
                     "Input[1] data type must DT_INT32, but got data type[%s].", DTypeStr(input1_datatype).c_str());

  // check input dimension
  KERNEL_CHECK_FALSE((input_0->GetTensorShape()->GetDims() == 2), KERNEL_STATUS_PARAM_INVALID,
                     "Input[0] should be a matrix, but got rank [%d].", input_0->GetTensorShape()->GetDims());
  // scalar input is converted to rank-zero tensor in the dynamic input scenario.
  // rank-zero tensor from ms has a dim of 1, so limit input1 dim so that it's smaller than 1.
  KERNEL_CHECK_FALSE((input_1->GetTensorShape()->GetDims() <= 1), KERNEL_STATUS_PARAM_INVALID,
                     "Input[1] should be a scalar, but got rank [%d].", input_1->GetTensorShape()->GetDims());

  // check num_classes positive
  auto num_classes = input_0->GetTensorShape()->GetDimSize(1);  // int64_t
  KERNEL_CHECK_FALSE((num_classes > 0), KERNEL_STATUS_PARAM_INVALID, "num_classes should be positive, but got [%d].",
                     num_classes);

  // check num_samples nonnegative
  auto *num_samples_ptr = reinterpret_cast<int32_t *>(input_1->GetData());  // int32_t
  KERNEL_CHECK_FALSE((*num_samples_ptr >= 0), KERNEL_STATUS_PARAM_INVALID,
                     "num_samples should be nonnegative, but got [%d].", *num_samples_ptr);

  // set attr dtype to default val
  DataType data_type = DT_INT64;
  auto attr_dtype = ctx.GetAttr("dtype");
  if (attr_dtype != nullptr) {
    // check attr dtype
    data_type = static_cast<DataType>(attr_dtype->GetDataType());
    KERNEL_CHECK_FALSE((data_type == DT_INT32 || data_type == DT_INT64), KERNEL_STATUS_PARAM_INVALID,
                       "attr[dtype] must DT_INT32 or DT_INT64,"
                       "but got data type[%s].",
                       DTypeStr(data_type).c_str());
  }

  // attr dtype & output type match
  if (data_type != output->GetDataType()) {
    KERNEL_LOG_ERROR(
      "Multinomial kernel data type not matched, dtype is [%s], "
      "out_data_type is [%s].",
      DTypeStr(data_type).c_str(), DTypeStr(output->GetDataType()).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  SetMap();
  calls_[input0_datatype][data_type](input_0, input_1, input_count, input_state, output, ctx);
  calls_.clear();
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kMultinomial, MultinomialCpuKernel);
}  // namespace aicpu