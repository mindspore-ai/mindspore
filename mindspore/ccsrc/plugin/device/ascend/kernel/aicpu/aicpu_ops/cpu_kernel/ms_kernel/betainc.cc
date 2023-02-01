/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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

#include "betainc.h"

#include <algorithm>
#include <iostream>
#include <vector>

#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
using namespace std;

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 3;
constexpr int64_t kParallelDataNums = 64;
const char *const kBetainc = "Betainc";
#define BETAINC_COMPUTE_CASE(DTYPE, TYPE, CTX)            \
  case (DTYPE): {                                         \
    uint32_t result = BetaincCompute<TYPE>(CTX);          \
    if (result != KERNEL_STATUS_OK) {                     \
      KERNEL_LOG_ERROR("Betainc kernel compute failed."); \
      return result;                                      \
    }                                                     \
    break;                                                \
  }
}  // namespace

namespace aicpu {
uint32_t BetaincCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "Betainc check input and output number failed.");
  auto a_shape = ctx.Input(0)->GetTensorShape();
  auto b_shape = ctx.Input(1)->GetTensorShape();
  auto x_shape = ctx.Input(2)->GetTensorShape();

  // dims check
  if (a_shape->GetDims() > 0 && b_shape->GetDims() > 0) {
    KERNEL_CHECK_FALSE((a_shape->GetDimSizes() == b_shape->GetDimSizes()), KERNEL_STATUS_PARAM_INVALID,
                       "Shapes of a and b are inconsistent")
  }
  if (a_shape->GetDims() > 0 && x_shape->GetDims() > 0) {
    KERNEL_CHECK_FALSE((a_shape->GetDimSizes() == x_shape->GetDimSizes()), KERNEL_STATUS_PARAM_INVALID,
                       "Shapes of a and x are inconsistent")
  }

  // check input datatype
  DataType a_dtype = ctx.Input(0)->GetDataType();
  DataType b_dtype = ctx.Input(1)->GetDataType();
  DataType x_dtype = ctx.Input(2)->GetDataType();

  KERNEL_CHECK_FALSE((b_dtype == a_dtype), KERNEL_STATUS_PARAM_INVALID,
                     "The data type of input[1] [%s] need be same with input[0] [%s].", DTypeStr(b_dtype).c_str(),
                     DTypeStr(a_dtype).c_str());
  KERNEL_CHECK_FALSE((x_dtype == a_dtype), KERNEL_STATUS_PARAM_INVALID,
                     "The data type of input[2] [%s] need be same with input[0] [%s].", DTypeStr(x_dtype).c_str(),
                     DTypeStr(a_dtype).c_str());

  switch (a_dtype) {
    BETAINC_COMPUTE_CASE(DT_FLOAT, float, ctx)
    BETAINC_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    default:
      KERNEL_LOG_ERROR("Betainc kernel data type [%s] not support.", DTypeStr(a_dtype).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

uint32_t SwitchParallel(const std::function<void(int64_t, int64_t)> &func, int64_t end_num, const CpuKernelContext &ctx,
                        int64_t max_core_num, int64_t data_num) {
  if (data_num <= kParallelDataNums) {
    func(0, end_num);
  } else {
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, end_num, end_num / max_core_num, func),
                        "Betainc func Compute failed.");
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t RunParallel(const CpuKernelContext &ctx, std::vector<T *> data_pointers, int data_num) {
  uint32_t min_core_num = 1;
  int64_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
  if (max_core_num > data_num) {
    max_core_num = data_num;
  }
  auto shard_betainc = [&](int64_t start, int64_t end) {
    Eigen::TensorMap<Eigen::Tensor<T, 1>> input_0(data_pointers[0] + start, end - start);
    Eigen::TensorMap<Eigen::Tensor<T, 1>> input_1(data_pointers[1] + start, end - start);
    Eigen::TensorMap<Eigen::Tensor<T, 1>> input_2(data_pointers[2] + start, end - start);
    Eigen::TensorMap<Eigen::Tensor<T, 1>> output(data_pointers[3] + start, end - start);
    output = Eigen::betainc(input_0, input_1, input_2);
  };

  return SwitchParallel(shard_betainc, data_num, ctx, max_core_num, data_num);
}

template <typename T>
uint32_t BetaincCpuKernel::BetaincCompute(CpuKernelContext &ctx) {
  auto input_a = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto input_b = reinterpret_cast<T *>(ctx.Input(1)->GetData());
  auto input_x = reinterpret_cast<T *>(ctx.Input(2)->GetData());
  auto output_z = reinterpret_cast<T *>(ctx.Output(0)->GetData());

  auto a_shape = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  auto b_shape = ctx.Input(1)->GetTensorShape()->GetDimSizes();
  auto x_shape = ctx.Input(2)->GetTensorShape()->GetDimSizes();

  int64_t bcast_size = 1;

  T *a_data = nullptr;
  T *b_data = nullptr;
  T *x_data = nullptr;
  std::vector<T *> data_pointers = {input_a, input_b, input_x, output_z};

  if (a_shape != b_shape || b_shape != x_shape) {
    // pseudo broadcast: bcast scalar
    std::vector<int64_t> bcast_shape = {1};

    bcast_shape = a_shape.empty() ? bcast_shape : a_shape;
    bcast_shape = b_shape.empty() ? bcast_shape : b_shape;
    bcast_shape = x_shape.empty() ? bcast_shape : x_shape;

    for (size_t i = 0; i < bcast_shape.size(); i++) {
      bcast_size *= bcast_shape[i];
    }
    cout << "bcast_size = " << bcast_size << endl;

    if (a_shape.empty()) {
      a_data = new T[bcast_size];
      (void)std::fill_n(a_data, bcast_size, *input_a);
      data_pointers[0] = a_data;
    }

    if (b_shape.empty()) {
      b_data = new T[bcast_size];
      (void)std::fill_n(b_data, bcast_size, *input_b);
      data_pointers[1] = b_data;
    }

    if (x_shape.empty()) {
      x_data = new T[bcast_size];
      (void)std::fill_n(x_data, bcast_size, *input_x);
      data_pointers[2] = x_data;
    }
  } else {
    bcast_size = ctx.Input(0)->NumElements();
  }

  uint32_t result = RunParallel<T>(ctx, data_pointers, static_cast<int>(bcast_size));

  if (a_data != nullptr) {
    delete[] a_data;
  }
  if (x_data != nullptr) {
    delete[] x_data;
  }
  if (b_data != nullptr) {
    delete[] b_data;
  }

  return result;
}

REGISTER_CPU_KERNEL(kBetainc, BetaincCpuKernel);
}  // namespace aicpu
