/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
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
#include "cpu_kernel/ms_kernel/affine_grid_grad.h"
#include <Eigen/Dense>
#include <algorithm>
#include <vector>
#include "cpu_kernel/common/cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace {
constexpr uint32_t kAffineGridGradInputNum = 2;
constexpr uint32_t kAffineGridGradOutputNum = 1;
const char *kAffineGridGrad = "AffineGridGrad";
const int row_num_1 = 1;
const int col_num_1 = 1;
const int y_grad_4_value_4D = 2;
const int row_num_2 = 2;
const int col_num_2 = 2;
const int y_grad_5_value_5D = 3;
const int row_num_3 = 3;
const int col_num_3 = 3;
const int row_num_4 = 4;
const int col_num_4 = 4;
const int int64_size = 8;
const int len_x_size_4D = 4;
const int len_x_size_5D = 5;
const int N = 0;
const int y_grad_H_4D = 1;
const int y_grad_W_4D = 2;
const int y_grad_3_4D = 3;
const int x_size_H_4D = 2;
const int x_size_W_4D = 3;
const int y_grad_D_5D = 1;
const int y_grad_H_5D = 2;
const int y_grad_W_5D = 3;
const int y_grad_4_5D = 4;
const int x_size_D_5D = 2;
const int x_size_H_5D = 3;
const int x_size_W_5D = 4;

// when input data size is more than kParallelDataNumSameShape, use Parallel
// func
const int64_t kParallelDataNumMid = 16 * 1024;
const int64_t kParallelDataNumSameShape = 10 * 1024;

#define AFFINEGRIDGRAD_COMPUTE_CASE(DTYPE, TYPE, DTYPE0, CTX)    \
  case (DTYPE): {                                                \
    uint32_t result;                                             \
    if ((DTYPE0) == DT_INT32) {                                  \
      result = AffineGridGradCompute<TYPE, int32_t>(CTX);        \
    } else {                                                     \
      result = AffineGridGradCompute<TYPE, int64_t>(CTX);        \
    }                                                            \
    if (result != KERNEL_STATUS_OK) {                            \
      KERNEL_LOG_ERROR("AffineGridGrad kernel compute failed."); \
      return result;                                             \
    }                                                            \
    break;                                                       \
  }
}  // namespace

namespace aicpu {
uint32_t AffineGridGradCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kAffineGridGradInputNum, kAffineGridGradOutputNum),
                      "[%s] check input and output failed.", kAffineGridGrad);
  auto data_type0 = static_cast<DataType>(ctx.Input(kFirstInputIndex)->GetDataType());
  auto data_type1 = static_cast<DataType>(ctx.Input(kSecondInputIndex)->GetDataType());
  auto data_type2 = static_cast<DataType>(ctx.Output(kFirstOutputIndex)->GetDataType());
  if ((data_type1 != DT_INT32) && (data_type1 != DT_INT64)) {
    KERNEL_LOG_ERROR(
      "[%s] Data type of x_size requires int32 or int64, but got data type "
      "[%s].",
      ctx.GetOpType().c_str(), DTypeStr(data_type1).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (data_type0 != data_type2) {
    KERNEL_LOG_ERROR(
      "[%s] Data type of y_grad and x_grad requires same, but got data type "
      "[%s] and [%s].",
      ctx.GetOpType().c_str(), DTypeStr(data_type0).c_str(), DTypeStr(data_type2).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (data_type1 == DT_INT32) {
    KERNEL_HANDLE_ERROR(AffineGridGradCheck<int32_t>(ctx), "[%s] check params failed.", kAffineGridGrad);
  } else if (data_type1 == DT_INT64) {
    KERNEL_HANDLE_ERROR(AffineGridGradCheck<int64_t>(ctx), "[%s] check params failed.", kAffineGridGrad);
  }
  switch (data_type0) {
    AFFINEGRIDGRAD_COMPUTE_CASE(DT_FLOAT16, Eigen::half, data_type1, ctx)
    AFFINEGRIDGRAD_COMPUTE_CASE(DT_FLOAT, float, data_type1, ctx)
    default:
      KERNEL_LOG_ERROR("AffineGridGrad kernel data type [%s] not support.", DTypeStr(data_type0).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <typename T0>
uint32_t AffineGridGradCpuKernel::AffineGridGradCheck(const CpuKernelContext &ctx) {
  auto input_0 = ctx.Input(kFirstInputIndex);
  auto input_1 = ctx.Input(kSecondInputIndex);
  auto output_0 = ctx.Output(kFirstOutputIndex);
  KERNEL_CHECK_NULLPTR(input_0->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input 0 data failed.")
  KERNEL_CHECK_NULLPTR(input_1->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input 1 data failed.")
  KERNEL_CHECK_NULLPTR(output_0->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get output 0 data failed.")
  std::vector<int64_t> outputsize = ctx.Input(1)->GetTensorShape()->GetDimSizes();
  std::vector<int64_t> input0 = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  auto input1 = reinterpret_cast<T0 *>(ctx.Input(1)->GetData());

  if (outputsize[0] == len_x_size_4D) {
    if ((input0[N] != input1[N]) || (input0[y_grad_H_4D] != input1[x_size_H_4D]) ||
        (input0[y_grad_W_4D] != input1[x_size_W_4D]) || (input0[y_grad_3_4D] != y_grad_4_value_4D)) {
      KERNEL_LOG_ERROR("There are some dimensional constraints between input0 and input1");
      return KERNEL_STATUS_PARAM_INVALID;
    }
  } else if (outputsize[0] == len_x_size_5D) {
    if ((input0[N] != input1[N]) || (input0[y_grad_D_5D] != input1[x_size_D_5D]) ||
        (input0[y_grad_H_5D] != input1[x_size_H_5D]) || (input0[y_grad_W_5D] != input1[x_size_W_5D]) ||
        (input0[y_grad_4_5D] != y_grad_5_value_5D)) {
      KERNEL_LOG_ERROR("There are some dimensional constraints between input0 and input1");
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }
  return KERNEL_STATUS_OK;
}

template <typename T, typename T0>
uint32_t AffineGridGradCpuKernel::AffineGridGradCompute(const CpuKernelContext &ctx) {
  bool align_corners = false;
  AttrValue *align_corners_attr_ptr = ctx.GetAttr("align_corners");
  KERNEL_CHECK_NULLPTR(align_corners_attr_ptr, KERNEL_STATUS_PARAM_INVALID, "'align_corners' is nullptr.")
  if (align_corners_attr_ptr) {
    align_corners = align_corners_attr_ptr->GetBool();
  }
  std::vector<int64_t> outputsize = ctx.Input(1)->GetTensorShape()->GetDimSizes();
  if (outputsize[0] == len_x_size_4D) {
    AffineGridGradCompute_4D<T, T0>(ctx, align_corners);
  } else if (outputsize[0] == len_x_size_5D) {
    AffineGridGradCompute_5D<T, T0>(ctx, align_corners);
  }
  return KERNEL_STATUS_OK;
}

template <typename T, typename T0>
uint32_t AffineGridGradCpuKernel::AffineGridGradCompute_5D(const CpuKernelContext &ctx, bool align_corners) {
  auto *data_out_size = reinterpret_cast<T0 *>(ctx.Input(1)->GetData());
  auto D = *(data_out_size + x_size_D_5D);
  auto H = *(data_out_size + x_size_H_5D);
  auto W = *(data_out_size + x_size_W_5D);

  Eigen::VectorXd vecX;
  Eigen::VectorXd vecY;
  Eigen::VectorXd vecZ;
  vecX.setZero(W, 1);
  vecY.setZero(H, 1);
  vecZ.setZero(D, 1);
  if (W != 1) {
    vecX = Eigen::VectorXd::LinSpaced(vecX.size(), -1.0, 1.0);
  }
  if (H != 1) {
    vecY = Eigen::VectorXd::LinSpaced(vecY.size(), -1.0, 1.0);
  }
  if (D != 1) {
    vecZ = Eigen::VectorXd::LinSpaced(vecZ.size(), -1.0, 1.0);
  }
  double x_ = 1;
  double y_ = 1;
  double z_ = 1;
  if (align_corners == 0) {
    x_ = static_cast<double>(W - 1) / static_cast<double>(W);
    y_ = static_cast<double>(H - 1) / static_cast<double>(H);
    z_ = static_cast<double>(D - 1) / static_cast<double>(D);
  }
  for (int64_t i = 0; i < W; i++) {
    vecX[i] = vecX[i] * x_;
  }
  for (int64_t i = 0; i < H; i++) {
    vecY[i] = vecY[i] * y_;
  }
  for (int64_t i = 0; i < D; i++) {
    vecZ[i] = vecZ[i] * z_;
  }

  Eigen::MatrixXf all(row_num_4, D * W * H);
  all = make_base_grid_5D<T0>(ctx, vecX, vecY, vecZ);
  DoCompute_5D<T, T0>(ctx, all);
  return KERNEL_STATUS_OK;
}

template <typename T0>
Eigen::MatrixXf AffineGridGradCpuKernel::make_base_grid_5D(const CpuKernelContext &ctx, Eigen::VectorXd vecX,
                                                           Eigen::VectorXd vecY, Eigen::VectorXd vecZ) {
  auto *data_out_size = reinterpret_cast<T0 *>(ctx.Input(1)->GetData());
  auto D = *(data_out_size + x_size_D_5D);
  auto H = *(data_out_size + x_size_H_5D);
  auto W = *(data_out_size + x_size_W_5D);
  Eigen::MatrixXf all(row_num_4, D * W * H);

  int64_t datanums = D * H * W;
  if (datanums * int64_size >= kParallelDataNumSameShape) {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
    if (datanums <= kParallelDataNumMid) {
      max_core_num = std::min(max_core_num, 4U);
    }
    if (max_core_num > datanums) {
      max_core_num = datanums;
    }
    auto sharder_affine_grid_grad_5D = [&](int64_t start, int64_t end) {
      for (int64_t t = start; t < end; t++) {
        int64_t i = t / (H * W);
        int64_t j = (t % (H * W)) / W;
        int64_t k = (t % (H * W)) % W;
        all(0, i * H * W + j * W + k) = vecX(k);
        all(row_num_1, i * H * W + j * W + k) = vecY(j);
        all(row_num_2, i * H * W + j * W + k) = vecZ(i);
        all(row_num_3, i * H * W + j * W + k) = 1.0;
      }
    };
    CpuKernelUtils::ParallelFor(ctx, datanums, datanums / max_core_num, sharder_affine_grid_grad_5D);
  } else {
    int i_j_k = 0;
    for (int64_t i = 0; i < D; i++) {
      for (int64_t t = 0; t < H * W; t++) {
        int64_t j = t / W;
        int64_t k = t % W;
        all(0, i_j_k) = vecX(k);
        all(row_num_1, i_j_k) = vecY(j);
        all(row_num_2, i_j_k) = vecZ(i);
        all(row_num_3, i_j_k) = 1.0;
        i_j_k += 1;
      }
    }
  }
  return all;
}

template <typename T, typename T0>
uint32_t AffineGridGradCpuKernel::DoCompute_5D(const CpuKernelContext &ctx, Eigen::MatrixXf all) {
  auto *data_out_size = reinterpret_cast<T0 *>(ctx.Input(1)->GetData());
  auto N = data_out_size[0];
  auto D = *(data_out_size + x_size_D_5D);
  auto H = *(data_out_size + x_size_H_5D);
  auto W = *(data_out_size + x_size_W_5D);
  auto data_y_grad = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto output = reinterpret_cast<T *>(ctx.Output(0)->GetData());

  Eigen::MatrixXf y_grad(D * H * W, col_num_3);
  Eigen::MatrixXf result(row_num_4, col_num_3);
  int64_t k_num = 0;

  for (int64_t n = 0; n < N; n++) {
    for (int64_t k = 0; k < D * H * W; k++) {
      y_grad(k, 0) = static_cast<float>(*(data_y_grad + (n * D * H * W * col_num_3 + k * col_num_3) + 0));
      y_grad(k, col_num_1) =
        static_cast<float>(*(data_y_grad + (n * D * H * W * col_num_3 + k * col_num_3) + col_num_1));
      y_grad(k, col_num_2) =
        static_cast<float>(*(data_y_grad + (n * D * H * W * col_num_3 + k * col_num_3) + col_num_2));
    }
    result = all * y_grad;
    for (int64_t k = 0; k < col_num_3; k++) {
      float result_0 = result(0, k);
      float result_1 = result(row_num_1, k);
      float result_2 = result(row_num_2, k);
      float result_3 = result(row_num_3, k);
      *(output + k_num) = static_cast<T>(result_0);
      *(output + k_num + col_num_1) = static_cast<T>(result_1);
      *(output + k_num + col_num_2) = static_cast<T>(result_2);
      *(output + k_num + col_num_3) = static_cast<T>(result_3);
      k_num += col_num_4;
    }
  }
  return KERNEL_STATUS_OK;
}

template <typename T, typename T0>
uint32_t AffineGridGradCpuKernel::AffineGridGradCompute_4D(const CpuKernelContext &ctx, bool align_corners) {
  auto *data_out_size = reinterpret_cast<T0 *>(ctx.Input(1)->GetData());
  auto H = *(data_out_size + x_size_H_4D);
  auto W = *(data_out_size + x_size_W_4D);

  Eigen::VectorXd vecX;
  Eigen::VectorXd vecY;
  vecX.setZero(W, 1);
  vecY.setZero(H, 1);
  if (W != 1) {
    vecX = Eigen::VectorXd::LinSpaced(vecX.size(), -1.0, 1.0);
  }
  if (H != 1) {
    vecY = Eigen::VectorXd::LinSpaced(vecY.size(), -1.0, 1.0);
  }
  double x_ = 1;
  double y_ = 1;
  if (align_corners == 0) {
    x_ = static_cast<double>((W - 1)) / double(W);
    y_ = static_cast<double>((H - 1)) / double(H);
  }
  for (int64_t i = 0; i < W; i++) {
    vecX[i] = vecX[i] * x_;
  }
  for (int64_t i = 0; i < H; i++) {
    vecY[i] = vecY[i] * y_;
  }

  Eigen::MatrixXf all(row_num_3, W * H);
  all = make_base_grid_4D<T0>(ctx, vecX, vecY);
  DoCompute_4D<T, T0>(ctx, all);
  return KERNEL_STATUS_OK;
}

template <typename T0>
Eigen::MatrixXf AffineGridGradCpuKernel::make_base_grid_4D(const CpuKernelContext &ctx, Eigen::VectorXd vecX,
                                                           Eigen::VectorXd vecY) {
  auto *data_out_size = reinterpret_cast<T0 *>(ctx.Input(1)->GetData());
  auto H = *(data_out_size + x_size_H_4D);
  auto W = *(data_out_size + x_size_W_4D);
  Eigen::MatrixXf all(row_num_3, W * H);

  int64_t datanums = H * W;
  if (datanums * int64_size >= kParallelDataNumSameShape) {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
    if (datanums <= kParallelDataNumMid) {
      max_core_num = std::min(max_core_num, 4U);
    }
    if (max_core_num > datanums) {
      max_core_num = datanums;
    }
    auto sharder_affine_grid_grad_4D = [&](int64_t start, int64_t end) {
      for (int64_t t = start; t < end; t++) {
        int64_t j = t % W;
        int64_t i = t / W;
        all(0, i * W + j) = vecX(j);
        all(row_num_1, i * W + j) = vecY(i);
        all(row_num_2, i * W + j) = 1.0;
      }
    };
    CpuKernelUtils::ParallelFor(ctx, datanums, datanums / max_core_num, sharder_affine_grid_grad_4D);
  } else {
    for (int64_t i = 0; i < H; i++) {
      for (int64_t j = 0; j < W; j++) {
        all(0, i * W + j) = vecX(j);
        all(row_num_1, i * W + j) = vecY(i);
        all(row_num_2, i * W + j) = 1.0;
      }
    }
  }
  return all;
}

template <typename T, typename T0>
uint32_t AffineGridGradCpuKernel::DoCompute_4D(const CpuKernelContext &ctx, Eigen::MatrixXf all) {
  auto *data_out_size = reinterpret_cast<T0 *>(ctx.Input(1)->GetData());
  auto N = data_out_size[0];
  auto H = *(data_out_size + x_size_H_4D);
  auto W = *(data_out_size + x_size_W_4D);
  auto data_y_grad = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto output = reinterpret_cast<T *>(ctx.Output(0)->GetData());

  Eigen::MatrixXf y_grad(H * W, col_num_2);
  Eigen::MatrixXf result(row_num_3, col_num_2);
  int64_t k_num = 0;

  for (int64_t n = 0; n < N; n++) {
    for (int64_t k = 0; k < H * W; k++) {
      y_grad(k, 0) = static_cast<float>(*(data_y_grad + (n * H * W * col_num_2 + k * col_num_2)));
      y_grad(k, col_num_1) = static_cast<float>(*(data_y_grad + (n * H * W * col_num_2 + k * col_num_2) + col_num_1));
    }
    result = all * y_grad;

    for (int64_t k = 0; k < col_num_2; k++) {
      float result_0 = result(0, k);
      float result_1 = result(row_num_1, k);
      float result_2 = result(row_num_2, k);
      *(output + k_num) = static_cast<T>(result_0);
      *(output + k_num + col_num_1) = static_cast<T>(result_1);
      *(output + k_num + col_num_2) = static_cast<T>(result_2);
      k_num += col_num_3;
    }
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kAffineGridGrad, AffineGridGradCpuKernel);
}  // namespace aicpu
