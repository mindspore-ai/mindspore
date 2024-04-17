/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
#include "cpu_kernel/ms_kernel/affine_grid.h"
#include <Eigen/Dense>
#include <Eigen/Eigen>
#include <algorithm>
#include <vector>
#include "Eigen/Core"
#include "context/inc/cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#include "utils/bcast.h"
using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace {
constexpr uint32_t kAffineGridInputNum = 2;
constexpr uint32_t kAffineGridOutputNum = 1;
const char *kAffineGrid = "AffineGrid";
constexpr int64_t kParallelDataNums = 1 * 1024;
const int64_t kParallelDataNumSameShape = 1 * 1024;
const int64_t kParallelDataNumSameShapeMid = 3 * 1024;
const int64_t SIZE_BOUNDARY = 100000;
enum kColNum : size_t {
  kColNum0 = 0,
  kColNum1,
  kColNum2,
  kColNum3,
  kColNum4,
  kColNum5,
  kColNum6,
  kColNum7,
  kColNum8,
  kColNum9,
  kColNum10,
  kColNum11,
  kColNum12
};

std::vector<int64_t> GetOutputSizeData(aicpu::DataType type_out_size, int64_t len_out_size, void *data) {
  std::vector<int64_t> data_out_size;
  if (type_out_size == aicpu::DT_INT64) {
    auto temp = reinterpret_cast<int64_t *>(data);
    for (int64_t i = 0; i < len_out_size; i++) {
      data_out_size.push_back(temp[i]);
    }
  } else {
    auto temp = reinterpret_cast<int32_t *>(data);
    for (int64_t i = 0; i < len_out_size; i++) {
      data_out_size.push_back(temp[i]);
    }
  }
  return data_out_size;
}

#define AFFINEGRID_COMPUTE_CASE(DTYPE, TYPE, CTX)                      \
  case (DTYPE): {                                                      \
    uint32_t result = AffineGridCompute<TYPE>(CTX);                    \
    ;                                                                  \
    if (result != KERNEL_STATUS_OK) {                                  \
      CUST_KERNEL_LOG_ERROR(ctx, "AffineGrid kernel compute failed."); \
      return result;                                                   \
    }                                                                  \
    break;                                                             \
  }
}  // namespace

namespace aicpu {
uint32_t AffineGridCpuKernel::Compute(CpuKernelContext &ctx) {
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kAffineGridInputNum, kAffineGridOutputNum),
                           "[%s] check input and output failed.", kAffineGrid);
  DataType data_type = ctx.Output(0)->GetDataType();
  switch (data_type) {
    AFFINEGRID_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    AFFINEGRID_COMPUTE_CASE(DT_FLOAT, float, ctx)
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "AffineGrid kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
      break;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
void AffineGridCpuKernel::SpecialCompute(bool theta_3D, Eigen::MatrixXf &all, T *data_theta, int64_t start, int64_t end,
                                         int64_t result_row, T *output) {
  if (theta_3D) {
    Eigen::MatrixXf theta(kColNum3, kColNum2);
    Eigen::MatrixXf result(result_row, kColNum2);
    float result_0 = 0.0;
    float result_1 = 0.0;
    for (int64_t n = start; n < end; n++) {
      theta(kColNum0, kColNum0) = static_cast<float>(*(data_theta + (n * kColNum6) + kColNum0));
      theta(kColNum1, kColNum0) = static_cast<float>(*(data_theta + (n * kColNum6) + kColNum1));
      theta(kColNum2, kColNum0) = static_cast<float>(*(data_theta + (n * kColNum6) + kColNum2));
      theta(kColNum0, kColNum1) = static_cast<float>(*(data_theta + (n * kColNum6) + kColNum3));
      theta(kColNum1, kColNum1) = static_cast<float>(*(data_theta + (n * kColNum6) + kColNum4));
      theta(kColNum2, kColNum1) = static_cast<float>(*(data_theta + (n * kColNum6) + kColNum5));
      result = all * theta;
      int64_t k_num = n * kColNum2 * result_row;
      for (int64_t k = 0; k < result_row; k++) {
        result_0 = result(k, kColNum0);
        result_1 = result(k, kColNum1);
        *(output + k_num) = static_cast<T>(result_0);
        *(output + k_num + kColNum1) = static_cast<T>(result_1);
        k_num += kColNum2;
      }
    }
  } else {
    Eigen::MatrixXf theta(kColNum4, kColNum3);
    Eigen::MatrixXf result(result_row, kColNum3);
    float result_0 = 0.0;
    float result_1 = 0.0;
    float result_2 = 0.0;
    for (int64_t n = start; n < end; n++) {
      theta(kColNum0, kColNum0) = static_cast<float>(*(data_theta + (n * kColNum12) + kColNum0));
      theta(kColNum1, kColNum0) = static_cast<float>(*(data_theta + (n * kColNum12) + kColNum1));
      theta(kColNum2, kColNum0) = static_cast<float>(*(data_theta + (n * kColNum12) + kColNum2));
      theta(kColNum3, kColNum0) = static_cast<float>(*(data_theta + (n * kColNum12) + kColNum3));
      theta(kColNum0, kColNum1) = static_cast<float>(*(data_theta + (n * kColNum12) + kColNum4));
      theta(kColNum1, kColNum1) = static_cast<float>(*(data_theta + (n * kColNum12) + kColNum5));
      theta(kColNum2, kColNum1) = static_cast<float>(*(data_theta + (n * kColNum12) + kColNum6));
      theta(kColNum3, kColNum1) = static_cast<float>(*(data_theta + (n * kColNum12) + kColNum7));
      theta(kColNum0, kColNum2) = static_cast<float>(*(data_theta + (n * kColNum12) + kColNum8));
      theta(kColNum1, kColNum2) = static_cast<float>(*(data_theta + (n * kColNum12) + kColNum9));
      theta(kColNum2, kColNum2) = static_cast<float>(*(data_theta + (n * kColNum12) + kColNum10));
      theta(kColNum3, kColNum2) = static_cast<float>(*(data_theta + (n * kColNum12) + kColNum11));
      result = all * theta;
      int64_t k_num = n * kColNum3 * result_row;
      for (int64_t k = 0; k < result_row; k++) {
        result_0 = result(k, kColNum0);
        result_1 = result(k, kColNum1);
        result_2 = result(k, kColNum2);
        *(output + k_num) = static_cast<T>(result_0);
        *(output + k_num + kColNum1) = static_cast<T>(result_1);
        *(output + k_num + kColNum2) = static_cast<T>(result_2);
        k_num += kColNum3;
      }
    }
  }
}

template <typename T>
uint32_t AffineGridCpuKernel::CommonCompute(CpuKernelContext &ctx, int64_t data_num, int64_t result_row, bool theta_3D,
                                            Eigen::MatrixXf &all) {
  auto output_y = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  auto data_theta = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  if (data_num >= kParallelDataNumSameShape) {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);

    if (data_num <= kParallelDataNumSameShapeMid) {
      max_core_num = std::min(max_core_num, 4U);  // up to 4 cpu cores
    }

    if (max_core_num > data_num) {
      max_core_num = data_num;
    }

    auto sharder_affiregrid = [&](int64_t start, int64_t end) {
      SpecialCompute<T>(theta_3D, all, data_theta, start, end, result_row, output_y);
    };

    if (max_core_num == 0) {
      CUST_KERNEL_LOG_ERROR(ctx, "max_core_num could not be 0.");
      return KERNEL_STATUS_INNER_ERROR;
    }
    CUST_KERNEL_HANDLE_ERROR(ctx,
                             CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, sharder_affiregrid),
                             "AffineGrid Compute failed.");
  } else {
    SpecialCompute<T>(theta_3D, all, data_theta, 0, data_num, result_row, output_y);
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t AffineGridCpuKernel::AffineGridCompute4D(CpuKernelContext &ctx, std::vector<int64_t> &data_out_size,
                                                  bool align_corners) {
  int64_t N = data_out_size[0];
  int64_t H = data_out_size[kColNum2];
  int64_t W = data_out_size[kColNum3];
  if (N > SIZE_BOUNDARY || H > SIZE_BOUNDARY || W > SIZE_BOUNDARY || N <= 0 || H <= 0 || W <= 0) {
    CUST_KERNEL_LOG_ERROR(ctx, "AffineGrid input size invalid.");
    return KERNEL_STATUS_INNER_ERROR;
  }
  Eigen::VectorXd vecX, vecY;
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
  if (align_corners == false) {
    x_ = static_cast<double>((W - 1)) / static_cast<double>(W);
    y_ = static_cast<double>((H - 1)) / static_cast<double>(H);
  }
  for (int64_t i = 0; i < W; i++) {
    vecX[i] = vecX[i] * x_;
  }
  for (int64_t i = 0; i < H; i++) {
    vecY[i] = vecY[i] * y_;
  }
  Eigen::MatrixXf all(W * H, kColNum3);
  for (int64_t i = 0; i < H; i++) {
    for (int64_t j = 0; j < W; j++) {
      all(i * W + j, kColNum0) = vecX(j);
      all(i * W + j, kColNum1) = vecY(i);
      all(i * W + j, kColNum2) = 1.0;
    }
  }
  int64_t result_row = H * W;
  bool theta_3D = true;
  int64_t data_num = N;
  return CommonCompute<T>(ctx, data_num, result_row, theta_3D, all);
}

template <typename T>
uint32_t AffineGridCpuKernel::AffineGridCompute5D(CpuKernelContext &ctx, std::vector<int64_t> &data_out_size,
                                                  bool align_corners) {
  int64_t N = data_out_size[0];
  int64_t D = data_out_size[kColNum2];
  int64_t H = data_out_size[kColNum3];
  int64_t W = data_out_size[kColNum4];
  if (N > SIZE_BOUNDARY || D > SIZE_BOUNDARY || H > SIZE_BOUNDARY || W > SIZE_BOUNDARY || N <= 0 || H <= 0 || D <= 0 ||
      W <= 0) {
    CUST_KERNEL_LOG_ERROR(ctx, "AffineGrid input size invalid.");
    return KERNEL_STATUS_INNER_ERROR;
  }
  Eigen::VectorXd vecX, vecY, vecZ;
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
    x_ = static_cast<double>((W - 1)) / static_cast<double>(W);
    y_ = static_cast<double>((H - 1)) / static_cast<double>(H);
    z_ = static_cast<double>((D - 1)) / static_cast<double>(D);
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
  Eigen::MatrixXf all(D * W * H, kColNum4);
  int i_j_k = 0;
  for (int64_t i = 0; i < D; i++) {
    for (int64_t j = 0; j < H; j++) {
      for (int64_t k = 0; k < W; k++) {
        all(i_j_k, 0) = vecX(k);
        all(i_j_k, kColNum1) = vecY(j);
        all(i_j_k, kColNum2) = vecZ(i);
        all(i_j_k, kColNum3) = 1.0;
        i_j_k += 1;
      }
    }
  }
  int64_t result_row = D * H * W;
  bool theta_3D = false;
  int64_t data_num = N;
  return CommonCompute<T>(ctx, data_num, result_row, theta_3D, all);
}

template <typename T>
uint32_t AffineGridCpuKernel::AffineGridCompute(CpuKernelContext &ctx) {
  BCalcInfo calc_info;
  calc_info.input_0 = ctx.Input(0);
  calc_info.input_1 = ctx.Input(1);
  calc_info.output = ctx.Output(0);
  auto type_out_size = calc_info.input_1->GetDataType();
  auto len_out_size = calc_info.input_1->NumElements();
  auto data = calc_info.input_1->GetData();
  auto data_out_size = GetOutputSizeData(type_out_size, len_out_size, data);
  bool align_corners = false;
  AttrValue *align_corners_attr_ptr = ctx.GetAttr("align_corners");
  if (align_corners_attr_ptr) {
    align_corners = ctx.GetAttr("align_corners")->GetBool();
  }
  std::vector<long int> outputsize = calc_info.input_1->GetTensorShape()->GetDimSizes();
  if (outputsize[0] == kColNum4) {
    return AffineGridCompute4D<T>(ctx, data_out_size, align_corners);
  } else if (outputsize[0] == kColNum5) {
    return AffineGridCompute5D<T>(ctx, data_out_size, align_corners);
  }
  return KERNEL_STATUS_OK;
}

REGISTER_MS_CPU_KERNEL(kAffineGrid, AffineGridCpuKernel);
}  // namespace aicpu
