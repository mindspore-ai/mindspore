/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
#include "matrix_exp.h"

#include <array>
#include <complex>
#include <cmath>
#include "cpu_kernel_utils.h"
#include "utils/kernel_util.h"

namespace {
constexpr uint32_t kMatrixExpInputNum = 1;
constexpr uint32_t kMatrixExpOutputNum = 1;
constexpr uint32_t kIndexTwo = 2;
const int64_t paralled_data_size = 8 * 1024;
const char *kMatrixExp = "MatrixExp";
constexpr int total_n_degs = 6;

// Coefficients for computing taylor approximant of order 8.
constexpr double sqrt_177 = 0.1330413469565007072504e+2, x3 = 2. / 3.;
constexpr double x1 = x3 * ((1. + sqrt_177) / 88.), x2 = x3 * ((1. + sqrt_177) / 352.);
constexpr double x4 = (-271. + 29. * sqrt_177) / (315. * x3), x5 = (-11. + 11. * sqrt_177) / (1260. * x3);
constexpr double x6 = (-99. + 11. * sqrt_177) / (5040. * x3), x7 = (89. - sqrt_177) / (5040. * x3);
constexpr double y2 = (857. - 58. * sqrt_177) / 630.;

template <typename T, int ROW, int COL>
using array2d = std::array<std::array<T, COL>, ROW>;

// Coefficients for computing taylor approximant of order 12.
constexpr int num_prods_12 = 4;
array2d<double, num_prods_12, num_prods_12> b12 = {
  {{9.0198e-16, 0.46932117595418237389, -0.20099424927047284052, -0.04623946134063071740},
   {5.31597895759871264183, 1.19926790417132231573, 0.01179296240992997031, 0.01108844528519167989},
   {0.18188869982170434744, 0.05502798439925399070, 0.09351590770535414968, 0.00610700528898058230},
   {-2.0861320e-13, -0.13181061013830184015, -0.02027855540589259079, -0.00675951846863086359}}};

// Coefficients for computing taylor approximant of order 18.
constexpr int num_prods_18 = 5;
array2d<double, num_prods_18, num_prods_18> b18 = {
  {{0., -1.00365581030144618291e-01, -8.02924648241156932449e-03, -8.92138498045729985177e-04, 0.},
   {0., 3.97849749499645077844e-01, 1.36783778460411720168e+00, 4.98289622525382669416e-01,
    -6.37898194594723280150e-04},
   {-1.09676396052962061844e+01, 1.68015813878906206114e+00, 5.71779846478865511061e-02, -6.98210122488052056106e-03,
    3.34975017086070470649e-05},
   {-9.04316832390810593223e-02, -6.76404519071381882256e-02, 6.75961301770459654925e-02, 2.95552570429315521194e-02,
    -1.39180257516060693404e-05},
   {0., 0., -9.23364619367118555360e-02, -1.69364939002081722752e-02, -1.40086798182036094347e-05}}};

// Threshold for different order of taylor approximant.
constexpr std::array<float, total_n_degs> thetas_float = {1.192092800768788e-07, 5.978858893805233e-04,
                                                          5.116619363445086e-02, 5.800524627688768e-01,
                                                          1.461661507209034e+00, 3.010066362817634e+00};

// Threshold for different order of taylor approximant.
constexpr std::array<double, total_n_degs> thetas_double = {2.220446049250313e-16, 2.580956802971767e-08,
                                                            3.397168839976962e-04, 4.991228871115323e-02,
                                                            2.996158913811580e-01, 1.090863719290036e+00};

#define MATRIX_EXP_COMPUTE_CASE(DTYPE, TYPE, CTX)           \
  case (DTYPE): {                                           \
    uint32_t result = MatrixExpCompute<TYPE>(CTX);          \
    if (result != KERNEL_STATUS_OK) {                       \
      KERNEL_LOG_ERROR("MatrixExp kernel compute failed."); \
      return result;                                        \
    }                                                       \
    break;                                                  \
  }

#define MATRIX_EXP_COMPUTE_DIFF_CASE(DTYPE, TYPE, CTX)      \
  case (DTYPE): {                                           \
    uint32_t result = MatrixExpDiffTypeCompute<TYPE>(CTX);  \
    if (result != KERNEL_STATUS_OK) {                       \
      KERNEL_LOG_ERROR("MatrixExp kernel compute failed."); \
      return result;                                        \
    }                                                       \
    break;                                                  \
  }
}  // namespace

namespace aicpu {
uint32_t MatrixExpCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kMatrixExpInputNum, kMatrixExpOutputNum),
                      "[%s] check input and output number failed.", kMatrixExp);
  KERNEL_HANDLE_ERROR(MatrixExpCheck(ctx), "[%s] check params failed.", kMatrixExp);
  auto data_type = ctx.Input(0)->GetDataType();
  switch (data_type) {
    MATRIX_EXP_COMPUTE_CASE(DT_FLOAT, float, ctx)
    MATRIX_EXP_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    MATRIX_EXP_COMPUTE_CASE(DT_COMPLEX64, std::complex<float>, ctx)
    MATRIX_EXP_COMPUTE_CASE(DT_COMPLEX128, std::complex<double>, ctx)
    MATRIX_EXP_COMPUTE_DIFF_CASE(DT_FLOAT16, Eigen::half, ctx)
    default:
      KERNEL_LOG_ERROR("MatrixExp kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t MatrixExpCpuKernel::MatrixExpCheck(CpuKernelContext &ctx) {
  auto input_0 = ctx.Input(0);
  std::vector<int64_t> shape_x = input_0->GetTensorShape()->GetDimSizes();
  size_t shape_size_x = shape_x.size();
  KERNEL_CHECK_FALSE((shape_size_x > 1), KERNEL_STATUS_PARAM_INVALID, "Input x must be at least rank 2, got [%zu].",
                     shape_size_x)
  KERNEL_CHECK_FALSE((shape_x[shape_size_x - 1] > 0), KERNEL_STATUS_PARAM_INVALID,
                     "Input x's last dimension must be at least 1.")
  KERNEL_CHECK_FALSE((shape_x[shape_size_x - kIndexTwo] == shape_x[shape_size_x - 1]), KERNEL_STATUS_PARAM_INVALID,
                     "Input x's last two dimensions must be equal, but are [%lld] and [%lld].",
                     shape_x[shape_size_x - kIndexTwo], shape_x[shape_size_x - 1])
  return KERNEL_STATUS_OK;
}

template <typename Derived1, typename Derived2, typename Derived3>
void MatrixExpCpuKernel::MTaylorApproximant(const Eigen::MatrixBase<Derived1> &A, const Eigen::MatrixBase<Derived2> &I,
                                            int order, Eigen::MatrixBase<Derived3> &E) {
  constexpr int expension_order_1 = 1;
  constexpr int expension_order_2 = 2;
  constexpr int expension_order_4 = 4;
  constexpr int expension_order_8 = 8;
  constexpr int expension_order_12 = 12;
  auto A2 = A * A;
  auto A3 = A * A2;
  if (order == expension_order_1) {
    E = I + A;
  } else if (order == expension_order_2) {
    constexpr int A2_divisor = 2;
    E = I + A + A2 / A2_divisor;
  } else if (order == expension_order_4) {
    constexpr int I_divisor = 2;
    constexpr int A_divisor = 6;
    constexpr int A2_divisor = 24;
    E = I + A + A2 * (I / I_divisor + A / A_divisor + A2 / A2_divisor);
  } else if (order == expension_order_8) {
    auto A4 = A2 * (x1 * A + x2 * A2);
    auto A8 = (x3 * A2 + A4) * (x4 * I + x5 * A + x6 * A2 + x7 * A4);
    E = I + A + y2 * A2 + A8;
  } else if (order == expension_order_12) {
    auto q31 = b12[0][0] * I + b12[0][1] * A + b12[0][2] * A2 + b12[0][3] * A3;
    auto q32 = b12[1][0] * I + b12[1][1] * A + b12[1][2] * A2 + b12[1][3] * A3;
    auto q33 = b12[2][0] * I + b12[2][1] * A + b12[2][2] * A2 + b12[2][3] * A3;
    auto q34 = b12[3][0] * I + b12[3][1] * A + b12[3][2] * A2 + b12[3][3] * A3;
    auto q61 = q33 + q34 * q34;
    E = q31 + (q32 + q61) * q61;
  } else {
    auto A6 = A3 * A3;
    auto q31 = b18[0][0] * I + b18[0][1] * A + b18[0][2] * A2 + b18[0][3] * A3 + b18[0][4] * A6;
    auto q61 = b18[1][0] * I + b18[1][1] * A + b18[1][2] * A2 + b18[1][3] * A3 + b18[1][4] * A6;
    auto q62 = b18[2][0] * I + b18[2][1] * A + b18[2][2] * A2 + b18[2][3] * A3 + b18[2][4] * A6;
    auto q63 = b18[3][0] * I + b18[3][1] * A + b18[3][2] * A2 + b18[3][3] * A3 + b18[3][4] * A6;
    auto q64 = b18[4][0] * I + b18[4][1] * A + b18[4][2] * A2 + b18[4][3] * A3 + b18[4][4] * A6;
    auto q91 = q31 * q64 + q63;
    E = q61 + (q62 + q91) * q91;
  }
}

template <typename Derived1, typename Derived2>
void MatrixExpCpuKernel::MexpImpl(const Eigen::MatrixBase<Derived1> &A, const Eigen::MatrixBase<Derived2> &I,
                                  Eigen::MatrixBase<Derived1> &mexp, CpuKernelContext &ctx) {
  const auto norm = A.cwiseAbs().colwise().sum().maxCoeff();
  constexpr std::array<int, total_n_degs> m_vals = {1, 2, 4, 8, 12, 18};
  constexpr int cut_deg = 2;
  int64_t s = -1;
  auto data_type = ctx.Input(0)->GetDataType();
  if (data_type == DT_FLOAT16 || data_type == DT_FLOAT || data_type == DT_COMPLEX64) {
    for (int i = 0; i < total_n_degs - 1; i++) {
      if (norm <= thetas_float[i]) {
        MTaylorApproximant(A, I, m_vals[i], mexp);
        break;
      }
    }
    if (norm >= thetas_float[total_n_degs - cut_deg]) {
      s = ceil(log2(norm / thetas_float[total_n_degs - 1]));
      if (s <= 0) {
        s = 0;
      }
    }
  } else {
    for (int i = 0; i < total_n_degs - 1; i++) {
      if (norm <= thetas_double[i]) {
        MTaylorApproximant(A, I, m_vals[i], mexp);
        break;
      }
    }
    if (norm >= thetas_double[total_n_degs - cut_deg]) {
      s = ceil(log2(norm / thetas_double[total_n_degs - 1]));
      if (s <= 0) {
        s = 0;
      }
    }
  }
  if (s >= 0) {
    const auto pow2s = pow(2, s);
    const auto A_scaled = A / pow2s;
    MTaylorApproximant(A_scaled, I, m_vals[total_n_degs - 1], mexp);
    for (int k = 0; k < s; k++) {
      mexp = mexp * mexp;
    }
  }
}

template <typename T>
uint32_t MatrixExpCpuKernel::MatrixExpCompute(CpuKernelContext &ctx) {
  auto input_x = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto output_y = reinterpret_cast<T *>(ctx.Output(0)->GetData());

  std::vector<int64_t> shape_x = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  size_t shape_size = shape_x.size();
  int64_t m = shape_x[shape_size - 1];
  int64_t size_mm = m * m;
  typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXd;
  MatrixXd I(m, m);
  I.setIdentity();
  int64_t matrix_num = ctx.Input(0)->NumElements() / size_mm;
  int64_t data_size = ctx.Input(0)->NumElements() * sizeof(T);
  if (data_size <= paralled_data_size) {
    for (int64_t i = 0; i < matrix_num; i++) {
      Eigen::Map<MatrixXd> matrix_x(input_x + i * m * m, m, m);
      Eigen::Map<MatrixXd> matrix_y(output_y + i * m * m, m, m);
      if (matrix_x.size() > 0) {
        MexpImpl(matrix_x, I, matrix_y, ctx);
      }
    }
  } else {
    uint32_t min_core_num = 1;
    int64_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
    if (max_core_num == 0) {
      return KERNEL_STATUS_PARAM_INVALID;
    }
    if (max_core_num > matrix_num) {
      max_core_num = matrix_num;
    }
    auto shard_work = [&](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        Eigen::Map<MatrixXd> matrix_x(input_x + i * m * m, m, m);
        Eigen::Map<MatrixXd> matrix_y(output_y + i * m * m, m, m);
        if (matrix_x.size() > 0) {
          MexpImpl(matrix_x, I, matrix_y, ctx);
        }
      }
    };
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, matrix_num, matrix_num / max_core_num, shard_work),
                        "MatrixExp Compute failed.");
  }
  return KERNEL_STATUS_OK;
}

void MatrixExpCpuKernel::TyepChangeForFp16(int64_t i, int64_t m, Eigen::half *input_x, Eigen::half *output_y,
                                           CpuKernelContext &ctx) {
  typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXd;
  MatrixXd I(m, m);
  (void)I.setIdentity();
  MatrixXd matrix_x(m, m);
  MatrixXd matrix_y(m, m);
  int64_t size_mm = m * m;
  for (int p = 0; p < m; p++) {
    for (int q = 0; q < m; q++) {
      matrix_x(p, q) = static_cast<float>(input_x[i * size_mm + p * m + q]);
    }
  }
  if (matrix_x.size() > 0) {
    MexpImpl(matrix_x, I, matrix_y, ctx);
  }
  for (int p = 0; p < m; p++) {
    for (int q = 0; q < m; q++) {
      output_y[i * size_mm + p * m + q] = static_cast<Eigen::half>(matrix_y(p, q));
    }
  }
}

template <typename T>
uint32_t MatrixExpCpuKernel::MatrixExpDiffTypeCompute(CpuKernelContext &ctx) {
  T *input_x = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto output_y = reinterpret_cast<T *>(ctx.Output(0)->GetData());

  std::vector<int64_t> shape_x = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  size_t shape_size = shape_x.size();
  int64_t m = shape_x[shape_size - 1];
  int64_t size_mm = m * m;
  int64_t matrix_num = ctx.Input(0)->NumElements() / size_mm;
  int64_t data_size = ctx.Input(0)->NumElements() * sizeof(T);
  if (data_size <= paralled_data_size) {
    for (int64_t i = 0; i < matrix_num; i++) {
      TyepChangeForFp16(i, m, input_x, output_y, ctx);
    }
  } else {
    uint32_t min_core_num = 1;
    int64_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
    if (max_core_num == 0) {
      return KERNEL_STATUS_PARAM_INVALID;
    }
    if (max_core_num > matrix_num) {
      max_core_num = matrix_num;
    }
    auto shard_work = [&](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        TyepChangeForFp16(i, m, input_x, output_y, ctx);
      }
    };
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, matrix_num, matrix_num / max_core_num, shard_work),
                        "MatrixExp Compute failed.");
  }
  // }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kMatrixExp, MatrixExpCpuKernel);
}  // namespace aicpu
