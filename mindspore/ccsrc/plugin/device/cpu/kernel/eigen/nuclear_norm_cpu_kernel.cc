/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "plugin/device/cpu/kernel/eigen/nuclear_norm_cpu_kernel.h"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kNuclearNormInputsNum = 1;
constexpr size_t kNuclearNormOutputsNum = 1;
constexpr int64_t kDimIsNone = 1000;
const size_t DIM_SIZE2 = 2;
const size_t DIM_SIZE3 = 3;
const size_t DIM_SIZE4 = 4;
const size_t DIM_SIZE5 = 5;
const size_t DIM_SIZE6 = 6;
const size_t DIM_SIZE7 = 7;
const size_t DIM_SIZE8 = 8;
}  // namespace

bool NuclearNormCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kNuclearNormInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kNuclearNormOutputsNum, kernel_name_);
  auto prim = base_operator->GetPrim();
  MS_EXCEPTION_IF_NULL(prim);

  // Attr dim is the optional attribute. Default:[0, 1]
  if (prim->HasAttr("dim")) {
    dim_ = GetValue<std::vector<int64_t>>(prim->GetAttr("dim"));
    if (dim_.size() == 1 && dim_[0] == kDimIsNone) {
      dim_.clear();
      dim_.push_back(0);
      dim_.push_back(1);
    }
  }

  // Attr keepdim is the optional attribute. Default:false
  if (prim->HasAttr("keepdim")) {
    keepdim = GetValue<bool>(prim->GetAttr("keepdim"));
  }

  return MatchKernelFunc(base_operator, inputs, outputs);
}

int NuclearNormCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs,
                                    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }

  input_shape = inputs[kIndex0]->GetDeviceShapeAdaptively();
  input_dtype = inputs[kIndex0]->GetDtype();

  size_t x_type_size = (input_dtype == kNumberTypeFloat32) ? sizeof(float) : sizeof(double);
  const size_t input_dimnum = input_shape.size();
  dim_[0] = (dim_[0] < 0) ? dim_[0] + static_cast<int64_t>(input_dimnum) : dim_[0];
  dim_[1] = (dim_[1] < 0) ? dim_[1] + static_cast<int64_t>(input_dimnum) : dim_[1];
  const size_t dimsize0 = static_cast<size_t>(input_shape[dim_[0]]);
  const size_t dimsize1 = static_cast<size_t>(input_shape[dim_[1]]);
  // Init workspace size list
  // This workspace size used for mat nuclearnorm calculation
  size_t mat_size = x_type_size * dimsize0 * dimsize1;
  (void)workspace_size_list_.emplace_back(mat_size);
  // This workspace size used for ComputeMatrixNuclearNorm
  (void)workspace_size_list_.emplace_back(mat_size);
  size_t m = dimsize1;
  size_t n = dimsize0;
  size_t k = (m < n ? m : n);
  size_t tU_size = x_type_size * m * k;
  (void)workspace_size_list_.emplace_back(tU_size);
  size_t tS_size = x_type_size * k;
  (void)workspace_size_list_.emplace_back(tS_size);
  size_t tVT_size = x_type_size * k * n;
  (void)workspace_size_list_.emplace_back(tVT_size);
  // This workspace size used for svd
  const size_t dim[2] = {std::max(dimsize0, dimsize1), std::min(dimsize0, dimsize1)};
  size_t U_size = x_type_size * dim[0] * dim[0];
  (void)workspace_size_list_.emplace_back(U_size);
  size_t V_size = x_type_size * dim[1] * dim[1];
  (void)workspace_size_list_.emplace_back(V_size);
  size_t S_size = x_type_size * dim[0] * dim[1];
  (void)workspace_size_list_.emplace_back(S_size);
  return KRET_OK;
}

template <typename T>
void NuclearNormCpuKernelMod::GivensL(T *S_, const size_t dim[], const size_t m, const T a, const T b) {
  T r = sqrt(a * a + b * b);
  T c = 0.0;
  T s = 0.0;
  if (r != 0) {
    c = a / r;
    s = -b / r;
  }

  for (size_t i = 0; i < dim[1]; i++) {
    T S0 = S_[(m + 0) * dim[1] + (i)];
    T S1 = S_[(m + 1) * dim[1] + (i)];
    S_[(m)*dim[1] + (i)] += S0 * (c - 1);
    S_[(m)*dim[1] + (i)] += S1 * (-s);

    S_[(m + 1) * dim[1] + (i)] += S0 * (s);
    S_[(m + 1) * dim[1] + (i)] += S1 * (c - 1);
  }
}

template <typename T>
void NuclearNormCpuKernelMod::GivensR(T *S_, const size_t dim[], const size_t m, const T a, const T b) {
  T r = sqrt(a * a + b * b);
  T c = 0.0;
  T s = 0.0;
  if (r != 0) {
    c = a / r;
    s = -b / r;
  }
  for (size_t i = 0; i < dim[0]; i++) {
    T S0 = S_[(i)*dim[1] + (m + 0)];
    T S1 = S_[(i)*dim[1] + (m + 1)];
    S_[(i)*dim[1] + (m)] += S0 * (c - 1);
    S_[(i)*dim[1] + (m)] += S1 * (-s);

    S_[(i)*dim[1] + (m + 1)] += S0 * (s);
    S_[(i)*dim[1] + (m + 1)] += S1 * (c - 1);
  }
}

template <typename T>
void NuclearNormCpuKernelMod::SVD_head(size_t i, std::vector<T> *house_vec, const size_t dim[], T *U_, T *S_) {
  T x1 = S_[(i)*dim[1] + (i)] < 0 ? -S_[(i)*dim[1] + (i)] : S_[(i)*dim[1] + (i)];
  T x_inv_norm = 0;
  for (size_t j = i; j < dim[0]; j++) {
    x_inv_norm += S_[(j)*dim[1] + (i)] * S_[(j)*dim[1] + (i)];
  }
  if (x_inv_norm > 0) {
    x_inv_norm = 1 / sqrt(x_inv_norm);
  }
  T alpha = sqrt(1 + x1 * x_inv_norm);
  T beta = 0.0;
  MS_EXCEPTION_IF_ZERO("alpha", alpha);
  beta = x_inv_norm / alpha;
  (*house_vec)[i] = -alpha;
  for (size_t j = i + 1; j < dim[0]; j++) {
    (*house_vec)[j] = -beta * S_[(j)*dim[1] + (i)];
  }
  if (S_[(i)*dim[1] + (i)] < 0) {
    for (size_t j = i + 1; j < dim[0]; j++) {
      (*house_vec)[j] = -(*house_vec)[j];
    }
  }
  for (size_t k = i; k < dim[1]; k++) {
    T dot_prod = 0;
    for (size_t j = i; j < dim[0]; j++) {
      dot_prod += S_[(j)*dim[1] + (k)] * (*house_vec)[j];
    }
    for (size_t j = i; j < dim[0]; j++) {
      S_[(j)*dim[1] + (k)] -= dot_prod * (*house_vec)[j];
    }
  }
  for (size_t k = 0; k < dim[0]; k++) {
    T dot_prod = 0;
    for (size_t j = i; j < dim[0]; j++) {
      dot_prod += U_[(k)*dim[0] + (j)] * (*house_vec)[j];
    }
    for (size_t j = i; j < dim[0]; j++) {
      U_[(k)*dim[0] + (j)] -= dot_prod * (*house_vec)[j];
    }
  }
}

template <typename T>
void NuclearNormCpuKernelMod::SVD(const size_t dim[], T *U_, T *S_, T *V_, T eps) {
  std::vector<T> house_vec(std::max(dim[0], dim[1]));
  for (size_t i = 0; i < std::min(dim[0], dim[1]); i++) {
    SVD_head<T>(i, &house_vec, dim, U_, S_);
    size_t min_dimnum = static_cast<size_t>(static_cast<int64_t>(std::min(dim[0], dim[1])) - 1);
    if (i >= min_dimnum) {
      continue;
    }
    T x1 = S_[(i)*dim[1] + (i + 1)] < 0 ? -S_[(i)*dim[1] + (i + 1)] : S_[(i)*dim[1] + (i + 1)];
    T x_inv_norm = 0;
    for (size_t j = i + 1; j < dim[1]; j++) {
      x_inv_norm += S_[(i)*dim[1] + (j)] * S_[(i)*dim[1] + (j)];
    }
    if (x_inv_norm > 0) {
      x_inv_norm = 1 / sqrt(x_inv_norm);
    }
    T alpha = sqrt(1 + x1 * x_inv_norm);
    T beta = 0.0;
    MS_EXCEPTION_IF_ZERO("alpha", alpha);
    beta = x_inv_norm / alpha;
    house_vec[i + 1] = -alpha;
    for (size_t j = i + 2; j < dim[1]; j++) {
      house_vec[j] = -beta * S_[(i)*dim[1] + (j)];
    }
    if (S_[(i)*dim[1] + (i + 1)] < 0) {
      for (size_t j = i + 2; j < dim[1]; j++) {
        house_vec[j] = -house_vec[j];
      }
    }

    for (size_t k = i; k < dim[0]; k++) {
      T dot_prod = 0;
      for (size_t j = i + 1; j < dim[1]; j++) {
        dot_prod += S_[(k)*dim[1] + (j)] * house_vec[j];
      }
      for (size_t j = i + 1; j < dim[1]; j++) {
        S_[(k)*dim[1] + (j)] -= dot_prod * house_vec[j];
      }
    }
    for (size_t k = 0; k < dim[1]; k++) {
      T dot_prod = 0;
      for (size_t j = i + 1; j < dim[1]; j++) {
        dot_prod += V_[(j)*dim[1] + (k)] * house_vec[j];
      }
      for (size_t j = i + 1; j < dim[1]; j++) {
        V_[(j)*dim[1] + (k)] -= dot_prod * house_vec[j];
      }
    }
  }
  SVD_tail<T>(dim, U_, S_, V_, eps);
}

template <typename T>
void NuclearNormCpuKernelMod::SVD_tail(const size_t dim[], T *U_, T *S_, T *V_, T eps) {
  size_t k0 = 0;
  if (eps < 0) {
    eps = 1.0;
    const T EPSDOT5 = 0.5;
    const T EPS64 = 64.0;
    while (eps + static_cast<T>(1.0) > 1.0) {
      eps *= EPSDOT5;
    }
    eps *= EPS64;
  }
  while (k0 < static_cast<size_t>(static_cast<int64_t>(dim[1]) - 1)) {
    T S_max = 0.0;
    for (size_t i = 0; i < dim[1]; i++) {
      S_max = (S_max > S_[(i)*dim[1] + (i)] ? S_max : S_[(i)*dim[1] + (i)]);
    }

    while (k0 < static_cast<size_t>(static_cast<int64_t>(dim[1]) - 1) &&
           fabs(S_[(k0)*dim[1] + (k0 + 1)]) <= eps * S_max) {
      k0++;
    }
    if (k0 == static_cast<size_t>(static_cast<int64_t>(dim[1]) - 1)) {
      continue;
    }

    size_t n = k0 + 2;
    while (n < dim[1] && fabs(S_[(n - 1) * dim[1] + (n)]) > eps * S_max) {
      n++;
    }

    T alpha = 0;
    T beta = 0;
    T C[2][2];
    C[0][0] = S_[(n - DIM_SIZE2) * dim[1] + (n - DIM_SIZE2)] * S_[(n - DIM_SIZE2) * dim[1] + (n - DIM_SIZE2)];
    const int DIFF2 = 2;
    if (n - k0 > DIFF2) {
      C[0][0] += S_[(n - DIM_SIZE3) * dim[1] + (n - DIM_SIZE2)] * S_[(n - DIM_SIZE3) * dim[1] + (n - DIM_SIZE2)];
    }
    C[0][1] = S_[(n - DIM_SIZE2) * dim[1] + (n - DIM_SIZE2)] * S_[(n - DIM_SIZE2) * dim[1] + (n - 1)];
    C[1][0] = S_[(n - DIM_SIZE2) * dim[1] + (n - DIM_SIZE2)] * S_[(n - DIM_SIZE2) * dim[1] + (n - 1)];
    C[1][1] = S_[(n - 1) * dim[1] + (n - 1)] * S_[(n - 1) * dim[1] + (n - 1)] +
              S_[(n - DIM_SIZE2) * dim[1] + (n - 1)] * S_[(n - DIM_SIZE2) * dim[1] + (n - 1)];

    T b = -(C[0][0] + C[1][1]) / 2;
    T c = C[0][0] * C[1][1] - C[0][1] * C[1][0];
    T d = 0;
    if (b * b - c > 0) {
      d = sqrt(b * b - c);
    } else {
      T bb = (C[0][0] - C[1][1]) / 2;
      T cc = -C[0][1] * C[1][0];
      if (bb * bb - cc > 0) {
        d = sqrt(bb * bb - cc);
      }
    }

    T lambda1 = -b + d;
    T lambda2 = -b - d;

    T d1 = lambda1 - C[1][1];
    d1 = (d1 < 0 ? -d1 : d1);
    T d2 = lambda2 - C[1][1];
    d2 = (d2 < 0 ? -d2 : d2);
    T mu = (d1 < d2 ? lambda1 : lambda2);

    alpha = S_[(k0)*dim[1] + (k0)] * S_[(k0)*dim[1] + (k0)] - mu;
    beta = S_[(k0)*dim[1] + (k0)] * S_[(k0)*dim[1] + (k0 + 1)];
    SVD_tail_cal<T>(dim, U_, S_, V_, eps, n, k0, alpha, beta, S_max);
  }
}

template <typename T>
void NuclearNormCpuKernelMod::SVD_tail_cal(const size_t dim[], T *U_, T *S_, T *V_, const T eps, const size_t n,
                                           const size_t k0, T alpha, T beta, const T S_max) {
  for (size_t k = k0; k < static_cast<size_t>(static_cast<int64_t>(n) - 1); k++) {
    size_t dimU[2] = {dim[0], dim[0]};
    size_t dimV[2] = {dim[1], dim[1]};

    GivensR<T>(S_, dim, k, alpha, beta);
    GivensL<T>(V_, dimV, k, alpha, beta);
    GivensL<T>(S_, dim, k, S_[(k)*dim[1] + (k)], S_[(k + 1) * dim[1] + (k)]);
    GivensR<T>(U_, dimU, k, S_[(k)*dim[1] + (k)], S_[(k + 1) * dim[1] + (k)]);

    alpha = S_[(k)*dim[1] + (k + 1)];
    beta = S_[(k)*dim[1] + (k + DIM_SIZE2)];
  }
  for (size_t i0 = k0; i0 < static_cast<size_t>(static_cast<int64_t>(n) - 1); i0++) {
    for (size_t i1 = 0; i1 < dim[1]; i1++) {
      if (i0 > i1 || i0 + 1 < i1) {
        S_[(i0)*dim[1] + (i1)] = 0;
      }
    }
  }
  for (size_t i0 = 0; i0 < dim[0]; i0++) {
    for (size_t i1 = k0; i1 < static_cast<size_t>(static_cast<int64_t>(n) - 1); i1++) {
      if (i0 > i1 || i0 + 1 < i1) {
        S_[(i0)*dim[1] + (i1)] = 0;
      }
    }
  }
  for (size_t i = 0; i < static_cast<size_t>(static_cast<int64_t>(dim[1]) - 1); i++) {
    if (fabs(S_[(i)*dim[1] + (i + 1)]) <= eps * S_max) {
      S_[(i)*dim[1] + (i + 1)] = 0;
    }
  }
}

template <typename T>
void NuclearNormCpuKernelMod::svd(int *M, int *N, const T *A, const int *LDA, T *S, T *U, const int *LDU, T *VT,
                                  const int *LDVT, const std::vector<kernel::AddressPtr> &workspace) {
  const size_t dim[2] = {std::max(static_cast<size_t>(*N), static_cast<size_t>(*M)),
                         std::min(static_cast<size_t>(*N), static_cast<size_t>(*M))};
  T *U_ = GetDeviceAddress<T>(workspace, kIndex5);
  auto ret0 = memset_s(U_, dim[0] * dim[0] * sizeof(T), 0, dim[0] * dim[0] * sizeof(T));
  if (ret0 != EOK) {
    MS_LOG(EXCEPTION) << "For 'NuclearNorm', it does memset_s failed. Error no: " << ret0;
  }
  T *V_ = GetDeviceAddress<T>(workspace, kIndex6);
  auto ret1 = memset_s(V_, dim[1] * dim[1] * sizeof(T), 0, dim[1] * dim[1] * sizeof(T));
  if (ret1 != EOK) {
    MS_LOG(EXCEPTION) << "For 'NuclearNorm', it does memset_s failed. Error no: " << ret1;
  }
  T *S_ = GetDeviceAddress<T>(workspace, kIndex7);

  const size_t lda = static_cast<size_t>(*LDA);

  if (dim[1] == static_cast<size_t>(*M)) {
    for (size_t i = 0; i < dim[0]; i++) {
      for (size_t j = 0; j < dim[1]; j++) {
        S_[i * dim[1] + j] = A[i * lda + j];
      }
    }
  } else {
    for (size_t i = 0; i < dim[0]; i++) {
      for (size_t j = 0; j < dim[1]; j++) {
        S_[i * dim[1] + j] = A[j * lda + i];
      }
    }
  }
  for (size_t i = 0; i < dim[0]; i++) {
    U_[i * dim[0] + i] = 1;
  }
  for (size_t i = 0; i < dim[1]; i++) {
    V_[i * dim[1] + i] = 1;
  }

  SVD<T>(dim, U_, S_, V_, static_cast<T>(-1));
  svd_tail<T>(M, N, S, S_, U, VT, U_, V_, dim, LDU, *LDVT);
}

template <typename T>
void NuclearNormCpuKernelMod::svd_tail(const int *M, const int *N, T *S, const T *S_, T *U, T *VT, const T *U_,
                                       const T *V_, const size_t dim[], const int *LDU, const int LDVT) {
  const size_t ldu = static_cast<size_t>(*LDU);
  const size_t ldv = static_cast<size_t>(LDVT);
  for (size_t i = 0; i < dim[1]; i++) {
    S[i] = S_[i * dim[1] + i];
  }
  if (dim[1] == static_cast<size_t>(*M)) {
    for (size_t i = 0; i < dim[1]; i++) {
      for (size_t j = 0; j < static_cast<size_t>(*M); j++) {
        U[j + ldu * i] = V_[j + i * dim[1]] * (S[i] < 0.0 ? -1.0 : 1.0);
      }
    }
  } else {
    for (size_t i = 0; i < dim[1]; i++) {
      for (size_t j = 0; j < static_cast<size_t>(*M); j++) {
        U[j + ldu * i] = U_[i + j * dim[0]] * static_cast<T>(S[i] < 0.0 ? -1.0 : 1.0);
      }
    }
  }
  if (dim[0] == static_cast<size_t>(*N)) {
    for (size_t i = 0; i < static_cast<size_t>(*N); i++) {
      for (size_t j = 0; j < dim[1]; j++) {
        VT[j + ldv * i] = U_[j + i * dim[0]];
      }
    }
  } else {
    for (size_t i = 0; i < static_cast<size_t>(*N); i++) {
      for (size_t j = 0; j < dim[1]; j++) {
        VT[j + ldv * i] = V_[i + j * dim[1]];
      }
    }
  }
  for (size_t i = 0; i < dim[1]; i++) {
    S[i] = S[i] * static_cast<T>(S[i] < 0.0 ? -1.0 : 1.0);
  }
}

template <typename T>
T NuclearNormCpuKernelMod::ComputeMatrixNuclearNorm(size_t dim0, size_t dim1, const T *mat,
                                                    const std::vector<kernel::AddressPtr> &workspace) {
  size_t n1 = dim0, n2 = dim1;
  T *M = GetDeviceAddress<T>(workspace, kIndex1);
  size_t copy_size = dim0 * dim1 * sizeof(T);
  auto ret0 = memcpy_s(M, copy_size, &mat[0], copy_size);
  if (ret0 != EOK) {
    MS_LOG(EXCEPTION) << "For 'NuclearNorm', it does memcpy_s failed. Error no: " << ret0;
  }
  int m = n2;
  int n = n1;
  int k = (m < n ? m : n);
  T *tU = GetDeviceAddress<T>(workspace, kIndex2);
  T *tS = GetDeviceAddress<T>(workspace, kIndex3);
  T *tVT = GetDeviceAddress<T>(workspace, kIndex4);
  svd<T>(&m, &n, &M[0], &m, &tS[0], &tU[0], &m, &tVT[0], &k, workspace);
  T nuclear_norm = 0.0;
  for (int i = 0; i < k; i++) {
    nuclear_norm += *(tS + i);
  }

  return nuclear_norm;
}

template <typename T, int32_t RANK>
bool NuclearNormCpuKernelMod::ComputeTensorNuclearNorm(const std::vector<kernel::AddressPtr> &inputs,
                                                       const std::vector<kernel::AddressPtr> &workspace,
                                                       const std::vector<kernel::AddressPtr> &outputs) {
  const size_t input_dimnum = input_shape.size();
  T *input_data_ptr = reinterpret_cast<T *>(inputs[0]->addr);
  size_t value_num_ = 1;
  for (size_t i = 0; i < input_dimnum; i++) {
    value_num_ *= static_cast<size_t>(input_shape[i]);
  }

  size_t total_copy_size = value_num_ * sizeof(T);
  Eigen::Tensor<T, 1, Eigen::RowMajor> eigen_tensor(value_num_);
  auto ret1 = memcpy_s(&eigen_tensor(0), total_copy_size, input_data_ptr, total_copy_size);
  if (ret1 != EOK) {
    MS_LOG(EXCEPTION) << "For 'NuclearNorm', it does memcpy_s failed. Error no: " << ret1;
  }

  std::array<Eigen::DenseIndex, RANK> dim_array;
  for (size_t i = 0; i < input_dimnum; i++) {
    dim_array.at(i) = static_cast<size_t>(input_shape[i]);
  }
  Eigen::Tensor<T, RANK, Eigen::RowMajor> reshaped_tensor = eigen_tensor.reshape(dim_array);

  dim_[0] = (dim_[0] < 0) ? dim_[0] + static_cast<int64_t>(input_dimnum) : dim_[0];
  dim_[1] = (dim_[1] < 0) ? dim_[1] + static_cast<int64_t>(input_dimnum) : dim_[1];

  int32_t j = 0;
  for (size_t i = 0; i < input_dimnum; i++) {
    if (i != static_cast<size_t>(dim_[0]) && i != static_cast<size_t>(dim_[1])) {
      dim_array.at(j) = static_cast<size_t>(i);
      j++;
    }
  }
  dim_array.at(j) = static_cast<size_t>(dim_[0]);
  dim_array.at(j + 1) = static_cast<size_t>(dim_[1]);

  Eigen::Tensor<T, RANK, Eigen::RowMajor> shuffled_tensor = reshaped_tensor.shuffle(dim_array);

  const size_t dimsize0 = static_cast<size_t>(input_shape[dim_[0]]);
  const size_t dimsize1 = static_cast<size_t>(input_shape[dim_[1]]);
  size_t iter_number = value_num_ / (dimsize0 * dimsize1);
  std::array<Eigen::DenseIndex, DIM_SIZE3> dim_array_last;
  const size_t DIM_INDEX0 = 0;
  const size_t DIM_INDEX1 = 1;
  const size_t DIM_INDEX2 = 2;
  dim_array_last.at(DIM_INDEX0) = iter_number;
  dim_array_last.at(DIM_INDEX1) = dimsize0;
  dim_array_last.at(DIM_INDEX2) = dimsize1;
  Eigen::Tensor<T, DIM_SIZE3, Eigen::RowMajor> permuted_tensor = shuffled_tensor.reshape(dim_array_last);

  auto output_data_ptr = reinterpret_cast<T *>(outputs[0]->addr);
  size_t copy_size = (dimsize0 * dimsize1) * sizeof(T);
  auto task = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; ++i) {
      T *mat = GetDeviceAddress<T>(workspace, kIndex0);
      auto ret2 = memcpy_s(mat, copy_size, &permuted_tensor(i, 0, 0), copy_size);
      if (ret2 != EOK) {
        MS_LOG(EXCEPTION) << "For 'NuclearNorm', it does memcpy_s failed. Error no: " << ret2;
      }
      T nuclear_norm = ComputeMatrixNuclearNorm<T>(dimsize0, dimsize1, mat, workspace);
      *(output_data_ptr + i) = nuclear_norm;
    }
  };
  ParallelLaunchAutoSearch(task, iter_number, this, &parallel_search_info_);
  return true;
}

template <typename T>
bool NuclearNormCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                           const std::vector<kernel::AddressPtr> &workspace,
                                           const std::vector<kernel::AddressPtr> &outputs) {
  bool res = true;
  switch (input_shape.size()) {
    case DIM_SIZE2:
      res = ComputeTensorNuclearNorm<T, DIM_SIZE2>(inputs, workspace, outputs);
      break;
    case DIM_SIZE3:
      res = ComputeTensorNuclearNorm<T, DIM_SIZE3>(inputs, workspace, outputs);
      break;
    case DIM_SIZE4:
      res = ComputeTensorNuclearNorm<T, DIM_SIZE4>(inputs, workspace, outputs);
      break;
    case DIM_SIZE5:
      res = ComputeTensorNuclearNorm<T, DIM_SIZE5>(inputs, workspace, outputs);
      break;
    case DIM_SIZE6:
      res = ComputeTensorNuclearNorm<T, DIM_SIZE6>(inputs, workspace, outputs);
      break;
    case DIM_SIZE7:
      res = ComputeTensorNuclearNorm<T, DIM_SIZE7>(inputs, workspace, outputs);
      break;
    case DIM_SIZE8:
      res = ComputeTensorNuclearNorm<T, DIM_SIZE8>(inputs, workspace, outputs);
      break;
    default:
      MS_LOG(EXCEPTION) << "Only tensors with ranks between 2 and 8 are "
                           "currently supported. Tensor rank:"
                        << input_shape.size();
      res = false;
      break;
  }
  return res;
}

const std::vector<std::pair<KernelAttr, NuclearNormCpuKernelMod::KernelRunFunc>> &NuclearNormCpuKernelMod::GetFuncList()
  const {
  static const std::vector<std::pair<KernelAttr, NuclearNormCpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &NuclearNormCpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &NuclearNormCpuKernelMod::LaunchKernel<double>},
  };
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, NuclearNorm, NuclearNormCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
