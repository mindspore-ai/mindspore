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
#define U(i, j) U_[(i)*dim[0] + (j)]
#define S(i, j) S_[(i)*dim[1] + (j)]
#define V(i, j) V_[(i)*dim[1] + (j)]
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
void NuclearNormCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);

  // Attr dim is the optional attribute. Default:[0, 1]
  if (common::AnfAlgo::HasNodeAttr("dim", kernel_node)) {
    dim = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, "dim");
    if (dim.size() == 1 && dim[0] == kDimIsNone) {
      dim.clear();
      dim.push_back(0);
      dim.push_back(1);
    }
  }
  // Attr keepdim is the optional attribute. Default:false
  if (common::AnfAlgo::HasNodeAttr("keepdim", kernel_node)) {
    keepdim = common::AnfAlgo::GetNodeAttr<bool>(kernel_node, "keepdim");
  }

  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  CHECK_KERNEL_INPUTS_NUM(input_num, kNuclearNormInputsNum, kernel_name_);
  size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
  CHECK_KERNEL_OUTPUTS_NUM(output_num, kNuclearNormOutputsNum, kernel_name_);

  input_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  input_dtype = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  if (input_shape.size() < DIM_SIZE2 || input_shape.size() > DIM_SIZE8) {
    MS_LOG_EXCEPTION << "For '" << kernel_name_ << "', the rank of parameter 'a' must be in [2, 8], but got "
                     << input_shape.size() << " dimensions.";
  }

  auto kernel_attr = GetKernelAttrFromNode(kernel_node);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "NuclearNorm does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
}

template <typename T>
void NuclearNormCpuKernelMod::GivensL(T *S_, const size_t dim[2], size_t m, T a, T b) {
  T r = sqrt(a * a + b * b);
  T c = 0.0;
  T s = 0.0;
  if (r != 0) {
    c = a / r;
    s = -b / r;
  }

  for (size_t i = 0; i < (size_t)dim[1]; i++) {
    T S0 = S(m + 0, i);
    T S1 = S(m + 1, i);
    S(m, i) += S0 * (c - 1);
    S(m, i) += S1 * (-s);

    S(m + 1, i) += S0 * (s);
    S(m + 1, i) += S1 * (c - 1);
  }
}

template <typename T>
void NuclearNormCpuKernelMod::GivensR(T *S_, const size_t dim[2], size_t m, T a, T b) {
  T r = sqrt(a * a + b * b);
  T c = 0.0;
  T s = 0.0;
  if (r != 0) {
    c = a / r;
    s = -b / r;
  }
  for (size_t i = 0; i < (size_t)dim[0]; i++) {
    T S0 = S(i, m + 0);
    T S1 = S(i, m + 1);
    S(i, m) += S0 * (c - 1);
    S(i, m) += S1 * (-s);

    S(i, m + 1) += S0 * (s);
    S(i, m + 1) += S1 * (c - 1);
  }
}

template <typename T>
void NuclearNormCpuKernelMod::SVD_head(size_t i, std::vector<T> *house_vec, const size_t dim[2], T *U_, T *S_) {
  T x1 = S(i, i) < 0 ? -S(i, i) : S(i, i);
  T x_inv_norm = 0;
  for (size_t j = i; j < (size_t)dim[0]; j++) {
    x_inv_norm += S(j, i) * S(j, i);
  }
  if (x_inv_norm > 0) {
    x_inv_norm = 1 / sqrt(x_inv_norm);
  }
  T alpha = sqrt(1 + x1 * x_inv_norm);
  T beta = 0.0;
  MS_EXCEPTION_IF_ZERO("alpha", alpha);
  beta = x_inv_norm / alpha;
  (*house_vec)[i] = -alpha;
  for (size_t j = i + 1; j < (size_t)dim[0]; j++) {
    (*house_vec)[j] = -beta * S(j, i);
  }
  if (S(i, i) < 0) {
    for (size_t j = i + 1; j < (size_t)dim[0]; j++) {
      (*house_vec)[j] = -(*house_vec)[j];
    }
  }
  for (size_t k = i; k < (size_t)dim[1]; k++) {
    T dot_prod = 0;
    for (size_t j = i; j < (size_t)dim[0]; j++) {
      dot_prod += S(j, k) * (*house_vec)[j];
    }
    for (size_t j = i; j < (size_t)dim[0]; j++) {
      S(j, k) -= dot_prod * (*house_vec)[j];
    }
  }
  for (size_t k = 0; k < (size_t)dim[0]; k++) {
    T dot_prod = 0;
    for (size_t j = i; j < (size_t)dim[0]; j++) {
      dot_prod += U(k, j) * (*house_vec)[j];
    }
    for (size_t j = i; j < (size_t)dim[0]; j++) {
      U(k, j) -= dot_prod * (*house_vec)[j];
    }
  }
}

template <typename T>
void NuclearNormCpuKernelMod::SVD(const size_t dim[2], T *U_, T *S_, T *V_, T eps) {
  std::vector<T> house_vec(std::max(dim[0], dim[1]));
  for (size_t i = 0; i < (size_t)std::min(dim[0], dim[1]); i++) {
    SVD_head(i, &house_vec, dim, U_, S_);
    if (i >= std::min(dim[0], dim[1]) - 1) {
      continue;
    }
    T x1 = S(i, i + 1) < 0 ? -S(i, i + 1) : S(i, i + 1);
    T x_inv_norm = 0;
    for (size_t j = i + 1; j < (size_t)dim[1]; j++) {
      x_inv_norm += S(i, j) * S(i, j);
    }
    if (x_inv_norm > 0) {
      x_inv_norm = 1 / sqrt(x_inv_norm);
    }
    T alpha = sqrt(1 + x1 * x_inv_norm);
    T beta = 0.0;
    MS_EXCEPTION_IF_ZERO("alpha", alpha);
    beta = x_inv_norm / alpha;
    house_vec[i + 1] = -alpha;
    for (size_t j = i + 2; j < (size_t)dim[1]; j++) {
      house_vec[j] = -beta * S(i, j);
    }
    if (S(i, i + 1) < 0) {
      for (size_t j = i + 2; j < (size_t)dim[1]; j++) {
        house_vec[j] = -house_vec[j];
      }
    }

    for (size_t k = i; k < (size_t)dim[0]; k++) {
      T dot_prod = 0;
      for (size_t j = i + 1; j < (size_t)dim[1]; j++) {
        dot_prod += S(k, j) * house_vec[j];
      }
      for (size_t j = i + 1; j < (size_t)dim[1]; j++) {
        S(k, j) -= dot_prod * house_vec[j];
      }
    }
    for (size_t k = 0; k < (size_t)dim[1]; k++) {
      T dot_prod = 0;
      for (size_t j = i + 1; j < (size_t)dim[1]; j++) {
        dot_prod += V(j, k) * house_vec[j];
      }
      for (size_t j = i + 1; j < (size_t)dim[1]; j++) {
        V(j, k) -= dot_prod * house_vec[j];
      }
    }
  }
  SVD_tail<T>(dim, U_, S_, V_, eps);
}

template <typename T>
void NuclearNormCpuKernelMod::SVD_tail(const size_t dim[2], T *U_, T *S_, T *V_, T eps) {
  size_t k0 = 0;
  if (eps < 0) {
    eps = 1.0;
    const T EPSDOT5 = 0.5;
    const T EPS64 = 64.0;
    while (eps + (T)1.0 > 1.0) {
      eps *= EPSDOT5;
    }
    eps *= EPS64;
  }
  while (k0 < dim[1] - 1) {
    T S_max = 0.0;
    for (size_t i = 0; i < (size_t)dim[1]; i++) {
      S_max = (S_max > S(i, i) ? S_max : S(i, i));
    }

    while (k0 < dim[1] - 1 && fabs(S(k0, k0 + 1)) <= eps * S_max) {
      k0++;
    }
    if (k0 == dim[1] - 1) {
      continue;
    }

    size_t n = k0 + 2;
    while (n < dim[1] && fabs(S(n - 1, n)) > eps * S_max) {
      n++;
    }

    T alpha = 0;
    T beta = 0;
    T C[2][2];
    C[0][0] = S(n - 2, n - 2) * S(n - 2, n - 2);
    const int DIFF2 = 2;
    if (n - k0 > DIFF2) {
      C[0][0] += S(n - 3, n - 2) * S(n - 3, n - 2);
    }
    C[0][1] = S(n - 2, n - 2) * S(n - 2, n - 1);
    C[1][0] = S(n - 2, n - 2) * S(n - 2, n - 1);
    C[1][1] = S(n - 1, n - 1) * S(n - 1, n - 1) + S(n - 2, n - 1) * S(n - 2, n - 1);

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

    alpha = S(k0, k0) * S(k0, k0) - mu;
    beta = S(k0, k0) * S(k0, k0 + 1);
    SVD_tail_cal(dim, U_, S_, V_, eps, n, k0, alpha, beta, S_max);
  }
}

template <typename T>
void NuclearNormCpuKernelMod::SVD_tail_cal(const size_t dim[2], T *U_, T *S_, T *V_, T eps, size_t n, size_t k0,
                                           T alpha, T beta, T S_max) {
  for (size_t k = k0; k < (size_t)n - 1; k++) {
    size_t dimU[2] = {dim[0], dim[0]};
    size_t dimV[2] = {dim[1], dim[1]};

    GivensR(S_, dim, k, alpha, beta);
    GivensL(V_, dimV, k, alpha, beta);
    GivensL(S_, dim, k, S(k, k), S(k + 1, k));
    GivensR(U_, dimU, k, S(k, k), S(k + 1, k));

    alpha = S(k, k + 1);
    beta = S(k, k + 2);
  }
  for (size_t i0 = k0; i0 < (size_t)n - 1; i0++) {
    for (size_t i1 = 0; i1 < (size_t)dim[1]; i1++) {
      if (i0 > i1 || i0 + 1 < i1) {
        S(i0, i1) = 0;
      }
    }
  }
  for (size_t i0 = 0; i0 < (size_t)dim[0]; i0++) {
    for (size_t i1 = k0; i1 < (size_t)n - 1; i1++) {
      if (i0 > i1 || i0 + 1 < i1) {
        S(i0, i1) = 0;
      }
    }
  }
  for (size_t i = 0; i < (size_t)dim[1] - 1; i++) {
    if (fabs(S(i, i + 1)) <= eps * S_max) {
      S(i, i + 1) = 0;
    }
  }
}

template <typename T>
void NuclearNormCpuKernelMod::svd(int *M, int *N, T *A, int *LDA, T *S, T *U, int *LDU, T *VT, int *LDVT) {
  const size_t dim[2] = {std::max((size_t)*N, (size_t)*M), std::min((size_t)*N, (size_t)*M)};
  T *U_ = new T[dim[0] * dim[0]];
  memset(U_, 0, dim[0] * dim[0] * sizeof(T));
  T *V_ = new T[dim[1] * dim[1]];
  memset(V_, 0, dim[1] * dim[1] * sizeof(T));
  T *S_ = new T[dim[0] * dim[1]];

  const size_t lda = *LDA;

  if (dim[1] == (size_t)*M) {
    for (size_t i = 0; i < (size_t)dim[0]; i++) {
      for (size_t j = 0; j < (size_t)dim[1]; j++) {
        S_[i * dim[1] + j] = A[i * lda + j];
      }
    }
  } else {
    for (size_t i = 0; i < (size_t)dim[0]; i++) {
      for (size_t j = 0; j < (size_t)dim[1]; j++) {
        S_[i * dim[1] + j] = A[j * lda + i];
      }
    }
  }
  for (size_t i = 0; i < (size_t)dim[0]; i++) {
    U_[i * dim[0] + i] = 1;
  }
  for (size_t i = 0; i < (size_t)dim[1]; i++) {
    V_[i * dim[1] + i] = 1;
  }

  SVD<T>(dim, U_, S_, V_, (T)-1);
  svd_tail<T>(M, N, S, S_, U, VT, U_, V_, dim, LDU, LDVT);

  delete[] U_;
  delete[] S_;
  delete[] V_;
}

template <typename T>
void NuclearNormCpuKernelMod::svd_tail(int *M, int *N, T *S, T *S_, T *U, T *VT, T *U_, T *V_, const size_t dim[2],
                                       int *LDU, int *LDVT) {
  const size_t ldu = *LDU;
  const size_t ldv = *LDVT;
  for (size_t i = 0; i < (size_t)dim[1]; i++) {
    S[i] = S_[i * dim[1] + i];
  }
  if (dim[1] == (size_t)*M) {
    for (size_t i = 0; i < (size_t)dim[1]; i++) {
      for (size_t j = 0; j < (size_t)*M; j++) {
        U[j + ldu * i] = V_[j + i * dim[1]] * (S[i] < 0.0 ? -1.0 : 1.0);
      }
    }
  } else {
    for (size_t i = 0; i < (size_t)dim[1]; i++) {
      for (size_t j = 0; j < (size_t)*M; j++) {
        U[j + ldu * i] = U_[i + j * dim[0]] * (S[i] < 0.0 ? -1.0 : 1.0);
      }
    }
  }
  if (dim[0] == (size_t)*N) {
    for (size_t i = 0; i < (size_t)*N; i++) {
      for (size_t j = 0; j < (size_t)dim[1]; j++) {
        VT[j + ldv * i] = U_[j + i * dim[0]];
      }
    }
  } else {
    for (size_t i = 0; i < (size_t)*N; i++) {
      for (size_t j = 0; j < (size_t)dim[1]; j++) {
        VT[j + ldv * i] = V_[i + j * dim[1]];
      }
    }
  }
  for (size_t i = 0; i < (size_t)dim[1]; i++) {
    S[i] = S[i] * (S[i] < 0.0 ? -1.0 : 1.0);
  }
}

template <typename T>
T NuclearNormCpuKernelMod::ComputeMatrixNuclearNorm(int dim0, int dim1, T *mat) {
  int n1 = dim0, n2 = dim1;
  T *M = new T[n1 * n2];
  size_t copy_size = dim0 * dim1 * sizeof(T);
  auto ret = memcpy_s(M, copy_size, &mat[0], copy_size);
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << " For 'NuclearNorm', it does memcpy_s failed. Error no: " << ret;
  }

  int m = n2;
  int n = n1;
  int k = (m < n ? m : n);
  T *tU = new T[m * k];
  T *tS = new T[k];
  T *tVT = new T[k * n];
  svd(&m, &n, &M[0], &m, &tS[0], &tU[0], &m, &tVT[0], &k);
  T nuclear_norm = 0.0;
  for (int i = 0; i < k; i++) {
    nuclear_norm += *(tS + i);
  }
  delete[] tU;
  delete[] tS;
  delete[] tVT;
  delete[] M;

  return nuclear_norm;
}

template <typename T, int32_t RANK>
bool NuclearNormCpuKernelMod::ComputeTensorNuclearNorm(const std::vector<kernel::AddressPtr> &inputs,
                                                       const std::vector<kernel::AddressPtr> &outputs) {
  const int32_t input_dimnum = input_shape.size();
  T *input_data_ptr = reinterpret_cast<T *>(inputs[0]->addr);
  size_t value_num_ = 1;
  for (int32_t i = 0; i < input_dimnum; i++) {
    value_num_ *= input_shape[i];
  }

  size_t total_copy_size = value_num_ * sizeof(T);
  Eigen::Tensor<T, 1, Eigen::RowMajor> eigen_tensor(value_num_);
  auto ret1 = memcpy_s(&eigen_tensor(0), total_copy_size, input_data_ptr, total_copy_size);
  if (ret1 != EOK) {
    MS_LOG(EXCEPTION) << " For 'NuclearNorm', it does memcpy_s failed. Error no: " << ret1;
  }

  std::array<Eigen::DenseIndex, RANK> dim_array;
  for (int32_t i = 0; i < input_dimnum; i++) {
    dim_array.at(i) = input_shape[i];
  }
  Eigen::Tensor<T, RANK, Eigen::RowMajor> reshaped_tensor = eigen_tensor.reshape(dim_array);

  dim[0] = (dim[0] < 0) ? dim[0] + input_dimnum : dim[0];
  dim[1] = (dim[1] < 0) ? dim[1] + input_dimnum : dim[1];

  int32_t j = 0;
  for (int32_t i = 0; i < input_dimnum; i++) {
    if (i != dim[0] && i != dim[1]) {
      dim_array.at(j) = i;
      j++;
    }
  }
  dim_array.at(j) = dim[0];
  dim_array.at(j + 1) = dim[1];

  Eigen::Tensor<T, RANK, Eigen::RowMajor> shuffled_tensor = reshaped_tensor.shuffle(dim_array);

  const int64_t dimsize0 = input_shape[dim[0]];
  const int64_t dimsize1 = input_shape[dim[1]];
  int64_t iter_number = value_num_ / (dimsize0 * dimsize1);
  const size_t DIM_SIZE3 = 3;
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
      T *mat = new T[dimsize0 * dimsize1];
      auto ret2 = memcpy_s(mat, copy_size, &permuted_tensor(i, 0, 0), copy_size);
      if (ret2 != EOK) {
        MS_LOG(EXCEPTION) << " For 'NuclearNorm', it does memcpy_s failed. Error no: " << ret2;
      }
      T nuclear_norm = ComputeMatrixNuclearNorm<T>(dimsize0, dimsize1, mat);
      *(output_data_ptr + i) = nuclear_norm;
    }
  };
  ParallelLaunchAutoSearch(task, iter_number, this, &parallel_search_info_);
  return true;
}

template <typename T>
bool NuclearNormCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                           const std::vector<kernel::AddressPtr> &outputs) {
  bool res = true;
  switch (input_shape.size()) {
    case DIM_SIZE2:
      res = ComputeTensorNuclearNorm<T, DIM_SIZE2>(inputs, outputs);
      break;
    case DIM_SIZE3:
      res = ComputeTensorNuclearNorm<T, DIM_SIZE3>(inputs, outputs);
      break;
    case DIM_SIZE4:
      res = ComputeTensorNuclearNorm<T, DIM_SIZE4>(inputs, outputs);
      break;
    case DIM_SIZE5:
      res = ComputeTensorNuclearNorm<T, DIM_SIZE5>(inputs, outputs);
      break;
    case DIM_SIZE6:
      res = ComputeTensorNuclearNorm<T, DIM_SIZE6>(inputs, outputs);
      break;
    case DIM_SIZE7:
      res = ComputeTensorNuclearNorm<T, DIM_SIZE7>(inputs, outputs);
      break;
    case DIM_SIZE8:
      res = ComputeTensorNuclearNorm<T, DIM_SIZE8>(inputs, outputs);
      break;
    default:
      MS_LOG(EXCEPTION) << "Only tensors with ranks between 2 and 8 are "
                           "currently supported. Tensor rank:"
                        << input_shape.size();
      return false;
  }
  return res;
}

std::vector<std::pair<KernelAttr, NuclearNormCpuKernelMod::NuclearNormFunc>> NuclearNormCpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &NuclearNormCpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
   &NuclearNormCpuKernelMod::LaunchKernel<double>}};

std::vector<KernelAttr> NuclearNormCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, NuclearNormFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, NuclearNorm, NuclearNormCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
