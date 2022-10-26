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

#include "plugin/device/cpu/kernel/sspaddmm_cpu_kernel.h"
#include <algorithm>
#include <complex>
#include <map>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kInputsNum = 9;
constexpr size_t kOutputsNum = 3;
constexpr char kKernelName[] = "Sspaddmm";

constexpr size_t kOutputShapeIndex = 2;
constexpr size_t kInputShapeIndex = 2;
constexpr size_t kMat1IndiceIndex = 3;
constexpr size_t kMat1ValueIndex = 4;
constexpr size_t kMat1ShapeIndex = 5;
constexpr size_t kMat2Index = 6;
constexpr size_t kAlphaIndex = 7;
constexpr size_t kBetaIndex = 8;
}  // namespace

void SspaddmmCPUKernelMod::CheckSparseIndices(const TypeId &indices_dtype, void *indices_addr, void *shape_addr,
                                              size_t num, const std::string &x_name) const {
  if (indices_dtype == kNumberTypeInt32) {
    CheckSparseIndicesLegal<int32_t, int32_t>(indices_addr, shape_addr, num, x_name);
  } else {
    CheckSparseIndicesLegal<int64_t, int64_t>(indices_addr, shape_addr, num, x_name);
  }
}

bool SspaddmmCPUKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);
  output_values_dtype_ = inputs.at(kIndex1)->GetDtype();
  input_indices_dtype_ = inputs.at(kIndex0)->GetDtype();
  input_shape_dtype_ = inputs.at(kIndex2)->GetDtype();
  mat1_indices_dtype_ = inputs.at(kIndex3)->GetDtype();
  mat1_shape_dtype_ = inputs.at(kIndex5)->GetDtype();
  alpha_dtype_ = inputs.at(kIndex7)->GetDtype();
  beta_dtype_ = inputs.at(kIndex8)->GetDtype();
  return true;
}

int SspaddmmCPUKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs,
                                 const std::map<uint32_t, tensor::TensorPtr> &) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  auto input_indices_shape = inputs.at(kIndex0)->GetShapeVector();
  auto mat1_indices_shape = inputs.at(kIndex3)->GetShapeVector();
  auto mat2_shape = inputs.at(kIndex6)->GetShapeVector();
  auto y_indices_shape = outputs.at(kIndex0)->GetDeviceShapeAdaptively();
  input_values_num_ = LongToSize(input_indices_shape[1]);
  mat1_values_num_ = LongToSize(mat1_indices_shape[1]);
  y_values_num_ = LongToSize(y_indices_shape[1]);
  mat2_row_ = LongToSize(mat2_shape[0]);
  mat2_col_ = LongToSize(mat2_shape[1]);
  return KRET_OK;
}

bool SspaddmmCPUKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                  const std::vector<AddressPtr> &outputs) {
  switch (output_values_dtype_) {
    case kNumberTypeUInt8: {
      LaunchKernel<uint8_t>(inputs, outputs);
      break;
    }
    case kNumberTypeInt8: {
      LaunchKernel<int8_t>(inputs, outputs);
      break;
    }
    case kNumberTypeInt16: {
      LaunchKernel<int16_t>(inputs, outputs);
      break;
    }
    case kNumberTypeInt32: {
      LaunchKernel<int32_t>(inputs, outputs);
      break;
    }
    case kNumberTypeInt64: {
      LaunchKernel<int64_t>(inputs, outputs);
      break;
    }
    case kNumberTypeFloat32: {
      LaunchKernel<float>(inputs, outputs);
      break;
    }
    case kNumberTypeFloat64: {
      LaunchKernel<double>(inputs, outputs);
      break;
    }
    default: {
      MS_EXCEPTION(TypeError) << "For Sspaddmm, The output dtype error.";
    }
  }
  return true;
}

template <typename T>
void SspaddmmCPUKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs) {
  auto input_indices_addr = inputs[0]->addr;
  auto input_values_addr = GetDeviceAddress<T>(inputs, 1);
  auto input_shape_addr = inputs[kInputShapeIndex]->addr;
  auto mat1_indices_addr = inputs[kMat1IndiceIndex]->addr;
  auto mat1_values_addr = GetDeviceAddress<T>(inputs, kMat1ValueIndex);
  auto mat1_shape_addr = inputs[kMat1ShapeIndex]->addr;
  auto mat2_addr = GetDeviceAddress<T>(inputs, kMat2Index);
  auto alpha_val_addr = inputs[kAlphaIndex]->addr;
  auto beta_val_addr = inputs[kBetaIndex]->addr;
  auto y_indices_addr = GetDeviceAddress<int64_t>(outputs, 0);
  auto y_values_addr = GetDeviceAddress<T>(outputs, 1);
  auto y_shape_addr = GetDeviceAddress<int64_t>(outputs, kOutputShapeIndex);

  const std::string x1_name = "x1";
  const std::string x2_name = "x2";
  CheckSparseIndices(input_indices_dtype_, input_indices_addr, input_shape_addr, input_values_num_, x1_name);
  CheckSparseIndices(mat1_indices_dtype_, mat1_indices_addr, mat1_shape_addr, mat1_values_num_, x2_name);

  int64_t mat1_row, mat1_col, input_row, input_col;
  if (mat1_shape_dtype_ == kNumberTypeInt32) {
    auto mat1_shape_val = static_cast<int32_t *>(mat1_shape_addr);
    mat1_row = static_cast<int64_t>(mat1_shape_val[0]);
    mat1_col = static_cast<int64_t>(mat1_shape_val[1]);
  } else {
    auto mat1_shape_val = static_cast<int64_t *>(mat1_shape_addr);
    mat1_row = mat1_shape_val[0];
    mat1_col = mat1_shape_val[1];
  }
  if (input_shape_dtype_ == kNumberTypeInt32) {
    auto input_shape_val = static_cast<int32_t *>(input_shape_addr);
    input_row = static_cast<int64_t>(input_shape_val[0]);
    input_col = static_cast<int64_t>(input_shape_val[1]);
  } else {
    auto input_shape_val = static_cast<int64_t *>(input_shape_addr);
    input_row = input_shape_val[0];
    input_col = input_shape_val[1];
  }

  if (mat1_col != static_cast<int64_t>(mat2_row_)) {
    MS_EXCEPTION(ValueError) << "For Sspaddmm, the sparse tensor x2's dense shape col:" << mat1_col
                             << " should be equal to x3_dense shape row:" << mat2_row_ << ".";
  }
  if (mat1_row != input_row || static_cast<int64_t>(mat2_col_) != input_col) {
    MS_EXCEPTION(ValueError) << "For Sspaddmm, the sparse x1's dense shape "
                             << "[" << input_row << "," << input_col << "]  should equal to x2@x3_dense shape ["
                             << mat1_row << "," << mat2_col_ << "].";
  }

  if (input_shape_dtype_ == kNumberTypeInt32) {
    InitShape<int32_t>(input_shape_addr, y_shape_addr);
  } else {
    InitShape<int64_t>(input_shape_addr, y_shape_addr);
  }

  ClearSparseValues<T>(y_values_addr, y_values_num_);

  // scalar * sparse inplace
  T *input_values_addr_bak = ScalarSparseMul<T>(input_values_addr, beta_val_addr, input_values_num_, beta_dtype_);
  T *mat1_values_addr_bak = ScalarSparseMul<T>(mat1_values_addr, alpha_val_addr, mat1_values_num_, alpha_dtype_);

  // sparse + sparse
  if (input_indices_dtype_ == kNumberTypeInt32) {
    SparseAddSparse<int32_t, T>(input_indices_addr, input_values_addr_bak, input_values_num_, y_indices_addr,
                                y_values_addr, y_values_num_);
  } else {
    SparseAddSparse<int64_t, T>(input_indices_addr, input_values_addr_bak, input_values_num_, y_indices_addr,
                                y_values_addr, y_values_num_);
  }
  auto col = input_col;
  if (mat1_indices_dtype_ == kNumberTypeInt32) {
    SparseMulDense<int32_t, T>(mat1_indices_addr, mat1_values_addr_bak, mat1_values_num_, mat2_addr, y_indices_addr,
                               y_values_addr, y_values_num_, col);
  } else {
    SparseMulDense<int64_t, T>(mat1_indices_addr, mat1_values_addr_bak, mat1_values_num_, mat2_addr, y_indices_addr,
                               y_values_addr, y_values_num_, col);
  }
}

template <typename T, typename S>
void SspaddmmCPUKernelMod::CheckSparseIndicesLegal(void *indices_addr, void *shape_addr, size_t num,
                                                   const std::string &x_name) const {
  auto indices_val = static_cast<T *>(indices_addr);
  auto shape_val = static_cast<S *>(shape_addr);
  int shape_num = 2;
  for (int i = 0; i < shape_num; i++) {
    if (shape_val[i] <= 0) {
      MS_EXCEPTION(ValueError) << "For Sspaddmm, the " << x_name << "_value should be positive"
                               << " while get shape [" << shape_val[0] << ", " << shape_val[1] << "]";
    }
  }
  for (size_t i = 0; i < num; i++) {
    int64_t row = static_cast<int64_t>(shape_val[0]);
    int64_t col = static_cast<int64_t>(shape_val[1]);
    int64_t indices_row = static_cast<int64_t>(indices_val[i]);
    int64_t indices_col = static_cast<int64_t>(indices_val[i + num]);
    if ((indices_row >= row) || indices_col >= col || indices_row < 0 || indices_col < 0) {
      MS_EXCEPTION(ValueError) << "For Sspaddmm, the " << x_name << "_indices"
                               << " row value:" << indices_row << ", col value: " << indices_col << " out of bounds."
                               << " Row should between [0," << row << "].Col should between [0," << col << "].";
    }
  }
}

template <typename T>
void SspaddmmCPUKernelMod::InitShape(void *input_shape, int64_t *y_shape) const {
  auto input_shape_val = static_cast<T *>(input_shape);
  size_t shape_num = 2;
  for (size_t i = 0; i < shape_num; i++) {
    y_shape[i] = static_cast<int64_t>(input_shape_val[i]);
  }
}

template <typename T>
void SspaddmmCPUKernelMod::ClearSparseValues(T *sparse_val, size_t data_num) {
  auto task = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      sparse_val[i] = static_cast<T>(0);
    }
  };

  ParallelLaunchAutoSearch(task, data_num, this, &parallel_search_info_);
}

// scalar * sparse matrix for beta * input alpha * mat1
template <typename T>
T *SspaddmmCPUKernelMod::ScalarSparseMul(const T *sparse_val, void *scalar_val, size_t data_num, const TypeId &tid) {
  T val;
  if (!(data_num > 0)) {
    MS_EXCEPTION(ValueError) << "For Sspaddmm, datanum value error. ";
  }
  T *sparse_val_bak = new T[data_num];
  switch (tid) {
    case kNumberTypeUInt8:
      val = static_cast<T>(static_cast<uint8_t *>(scalar_val)[0]);
      break;
    case kNumberTypeUInt16:
      val = static_cast<T>(static_cast<uint16_t *>(scalar_val)[0]);
      break;
    case kNumberTypeUInt32:
      val = static_cast<T>(static_cast<uint32_t *>(scalar_val)[0]);
      break;
    case kNumberTypeUInt64:
      val = static_cast<T>(static_cast<uint64_t *>(scalar_val)[0]);
      break;
    case kNumberTypeInt8:
      val = static_cast<T>(static_cast<int8_t *>(scalar_val)[0]);
      break;
    case kNumberTypeInt16:
      val = static_cast<T>(static_cast<int16_t *>(scalar_val)[0]);
      break;
    case kNumberTypeInt32:
      val = static_cast<T>(static_cast<int32_t *>(scalar_val)[0]);
      break;
    case kNumberTypeInt64:
      val = static_cast<T>(static_cast<int64_t *>(scalar_val)[0]);
      break;
    case kNumberTypeFloat16:
      val = static_cast<T>(static_cast<float16 *>(scalar_val)[0]);
      break;
    case kNumberTypeFloat32:
      val = static_cast<T>(static_cast<float *>(scalar_val)[0]);
      break;
    case kNumberTypeFloat64:
      val = static_cast<T>(static_cast<double *>(scalar_val)[0]);
      break;
    case kNumberTypeBool:
      val = static_cast<T>(static_cast<bool *>(scalar_val)[0]);
      break;
    case kNumberTypeComplex64:
      val = static_cast<T>(static_cast<std::complex<float> *>(scalar_val)[0].real());
      break;
    case kNumberTypeComplex128:
      val = static_cast<T>(static_cast<std::complex<double> *>(scalar_val)[0].real());
      break;
    default:
      MS_EXCEPTION(TypeError) << "For Sspaddmm, dtype not support. ";
  }
  auto task = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      sparse_val_bak[i] = sparse_val[i] * val;
    }
  };
  ParallelLaunchAutoSearch(task, data_num, this, &parallel_search_info_);
  return sparse_val_bak;
}

// sparse matrix add sparse matrix
// input + mat1 @ mat2
template <typename T, typename S>
void SspaddmmCPUKernelMod::SparseAddSparse(void *input_indices, const S *input_values, size_t input_num,
                                           int64_t *y_indices, S *y_values, size_t y_num) {
  // to implement m1[row][col] = vals
  auto input_ids = static_cast<T *>(input_indices);
  this->cnt_ = input_num;
  // get output vals and index addr
  auto task = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      auto row = input_ids[i];
      auto col = input_ids[i + input_num];
      y_values[i] = input_values[i];
      y_indices[i] = static_cast<int64_t>(row);
      y_indices[i + y_num] = static_cast<int64_t>(col);
    }
  };
  ParallelLaunchAutoSearch(task, input_num, this, &parallel_search_info_);
}

template <typename T, typename S>
void SspaddmmCPUKernelMod::SparseMulDense(void *mat1_indices, const S *mat1_values, size_t mat1_vals_num,
                                          const S *mat2_addr, int64_t *y_indices, S *y_values, size_t y_vals_num,
                                          int64_t mat2_col) {
  // the result of mat1 @ mat2 will write to output directly
  auto mat1_ids = static_cast<T *>(mat1_indices);

  std::unordered_map<T, std::unordered_map<int64_t, uint32_t>> idx_map_cnt;
  std::unordered_map<T, std::vector<T>> unrepeated;
  std::unordered_map<T, std::unordered_map<T, std::vector<S>>> co_map_idx;

  // unrepeated : [1 -> [0], 2 -> [1, 2]]
  // co_map_idx : [1][0] -> 0.3
  for (size_t i = 0; i < mat1_vals_num; i++) {
    T _row = mat1_ids[i];
    T _col = mat1_ids[i + mat1_vals_num];
    unrepeated[_row].push_back(_col);
    co_map_idx[_row][_col].push_back(mat1_values[i]);
    for (int64_t j = 0; j < mat2_col; j++) {
      if (idx_map_cnt[_row][j] == 0) {
        idx_map_cnt[_row][j] = SizeToUint(this->cnt_);
        this->cnt_++;
      }
    }
  }

  std::vector<T> res;
  for (auto it = unrepeated.begin(); it != unrepeated.end(); ++it) {
    res.push_back(it->first);
  }

  size_t n_unreapeat = unrepeated.size();
  auto task = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      // get val
      auto row_mat1 = res[i];
      for (auto row_mat2 : unrepeated[row_mat1]) {
        S val = co_map_idx[row_mat1][row_mat2].back();
        co_map_idx[row_mat1][row_mat2].pop_back();
        for (int64_t j = 0; j < mat2_col; j++) {
          // get val
          size_t idx = static_cast<size_t>(idx_map_cnt[row_mat1][j]);
          int64_t row_mat2_long = static_cast<int64_t>(row_mat2);
          *(y_values + idx) += val * mat2_addr[row_mat2_long * mat2_col + j];
          y_indices[idx] = static_cast<int64_t>(row_mat1);
          y_indices[idx + y_vals_num] = j;
        }
      }
    }
  };
  ParallelLaunchAutoSearch(task, n_unreapeat, this, &parallel_search_info_);
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Sspaddmm, SspaddmmCPUKernelMod);
}  // namespace kernel
}  // namespace mindspore
