/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
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
#include "sspaddmm.h"
#include <complex>
#include <iostream>
#include "utils/eigen_tensor.h"

namespace aicpu {

const char *SSPADDMM = "Sspaddmm";
#define SPADDMM_COMPUTE_CASE(DTYPE, TYPE, CTX)             \
  case (DTYPE): {                                          \
    uint32_t result = SspaddmmCompute<TYPE>(CTX);          \
    if (result != KERNEL_STATUS_OK) {                      \
      KERNEL_LOG_ERROR("Sspaddmm kernel compute failed."); \
      return result;                                       \
    }                                                      \
    break;                                                 \
  }

// scalar * sparse matrix for beta * input alpha * mat1
template <typename T>
T *SspaddmmCpuKernel::ScalarSparseMul(CpuKernelContext &ctx, Tensor *vals, Tensor *scalar) {
  T scalar_val;
  auto scalar_val_addr = scalar->GetData();
  switch (scalar->GetDataType()) {
    case DT_UINT8:
      scalar_val = static_cast<T>(reinterpret_cast<uint8_t *>(scalar_val_addr)[0]);
      break;
    case DT_UINT16:
      scalar_val = static_cast<T>(reinterpret_cast<uint16_t *>(scalar_val_addr)[0]);
      break;
    case DT_UINT32:
      scalar_val = static_cast<T>(reinterpret_cast<uint32_t *>(scalar_val_addr)[0]);
      break;
    case DT_UINT64:
      scalar_val = static_cast<T>(reinterpret_cast<uint64_t *>(scalar_val_addr)[0]);
      break;
    case DT_INT8:
      scalar_val = static_cast<T>(reinterpret_cast<int8_t *>(scalar_val_addr)[0]);
      break;
    case DT_INT16:
      scalar_val = static_cast<T>(reinterpret_cast<int16_t *>(scalar_val_addr)[0]);
      break;
    case DT_INT32:
      scalar_val = static_cast<T>(reinterpret_cast<int32_t *>(scalar_val_addr)[0]);
      break;
    case DT_INT64:
      scalar_val = static_cast<T>(reinterpret_cast<int64_t *>(scalar_val_addr)[0]);
      break;
    case DT_FLOAT16:
      scalar_val = static_cast<T>(reinterpret_cast<Eigen::half *>(scalar_val_addr)[0]);
      break;
    case DT_FLOAT:
      scalar_val = static_cast<T>(reinterpret_cast<float *>(scalar_val_addr)[0]);
      break;
    case DT_DOUBLE:
      scalar_val = static_cast<T>(reinterpret_cast<double *>(scalar_val_addr)[0]);
      break;
    case DT_BOOL:
      scalar_val = static_cast<T>(reinterpret_cast<bool *>(scalar_val_addr)[0]);
      break;
    case DT_COMPLEX64:
      scalar_val = static_cast<T>(reinterpret_cast<std::complex<float> *>(scalar_val_addr)[0].real());
      break;
    case DT_COMPLEX128:
      scalar_val = static_cast<T>(reinterpret_cast<std::complex<double> *>(scalar_val_addr)[0].real());
      break;
    default:
      KERNEL_LOG_ERROR("For Sspaddm, scalar dtype %s not support", DTypeStr(scalar->GetDataType()).c_str());
      return nullptr;
  }
  T *val_addr = reinterpret_cast<T *>(vals->GetData());
  uint32_t data_num = vals->GetTensorShape()->GetDimSize(0);
  T *val_addr_bak = new T[data_num];
  if (data_num >= kParallelDataNumSameShape_) {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, CpuKernelUtils::GetCPUNum(ctx));
    if (data_num <= kParallelDataNumSameShapeMid_) {
      max_core_num = std::min(max_core_num, 4U);
    }
    if (max_core_num > data_num) {
      max_core_num = data_num;
    }
    auto multi = [&val_addr, &val_addr_bak, scalar_val](uint32_t start, uint32_t end) {
      for (uint32_t idx = start; idx < end; idx++) {
        val_addr_bak[idx] = val_addr[idx] * scalar_val;
      }
    };
    max_core_num = max_core_num == 0 ? 1 : max_core_num;
    CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, multi);
  } else {
    // no match for ‘operator*=’ (operand types are ‘Eigen::half’ and ‘float’)
    for (uint32_t idx = 0; idx < data_num; idx++) {
      val_addr_bak[idx] = val_addr[idx] * scalar_val;
    }
  }
  return val_addr_bak;
}

template <typename T>
void SspaddmmCpuKernel::Clear(Tensor *tensor, CpuKernelContext &ctx) {
  T *addr = reinterpret_cast<T *>(tensor->GetData());
  uint32_t num = tensor->GetTensorShape()->GetDimSize(0);
  if (num >= kParallelDataNumSameShape_) {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, CpuKernelUtils::GetCPUNum(ctx));
    if (num <= kParallelDataNumSameShapeMid_) {
      max_core_num = std::min(max_core_num, 4U);
    }
    if (max_core_num > num) {
      max_core_num = num;
    }
    auto multi = [&addr](uint32_t start, uint32_t end) {
      for (uint32_t idx = start; idx < end; idx++) {
        addr[idx] = static_cast<T>(0);
      }
    };
    max_core_num = max_core_num == 0 ? 1 : max_core_num;
    CpuKernelUtils::ParallelFor(ctx, num, num / max_core_num, multi);
  } else {
    // no match for ‘operator*=’ (operand types are ‘Eigen::half’ and ‘float’)
    for (uint32_t idx = 0; idx < num; idx++) {
      addr[idx] = static_cast<T>(0);
    }
  }
}

template <typename T>
void SspaddmmCpuKernel::ClearIndices(Tensor *tensor, CpuKernelContext &ctx) {
  T *addr = reinterpret_cast<T *>(tensor->GetData());
  uint32_t num = 2 * tensor->GetTensorShape()->GetDimSize(1);
  if (num >= kParallelDataNumSameShape_) {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, CpuKernelUtils::GetCPUNum(ctx));
    if (num <= kParallelDataNumSameShapeMid_) {
      max_core_num = std::min(max_core_num, 4U);
    }
    if (max_core_num > num) {
      max_core_num = num;
    }
    auto multi = [&addr](uint32_t start, uint32_t end) {
      for (uint32_t idx = start; idx < end; idx++) {
        addr[idx] = static_cast<T>(0);
      }
    };
    max_core_num = max_core_num == 0 ? 1 : max_core_num;
    CpuKernelUtils::ParallelFor(ctx, num, num / max_core_num, multi);
  } else {
    // no match for ‘operator*=’ (operand types are ‘Eigen::half’ and ‘float’)
    for (uint32_t idx = 0; idx < num; idx++) {
      addr[idx] = static_cast<T>(0);
    }
  }
}

template <typename T1>
uint32_t SspaddmmCpuKernel::BoundaryCheck(Tensor *tensor, Tensor *shape_tensor, int64_t nums, CpuKernelContext &ctx) {
  int64_t row;
  int64_t col;
  if (shape_tensor->GetDataType() == DT_INT32) {
    int32_t *in_dim = reinterpret_cast<int32_t *>(shape_tensor->GetData());
    row = static_cast<int64_t>(in_dim[0]);
    col = static_cast<int64_t>(in_dim[1]);
  } else {
    int64_t *in_dim = reinterpret_cast<int64_t *>(shape_tensor->GetData());
    row = in_dim[0];
    col = in_dim[1];
  }
  if (row <= 0 || col <= 0) {
    KERNEL_LOG_ERROR("For sspaddmm, sparse tensor shape should be positive num but get [%d, %d]", row, col);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  int64_t row_tmp, col_tmp;
  T1 *addr = reinterpret_cast<T1 *>(tensor->GetData());
  uint32_t data_num = static_cast<uint32_t>(nums);
  if (data_num >= kParallelDataNumSameShape_) {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, CpuKernelUtils::GetCPUNum(ctx));
    if (data_num <= kParallelDataNumSameShapeMid_) {
      max_core_num = std::min(max_core_num, 4U);
    }
    if (max_core_num > data_num) {
      max_core_num = data_num;
    }
    auto multi = [&](uint32_t start, uint32_t end) {
      for (uint32_t i = start; i < end; i++) {
        row_tmp = static_cast<int64_t>(addr[i]);
        col_tmp = static_cast<int64_t>(addr[i + data_num]);
        if (row_tmp >= row || row_tmp < 0) {
          KERNEL_LOG_ERROR("For sspaddmm, sparse tensor indices row index [%d] out of range[0, %d]", row_tmp, row);
          return KERNEL_STATUS_PARAM_INVALID;
        }
        if (col_tmp >= col || col_tmp < 0) {
          KERNEL_LOG_ERROR("For sspaddmm, sparse tensor indices col index [%d] out of range[0, %d]", col_tmp, col);
          return KERNEL_STATUS_PARAM_INVALID;
        }
      }
      return KERNEL_STATUS_PARAM_INVALID;
    };
    max_core_num = max_core_num == 0 ? 1 : max_core_num;
    CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, multi);
    return KERNEL_STATUS_OK;
  } else {
    for (uint32_t i = 0; i < data_num; i++) {
      row_tmp = static_cast<int64_t>(addr[i]);
      col_tmp = static_cast<int64_t>(addr[i + data_num]);
      if (row_tmp >= row || row_tmp < 0) {
        KERNEL_LOG_ERROR("For sspaddmm, sparse tensor indices row index [%d] out of range[0, %d]", row_tmp, row);
        return KERNEL_STATUS_PARAM_INVALID;
      }
      if (col_tmp >= col || col_tmp < 0) {
        KERNEL_LOG_ERROR("For sspaddmm, sparse tensor indices col index [%d] out of range[0, %d]", col_tmp, col);
        return KERNEL_STATUS_PARAM_INVALID;
      }
    }
    return KERNEL_STATUS_OK;
  }
}

// sparse matrix multiply dense matrix
template <typename T_idx, typename T>
uint32_t SspaddmmCpuKernel::SparseMulDense(CpuKernelContext &ctx, Tensor *mat1_indices_tensor, T *mat1_val_addr,
                                           Tensor *mat2_values_tensor, Tensor *output_indices_tensor,
                                           Tensor *output_values_tensor, const int64_t row, const int64_t mat2_col) {
  const int mat1_vals_num = mat1_indices_tensor->GetTensorShape()->GetDimSize(1);

  // the result of mat1 @ mat2 will write to output directly
  T_idx *mat1_idx_addr = reinterpret_cast<T_idx *>(mat1_indices_tensor->GetData());
  T *mat2_val_addr = reinterpret_cast<T *>(mat2_values_tensor->GetData());
  int64_t *out_idx_addr = reinterpret_cast<int64_t *>(output_indices_tensor->GetData());
  T *out_val_addr = reinterpret_cast<T *>(output_values_tensor->GetData());
  int out_num = output_indices_tensor->GetTensorShape()->GetDimSize(1);
  std::unordered_map<T_idx, std::unordered_map<int64_t, uint32_t>> idx_map_cnt;
  std::unordered_map<T_idx, std::vector<T_idx>> unrepeated;
  std::unordered_map<T_idx, std::unordered_map<T_idx, std::vector<T>>> co_map_idx;

  // unrepeated : [1 -> [0], 2 -> [1, 2]]
  // co_map_idx : [1][0] -> 0.3
  for (int64_t i = 0; i < mat1_vals_num; i++) {
    T_idx _row = mat1_idx_addr[i];
    T_idx _col = mat1_idx_addr[i + mat1_vals_num];
    unrepeated[_row].push_back(_col);
    co_map_idx[_row][_col].push_back(mat1_val_addr[i]);
    for (uint32_t j = 0; j < mat2_col; j++) {
      if (idx_map_cnt[_row][j] == 0) {
        idx_map_cnt[_row][j] = this->cnt_;
        this->cnt_++;
      }
    }
  }

  std::vector<T_idx> res;
  for (auto it = unrepeated.begin(); it != unrepeated.end(); it++) {
    res.push_back(it->first);
  }

  uint32_t n_unreapeat = unrepeated.size();
  if (n_unreapeat * mat2_col > kParallelDataNumSameShape_) {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, CpuKernelUtils::GetCPUNum(ctx));
    if (n_unreapeat <= kParallelDataNumSameShape_) {
      max_core_num = std::min(max_core_num, 4U);
    }
    if (max_core_num > n_unreapeat) {
      max_core_num = n_unreapeat;
    }
    auto multi = [&](uint32_t start, uint32_t end) {
      for (uint32_t i = start; i < end; i++) {
        // get val
        auto row_mat1 = res[i];
        for (auto row_mat2 : unrepeated[row_mat1]) {
          T val = co_map_idx[row_mat1][row_mat2].back();
          co_map_idx[row_mat1][row_mat2].pop_back();
          for (int64_t j = 0; j < mat2_col; j++) {
            // get val
            T_idx idx = idx_map_cnt[row_mat1][j];
            *(out_val_addr + idx) += val * mat2_val_addr[row_mat2 * mat2_col + j];
            out_idx_addr[idx] = static_cast<int64_t>(row_mat1);
            out_idx_addr[idx + out_num] = j;
          }
        }
      }
    };
    max_core_num = max_core_num == 0 ? 1 : max_core_num;
    CpuKernelUtils::ParallelFor(ctx, n_unreapeat, n_unreapeat / max_core_num, multi);
  } else {
    for (uint32_t i = 0; i < n_unreapeat; i++) {
      // get val
      auto row_mat1 = res[i];
      for (auto row_mat2 : unrepeated[row_mat1]) {
        T val = co_map_idx[row_mat1][row_mat2].back();
        co_map_idx[row_mat1][row_mat2].pop_back();
        for (int64_t j = 0; j < mat2_col; j++) {
          // get val
          T_idx idx = idx_map_cnt[row_mat1][j];
          *(out_val_addr + idx) += val * mat2_val_addr[row_mat2 * mat2_col + j];
          out_idx_addr[idx] = static_cast<int64_t>(row_mat1);
          out_idx_addr[idx + out_num] = j;
        }
      }
    }
  }

  return KERNEL_STATUS_OK;
}

// sparse matrix add sparse matrix
// input + mat1 @ mat2
template <typename T_idx, typename T>
uint32_t SspaddmmCpuKernel::SparseAddSparse(CpuKernelContext &ctx, Tensor *input_indices_tensor, T *in_val_addr,
                                            Tensor *output_indices_tensor, Tensor *output_values_tensor) {
  // to implement m1[row][col] = vals
  uint32_t input_nums = input_indices_tensor->GetTensorShape()->GetDimSize(1);
  this->cnt_ = input_nums;
  // get output vals and index addr
  T *out_val_addr = reinterpret_cast<T *>(output_values_tensor->GetData());

  int64_t *out_idx_addr = reinterpret_cast<int64_t *>(output_indices_tensor->GetData());
  int out_num = output_indices_tensor->GetTensorShape()->GetDimSize(1);
  // if input idx not in output, will append at the end of output

  T_idx *input_addr = reinterpret_cast<T_idx *>(input_indices_tensor->GetData());
  if (input_nums >= kParallelDataNumSameShape_) {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, CpuKernelUtils::GetCPUNum(ctx));
    if (input_nums <= kParallelDataNumSameShapeMid_) {
      max_core_num = std::min(max_core_num, 4U);
    }
    if (max_core_num > input_nums) {
      max_core_num = input_nums;
    }
    auto multi = [&](uint32_t start, uint32_t end) {
      for (uint32_t i = start; i < end; i++) {
        auto row = input_addr[i];
        auto col = input_addr[i + input_nums];
        // else append it at the end
        out_val_addr[i] = in_val_addr[i];
        // copy indices[0]
        out_idx_addr[i] = row;
        // copy indices[1]
        out_idx_addr[i + out_num] = col;
      }
    };
    max_core_num = max_core_num == 0 ? 1 : max_core_num;
    CpuKernelUtils::ParallelFor(ctx, input_nums, input_nums / max_core_num, multi);
  } else {
    for (uint32_t i = 0; i < input_nums; i++) {
      auto row = input_addr[i];
      auto col = input_addr[i + input_nums];

      // else append it at the end
      out_val_addr[i] = in_val_addr[i];
      // copy indices[0]
      out_idx_addr[i] = row;
      // copy indices[1]
      out_idx_addr[i + out_num] = col;
    }
  }
  return KERNEL_STATUS_OK;
}

int64_t SspaddmmCpuKernel::GetIndicesNum(Tensor *tensor) {
  if (tensor->GetDataType() == DT_INT32) {
    int32_t *a = reinterpret_cast<int32_t *>(tensor->GetData());
    return static_cast<int64_t>(a[1]);
  }
  int64_t *a = reinterpret_cast<int64_t *>(tensor->GetData());
  return a[1];
}

template <typename T>
uint32_t SspaddmmCpuKernel::SspaddmmCompute(CpuKernelContext &ctx) {
  Tensor *input_indices_tensor = ctx.Input(0);
  Tensor *input_values_tensor = ctx.Input(1);
  Tensor *input_shapes_tensor = ctx.Input(2);

  Tensor *mat1_indices_tensor = ctx.Input(3);
  Tensor *mat1_values_tensor = ctx.Input(4);
  Tensor *mat1_shapes_tensor = ctx.Input(5);

  Tensor *mat2_values_tensor = ctx.Input(6);
  Tensor *alpha_tensor = ctx.Input(7);
  Tensor *beta_tensor = ctx.Input(8);

  Tensor *output_indices_tensor = ctx.Output(0);
  Tensor *output_values_tensor = ctx.Output(1);

  Clear<T>(output_values_tensor, ctx);
  ClearIndices<int64_t>(output_indices_tensor, ctx);

  // scalar * sparse inplace
  T *input_values_addr_bak = ScalarSparseMul<T>(ctx, input_values_tensor, beta_tensor);
  T *mat1_values_addr_bak = ScalarSparseMul<T>(ctx, mat1_values_tensor, alpha_tensor);
  // sparse * mat write to output directly
  auto row = GetIndicesNum(mat1_shapes_tensor);
  auto col = GetIndicesNum(input_shapes_tensor);

  // sparse + sparse
  if (input_indices_tensor->GetDataType() == DT_INT32) {
    SparseAddSparse<int32_t, T>(ctx, input_indices_tensor, input_values_addr_bak, output_indices_tensor,
                                output_values_tensor);
  } else {
    SparseAddSparse<int64_t, T>(ctx, input_indices_tensor, input_values_addr_bak, output_indices_tensor,
                                output_values_tensor);
  }
  if (mat1_indices_tensor->GetDataType() == DT_INT32) {
    SparseMulDense<int32_t, T>(ctx, mat1_indices_tensor, mat1_values_addr_bak, mat2_values_tensor,
                               output_indices_tensor, output_values_tensor, row, col);
  } else {
    SparseMulDense<int64_t, T>(ctx, mat1_indices_tensor, mat1_values_addr_bak, mat2_values_tensor,
                               output_indices_tensor, output_values_tensor, row, col);
  }
  return KERNEL_STATUS_OK;
}

uint32_t SspaddmmCpuKernel::ValidParam(CpuKernelContext &ctx) {
  // valid input and output nullptr
  Tensor *input_indices_tensor = ctx.Input(0);
  Tensor *input_values_tensor = ctx.Input(1);
  Tensor *input_shapes_tensor = ctx.Input(2);

  Tensor *mat1_indices_tensor = ctx.Input(3);
  Tensor *mat1_values_tensor = ctx.Input(4);
  Tensor *mat1_shapes_tensor = ctx.Input(5);
  Tensor *mat2_tensor = ctx.Input(6);
  Tensor *alpha_tensor = ctx.Input(7);
  Tensor *beta_tensor = ctx.Input(8);

  Tensor *output_indices_tensor = ctx.Output(0);
  Tensor *output_values_tensor = ctx.Output(1);
  Tensor *output_shapes_tensor = ctx.Output(2);

  // valid shape nullptr
  auto mat1_values_shape = mat1_values_tensor->GetTensorShape();
  auto mat1_shapes_shape = mat1_shapes_tensor->GetTensorShape();
  auto mat1_indices_shape = mat1_indices_tensor->GetTensorShape();
  auto mat2_shapes_shape = mat2_tensor->GetTensorShape();
  auto input_values_shape = input_values_tensor->GetTensorShape();
  auto input_shapes_shape = input_shapes_tensor->GetTensorShape();
  auto input_indices_shape = input_indices_tensor->GetTensorShape();
  auto output_values_shape = output_values_tensor->GetTensorShape();
  auto output_shapes_shape = output_shapes_tensor->GetTensorShape();
  auto output_indices_shape = output_indices_tensor->GetTensorShape();
  auto alpha_shape = alpha_tensor->GetTensorShape();
  auto beta_shape = beta_tensor->GetTensorShape();
  // sparse_indices
  // GetDims() will return dims number, uint32_t
  if (mat1_indices_shape->GetDims() != 2) {
    KERNEL_LOG_ERROR(
      "Mat1 sparse_indices should be 2D, got dim "
      "size [%d].",
      mat1_indices_shape->GetDims());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (input_indices_shape->GetDims() != 2) {
    KERNEL_LOG_ERROR(
      "Input sparse_indices should be 2D, got dim "
      "size [%d].",
      input_indices_shape->GetDims());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (output_indices_shape->GetDims() != 2) {
    KERNEL_LOG_ERROR(
      "Output sparse_indices should be 2D, got dim "
      "size [%d].",
      input_indices_shape->GetDims());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  // valid data type
  int32_t mat1_IndiceType = mat1_indices_tensor->GetDataType();
  int32_t input_IndiceType = input_indices_tensor->GetDataType();
  int32_t output_IndiceType = output_indices_tensor->GetDataType();
  int32_t mat1_ShapeType = mat1_shapes_tensor->GetDataType();
  int32_t input_ShapeType = input_shapes_tensor->GetDataType();
  int32_t output_ShapeType = output_shapes_tensor->GetDataType();

  bool validIndiceType = ((mat1_IndiceType == DT_INT32) || (mat1_IndiceType == DT_INT64)) &&
                         ((output_IndiceType == DT_INT32) || (output_IndiceType == DT_INT64)) &&
                         ((input_IndiceType == DT_INT32) || (input_IndiceType == DT_INT64));
  bool validShapeType = ((mat1_ShapeType == DT_INT32) || (mat1_ShapeType == DT_INT64)) &&
                        ((output_ShapeType == DT_INT32) || (output_ShapeType == DT_INT64)) &&
                        ((input_ShapeType == DT_INT32) || (input_ShapeType == DT_INT64));
  if (!validShapeType || !validIndiceType) {
    KERNEL_LOG_ERROR(
      "Valid indice and shape data type failed, "
      "indiceType and shapeType should be INT32 or INT64");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  // sparse_values' number check
  int32_t mat1_values_dims_size = mat1_values_shape->GetDims();
  int32_t input_values_dims_size = input_values_shape->GetDims();

  if ((mat1_values_dims_size != 0) && (mat1_values_dims_size != 1)) {
    KERNEL_LOG_ERROR(
      "mat1 values_shape should be a scalar or a vector, "
      "got dim size [%d].",
      mat1_values_shape->GetDims());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  if ((input_values_dims_size != 0) && (input_values_dims_size != 1)) {
    KERNEL_LOG_ERROR(
      "input values_shape should be a scalar or a vector, "
      "got dim size [%d].",
      input_values_shape->GetDims());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  int64_t mat1_elems_num = mat1_indices_shape->GetDims() > 0 ? mat1_indices_shape->GetDimSize(1) : 1;
  int64_t input_elems_num = input_indices_shape->GetDims() > 0 ? input_indices_shape->GetDimSize(1) : 1;

  if ((mat1_values_dims_size == 1) && (mat1_values_tensor->NumElements() != mat1_elems_num)) {
    KERNEL_LOG_ERROR(
      "mat1 values_shape has incorrect number of elements [%lld], "
      "should be [%lld]",
      mat1_values_tensor->NumElements(), mat1_elems_num);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if ((input_values_dims_size == 1) && (input_values_tensor->NumElements() != input_elems_num)) {
    KERNEL_LOG_ERROR(
      "input values_shape has incorrect number of elements [%lld], "
      "should be [%lld]",
      input_values_tensor->NumElements(), input_elems_num);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (alpha_shape->GetDims() > 1) {
    KERNEL_LOG_ERROR("alpha should be a scalar or vector but get dim num [%d]", alpha_shape->GetDims());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (beta_shape->GetDims() > 1) {
    KERNEL_LOG_ERROR("beta should be a scalar or vector but get dim num [%d]", alpha_shape->GetDims());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  uint32_t status = KERNEL_STATUS_OK;
  if (input_indices_tensor->GetDataType() == DT_INT32) {
    status = BoundaryCheck<int32_t>(input_indices_tensor, input_shapes_tensor, input_values_tensor->NumElements(), ctx);
  } else {
    status = BoundaryCheck<int64_t>(input_indices_tensor, input_shapes_tensor, input_values_tensor->NumElements(), ctx);
  }
  if (status != KERNEL_STATUS_OK) {
    return status;
  }
  if (mat1_indices_tensor->GetDataType() == DT_INT32) {
    status = BoundaryCheck<int32_t>(mat1_indices_tensor, mat1_shapes_tensor, mat1_values_tensor->NumElements(), ctx);
  } else {
    status = BoundaryCheck<int64_t>(mat1_indices_tensor, mat1_shapes_tensor, mat1_values_tensor->NumElements(), ctx);
  }
  return status;
}

uint32_t SspaddmmCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, this->kInputNum, this->kOutputNum),
                      "Sspaddmm check input and output number failed.");
  if (ValidParam(ctx) != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("Valid sparse to dense param error.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  Tensor *input_shapes_tensor = ctx.Input(2);
  Tensor *output_values_tensor = ctx.Output(1);
  Tensor *output_shapes_tensor = ctx.Output(2);
  int64_t *ou_dim = reinterpret_cast<int64_t *>(output_shapes_tensor->GetData());
  if (input_shapes_tensor->GetDataType() == DT_INT32) {
    int32_t *in_dim = reinterpret_cast<int32_t *>(input_shapes_tensor->GetData());
    for (int32_t index = 0; index < 2; ++index) {
      ou_dim[index] = in_dim[index];
    }
  } else {
    int64_t *in_dim = reinterpret_cast<int64_t *>(input_shapes_tensor->GetData());
    for (int32_t index = 0; index < 2; ++index) {
      ou_dim[index] = in_dim[index];
    }
  }
  auto output_dtype = output_values_tensor->GetDataType();
  switch (output_dtype) {
    SPADDMM_COMPUTE_CASE(DT_UINT8, uint8_t, ctx)
    SPADDMM_COMPUTE_CASE(DT_INT8, int8_t, ctx)
    SPADDMM_COMPUTE_CASE(DT_INT16, int16_t, ctx)
    SPADDMM_COMPUTE_CASE(DT_INT32, int32_t, ctx)
    SPADDMM_COMPUTE_CASE(DT_INT64, int64_t, ctx)
    SPADDMM_COMPUTE_CASE(DT_FLOAT, float_t, ctx)
    SPADDMM_COMPUTE_CASE(DT_DOUBLE, double_t, ctx)
    default:
      KERNEL_LOG_ERROR("Sspaddmm kernel data type [%s] not support.", DTypeStr(output_dtype).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(SSPADDMM, SspaddmmCpuKernel);
}  // namespace aicpu