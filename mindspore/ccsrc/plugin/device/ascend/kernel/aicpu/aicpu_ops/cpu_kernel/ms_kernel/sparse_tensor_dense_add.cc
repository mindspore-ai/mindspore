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
#include "sparse_tensor_dense_add.h"
#include <float.h>
#include <securec.h>
#include <complex>
#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "iostream"
#include "kernel_log.h"
#include "status.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
namespace {
const char *kSparseTensorDenseAdd = "SparseTensorDenseAdd";
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 4;
// when input data size is more than kParallelDataNum, use Parallel func
constexpr uint64_t kParallelDataNums = 256 * 1024;
}  // namespace

namespace aicpu {
uint32_t SparseTensorDenseAddCpuKernel::Compute(CpuKernelContext &ctx) {
  if (NormalCheck(ctx, kInputNum, kOutputNum) != KERNEL_STATUS_OK) {
    KERNEL_LOG_ERROR("Check SparseTensorDenseAdd params failed.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  uint32_t res = ValidateInputs(ctx);
  if (res != KERNEL_STATUS_OK) {
    return KERNEL_STATUS_PARAM_INVALID;
  }
  DataType data_type = static_cast<DataType>(ctx.Input(1)->GetDataType());
  switch (data_type) {
    case DT_INT8:
      return DoCompute<int8_t>(ctx);
    case DT_UINT8:
      return DoCompute<uint8_t>(ctx);
    case DT_INT16:
      return DoCompute<int16_t>(ctx);
    case DT_UINT16:
      return DoCompute<uint16_t>(ctx);
    case DT_INT32:
      return DoCompute<int32_t>(ctx);
    case DT_INT64:
      return DoCompute<int64_t>(ctx);
    case DT_FLOAT16:
      return DoCompute<Eigen::half>(ctx);
    case DT_FLOAT:
      return DoCompute<float>(ctx);
    case DT_DOUBLE:
      return DoCompute<double>(ctx);
    case DT_COMPLEX64:
      return DoCompute<std::complex<float>>(ctx);
    case DT_COMPLEX128:
      return DoCompute<std::complex<double>>(ctx);
    default:
      KERNEL_LOG_ERROR("Unsupported input data type [%s].", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}

template <typename T>
uint32_t SparseTensorDenseAddCpuKernel::DoCompute(CpuKernelContext &ctx) {
  Tensor *a_indices = ctx.Input(0);
  Tensor *a_values = ctx.Input(1);
  Tensor *b = ctx.Input(3);
  Tensor *out = ctx.Output(0);
  const int NDIMS = static_cast<int>(a_indices->GetTensorShape()->GetDimSize(1));
  auto b_data = reinterpret_cast<T *>(b->GetData());
  auto out_data = reinterpret_cast<T *>(out->GetData());
  auto value_data = reinterpret_cast<T *>(a_values->GetData());
  const auto ix_ = std::make_shared<EigenTensor>(a_indices, a_indices->GetData());
  DataType dt = static_cast<DataType>(a_indices->GetDataType());
  uint32_t data_num = out->NumElements();
  if (data_num <= kParallelDataNums) {
    for (size_t i = 0; i < data_num; i++) {
      out_data[i] = b_data[i];
    }
  } else {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
    if (max_core_num > data_num) {
      max_core_num = data_num;
    }
    auto shared_sparsetensordenseadd = [&](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        out_data[i] = b_data[i];
      }
    };
    if (max_core_num == 0) {
      KERNEL_LOG_ERROR("max_core_num could not be 0.");
    }
    CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, shared_sparsetensordenseadd);
  }
  const int num_nnz = static_cast<int>(a_indices->GetTensorShape()->GetDimSize(0));
  std::vector<int64_t> strides(NDIMS);
  if (NDIMS > 0) {
    strides[NDIMS - 1] = 1;
  }
  for (int d = NDIMS - 2; d >= 0; --d) {
    const int64_t dimsize = b->GetTensorShape()->GetDimSize(d + 1);
    strides[d] = strides[d + 1] * dimsize;
  }
  for (int i = 0; i < num_nnz; ++i) {
    int64_t ix = 0;
    for (int d = 0; d < NDIMS; ++d) {
      int64_t ix_i_d = 0;
      if (dt == DT_INT32) {
        auto a_indices_mat = ix_->matrix<int32_t>();
        ix_i_d = a_indices_mat(i, d);
      } else {
        auto a_indices_mat = ix_->matrix<int64_t>();
        ix_i_d = a_indices_mat(i, d);
      }
      ix += strides[d] * ix_i_d;
    }
    out_data[ix] += value_data[i];
  }
  return KERNEL_STATUS_OK;
}

uint32_t SparseTensorDenseAddCpuKernel::ValidateInputs(CpuKernelContext &ctx) {
  Tensor *a_indices_t = ctx.Input(0);
  Tensor *a_values_t = ctx.Input(1);
  Tensor *a_shape_t = ctx.Input(2);
  Tensor *b_t = ctx.Input(3);
  Tensor *out_t = ctx.Output(0);
  const int a_indices_shape_dims = 2;
  DataType input0_dt = a_values_t->GetDataType();
  DataType input1_dt = b_t->GetDataType();
  DataType input2_dt = out_t->GetDataType();
  const int ndims = static_cast<int>(a_indices_t->GetTensorShape()->GetDimSize(1));
  const int min_ndims = 1;
  const int max_ndims = 5;
  if (ndims < min_ndims || ndims > max_ndims) {
    KERNEL_LOG_ERROR("Only tensors with ranks between 1 and 5 are currently supported. Tensor rank: [%d]", ndims);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  // valid data type
  if (input0_dt != input1_dt || input1_dt != input2_dt) {
    KERNEL_LOG_ERROR("x1_values data type[%s], x2 data type[%s] and y data type[%s] must be same.",
                     DTypeStr(input0_dt).c_str(), DTypeStr(input1_dt).c_str(), DTypeStr(input2_dt).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  int32_t IndiceType = a_indices_t->GetDataType();
  int32_t ShapeType = a_shape_t->GetDataType();
  bool validIndiceType = (IndiceType != DT_INT64 && IndiceType != DT_INT32);
  bool validShapeType = (ShapeType != DT_INT64 && ShapeType != DT_INT32);
  if (validShapeType || validIndiceType) {
    KERNEL_LOG_ERROR(
      "Valid indice or shape data type failed, indiceType [%d], shapeType "
      "[%d].",
      IndiceType, ShapeType);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (IndiceType != ShapeType) {
    KERNEL_LOG_ERROR(
      "Indice type and shape type should be same, indiceType [%d], shapeType "
      "[%d].",
      IndiceType, ShapeType);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  //  valid data shape
  if (a_indices_t->GetTensorShape()->GetDims() != a_indices_shape_dims) {
    KERNEL_LOG_ERROR("Input a_indices should be a matrix but get dim size: [%d].",
                     a_indices_t->GetTensorShape()->GetDims());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (a_values_t->GetTensorShape()->GetDims() != 1 || a_shape_t->GetTensorShape()->GetDims() != 1) {
    KERNEL_LOG_ERROR(
      "Inputs a_values and a_shape should be vectors but received shapes: "
      "[%d] and [%d]",
      a_values_t->GetTensorShape()->GetDims(), a_shape_t->GetTensorShape()->GetDims());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (a_shape_t->NumElements() != b_t->GetTensorShape()->GetDims() ||
      out_t->GetTensorShape()->GetDims() != b_t->GetTensorShape()->GetDims()) {
    KERNEL_LOG_ERROR(
      "Three operands have different ranks; received: [%lld] , [%lld] and "
      "[%lld]",
      a_shape_t->NumElements(), b_t->GetTensorShape()->GetDims(), out_t->GetTensorShape()->GetDims());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  std::shared_ptr<EigenTensor> a_shape_ = std::make_shared<EigenTensor>(a_shape_t, a_shape_t->GetData());
  if (IndiceType == DT_INT32) {
    auto a_shape_flat = a_shape_->vec<int32_t>();
    for (int i = 0; i < b_t->GetTensorShape()->GetDims(); i++) {
      if (out_t->GetTensorShape()->GetDimSize(i) != b_t->GetTensorShape()->GetDimSize(i) ||
          a_shape_flat(i) != b_t->GetTensorShape()->GetDimSize(i)) {
        KERNEL_LOG_ERROR(
          "Dimension [%d] does not equal (no broadcasting is supported): y "
          "side [%lld] vs x2 shape side [%lld] vs x1 shape side [%lld]",
          i, out_t->GetTensorShape()->GetDimSize(i), b_t->GetTensorShape()->GetDimSize(i), a_shape_flat(i));
        return KERNEL_STATUS_PARAM_INVALID;
      }
    }
  } else {
    auto a_shape_flat = a_shape_->vec<int64_t>();
    for (int i = 0; i < b_t->GetTensorShape()->GetDims(); i++) {
      if (out_t->GetTensorShape()->GetDimSize(i) != b_t->GetTensorShape()->GetDimSize(i) ||
          a_shape_flat(i) != b_t->GetTensorShape()->GetDimSize(i)) {
        KERNEL_LOG_ERROR(
          "Dimension [%d] does not equal (no broadcasting is supported): y "
          "side [%lld] vs x2 shape side [%lld] vs x1 shape side [%lld]",
          i, out_t->GetTensorShape()->GetDimSize(i), b_t->GetTensorShape()->GetDimSize(i), a_shape_flat(i));
        return KERNEL_STATUS_PARAM_INVALID;
      }
    }
  }
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kSparseTensorDenseAdd, SparseTensorDenseAddCpuKernel);
}  // namespace aicpu