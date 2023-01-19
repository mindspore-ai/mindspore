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

#include "transpose.h"

#include "cpu_kernel_utils.h"
#include "securec.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 2;
const char *kTranspose = "Transpose";

#define TRANSPOSE_COMPUTE_CASE(DTYPE, TYPE, CTX)            \
  case (DTYPE): {                                           \
    uint32_t result = TransposeCompute<TYPE>(CTX);          \
    if (result != KERNEL_STATUS_OK) {                       \
      KERNEL_LOG_ERROR("Transpose kernel compute failed."); \
      return result;                                        \
    }                                                       \
    break;                                                  \
  }
}  // namespace

namespace aicpu {
uint32_t TransposeCpuKernel::GetTransposeValue(Tensor *tensor, std::vector<int64_t> &value) {
  auto type = tensor->GetDataType();
  if (type == DT_INT32) {
    auto data = reinterpret_cast<int32_t *>(tensor->GetData());
    for (unsigned int i = 0; i < tensor->NumElements(); i++) {
      value.push_back(static_cast<int64_t>(*(data + i)));
    }
  } else if (type == DT_INT64) {
    auto data = reinterpret_cast<int64_t *>(tensor->GetData());
    for (unsigned int i = 0; i < tensor->NumElements(); i++) {
      value.push_back(*(data + i));
    }
  } else {
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t TransposeCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "[%s] check input and output failed.", kTranspose);
  KERNEL_HANDLE_ERROR(TransposeParamCheck(ctx), "[%s] check params failed.", kTranspose);
  auto x_type = ctx.Input(0)->GetDataType();
  switch (x_type) {
    TRANSPOSE_COMPUTE_CASE(DT_BOOL, bool, ctx)
    TRANSPOSE_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    TRANSPOSE_COMPUTE_CASE(DT_UINT8, uint8_t, ctx)
    TRANSPOSE_COMPUTE_CASE(DT_UINT16, uint16_t, ctx)
    TRANSPOSE_COMPUTE_CASE(DT_UINT32, uint32_t, ctx)
    TRANSPOSE_COMPUTE_CASE(DT_UINT64, uint64_t, ctx)
    TRANSPOSE_COMPUTE_CASE(DT_INT8, int8_t, ctx)
    TRANSPOSE_COMPUTE_CASE(DT_INT16, int16_t, ctx)
    TRANSPOSE_COMPUTE_CASE(DT_INT32, int32_t, ctx)
    TRANSPOSE_COMPUTE_CASE(DT_INT64, int64_t, ctx)
    TRANSPOSE_COMPUTE_CASE(DT_FLOAT, float, ctx)
    TRANSPOSE_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    TRANSPOSE_COMPUTE_CASE(DT_COMPLEX64, std::complex<float>, ctx)
    TRANSPOSE_COMPUTE_CASE(DT_COMPLEX128, std::complex<double>, ctx)
    default:
      KERNEL_LOG_ERROR("Transpose kernel data type [%s] not support.", DTypeStr(x_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

uint32_t TransposeCpuKernel::TransposeParamCheck(CpuKernelContext &ctx) {
  std::vector<int64_t> shape_x = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  std::vector<int64_t> shape_perm = ctx.Input(1)->GetTensorShape()->GetDimSizes();

  auto perm_tensor = ctx.Input(1);
  auto y_tensor = ctx.Output(0);

  KERNEL_CHECK_FALSE((shape_perm.size() == 1), KERNEL_STATUS_PARAM_INVALID,
                     "Expected perm to be 1-D tensors , but got [%zu]-D tensors.", shape_x.size())
  KERNEL_CHECK_FALSE((perm_tensor->NumElements() == (unsigned int)shape_x.size()), KERNEL_STATUS_PARAM_INVALID,
                     "Expected the size of perm to be [%zu], but got [%zu].", shape_x.size(),
                     perm_tensor->NumElements())
  KERNEL_CHECK_FALSE((GetTransposeValue(perm_tensor, perm) == KERNEL_STATUS_OK), KERNEL_STATUS_PARAM_INVALID,
                     "perm must be either int32 or int64, but got [%s].", DTypeStr(perm_tensor->GetDataType()).c_str())
  KERNEL_CHECK_FALSE((shape_x.size() > 1), KERNEL_STATUS_PARAM_INVALID,
                     "Expected the dimension of x to be greater than 1-D, but got [%zu].", shape_x.size())

  std::vector<int64_t> shape_y;
  for (size_t i = 0; i < shape_x.size(); ++i) {
    int64_t perm_value = perm.at(i);
    if (shape_x.at(i) == 0) {
      KERNEL_CHECK_FALSE((perm_value == 0), KERNEL_STATUS_PARAM_INVALID,
                         "Expected perm[%zu] == 0 (got %zu), when x shape[%zu] == 0.", i, perm_value, i)
    } else {
      KERNEL_CHECK_FALSE((0 <= perm_value && perm_value <= (unsigned int)shape_x.size() - 1),
                         KERNEL_STATUS_PARAM_INVALID, "Expected perm[%zu] in [0, %zu], but got %zu.", i, shape_x.size(),
                         perm_value)
    }
    int64_t temp_value = 0;
    for (size_t j = 0; j < shape_x.size(); ++j) {
      if ((unsigned int)perm.at(j) == i) {
        break;
      } else {
        temp_value = j + 1;
        KERNEL_CHECK_FALSE((temp_value < (unsigned int)shape_x.size()), KERNEL_STATUS_PARAM_INVALID,
                           "Expected perm value is unique.")
      }
    }
    shape_y.push_back(shape_x.at(perm_value));
  }
  y_tensor->GetTensorShape()->SetDimSizes(shape_y);

  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t TransposeCpuKernel::TransposeCompute(CpuKernelContext &ctx) {
  auto x_data = ctx.Input(0)->GetData();
  auto y_data = ctx.Output(0)->GetData();
  std::vector<int64_t> shape_x = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  std::vector<int64_t> shape_y = ctx.Output(0)->GetTensorShape()->GetDimSizes();
  auto input_data = reinterpret_cast<T *>(x_data);
  auto output_data = reinterpret_cast<T *>(y_data);
  int64_t input_dims = shape_x.size();
  switch (input_dims) {
    case 2: {
      typedef Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>, Eigen::Aligned> Eigen_Tensor_2D;
      Eigen_Tensor_2D input_2D(input_data, shape_x.at(0), shape_x.at(1));
      Eigen_Tensor_2D output_2D(output_data, shape_y.at(0), shape_y.at(1));
      Eigen::array<Eigen::DenseIndex, 2> perm_2D;
      for (size_t i = 0; i < 2; ++i) {
        perm_2D[i] = perm.at(i);
      }
      output_2D = input_2D.shuffle(perm_2D);
      break;
    }
    case 3: {
      typedef Eigen::TensorMap<Eigen::Tensor<T, 3, Eigen::RowMajor>, Eigen::Aligned> Eigen_Tensor_3D;
      Eigen_Tensor_3D input_3D(input_data, shape_x.at(0), shape_x.at(1), shape_x.at(2));
      Eigen_Tensor_3D output_3D(output_data, shape_y.at(0), shape_y.at(1), shape_y.at(2));
      Eigen::array<Eigen::DenseIndex, 3> perm_3D;
      for (size_t i = 0; i < 3; ++i) {
        perm_3D[i] = perm.at(i);
      }
      output_3D = input_3D.shuffle(perm_3D);
      break;
    }
    case 4: {
      typedef Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>, Eigen::Aligned> Eigen_Tensor_4D;
      Eigen_Tensor_4D input_4D(input_data, shape_x.at(0), shape_x.at(1), shape_x.at(2), shape_x.at(3));
      Eigen_Tensor_4D output_4D(output_data, shape_y.at(0), shape_y.at(1), shape_y.at(2), shape_y.at(3));
      Eigen::array<Eigen::DenseIndex, 4> perm_4D;
      for (size_t i = 0; i < 4; ++i) {
        perm_4D[i] = perm.at(i);
      }
      output_4D = input_4D.shuffle(perm_4D);
      break;
    }
    case 5: {
      typedef Eigen::TensorMap<Eigen::Tensor<T, 5, Eigen::RowMajor>, Eigen::Aligned> Eigen_Tensor_5D;
      Eigen_Tensor_5D input_5D(input_data, shape_x.at(0), shape_x.at(1), shape_x.at(2), shape_x.at(3), shape_x.at(4));
      Eigen_Tensor_5D output_5D(output_data, shape_y.at(0), shape_y.at(1), shape_y.at(2), shape_y.at(3), shape_y.at(4));
      Eigen::array<Eigen::DenseIndex, 5> perm_5D;
      for (size_t i = 0; i < 5; ++i) {
        perm_5D[i] = perm.at(i);
      }
      output_5D = input_5D.shuffle(perm_5D);
      break;
    }
    case 6: {
      typedef Eigen::TensorMap<Eigen::Tensor<T, 6, Eigen::RowMajor>, Eigen::Aligned> Eigen_Tensor_6D;
      Eigen_Tensor_6D input_6D(input_data, shape_x.at(0), shape_x.at(1), shape_x.at(2), shape_x.at(3), shape_x.at(4),
                               shape_x.at(5));
      Eigen_Tensor_6D output_6D(output_data, shape_y.at(0), shape_y.at(1), shape_y.at(2), shape_y.at(3), shape_y.at(4),
                                shape_y.at(5));
      Eigen::array<Eigen::DenseIndex, 6> perm_6D;
      for (size_t i = 0; i < 6; ++i) {
        perm_6D[i] = perm.at(i);
      }
      output_6D = input_6D.shuffle(perm_6D);
      break;
    }
    case 7: {
      typedef Eigen::TensorMap<Eigen::Tensor<T, 7, Eigen::RowMajor>, Eigen::Aligned> Eigen_Tensor_7D;
      Eigen_Tensor_7D input_7D(input_data, shape_x.at(0), shape_x.at(1), shape_x.at(2), shape_x.at(3), shape_x.at(4),
                               shape_x.at(5), shape_x.at(6));
      Eigen_Tensor_7D output_7D(output_data, shape_y.at(0), shape_y.at(1), shape_y.at(2), shape_y.at(3), shape_y.at(4),
                                shape_y.at(5), shape_y.at(6));
      Eigen::array<Eigen::DenseIndex, 7> perm_7D;
      for (size_t i = 0; i < 7; ++i) {
        perm_7D[i] = perm.at(i);
      }
      output_7D = input_7D.shuffle(perm_7D);
      break;
    }
    default:
      KERNEL_LOG_ERROR("[%s] : Unhandled input dimensions [%zu].", kTranspose, input_dims);
      return KERNEL_STATUS_INNER_ERROR;
  }

  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kTranspose, TransposeCpuKernel);
}  // namespace aicpu
