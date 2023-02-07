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
#include "slice.h"

#include "securec.h"
#include "cpu_kernel_utils.h"
#include "utils/kernel_util.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include <iostream>

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 3;
const char *kSlice = "Slice";

#define SLICE_COMPUTE_CASE(DTYPE, TYPE, CTX)            \
  case (DTYPE): {                                       \
    uint32_t result = SliceCompute<TYPE>(CTX);          \
    if (result != KERNEL_STATUS_OK) {                   \
      KERNEL_LOG_ERROR("Slice kernel compute failed."); \
      return result;                                    \
    }                                                   \
    break;                                              \
  }
}  // namespace

namespace aicpu {
uint32_t SliceCpuKernel::GetSliceValue(Tensor *tensor, std::vector<int64_t> &value) {
  auto type = tensor->GetDataType();
  if (type == DT_INT32) {
    auto data = reinterpret_cast<int32_t *>(tensor->GetData());
    for (int64_t i = 0; i < tensor->NumElements(); i++) {
      value.push_back(static_cast<int64_t>(*(data + i)));
    }
  } else if (type == DT_INT64) {
    auto data = reinterpret_cast<int64_t *>(tensor->GetData());
    for (int64_t i = 0; i < tensor->NumElements(); i++) {
      value.push_back(*(data + i));
    }
  } else {
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t SliceCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "[%s] check input and output failed.", kSlice);
  KERNEL_HANDLE_ERROR(SliceCheck(ctx), "[%s] check params failed.", kSlice);
  auto x_type = ctx.Input(0)->GetDataType();
  switch (x_type) {
    SLICE_COMPUTE_CASE(DT_BOOL, bool, ctx)
    SLICE_COMPUTE_CASE(DT_UINT16, uint16_t, ctx)
    SLICE_COMPUTE_CASE(DT_UINT32, uint32_t, ctx)
    SLICE_COMPUTE_CASE(DT_UINT64, uint64_t, ctx)
    SLICE_COMPUTE_CASE(DT_UINT8, uint8_t, ctx)
    SLICE_COMPUTE_CASE(DT_INT8, int8_t, ctx)
    SLICE_COMPUTE_CASE(DT_INT64, int64_t, ctx)
    SLICE_COMPUTE_CASE(DT_INT32, int32_t, ctx)
    SLICE_COMPUTE_CASE(DT_INT16, int16_t, ctx)
    SLICE_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
    SLICE_COMPUTE_CASE(DT_FLOAT, float, ctx)
    SLICE_COMPUTE_CASE(DT_DOUBLE, double, ctx)
    SLICE_COMPUTE_CASE(DT_COMPLEX64, std::complex<float>, ctx)
    SLICE_COMPUTE_CASE(DT_COMPLEX128, std::complex<double>, ctx)
    default:
      KERNEL_LOG_ERROR("Slice kernel data type [%s] not support.", DTypeStr(x_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

uint32_t SliceCpuKernel::SliceCheck(CpuKernelContext &ctx) {
  KERNEL_CHECK_NULLPTR(ctx.Input(0)->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input 0 data failed.")
  KERNEL_CHECK_NULLPTR(ctx.Input(1)->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input 1 data failed.")
  KERNEL_CHECK_NULLPTR(ctx.Input(kThirdInputIndex)->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input 2 data failed.")
  KERNEL_CHECK_NULLPTR(ctx.Output(0)->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get output 0 data failed.")

  KERNEL_CHECK_NULLPTR(ctx.Input(0)->GetTensorShape(), KERNEL_STATUS_PARAM_INVALID, "Get input 0 tensor shape failed.")
  KERNEL_CHECK_NULLPTR(ctx.Input(1)->GetTensorShape(), KERNEL_STATUS_PARAM_INVALID, "Get input 1 tensor shape failed.")
  KERNEL_CHECK_NULLPTR(ctx.Input(kThirdInputIndex)->GetTensorShape(), KERNEL_STATUS_PARAM_INVALID,
                       "Get input 2 tensor shape failed.")
  KERNEL_CHECK_NULLPTR(ctx.Output(0)->GetTensorShape(), KERNEL_STATUS_PARAM_INVALID,
                       "Get output 0 tensor shape failed.")

  std::vector<int64_t> shape_x = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  std::vector<int64_t> shape_offsets = ctx.Input(1)->GetTensorShape()->GetDimSizes();
  std::vector<int64_t> shape_size = ctx.Input(kThirdInputIndex)->GetTensorShape()->GetDimSizes();

  auto offsets_tensor = ctx.Input(1);
  auto size_tensor = ctx.Input(2);
  auto y_tensor = ctx.Output(0);

  KERNEL_CHECK_FALSE((offsets_tensor->NumElements() == static_cast<int64_t>(shape_x.size())),
                     KERNEL_STATUS_PARAM_INVALID, "Expected offsets to be 1-D tensors of size [%zu], but got [%zu].",
                     shape_x.size(), offsets_tensor->NumElements())
  KERNEL_CHECK_FALSE((size_tensor->NumElements() == static_cast<int64_t>(shape_x.size())), KERNEL_STATUS_PARAM_INVALID,
                     "Expected size to be 1-D tensors of size [%zu], but got [%zu].", shape_x.size(),
                     size_tensor->NumElements())

  KERNEL_CHECK_FALSE((GetSliceValue(offsets_tensor, offsets) == KERNEL_STATUS_OK), KERNEL_STATUS_PARAM_INVALID,
                     "Offsets must be either int32 or int64, but got [%s].",
                     DTypeStr(offsets_tensor->GetDataType()).c_str())
  KERNEL_CHECK_FALSE((GetSliceValue(size_tensor, size) == KERNEL_STATUS_OK), KERNEL_STATUS_PARAM_INVALID,
                     "Size must be either int32 or int64, but got [%s].", DTypeStr(size_tensor->GetDataType()).c_str())

  is_identity = true;
  slice_dim0 = true;
  std::vector<int64_t> shape_y;
  for (size_t i = 0; i < shape_x.size(); ++i) {
    if (size.at(i) == -1) {
      size.at(i) = shape_x.at(i) - offsets.at(i);
    }
    int64_t offset = offsets.at(i);
    int64_t size_dim = size.at(i);
    if (shape_x.at(i) == 0) {
      KERNEL_CHECK_FALSE((offset == 0 && size_dim == 0), KERNEL_STATUS_PARAM_INVALID,
                         "Expected offsets[%zu] == 0 (got %zu) and size[%zu] == 0 (got %zu),"
                         " when x shape[%zu] == 0.",
                         i, offset, i, size_dim, i)
    } else {
      KERNEL_CHECK_FALSE((0 <= offset && offset < shape_x.at(i)), KERNEL_STATUS_PARAM_INVALID,
                         "Expected offsets[%zu] in [0, %zu], but got %zu.", i, shape_x.at(i), offset)
      KERNEL_CHECK_FALSE((0 <= size_dim && offset + size_dim <= shape_x.at(i)), KERNEL_STATUS_PARAM_INVALID,
                         "Expected size[%zu] in [0, %zu], but got %zu.", i, shape_x.at(i) - offset, size_dim)
    }
    bool take_all = (offset == 0) && (size_dim == shape_x.at(i));
    is_identity &= take_all;
    slice_dim0 &= (i == 0) || take_all;
    shape_y.push_back(size_dim);
  }
  y_tensor->GetTensorShape()->SetDimSizes(shape_y);

  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t SliceCpuKernel::SliceCompute(CpuKernelContext &ctx) {
  auto x_data = ctx.Input(0)->GetData();
  auto y_data = ctx.Output(0)->GetData();
  int64_t num_output = ctx.Output(0)->NumElements();
  std::vector<int64_t> shape_x = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  std::vector<int64_t> shape_y = ctx.Output(0)->GetTensorShape()->GetDimSizes();
  if (num_output == 0) {
    return KERNEL_STATUS_OK;
  }
  if (is_identity) {
    int64_t input_size = ctx.Input(0)->GetDataSize();
    int cpret = memcpy_s(y_data, input_size, x_data, input_size);
    KERNEL_CHECK_FALSE((cpret == EOK), KERNEL_STATUS_INNER_ERROR, "[%s] memcpy_s to output failed, size [%llu].",
                       kSlice, input_size);
    return KERNEL_STATUS_OK;
  }
  if (slice_dim0) {
    int data_size = size.at(0);
    data_size = data_size * sizeof(T);
    int cpret = memcpy_s(y_data, data_size, static_cast<T *>(x_data) + offsets.at(0), data_size);
    KERNEL_CHECK_FALSE((cpret == EOK), KERNEL_STATUS_INNER_ERROR, "[%s] memcpy_s to output failed, size [%llu].",
                       kSlice, data_size);
    return KERNEL_STATUS_OK;
  }

  auto input_data = reinterpret_cast<T *>(x_data);
  auto output_data = reinterpret_cast<T *>(y_data);
  size_t input_dims = shape_x.size();
  switch (input_dims) {
    case INPUT_NUM2: {
      using Eigen_Tensor_2D =
        Eigen::TensorMap<Eigen::Tensor<T, static_cast<int>(INPUT_NUM2), Eigen::RowMajor>, Eigen::Aligned>;
      Eigen_Tensor_2D input_2D(input_data, shape_x.at(0), shape_x.at(1));
      Eigen_Tensor_2D output_2D(output_data, shape_y.at(0), shape_y.at(1));
      Eigen::array<Eigen::DenseIndex, INPUT_NUM2> offsets_2D;
      Eigen::array<Eigen::DenseIndex, INPUT_NUM2> size_2D;
      for (size_t i = 0; i < INPUT_NUM2; ++i) {
        offsets_2D[i] = offsets.at(i);
        size_2D[i] = size.at(i);
      }
      output_2D = input_2D.slice(offsets_2D, size_2D);
      break;
    }
    case INPUT_NUM3: {
      using Eigen_Tensor_3D =
        Eigen::TensorMap<Eigen::Tensor<T, static_cast<int>(INPUT_NUM3), Eigen::RowMajor>, Eigen::Aligned>;
      Eigen_Tensor_3D input_3D(input_data, shape_x.at(0), shape_x.at(1), shape_x.at(INPUT_NUM2));
      Eigen_Tensor_3D output_3D(output_data, shape_y.at(0), shape_y.at(1), shape_y.at(INPUT_NUM2));
      Eigen::array<Eigen::DenseIndex, INPUT_NUM3> offsets_3D;
      Eigen::array<Eigen::DenseIndex, INPUT_NUM3> size_3D;
      for (size_t i = 0; i < INPUT_NUM3; ++i) {
        offsets_3D[i] = offsets.at(i);
        size_3D[i] = size.at(i);
      }
      output_3D = input_3D.slice(offsets_3D, size_3D);
      break;
    }
    case INPUT_NUM4: {
      using Eigen_Tensor_4D =
        Eigen::TensorMap<Eigen::Tensor<T, static_cast<int>(INPUT_NUM4), Eigen::RowMajor>, Eigen::Aligned>;
      Eigen_Tensor_4D input_4D(input_data, shape_x.at(0), shape_x.at(1), shape_x.at(INPUT_NUM2),
                               shape_x.at(INPUT_NUM3));
      Eigen_Tensor_4D output_4D(output_data, shape_y.at(0), shape_y.at(1), shape_y.at(INPUT_NUM2),
                                shape_y.at(INPUT_NUM3));
      Eigen::array<Eigen::DenseIndex, INPUT_NUM4> offsets_4D;
      Eigen::array<Eigen::DenseIndex, INPUT_NUM4> size_4D;
      for (size_t i = 0; i < INPUT_NUM4; ++i) {
        offsets_4D[i] = offsets.at(i);
        size_4D[i] = size.at(i);
      }
      output_4D = input_4D.slice(offsets_4D, size_4D);
      break;
    }
    case INPUT_NUM5: {
      using Eigen_Tensor_5D =
        Eigen::TensorMap<Eigen::Tensor<T, static_cast<int>(INPUT_NUM5), Eigen::RowMajor>, Eigen::Aligned>;
      Eigen_Tensor_5D input_5D(input_data, shape_x.at(0), shape_x.at(1), shape_x.at(INPUT_NUM2), shape_x.at(INPUT_NUM3),
                               shape_x.at(INPUT_NUM4));
      Eigen_Tensor_5D output_5D(output_data, shape_y.at(0), shape_y.at(1), shape_y.at(INPUT_NUM2),
                                shape_y.at(INPUT_NUM3), shape_y.at(INPUT_NUM4));
      Eigen::array<Eigen::DenseIndex, INPUT_NUM5> offsets_5D;
      Eigen::array<Eigen::DenseIndex, INPUT_NUM5> size_5D;
      for (size_t i = 0; i < INPUT_NUM5; ++i) {
        offsets_5D[i] = offsets.at(i);
        size_5D[i] = size.at(i);
      }
      output_5D = input_5D.slice(offsets_5D, size_5D);
      break;
    }
    case INPUT_NUM6: {
      using Eigen_Tensor_6D =
        Eigen::TensorMap<Eigen::Tensor<T, static_cast<int>(INPUT_NUM6), Eigen::RowMajor>, Eigen::Aligned>;
      Eigen_Tensor_6D input_6D(input_data, shape_x.at(0), shape_x.at(1), shape_x.at(INPUT_NUM2), shape_x.at(INPUT_NUM3),
                               shape_x.at(INPUT_NUM4), shape_x.at(INPUT_NUM5));
      Eigen_Tensor_6D output_6D(output_data, shape_y.at(0), shape_y.at(1), shape_y.at(INPUT_NUM2),
                                shape_y.at(INPUT_NUM3), shape_y.at(INPUT_NUM4), shape_y.at(INPUT_NUM5));
      Eigen::array<Eigen::DenseIndex, INPUT_NUM6> offsets_6D;
      Eigen::array<Eigen::DenseIndex, INPUT_NUM6> size_6D;
      for (size_t i = 0; i < INPUT_NUM6; ++i) {
        offsets_6D[i] = offsets.at(i);
        size_6D[i] = size.at(i);
      }
      output_6D = input_6D.slice(offsets_6D, size_6D);
      break;
    }
    case INPUT_NUM7: {
      using Eigen_Tensor_7D =
        Eigen::TensorMap<Eigen::Tensor<T, static_cast<int>(INPUT_NUM7), Eigen::RowMajor>, Eigen::Aligned>;
      Eigen_Tensor_7D input_7D(input_data, shape_x.at(0), shape_x.at(1), shape_x.at(INPUT_NUM2), shape_x.at(INPUT_NUM3),
                               shape_x.at(INPUT_NUM4), shape_x.at(INPUT_NUM5), shape_x.at(INPUT_NUM6));
      Eigen_Tensor_7D output_7D(output_data, shape_y.at(0), shape_y.at(1), shape_y.at(INPUT_NUM2),
                                shape_y.at(INPUT_NUM3), shape_y.at(INPUT_NUM4), shape_y.at(INPUT_NUM5),
                                shape_y.at(INPUT_NUM6));
      Eigen::array<Eigen::DenseIndex, INPUT_NUM7> offsets_7D;
      Eigen::array<Eigen::DenseIndex, INPUT_NUM7> size_7D;
      for (size_t i = 0; i < INPUT_NUM7; ++i) {
        offsets_7D[i] = offsets.at(i);
        size_7D[i] = size.at(i);
      }
      output_7D = input_7D.slice(offsets_7D, size_7D);
      break;
    }
    default:
      KERNEL_LOG_ERROR("[%s] : Unhandled input dimensions [%zu].", kSlice, input_dims);
      return KERNEL_STATUS_INNER_ERROR;
  }

  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kSlice, SliceCpuKernel);
}  // namespace aicpu