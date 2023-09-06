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

#include "cpu_kernel/ms_kernel/col2im.h"

#include <vector>
#include <complex>

#include "cpu_kernel/inc/cpu_ops_kernel.h"
#include "cpu_kernel/common/cpu_kernel_utils.h"
#include "common/status.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kCol2imInputNum = 2;
const uint32_t kCol2imOutputNum = 1;
constexpr uint32_t kValue0 = 0;
constexpr uint32_t kValue1 = 1;
constexpr uint32_t kValue2 = 2;
constexpr uint32_t kValue4 = 4;
constexpr uint32_t kIndex0 = 0;
constexpr uint32_t kIndex1 = 1;
constexpr uint32_t kIndex2 = 2;
constexpr uint32_t kIndex3 = 3;
const char *kCol2im = "Col2im";
}  // namespace

namespace aicpu {
uint32_t Col2imCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kCol2imInputNum, kCol2imOutputNum), "[%s] check params failed.", kCol2im);
  KERNEL_HANDLE_ERROR(Col2imParamCheck(ctx), "[%s] check params failed.", kCol2im);
  auto data_type = ctx.Input(0)->GetDataType();
  uint32_t ret = KERNEL_STATUS_OK;
  switch (data_type) {
    case DT_COMPLEX64:
      ret = Col2imCompute<std::complex<float>>(ctx);
      break;
    case DT_COMPLEX128:
      ret = Col2imCompute<std::complex<double>>(ctx);
      break;
    case DT_DOUBLE:
      ret = Col2imCompute<double>(ctx);
      break;
    case DT_FLOAT:
      ret = Col2imCompute<float>(ctx);
      break;
    case DT_FLOAT16:
      ret = Col2imCompute<Eigen::half>(ctx);
      break;
    default:
      KERNEL_LOG_ERROR("Range kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      ret = KERNEL_STATUS_PARAM_INVALID;
      break;
  }

  return ret;
}

template <typename T>
static inline T div_rtn(T x, T y) {
  int q = x / y;
  int r = x % y;
  if ((r != 0) && ((r < 0) != (y < 0))) {
    --q;
  }
  return q;
}

uint32_t Col2imCpuKernel::Col2imParamCheck(const CpuKernelContext &ctx) {
  Tensor *input_ = ctx.Input(0);
  Tensor *output_size_ = ctx.Input(1);
  KERNEL_CHECK_NULLPTR(ctx.GetAttr("kernel_size"), KERNEL_STATUS_PARAM_INVALID,
                       "Get ctx.GetAttr(\"kernel_size\") failed.");
  KERNEL_CHECK_NULLPTR(ctx.GetAttr("dilation"), KERNEL_STATUS_PARAM_INVALID, "Get ctx.GetAttr(\"dilation\") failed.");
  KERNEL_CHECK_NULLPTR(ctx.GetAttr("padding"), KERNEL_STATUS_PARAM_INVALID, "Get ctx.GetAttr(\"padding\") failed.");
  KERNEL_CHECK_NULLPTR(ctx.GetAttr("stride"), KERNEL_STATUS_PARAM_INVALID, "Get ctx.GetAttr(\"stride\") failed.");
  std::vector<int64_t> kernel_size = ctx.GetAttr("kernel_size")->GetListInt();
  std::vector<int64_t> dilation = ctx.GetAttr("dilation")->GetListInt();
  std::vector<int64_t> padding = ctx.GetAttr("padding")->GetListInt();
  std::vector<int64_t> stride = ctx.GetAttr("stride")->GetListInt();
  auto output_size_shape = output_size_->GetTensorShape()->GetDimSizes();
  KERNEL_CHECK_FALSE((output_size_shape.size() == kValue1 && output_size_->NumElements() == kValue2),
                     KERNEL_STATUS_PARAM_INVALID,
                     "Expected 1D tensor for output_size with non-zero dimensions for and "
                     "output_size's size equals to 2, but "
                     "got %dD tensor for output_size and output_size's size equals to %d.",
                     output_size_shape.size(), output_size_->NumElements());
  KERNEL_CHECK_FALSE(kernel_size.size() == kValue2, KERNEL_STATUS_PARAM_INVALID,
                     "It is expected kernel_size's size equals to 2, but got size %d.", kernel_size.size());
  KERNEL_CHECK_FALSE(dilation.size() == kValue2, KERNEL_STATUS_PARAM_INVALID,
                     "It is expected dilation_size equals to 2, but got size %d.", dilation.size());
  KERNEL_CHECK_FALSE(padding.size() == kValue2, KERNEL_STATUS_PARAM_INVALID,
                     "It is expected padding_size equals to 2, but got size %d.", padding.size());
  KERNEL_CHECK_FALSE(stride.size() == kValue2, KERNEL_STATUS_PARAM_INVALID,
                     "It is expected stride_size equals to 2, but got size %d.", stride.size());
  int32_t *output_size_data = reinterpret_cast<int32_t *>(output_size_->GetData());
  std::vector<int64_t> output_size(kValue2, kValue0);
  output_size[kIndex0] = output_size_data[kIndex0];
  output_size[kIndex1] = output_size_data[kIndex1];
  const int64_t output_height = output_size.front();
  const int64_t output_width = output_size.back();
  const int64_t kernel_height = kernel_size.front();
  const int64_t kernel_width = kernel_size.back();
  const int64_t dilation_height = dilation.front();
  const int64_t dilation_width = dilation.back();
  const int64_t pad_height = padding.front();
  const int64_t pad_width = padding.back();
  const int64_t stride_height = stride.front();
  const int64_t stride_width = stride.back();
  KERNEL_CHECK_FALSE(output_width > kValue0 && output_height > kValue0, KERNEL_STATUS_PARAM_INVALID,
                     "output should be greater than zero, but got "
                     "output_height: %d output_width: %d.",
                     output_height, output_width);
  KERNEL_CHECK_FALSE(kernel_width > kValue0 && kernel_height > kValue0, KERNEL_STATUS_PARAM_INVALID,
                     "kernel should be greater than zero, but got "
                     "kernel_height: %d kernel_width: %d.",
                     kernel_height, kernel_width);
  KERNEL_CHECK_FALSE(dilation_width > kValue0 && dilation_height > kValue0, KERNEL_STATUS_PARAM_INVALID,
                     "dilation should be greater than zero, but got "
                     "dilation_height: %d dilation_width: %d.",
                     dilation_height, dilation_width);
  KERNEL_CHECK_FALSE(pad_width >= kValue0 && pad_height >= kValue0, KERNEL_STATUS_PARAM_INVALID,
                     "padding should be greater than zero, but got pad_height: "
                     "%d pad_width: %d.",
                     pad_height, pad_width);
  KERNEL_CHECK_FALSE(stride_width > kValue0 && stride_height > kValue0, KERNEL_STATUS_PARAM_INVALID,
                     "stride should be greater than zero, but got "
                     "stride_height: %d stride_width: %d.",
                     stride_height, stride_width);
  auto input_shape = input_->GetTensorShape()->GetDimSizes();
  KERNEL_CHECK_FALSE(
    (input_shape.size() == kValue4 && input_shape[kIndex0] != kValue0 && input_shape[kIndex1] != kValue0 &&
     input_shape[kIndex2] != kValue0 && input_shape[kIndex3] != kValue0),
    KERNEL_STATUS_PARAM_INVALID,
    "Expected 4D (batch mode) tensor for input with non-zero "
    "batch size and non-zero dimensions for input, but got %dD input: (%d %d "
    "%d %d).",
    input_shape.size(), input_shape[kIndex0], input_shape[kIndex1], input_shape[kIndex2], input_shape[kIndex3]);
  KERNEL_CHECK_FALSE(input_shape[kIndex2] == (kernel_width * kernel_height), KERNEL_STATUS_PARAM_INVALID,
                     "Expected size of input's dimension 2 to match the calculated "
                     "number of kernel_size, but got input_shape[2]=%d and kernel_size=(%d, "
                     "%d).",
                     input_shape[kIndex2], kernel_height, kernel_width);
  auto input_length = input_shape[kIndex3];
  int64_t n_blocks_height =
    div_rtn<int64_t>(output_height + 2 * pad_height - dilation_height * (kernel_height - 1) - 1, stride_height) + 1;
  int64_t n_blocks_width =
    div_rtn<int64_t>(output_width + 2 * pad_width - dilation_width * (kernel_width - 1) - 1, stride_width) + 1;
  KERNEL_CHECK_FALSE(input_length == (n_blocks_height * n_blocks_width), KERNEL_STATUS_PARAM_INVALID,
                     "Given output_size=(%d, %d), kernel_size=(%d, %d), dilation=(%d, %d",
                     "), padding=(%d, %d), stride=(%d, %d), expected size of input's "
                     "dimension 2 to match the calculated "
                     "number of sliding blocks %d * %d = %d, but got input.size(2)=%d.",
                     output_height, output_width, kernel_height, kernel_width, dilation_height, dilation_width,
                     pad_height, pad_width, stride_height, stride_width, n_blocks_height, n_blocks_width,
                     (n_blocks_height * n_blocks_width), input_length);
  return KERNEL_STATUS_OK;
}

template <typename T>
void Col2imCpuKernel::InnerCompute(int64_t c_col, int64_t input_offset, int64_t output_offset, T *input_data,
                                   T *output_data) {
  int64_t w_offset = c_col % kernel_width;
  int64_t h_offset = (c_col / kernel_width) % kernel_height;
  int64_t c_im = c_col / kernel_height / kernel_width;
  for (int64_t h_col = 0; h_col < height_col; ++h_col) {
    int64_t h_im = h_col * stride_height - pad_height + h_offset * dilation_height;
    for (int64_t w_col = 0; w_col < width_col; ++w_col) {
      int64_t w_im = w_col * stride_width - pad_width + w_offset * dilation_width;
      if (h_im >= 0 && h_im < output_height && w_im >= 0 && w_im < output_width) {
        output_data[output_offset + (c_im * output_height + h_im) * output_width + w_im] +=
          input_data[input_offset + (c_col * height_col + h_col) * width_col + w_col];
      }
    }
  }
}

template <typename T>
uint32_t Col2imCpuKernel::Col2imCompute(const CpuKernelContext &ctx) {
  Tensor *input_ = ctx.Input(0);
  Tensor *output_size_ = ctx.Input(1);
  Tensor *output_ = ctx.Output(0);
  int32_t *output_size_data = reinterpret_cast<int32_t *>(output_size_->GetData());
  std::vector<int64_t> output_size(kValue2, kValue0);
  output_size[kIndex0] = output_size_data[kIndex0];
  output_size[kIndex1] = output_size_data[kIndex1];

  std::vector<int64_t> kernel_size = ctx.GetAttr("kernel_size")->GetListInt();
  std::vector<int64_t> dilation = ctx.GetAttr("dilation")->GetListInt();
  std::vector<int64_t> padding = ctx.GetAttr("padding")->GetListInt();
  std::vector<int64_t> stride = ctx.GetAttr("stride")->GetListInt();

  output_height = output_size.front();
  output_width = output_size.back();
  kernel_height = kernel_size.front();
  kernel_width = kernel_size.back();
  dilation_height = dilation.front();
  dilation_width = dilation.back();
  pad_height = padding.front();
  pad_width = padding.back();
  stride_height = stride.front();
  stride_width = stride.back();

  auto input_shape = input_->GetTensorShape()->GetDimSizes();
  const int64_t batch_size = input_shape[kIndex0];
  const int64_t n_input_plane = input_shape[kIndex1];

  height_col =
    (output_height + kValue2 * pad_height - (dilation_height * (kernel_height - kValue1) + kValue1)) / stride_height +
    1;
  width_col =
    (output_width + kValue2 * pad_width - (dilation_width * (kernel_width - kValue1) + kValue1)) / stride_width + 1;

  T *input_data = reinterpret_cast<T *>(input_->GetData());
  T *output_data = reinterpret_cast<T *>(output_->GetData());
  std::fill_n(output_data, output_->NumElements(), T(0));
  channels_col = n_input_plane * kernel_height * kernel_width;
  batch_input_size = n_input_plane * kernel_height * kernel_width * height_col * width_col;
  batch_output_size = n_input_plane * output_height * output_width;
  for (int64_t elt = 0; elt < batch_size; ++elt) {
    int64_t input_offset = batch_input_size * elt;
    int64_t output_offset = batch_output_size * elt;
    for (int64_t c_col = 0; c_col < channels_col; ++c_col) {
      InnerCompute<T>(c_col, input_offset, output_offset, input_data, output_data);
    }
  }

  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kCol2im, Col2imCpuKernel);
}  // namespace aicpu
