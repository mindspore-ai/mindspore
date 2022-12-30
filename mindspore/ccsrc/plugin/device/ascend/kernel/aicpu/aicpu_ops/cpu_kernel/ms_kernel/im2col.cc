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
#include "im2col.h"

#include <algorithm>

#include "cpu_types.h"
#include "kernel_log.h"
#include "securec.h"
#include "status.h"
#include "utils/kernel_util.h"

namespace {
const char *kIm2col = "Im2col";
constexpr uint32_t kIm2colInputNum = 1;
constexpr uint32_t kIm2colOutputNum = 1;
constexpr uint32_t kValue0 = 0;
constexpr uint32_t kValue1 = 1;
constexpr uint32_t kValue2 = 2;
constexpr uint32_t kValue4 = 4;
constexpr uint32_t kIndex0 = 0;
constexpr uint32_t kIndex1 = 1;
constexpr uint32_t kIndex2 = 2;
constexpr uint32_t kIndex3 = 3;
#define NotNull(Ptr) ((Ptr) != nullptr)
}  // namespace

namespace aicpu {
// shape <= 2 and all values greater than 0
bool VectorShapeAndValueCheck(std::vector<int64_t> &values) {
  auto iter =
    std::find_if(values.begin(), values.end(), [&](const int64_t &item) -> bool { return (item <= kValue0); });
  return values.size() <= kValue2 && iter == values.end();
}

uint32_t Im2colCpuKernel::Im2colParamCheck(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kIm2colInputNum, kIm2colOutputNum), "[%s] check params failed.", kIm2col);
  // check the shape、format of input tensor x
  const Tensor *x = ctx.Input(0);
  KERNEL_CHECK_FALSE(x->GetTensorShape()->GetDims() == kValue4, KERNEL_STATUS_PARAM_INVALID,
                     "Input tensor x must be 4D tensor.");
  Format x_format = x->GetTensorShape()->GetFormat();
  KERNEL_CHECK_FALSE(x_format == FORMAT_NCHW || x_format == FORMAT_NHWC, KERNEL_STATUS_PARAM_INVALID,
                     "Input tensor x format only support NHWC, NCHW.");
  // ksizes check
  KERNEL_CHECK_NULLPTR(ctx.GetAttr("ksizes"), KERNEL_STATUS_PARAM_INVALID, "Attr 'ksizes' is necessary.");
  ksizes = ctx.GetAttr("ksizes")->GetListInt();
  KERNEL_CHECK_FALSE(VectorShapeAndValueCheck(ksizes), KERNEL_STATUS_PARAM_INVALID,
                     "The size of ksizes must be 1 or 2 and value > 0.");
  // strides check
  if (NotNull(ctx.GetAttr("strides"))) {
    strides = ctx.GetAttr("strides")->GetListInt();
    KERNEL_CHECK_FALSE(VectorShapeAndValueCheck(strides), KERNEL_STATUS_PARAM_INVALID,
                       "The size of strides must be 1 or 2 and value > 0.");
  }
  // dilations check
  if (NotNull(ctx.GetAttr("dilations"))) {
    dilations = ctx.GetAttr("dilations")->GetListInt();
    KERNEL_CHECK_FALSE(VectorShapeAndValueCheck(dilations), KERNEL_STATUS_PARAM_INVALID,
                       "The size of dilations must be 1 or 2 and value > 0.");
  }
  // padding_mode check
  if (NotNull(ctx.GetAttr("padding_mode"))) {
    padding_mode = ctx.GetAttr("padding_mode")->GetString();
    KERNEL_CHECK_FALSE(std::find(padding_modes.begin(), padding_modes.end(), padding_mode) != padding_modes.end(),
                       KERNEL_STATUS_PARAM_INVALID, "The padding_mode only support VALID, SAME and CALCULATED.");
  }
  // only padding_mode == "CALCULATED", to check the pads
  if (padding_mode == "CALCULATED" && NotNull(ctx.GetAttr("pads"))) {
    pads = ctx.GetAttr("pads")->GetListInt();
    auto iter = std::find_if(pads.begin(), pads.end(), [&](const int64_t &item) -> bool { return (item < 0); });
    KERNEL_CHECK_FALSE(iter == pads.end(), KERNEL_STATUS_PARAM_INVALID, "The values of pads must >= 0.");
    KERNEL_CHECK_FALSE(pads.size() == kValue1 || pads.size() == kValue2 || pads.size() == kValue4,
                       KERNEL_STATUS_PARAM_INVALID, "The size of pads must be 1, 2 or 4.");
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
void Im2colCpuKernel::InnerCompute(
  int64_t batch_idx, int64_t c_col,
  Eigen::TensorMap<Eigen::Tensor<T, kValue4, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned> x_4d,
  Eigen::TensorMap<Eigen::Tensor<T, kValue4, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned> y_4d) {
  int64_t w_offset = c_col % kernel_width;
  int64_t h_offset = (c_col / kernel_width) % kernel_height;
  int64_t c_im = c_col / kernel_height / kernel_width;
  for (int64_t h_col = 0; h_col < out_height; ++h_col) {
    int64_t h_im = h_col * stride_height - pad_height_top + h_offset * dilation_height;
    for (int64_t w_col = 0; w_col < out_width; ++w_col) {
      int64_t w_im = w_col * stride_width - pad_width_left + w_offset * dilation_width;
      if (is_NCHW) {
        y_4d(batch_idx, c_col, h_col, w_col) =
          (h_im >= kValue0 && w_im >= kValue0 && h_im < input_height && w_im < input_width)
            ? x_4d(batch_idx, c_im, h_im, w_im)
            : static_cast<T>(0);
      } else {
        y_4d(batch_idx, h_col, w_col, c_col) =
          (h_im >= kValue0 && w_im >= kValue0 && h_im < input_height && w_im < input_width)
            ? x_4d(batch_idx, h_im, w_im, c_im)
            : static_cast<T>(0);
      }
    }
  }
}

template <typename T>
uint32_t Im2colCpuKernel::Im2colCompute(CpuKernelContext &ctx) {
  Tensor *x = ctx.Input(0);
  Tensor *y = ctx.Output(0);
  auto y_col = reinterpret_cast<T *>(y->GetData());
  std::fill_n(y_col, y->NumElements(), T(0));
  std::vector<int64_t> y_shapes = y->GetTensorShape()->GetDimSizes();
  std::vector<int64_t> x_shapes = x->GetTensorShape()->GetDimSizes();
  Format x_format = x->GetTensorShape()->GetFormat();
  is_NCHW = (FORMAT_NCHW == x_format);

  int64_t batch_size = x_shapes[kIndex0];
  input_height = kValue0;
  input_width = kValue0;
  out_height = kValue0;
  out_width = kValue0;
  out_channel = kValue0;

  if (is_NCHW) {
    input_height = x_shapes[kIndex2];
    input_width = x_shapes[kIndex3];

    out_channel = y_shapes[kIndex1];
    out_height = y_shapes[kIndex2];
    out_width = y_shapes[kIndex3];
  } else {
    input_height = x_shapes[kIndex1];
    input_width = x_shapes[kIndex2];

    out_channel = y_shapes[kIndex3];
    out_height = y_shapes[kIndex1];
    out_width = y_shapes[kIndex2];
  }
  kernel_height = ksizes.front();
  kernel_width = ksizes.back();
  stride_height = strides.front();
  stride_width = strides.back();
  dilation_height = dilations.front();
  dilation_width = dilations.back();
  // pad distance
  pad_height_top = kValue0;
  pad_width_left = kValue0;

  if (padding_mode == "CALCULATED") {
    if (!pads.empty() && pads.size() <= kValue2) {
      pad_height_top = pads.front();
      pad_width_left = pads.back();
    } else if (!pads.empty() && pads.size() == kValue4) {
      pad_height_top = pads[kIndex0];
      pad_width_left = pads[kIndex2];
    }
  } else if (padding_mode == "SAME") {
    pad_height_top = (kernel_height - kValue1) / kValue2;
    pad_width_left = (kernel_width - kValue1) / kValue2;
  }  // else VALID no padding

  EigenTensor y_et(y, y->GetData());
  EigenTensor x_et(x, x->GetData());
  auto x_4d = x_et.tensor<T, kValue4>();
  auto y_4d = y_et.tensor<T, kValue4>();

  for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
    for (int64_t c_col = 0; c_col < out_channel; ++c_col) {
      InnerCompute<T>(batch_idx, c_col, x_4d, y_4d);
    }
  }
  return KERNEL_STATUS_OK;
}

uint32_t Im2colCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(Im2colParamCheck(ctx), "[%s] check params failed.", kIm2col);
  auto data_type = ctx.Input(0)->GetDataType();
  uint32_t ret = KERNEL_STATUS_OK;
  switch (data_type) {
    case DT_UINT8:
      ret = Im2colCompute<uint8_t>(ctx);
      break;
    case DT_INT8:
      ret = Im2colCompute<int8_t>(ctx);
      break;
    case DT_UINT16:
      ret = Im2colCompute<uint16_t>(ctx);
      break;
    case DT_INT16:
      ret = Im2colCompute<int16_t>(ctx);
      break;
    case DT_UINT32:
      ret = Im2colCompute<uint32_t>(ctx);
      break;
    case DT_INT32:
      ret = Im2colCompute<int32_t>(ctx);
      break;
    case DT_UINT64:
      ret = Im2colCompute<uint64_t>(ctx);
      break;
    case DT_INT64:
      ret = Im2colCompute<int64_t>(ctx);
      break;
    case DT_FLOAT16:
      ret = Im2colCompute<Eigen::half>(ctx);
      break;
    case DT_FLOAT:
      ret = Im2colCompute<float>(ctx);
      break;
    case DT_DOUBLE:
      ret = Im2colCompute<double>(ctx);
      break;
    default:
      KERNEL_LOG_ERROR("Im2col kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      ret = KERNEL_STATUS_PARAM_INVALID;
      break;
  }
  return ret;
}

REGISTER_CPU_KERNEL(kIm2col, Im2colCpuKernel);
}  // namespace aicpu