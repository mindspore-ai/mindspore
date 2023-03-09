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
void Im2colCpuKernel::InnerCompute(int64_t c_col, T *x_ptr, T *y_ptr) {
  int64_t w_offset = c_col % kernel_width;
  int64_t h_offset = (c_col / kernel_width) % kernel_height;
  int64_t c_im = c_col / kernel_height / kernel_width;
  for (int64_t h_col = 0; h_col < out_height; ++h_col) {
    int64_t h_im = h_col * stride_height - pad_height_top + h_offset * dilation_height;
    for (int64_t w_col = 0; w_col < out_width; ++w_col) {
      int64_t w_im = w_col * stride_width - pad_width_left + w_offset * dilation_width;
      if (is_NCHW) {
        y_ptr[(c_col * out_height + h_col) * out_width + w_col] =
          (h_im >= kValue0 && w_im >= kValue0 && h_im < input_height && w_im < input_width)
            ? x_ptr[(c_im * input_height + h_im) * input_width + w_im]
            : static_cast<T>(0);
      } else {
        y_ptr[(h_col * out_width + w_col) * out_plane + c_col] =
          (h_im >= kValue0 && w_im >= kValue0 && h_im < input_height && w_im < input_width)
            ? x_ptr[(h_im * input_width + w_im) * input_channel + c_im]
            : static_cast<T>(0);
      }
    }
  }
}

template <typename T>
uint32_t Im2colCpuKernel::Im2colCompute(CpuKernelContext &ctx) {
  Tensor *x = ctx.Input(0);
  Tensor *y = ctx.Output(0);
  std::vector<int64_t> y_shapes = y->GetTensorShape()->GetDimSizes();
  std::vector<int64_t> x_shapes = x->GetTensorShape()->GetDimSizes();
  Format x_format = x->GetTensorShape()->GetFormat();
  is_NCHW = (FORMAT_NCHW == x_format);

  int64_t batch_size = x_shapes[kIndex0];
  input_height = kValue0;
  input_width = kValue0;
  out_height = kValue0;
  out_width = kValue0;
  out_plane = kValue0;

  if (is_NCHW) {
    input_channel = x_shapes[kIndex1];
    input_height = x_shapes[kIndex2];
    input_width = x_shapes[kIndex3];

    out_plane = y_shapes[kIndex1] * y_shapes[kIndex2];
    total_block = y_shapes[kIndex3];
  } else {
    input_channel = x_shapes[kIndex3];
    input_height = x_shapes[kIndex1];
    input_width = x_shapes[kIndex2];

    out_plane = y_shapes[kIndex3] * y_shapes[kIndex1];
    total_block = y_shapes[kIndex2];
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

  if (padding_mode == "VALID") {
    int64_t effective_filter_h = (kernel_height - 1) * dilation_height + 1;
    int64_t effective_filter_w = (kernel_width - 1) * dilation_width + 1;
    out_height = (input_height - effective_filter_h + stride_height) / stride_height;
    out_width = (input_width - effective_filter_w + stride_width) / stride_width;
  } else if (padding_mode == "SAME") {
    pad_height_top = (kernel_height - kValue1) / kValue2;
    pad_width_left = (kernel_width - kValue1) / kValue2;
    out_height = (input_height + stride_height - kValue1) / stride_height;
    out_width = (input_width + stride_width - kValue1) / stride_width;
  } else if (padding_mode == "CALCULATED") {
    int64_t pad_h_top{0}, pad_h_bottom{0}, pad_w_before{0}, pad_w_after{0};
    if (!pads.empty() && pads.size() <= kValue2) {
      pad_height_top = pad_h_top = pad_h_bottom = pads.front();
      pad_width_left = pad_w_before = pad_w_after = pads.back();
    } else if (!pads.empty() && pads.size() == kValue4) {
      pad_height_top = pad_h_top = pads[kIndex0];
      pad_h_bottom = pads[kIndex1];
      pad_width_left = pad_w_before = pads[kIndex2];
      pad_w_after = pads[kIndex3];
    }
    out_height = (input_height + pad_h_top + pad_h_bottom - (dilation_height * (kernel_height - kValue1) + kValue1)) /
                   stride_height +
                 kValue1;
    out_width = (input_width + pad_w_before + pad_w_after - (dilation_width * (kernel_width - kValue1) + kValue1)) /
                  stride_width +
                kValue1;
  }

  KERNEL_CHECK_FALSE(total_block == out_width * out_height, KERNEL_STATUS_PARAM_INVALID,
                     "For 'Im2Col', the output shape's last dim must be equal to out_width * out_width");

  auto x_ptr = reinterpret_cast<T *>(x->GetData());
  auto y_ptr = reinterpret_cast<T *>(y->GetData());
  int64_t inner_size_x = input_height * input_channel * input_width;
  int64_t inner_size_y = out_plane * out_height * out_width;
  for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
    for (int64_t c_col = 0; c_col < out_plane; ++c_col) {
      InnerCompute<T>(c_col, x_ptr, y_ptr);
    }
    x_ptr += inner_size_x;
    y_ptr += inner_size_y;
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