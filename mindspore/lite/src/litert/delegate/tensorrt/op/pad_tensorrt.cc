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

#include <numeric>
#include <functional>
#include "src/litert/delegate/tensorrt/op/pad_tensorrt.h"
#include "src/litert/delegate/tensorrt/tensorrt_utils.h"

namespace mindspore::lite {
constexpr int PADDING_INDEX0 = 0;
constexpr int PADDING_INDEX1 = 1;
constexpr int PADDING_INDEX2 = 2;
constexpr int PADDING_INDEX3 = 3;
constexpr int PADDING_INDEX4 = 4;
constexpr int PADDING_INDEX5 = 5;
constexpr int PADDING_INDEX6 = 6;
constexpr int PADDING_INDEX7 = 7;
int PadTensorRT::IsSupport(const mindspore::schema::Primitive *primitive,
                           const std::vector<mindspore::MSTensor> &in_tensors,
                           const std::vector<mindspore::MSTensor> &out_tensors) {
  if (!IsShapeKnown()) {
    MS_LOG(ERROR) << "Unsupported input tensor unknown shape: " << op_name_;
    return RET_ERROR;
  }
  if (in_tensors.size() != INPUT_SIZE2 && in_tensors.size() != INPUT_SIZE3) {
    MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size();
    return RET_ERROR;
  }
  if (out_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported output tensor size, size is " << out_tensors.size();
    return RET_ERROR;
  }
  if (in_tensors_[1].Data() == nullptr) {
    MS_LOG(ERROR) << "invalid pad tensor for: " << op_name_;
    return RET_ERROR;
  }
  auto pad_primitive = this->GetPrimitive()->value_as_PadFusion();
  if (pad_primitive == nullptr) {
    MS_LOG(ERROR) << "convert PadFusion failed: " << op_name_;
    return RET_ERROR;
  }
  schema::PaddingMode padding_mode = pad_primitive->padding_mode();
  if (padding_mode != schema::PaddingMode::PaddingMode_CONSTANT) {
    MS_LOG(ERROR) << "Unsupported padding mode: " << schema::PaddingMode(padding_mode) << ", for op: " << op_name_;
    return RET_ERROR;
  }
  if (in_tensors[0].format() != Format::NHWC && in_tensors[0].format() != Format::NCHW) {
    MS_LOG(ERROR) << "Unsupported input tensor format of " << in_tensors[0].format();
    return RET_ERROR;
  }
  constant_value_ = pad_primitive->constant_value();
  return RET_OK;
}

int PadTensorRT::ParasePaddingParam(TensorRTContext *ctx, int element_cnt, nvinfer1::ITensor *pad_input,
                                    const int *padding_data) {
  if (element_cnt == index_NHWC_ * INPUT_SIZE2) {
    // only support pad at HW index
    if (SameDims(pad_input->getDimensions(), in_tensors_[0].Shape())) {
      // NCHW: 0: N_pre, 1: N_post, 2: C_pre, 3: C_post, 4: H_pre, 5: H_post, 6: W_pre, 7: W_post
      if (*padding_data != 0 || *(padding_data + 1) != 0 || *(padding_data + 2) != 0 || *(padding_data + 3) != 0) {
        MS_LOG(WARNING) << "tensorrt padding only support pad at HW index, unsupported padding value of: " << op_name_;
      }
      h_pre_ = PADDING_INDEX4, h_post_ = PADDING_INDEX5, w_pre_ = PADDING_INDEX6, w_post_ = PADDING_INDEX7;
    } else {
      // NHWC: 0: N_pre, 1: N_post, 2: H_pre, 3: H_post, 4: W_pre, 5: W_post, 6: C_pre, 7: C_post
      if (*padding_data != 0 || *(padding_data + 1) != 0 || *(padding_data + 6) != 0 || *(padding_data + 7) != 0) {
        MS_LOG(WARNING) << "tensorrt padding only support pad at HW index, unsupported padding value of: " << op_name_;
      }
      h_pre_ = PADDING_INDEX2, h_post_ = PADDING_INDEX3, w_pre_ = PADDING_INDEX4, w_post_ = PADDING_INDEX5;
    }
  } else if (element_cnt == index_NHC_ * INPUT_SIZE2) {
    if (input(ctx, 0).same_format_) {
      // NCHW: 0: N_pre, 1: N_post, 2: C_pre, 3: C_post, 4: H_pre, 5: H_post, 6: W_pre, 7: W_post
      if (*padding_data != 0 || *(padding_data + 1) != 0) {
        MS_LOG(WARNING) << "tensorrt padding only support pad at HW index, unsupported padding value of: " << op_name_;
      }
      h_pre_ = PADDING_INDEX2, h_post_ = PADDING_INDEX3, w_pre_ = PADDING_INDEX4, w_post_ = PADDING_INDEX5;
    } else {
      // NHWC: 0: N_pre, 1: N_post, 2: H_pre, 3: H_post, 4: W_pre, 5: W_post, 6: C_pre, 7: C_post
      if (*(padding_data + 4) != 0 || *(padding_data + 5) != 0) {
        MS_LOG(WARNING) << "tensorrt padding only support pad at HW index, unsupported padding value of: " << op_name_;
      }
      h_pre_ = PADDING_INDEX0, h_post_ = PADDING_INDEX1, w_pre_ = PADDING_INDEX2, w_post_ = PADDING_INDEX3;
    }
  } else {
    MS_LOG(ERROR) << "need check for pad_tensor dims: " << op_name_ << ", pad_tensor ElementNum";
    return RET_ERROR;
  }
  return RET_OK;
}

int PadTensorRT::AddInnerOp(TensorRTContext *ctx) {
  mindspore::MSTensor &pad_tensor = in_tensors_[1];
  int element_cnt = std::accumulate(pad_tensor.Shape().begin(), pad_tensor.Shape().end(), 1, std::multiplies<int>());
  if (element_cnt != input(ctx, 0).trt_tensor_->getDimensions().nbDims * INPUT_SIZE2) {
    MS_LOG(ERROR) << "pad tensor cnt is invalid. cnt: " << element_cnt
                  << ", input tensor dims cnt: " << input(ctx, 0).trt_tensor_->getDimensions().nbDims;
    return RET_ERROR;
  }

  nvinfer1::ITensor *pad_input = input(ctx, 0).trt_tensor_;
  MS_LOG(DEBUG) << "before transpose " << GetTensorFormat(pad_input, input(ctx, 0).format_, input(ctx, 0).same_format_);
  if (input(ctx, 0).trt_tensor_->getDimensions().nbDims == DIMENSION_4D && input(ctx, 0).format_ == Format::NHWC) {
    // transpose: NHWC->NCHW
    nvinfer1::IShuffleLayer *transpose_layer_in = NHWC2NCHW(ctx, *input(ctx, 0).trt_tensor_);
    if (transpose_layer_in == nullptr) {
      MS_LOG(ERROR) << "transpose: NHWC->NCHW failed";
      return RET_ERROR;
    }
    transpose_layer_in->setName((op_name_ + "_transpose2NCHW").c_str());
    this->transpose_layer_ = transpose_layer_in;
    pad_input = transpose_layer_in->getOutput(0);
    MS_LOG(DEBUG) << "after transpose " << GetTensorFormat(pad_input, Format::NCHW, false);
  }

  // trt 6 only support 2D padding
  const int *padding_data = reinterpret_cast<const int *>(in_tensors_[1].Data().get());
  MS_ASSERT(padding_data);
  nvinfer1::IPaddingLayer *padding_layer = nullptr;
  int padding_ret = ParasePaddingParam(ctx, element_cnt, pad_input, padding_data);
  if (padding_ret) {
    MS_LOG(ERROR) << "parase padding parameter failed";
    return RET_OK;
  }
  bool input_size3 = (input(ctx, 0).trt_tensor_->getDimensions().nbDims == INPUT_SIZE3);
  nvinfer1::DimsHW prePadding{*(padding_data + h_pre_), *(padding_data + w_pre_)};
  nvinfer1::DimsHW postPadding{*(padding_data + h_post_), *(padding_data + w_post_)};
  MS_LOG(DEBUG) << op_name_ << " prePadding: " << prePadding.d[0] << ", " << prePadding.d[1]
                << "; postPadding: " << postPadding.d[0] << ", " << postPadding.d[1];
  if (input_size3) {
    pad_input = ExpandDim(ctx, pad_input, 0);
  }
  padding_layer = ctx->network()->addPadding(*pad_input, prePadding, postPadding);
  if (padding_layer == nullptr) {
    MS_LOG(ERROR) << "add padding layer failed for " << op_name_;
    return RET_ERROR;
  }
  this->layer_ = padding_layer;
  padding_layer->setName(op_name_.c_str());
  nvinfer1::ITensor *out_tensor = padding_layer->getOutput(0);
  if (input_size3) {
    std::vector<int> axes{0};
    auto squeeze_shape = ctx->network()->addShape(*out_tensor)->getOutput(0);
    std::vector<int> subscripts(out_tensor->getDimensions().nbDims);
    std::iota(subscripts.begin(), subscripts.end(), 0);
    auto p = std::remove_if(subscripts.begin(), subscripts.end(),
                            [axes](int x) { return std::find(axes.begin(), axes.end(), x) != axes.end(); });
    subscripts.resize(p - subscripts.begin());
    auto subscripts_tensor = ctx->ConvertTo1DTensor(subscripts);
    auto newDims = ctx->network()->addGather(*squeeze_shape, *subscripts_tensor, 0)->getOutput(0);
    auto shuffle_layer = ctx->network()->addShuffle(*out_tensor);
    shuffle_layer->setInput(1, *newDims);
    out_tensor = shuffle_layer->getOutput(0);
  }
  bool same_format = SameDims(out_tensor->getDimensions(), out_tensors_[0].Shape()) &&
                     SameDims(input(ctx, 0).trt_tensor_->getDimensions(), in_tensors_[0].Shape());
  auto output_helper = ITensorHelper{out_tensor, Format::NCHW, same_format};
  ctx->RegisterTensor(output_helper, out_tensors_[0].Name());
  MS_LOG(DEBUG) << "after transpose " << GetTensorFormat(output_helper);
  return RET_OK;
}
REGISTER_TENSORRT_CREATOR(schema::PrimitiveType_PadFusion, PadTensorRT)
}  // namespace mindspore::lite
