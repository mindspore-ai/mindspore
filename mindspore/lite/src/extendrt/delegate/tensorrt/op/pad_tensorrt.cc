/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "src/extendrt/delegate/tensorrt/op/pad_tensorrt.h"
#include <numeric>
#include <functional>
#include "src/extendrt/delegate/tensorrt/tensorrt_utils.h"
#include "ops/fusion/pad_fusion.h"

namespace mindspore::lite {
int PadTensorRT::IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                           const std::vector<TensorInfo> &out_tensors) {
  if (in_tensors.size() != INPUT_SIZE2 && in_tensors.size() != INPUT_SIZE3) {
    MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size();
    return RET_ERROR;
  }
  if (out_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported output tensor size, size is " << out_tensors.size();
    return RET_ERROR;
  }
  if (!in_tensors_[1].IsConst()) {
    MS_LOG(ERROR) << "invalid pad tensor for: " << op_name_;
    return RET_ERROR;
  }
  auto pad_op = AsOps<ops::PadFusion>();
  if (pad_op == nullptr) {
    MS_LOG(ERROR) << "convert PadFusion failed: " << op_name_;
    return RET_ERROR;
  }
  if (pad_op->HasAttr(ops::kPaddingMode)) {
    padding_mode_ = pad_op->get_padding_mode();
  }
#if TRT_VERSION_GE(8, 0)
  if (padding_mode_ != PaddingMode::CONSTANT && padding_mode_ != PaddingMode::REFLECT) {
    MS_LOG(ERROR) << "Unsupported padding mode: " << PaddingMode(padding_mode_) << ", for op: " << op_name_;
    return RET_ERROR;
  }
#else
  if (padding_mode_ != PaddingMode::CONSTANT) {
    MS_LOG(ERROR) << "Unsupported padding mode: " << PaddingMode(padding_mode_) << ", for op: " << op_name_;
    return RET_ERROR;
  }
#endif
  if (in_tensors[0].format() != Format::NHWC && in_tensors[0].format() != Format::NCHW) {
    MS_LOG(ERROR) << "Unsupported input tensor format of " << in_tensors[0].format();
    return RET_ERROR;
  }
  if (pad_op->HasAttr(ops::kConstantValue)) {
    constant_value_ = pad_op->get_constant_value();
  }
  return RET_OK;
}

int PadTensorRT::AddInnerOpFix(TensorRTContext *ctx, const std::vector<int64_t> &input_shape,
                               nvinfer1::ITensor *pad_input, const std::vector<int> &pad_vec) {
#if TRT_VERSION_GE(8, 0)
  std::vector<int64_t> start_values;
  std::vector<int64_t> size_values;
  std::vector<int64_t> stride_values;
  for (size_t i = 0; i < pad_vec.size(); i += 2) {
    start_values.push_back(-pad_vec[i]);
    stride_values.push_back(1);
    size_values.push_back(input_shape[i / 2] + pad_vec[i] + pad_vec[i + 1]);
  }
  auto slice_layer = ctx->network()->addSlice(*pad_input, ConvertCudaDims(start_values), ConvertCudaDims(size_values),
                                              ConvertCudaDims(stride_values));
  if (slice_layer == nullptr) {
    MS_LOG(ERROR) << "Failed to add slice layer for op " << op_name_;
    return RET_ERROR;
  }
  if (padding_mode_ == PaddingMode::REFLECT) {
    slice_layer->setMode(nvinfer1::SliceMode::kREFLECT);
  } else if (padding_mode_ == PaddingMode::CONSTANT) {
    slice_layer->setMode(nvinfer1::SliceMode::kFILL);
    auto const_input =
      ConvertScalarToITensor(ctx, 1, &constant_value_, DataType::kNumberTypeFloat32, op_name_ + "_fill");
    if (const_input == nullptr) {
      MS_LOG(ERROR) << "Failed to create scalar tensor of constant value for op " << op_name_;
      return RET_ERROR;
    }
    constexpr int fill_input_index = 4;
    slice_layer->setInput(fill_input_index, *const_input);
  } else {
    MS_LOG(ERROR) << "Not support padding mode " << padding_mode_ << " for op " << op_name_;
    return RET_ERROR;
  }
  slice_layer->setName(op_name_.c_str());
  this->layer_ = slice_layer;
  auto out_tensor = slice_layer->getOutput(0);
  if (out_tensor == nullptr) {
    MS_LOG(ERROR) << "Failed to get output tensor of op " << op_name_;
    return RET_ERROR;
  }
  auto output_tensor = ITensorHelper{out_tensor, NCHW, true};
  ctx->RegisterTensor(output_tensor, out_tensors_[0].Name());
  return RET_OK;
#else
  MS_LOG(ERROR) << "Only support pad mode constant and input dims count 8 when trt version < 8.0, op:  " << op_name_;
  return RET_ERROR;
#endif
}

int PadTensorRT::AddInnerOpDynamic(TensorRTContext *ctx, const std::vector<int64_t> &input_shape,
                                   nvinfer1::ITensor *pad_input, const std::vector<int> &pad_vec) {
#if TRT_VERSION_GE(8, 0)
  std::vector<int> pre_values;
  std::vector<int> post_values;
  std::vector<int> stride_values;
  for (size_t i = 0; i < pad_vec.size(); i += 2) {
    pre_values.push_back(-pad_vec[i]);
    post_values.push_back(pad_vec[i + 1]);
    stride_values.push_back(1);
  }
  auto post_tensor = ctx->ConvertTo1DTensor(post_values);
  auto pre_tensor = ctx->ConvertTo1DTensor(pre_values);
  auto shape_tensor = ctx->network()->addShape(*pad_input)->getOutput(0);
  auto size_tensor =
    ctx->network()->addElementWise(*shape_tensor, *post_tensor, nvinfer1::ElementWiseOperation::kSUM)->getOutput(0);
  size_tensor =
    ctx->network()->addElementWise(*size_tensor, *pre_tensor, nvinfer1::ElementWiseOperation::kSUB)->getOutput(0);
  auto slice_layer =
    ctx->network()->addSlice(*pad_input, ConvertCudaDims(pre_values), {-1, {}}, ConvertCudaDims(stride_values));
  if (slice_layer == nullptr) {
    MS_LOG(ERROR) << "Failed to add slice layer for op " << op_name_;
    return RET_ERROR;
  }
  slice_layer->setInput(INPUT_SIZE2, *size_tensor);
  if (padding_mode_ == PaddingMode::REFLECT) {
    slice_layer->setMode(nvinfer1::SliceMode::kREFLECT);
  } else if (padding_mode_ == PaddingMode::CONSTANT) {
    slice_layer->setMode(nvinfer1::SliceMode::kFILL);
    auto const_input =
      ConvertScalarToITensor(ctx, 1, &constant_value_, DataType::kNumberTypeFloat32, op_name_ + "_fill");
    if (const_input == nullptr) {
      MS_LOG(ERROR) << "Failed to create scalar tensor of constant value for op " << op_name_;
      return RET_ERROR;
    }
    constexpr int fill_input_index = 4;
    slice_layer->setInput(fill_input_index, *const_input);
  } else {
    MS_LOG(ERROR) << "Not support padding mode " << padding_mode_ << " for op " << op_name_;
    return RET_ERROR;
  }
  slice_layer->setName(op_name_.c_str());
  this->layer_ = slice_layer;
  auto out_tensor = slice_layer->getOutput(0);
  if (out_tensor == nullptr) {
    MS_LOG(ERROR) << "Failed to get output tensor of op " << op_name_;
    return RET_ERROR;
  }
  auto output_tensor = ITensorHelper{out_tensor, NCHW, true};
  ctx->RegisterTensor(output_tensor, out_tensors_[0].Name());
  return RET_OK;
#else
  MS_LOG(ERROR) << "Only support pad mode constant and input dims count 8 when trt version < 8.0, op:  " << op_name_;
  return RET_ERROR;
#endif
}

int PadTensorRT::AddInnerOp(TensorRTContext *ctx) {
  auto pad_input = input(ctx, 0).trt_tensor_;
  if (pad_input == nullptr) {
    MS_LOG(ERROR) << "TensorRt Tensor of input 0 of pad " << op_name_ << " is nullptr";
    return RET_ERROR;
  }
  auto input_shape = ConvertMSShape(pad_input->getDimensions());
  if (!in_tensors_[1].IsConst()) {
    MS_LOG(ERROR) << "Input 1 of pad " << op_name_ << " is not constant";
    return RET_ERROR;
  }
  auto pad_vec = ConvertTensorAsIntVector(in_tensors_[1]);
  if (pad_vec.empty()) {
    MS_LOG(ERROR) << "Failed to get pad input, node: " << op_name_;
    return RET_ERROR;
  }
  constexpr size_t pad_multi_times = 2;
  if (pad_vec.size() % 2 != 0 && pad_vec.size() != input_shape.size() * pad_multi_times) {
    MS_LOG(ERROR) << "pad tensor is invalid, pad count: " << pad_vec.size()
                  << ", input dims count: " << input_shape.size() << ", op: " << op_name_;
    return RET_ERROR;
  }
  if (input_shape.size() == kDim4 && padding_mode_ == PaddingMode::CONSTANT) {
    return AddInnerOpOld(ctx);
  }
  if (IsDynamicInput(ctx, 0)) {
    return AddInnerOpDynamic(ctx, input_shape, pad_input, pad_vec);
  } else {
    return AddInnerOpFix(ctx, input_shape, pad_input, pad_vec);
  }
}

int PadTensorRT::AddInnerOpOld(TensorRTContext *ctx) {
  TensorInfo &pad_tensor = in_tensors_[1];
  int element_cnt = pad_tensor.ElementNum();
  if (element_cnt != input(ctx, 0).trt_tensor_->getDimensions().nbDims * INPUT_SIZE2) {
    MS_LOG(ERROR) << "pad tensor cnt is invalid. cnt: " << element_cnt
                  << ", input tensor dims cnt: " << input(ctx, 0).trt_tensor_->getDimensions().nbDims;
    return RET_ERROR;
  }

  nvinfer1::ITensor *pad_input = input(ctx, 0).trt_tensor_;
  MS_LOG(DEBUG) << "before transpose " << GetTensorFormat(pad_input, input(ctx, 0).format_, input(ctx, 0).same_format_);

  // trt 6 only support 2D padding
  auto pad_vec = ConvertTensorAsIntVector(in_tensors_[1]);
  if (pad_vec.empty()) {
    MS_LOG(ERROR) << "Failed to get pad input, node: " << op_name_;
    return RET_ERROR;
  }
  nvinfer1::IPaddingLayer *padding_layer = nullptr;
  constexpr size_t expect_pad_size = 8;  // NCHW dim number * 2
  if (pad_vec.size() == expect_pad_size) {
    // only support pad at HW index
    nvinfer1::DimsHW prePadding;
    nvinfer1::DimsHW postPadding;
    // NCHW: 0: N_pre, 1: N_post, 2: C_pre, 3: C_post, 4: H_pre, 5: H_post, 6: W_pre, 7: W_post
    constexpr size_t n_pre = 0;
    constexpr size_t n_post = 1;
    constexpr size_t c_pre = 2;
    constexpr size_t c_post = 3;
    constexpr size_t h_pre = 4;
    constexpr size_t h_post = 5;
    constexpr size_t w_pre = 6;
    constexpr size_t w_post = 7;
    if (pad_vec[n_pre] != 0 || pad_vec[n_post] != 0 || pad_vec[c_pre] != 0 || pad_vec[c_post] != 0) {
      MS_LOG(WARNING) << "tensorrt padding only support pad at HW index, unsupported padding value of: " << op_name_;
    }
    prePadding = nvinfer1::DimsHW{pad_vec[h_pre], pad_vec[w_pre]};
    postPadding = nvinfer1::DimsHW{pad_vec[h_post], pad_vec[w_post]};
    MS_LOG(DEBUG) << op_name_ << " prePadding: " << prePadding.d[0] << ", " << prePadding.d[1]
                  << "; postPadding: " << postPadding.d[0] << ", " << postPadding.d[1];

    padding_layer = ctx->network()->addPadding(*pad_input, prePadding, postPadding);
  } else {
    MS_LOG(ERROR) << "need check for pad_tensor dims: " << op_name_
                  << ", pad_tensor ElementNum: " << pad_tensor.ElementNum();
    return RET_ERROR;
  }
  if (padding_layer == nullptr) {
    MS_LOG(ERROR) << "add padding layer failed for " << op_name_;
    return RET_ERROR;
  }
  this->layer_ = padding_layer;
  padding_layer->setName(op_name_.c_str());
  nvinfer1::ITensor *out_tensor = padding_layer->getOutput(0);
  bool same_format = SameDims(out_tensor->getDimensions(), out_tensors_[0].Shape()) &&
                     SameDims(input(ctx, 0).trt_tensor_->getDimensions(), in_tensors_[0].Shape());
  auto output_helper = ITensorHelper{out_tensor, Format::NCHW, same_format};
  ctx->RegisterTensor(output_helper, out_tensors_[0].Name());
  MS_LOG(DEBUG) << "after transpose " << GetTensorFormat(output_helper);
  return RET_OK;
}

REGISTER_TENSORRT_CREATOR(ops::kNamePadFusion, PadTensorRT)
}  // namespace mindspore::lite
