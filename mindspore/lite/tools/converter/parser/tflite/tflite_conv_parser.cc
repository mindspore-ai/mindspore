/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "tools/converter/parser/tflite/tflite_conv_parser.h"
#include <vector>
#include <memory>
#include "ops/fusion/conv2d_fusion.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
namespace {
constexpr int kWeightChannelOut = 0;
constexpr int kWeightKernelH = 1;
constexpr int kWeightKernelW = 2;
constexpr int kWeightChannelIn = 3;
STATUS GetConvPaddingParam(const std::unique_ptr<tflite::TensorT> &tensor, mindspore::PadMode pad_mode,
                           const ops::Conv2DFusion *conv_prim, std::vector<int64_t> *params) {
  MSLITE_CHECK_PTR(tensor);
  MSLITE_CHECK_PTR(params);
  MSLITE_CHECK_PTR(conv_prim);
  if (tensor->shape.empty()) {
    MS_LOG(DEBUG) << "the tensor's shape is dynamic, which obtain only when running.";
    return RET_NO_CHANGE;
  }
  int pad_u = 0;
  int pad_d = 0;
  int pad_l = 0;
  int pad_r = 0;
  if (pad_mode == mindspore::PadMode::SAME) {
    auto shape = tensor->shape;
    MS_CHECK_TRUE_RET(shape.size() == DIMENSION_4D, RET_ERROR);
    int input_h = shape.at(kNHWC_H);
    int input_w = shape.at(kNHWC_W);
    auto strides = conv_prim->get_stride();
    MS_CHECK_TRUE_MSG(strides.size() > 1, RET_ERROR, "conv stride param is invalid.");
    auto dilates = conv_prim->get_dilation();
    MS_CHECK_TRUE_MSG(dilates.size() > 1, RET_ERROR, "conv dilation param is invalid.");
    auto kernel_size = conv_prim->get_kernel_size();
    MS_CHECK_TRUE_MSG(kernel_size.size() > 1, RET_ERROR, "conv kernel_size param is invalid.");
    int stride_h = strides[0];
    int stride_w = strides[1];
    int dilate_h = dilates[0];
    int dilate_w = dilates[1];
    int kernel_h = kernel_size[0];
    int kernel_w = kernel_size[1];
    MS_CHECK_TRUE_MSG(stride_h != 0, RET_ERROR, "stride_h shouldn't be 0");
    MS_CHECK_TRUE_MSG(stride_w != 0, RET_ERROR, "stride_w shouldn't be 0");
    int output_w = ceil(static_cast<float>(input_w) / static_cast<float>(stride_w));
    int output_h = ceil(static_cast<float>(input_h) / static_cast<float>(stride_h));
    if (INT_MUL_OVERFLOW(output_h - 1, stride_h) || INT_MUL_OVERFLOW(kernel_h - 1, dilate_h)) {
      MS_LOG(ERROR) << "int mul overflow";
      return RET_ERROR;
    }
    int pad_h_all = ((output_h - 1) * stride_h + (kernel_h - 1) * dilate_h + 1 - input_h);
    if (INT_MUL_OVERFLOW(output_w - 1, stride_w) || INT_MUL_OVERFLOW(kernel_w - 1, dilate_w)) {
      MS_LOG(ERROR) << "int mul overflow";
      return RET_ERROR;
    }
    int pad_w_all = ((output_w - 1) * stride_w + (kernel_w - 1) * dilate_w + 1 - input_w);
    if (pad_h_all < 0) {
      pad_u = pad_d = 0;
    } else {
      pad_u = pad_h_all / 2;
      pad_d = pad_h_all - pad_u;
    }
    if (pad_w_all < 0) {
      pad_l = pad_r = 0;
    } else {
      pad_l = pad_w_all / 2;
      pad_r = pad_w_all - pad_l;
    }
  }

  params->emplace_back(pad_u);
  params->emplace_back(pad_d);
  params->emplace_back(pad_l);
  params->emplace_back(pad_r);
  return RET_OK;
}
}  // namespace
PrimitiveCPtr TfliteConvParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                      const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                      const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::Conv2DFusion>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);

  prim->set_pad({0, 0, 0, 0});
  prim->set_group(1);
  prim->set_format(mindspore::Format::NHWC);

  const auto &tflite_attr = tflite_op->builtin_options.AsConv2DOptions();
  if (tflite_attr == nullptr) {
    MS_LOG(ERROR) << "get conv attr failed";
    return nullptr;
  }
  prim->set_stride({tflite_attr->stride_h, tflite_attr->stride_w});
  prim->set_dilation({tflite_attr->dilation_h_factor, tflite_attr->dilation_w_factor});
  auto padMode = GetPadMode(tflite_attr->padding);
  prim->set_pad_mode(padMode);
  prim->set_activation_type(GetActivationFunctionType(tflite_attr->fused_activation_function));

  // get weight tensor
  if (tflite_op->inputs.size() < kInputSize1) {
    MS_LOG(ERROR) << "the tflite_op shape is illegal";
    return nullptr;
  }
  MS_CHECK_TRUE_RET(static_cast<size_t>(tflite_op->inputs[SECOND_INPUT]) < tflite_subgraph->tensors.size(), nullptr);
  const auto &weight_tensor = tflite_subgraph->tensors.at(tflite_op->inputs[SECOND_INPUT]);
  if (weight_tensor == nullptr) {
    MS_LOG(ERROR) << "the weight tensor is null";
    return nullptr;
  }
  auto weight_shape = weight_tensor->shape;
  if (weight_shape.empty() || weight_shape.size() < DIMENSION_4D) {
    MS_LOG(ERROR) << "the weight shape is illegal";
    return nullptr;
  }
  prim->set_in_channel(weight_shape[kWeightChannelIn]);
  prim->set_out_channel(weight_shape[kWeightChannelOut]);
  prim->set_kernel_size({weight_shape[kWeightKernelH], weight_shape[kWeightKernelW]});

  // calculate pad params
  const auto &dataTensor = tflite_subgraph->tensors.at(tflite_op->inputs[FIRST_INPUT]);
  if (dataTensor == nullptr) {
    MS_LOG(ERROR) << "dataTensor is nullptr";
    return nullptr;
  }
  std::vector<int64_t> params;
  int status = GetConvPaddingParam(dataTensor, padMode, prim.get(), &params);
  if (status != RET_OK && status != RET_NO_CHANGE) {
    MS_LOG(ERROR) << "get padding params failed";
    return nullptr;
  } else if (status == RET_OK) {
    prim->set_pad_list(params);
  }

  return prim->GetPrim();
}

PrimitiveCPtr TfliteDepthwiseConv2DParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                                 const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                                 const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::Conv2DFusion>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);

  prim->set_pad({0, 0, 0, 0});
  prim->set_format(mindspore::Format::NHWC);

  const auto &tflite_attr = tflite_op->builtin_options.AsDepthwiseConv2DOptions();
  if (tflite_attr == nullptr) {
    MS_LOG(ERROR) << "get op de attr failed";
    return nullptr;
  }
  prim->set_stride({tflite_attr->stride_h, tflite_attr->stride_w});
  prim->set_dilation({tflite_attr->dilation_h_factor, tflite_attr->dilation_w_factor});
  auto padMode = GetPadMode(tflite_attr->padding);
  prim->set_pad_mode(padMode);
  prim->set_activation_type(GetActivationFunctionType(tflite_attr->fused_activation_function));

  // get weight tensor
  if (tflite_op->inputs.size() < kInputSize1) {
    MS_LOG(ERROR) << "the tflite_op shape is illegal";
    return nullptr;
  }
  const auto &weight_tensor = tflite_subgraph->tensors.at(tflite_op->inputs.at(SECOND_INPUT));
  if (weight_tensor == nullptr) {
    MS_LOG(ERROR) << "the weight tensor is null";
    return nullptr;
  }
  auto weight_shape = weight_tensor->shape;
  if (weight_shape.empty() || weight_shape.size() < DIMENSION_4D) {
    MS_LOG(ERROR) << "the weight shape is illegal";
    return nullptr;
  }
  prim->set_kernel_size({weight_shape[kWeightKernelH], weight_shape[kWeightKernelW]});
  prim->set_in_channel(weight_shape[kWeightChannelIn]);
  if (tflite_attr->depth_multiplier == 0) {
    MS_LOG(ERROR) << "depth_multiplier must not be zero!";
    return nullptr;
  }
  prim->set_group(weight_shape[kWeightChannelIn] / tflite_attr->depth_multiplier);

  // get data tensor
  const auto &data_tensor = tflite_subgraph->tensors.at(tflite_op->inputs.at(FIRST_INPUT));
  if (data_tensor == nullptr) {
    MS_LOG(ERROR) << "data_tensor is nullptr";
    return nullptr;
  }
  auto data_shape = data_tensor->shape;
  if (!data_shape.empty()) {
    MS_CHECK_GE(static_cast<int>(data_shape.size()), DIMENSION_4D, nullptr);
    auto multiplier = tflite_attr->depth_multiplier;
    if (INT_MUL_OVERFLOW(data_shape[kNHWC_C], multiplier)) {
      MS_LOG(ERROR) << "data_size overflow";
      return nullptr;
    }
    prim->set_out_channel(data_shape[kNHWC_C] * multiplier);
  }

  // calculate pad params
  std::vector<int64_t> params;
  int status = GetConvPaddingParam(data_tensor, padMode, prim.get(), &params);
  if (status != RET_OK && status != RET_NO_CHANGE) {
    MS_LOG(ERROR) << "get padding params failed";
    return nullptr;
  } else if (status == RET_OK) {
    prim->set_pad_list(params);
  }
  auto value_ptr = MakeValue<bool>(true);
  MS_CHECK_TRUE_RET(value_ptr != nullptr, nullptr);
  auto prim_c = prim->GetPrim();
  MS_CHECK_TRUE_RET(prim_c != nullptr, nullptr);
  (void)prim_c->AddAttr(ops::kIsDepthWise, value_ptr);

  return prim->GetPrim();
}

TfliteNodeRegister g_tfliteConv2DParser(tflite::BuiltinOperator_CONV_2D, new TfliteConvParser());
TfliteNodeRegister g_tfliteDepthwiseConv2DParser(tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
                                                 new TfliteDepthwiseConv2DParser());
}  // namespace lite
}  // namespace mindspore
