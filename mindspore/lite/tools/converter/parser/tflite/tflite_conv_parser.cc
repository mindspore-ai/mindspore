/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
}  // namespace
ops::PrimitiveC *TfliteConvParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                         const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                         const std::unique_ptr<tflite::ModelT> &tflite_model) {
  MS_CHECK_TRUE_RET(tflite_op != nullptr, nullptr);
  MS_CHECK_TRUE_RET(tflite_subgraph != nullptr, nullptr);
  MS_CHECK_TRUE_RET(tflite_model != nullptr, nullptr);
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
  const auto &weight_tensor = tflite_subgraph->tensors.at(tflite_op->inputs[1]);
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
  const auto &dataTensor = tflite_subgraph->tensors.at(tflite_op->inputs[0]);
  std::vector<int64_t> params;
  int status = getPaddingParam(dataTensor, padMode, tflite_attr->stride_h, tflite_attr->stride_w,
                               weight_shape[kWeightKernelH], weight_shape[kWeightKernelW], &params);
  if (status != RET_OK && status != RET_NO_CHANGE) {
    MS_LOG(ERROR) << "get padding params failed";
    return nullptr;
  } else if (status == RET_OK) {
    prim->set_pad_list(params);
  }

  return prim.release();
}

ops::PrimitiveC *TfliteDepthwiseConv2DParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                                    const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                                    const std::unique_ptr<tflite::ModelT> &tflite_model) {
  MS_CHECK_TRUE_RET(tflite_op != nullptr, nullptr);
  MS_CHECK_TRUE_RET(tflite_subgraph != nullptr, nullptr);
  MS_CHECK_TRUE_RET(tflite_model != nullptr, nullptr);
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
  const auto &weight_tensor = tflite_subgraph->tensors.at(tflite_op->inputs.at(1));
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
  const auto &data_tensor = tflite_subgraph->tensors.at(tflite_op->inputs.at(0));
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
  int status = getPaddingParam(data_tensor, padMode, tflite_attr->stride_h, tflite_attr->stride_w,
                               weight_shape[kWeightKernelH], weight_shape[kWeightKernelW], &params);
  if (status != RET_OK && status != RET_NO_CHANGE) {
    MS_LOG(ERROR) << "get padding params failed";
    return nullptr;
  } else if (status == RET_OK) {
    prim->set_pad_list(params);
  }
  auto value_ptr = MakeValue<bool>(true);
  MS_CHECK_TRUE_RET(value_ptr != nullptr, nullptr);
  prim->AddAttr(ops::kIsDepthWise, value_ptr);

  return prim.release();
}

TfliteNodeRegister g_tfliteConv2DParser(tflite::BuiltinOperator_CONV_2D, new TfliteConvParser());
TfliteNodeRegister g_tfliteDepthwiseConv2DParser(tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
                                                 new TfliteDepthwiseConv2DParser());
}  // namespace lite
}  // namespace mindspore
