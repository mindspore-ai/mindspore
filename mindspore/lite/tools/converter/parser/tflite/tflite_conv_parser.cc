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

namespace mindspore {
namespace lite {
ops::PrimitiveC *TfliteConvParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                         const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::Conv2DFusion>();

  prim->set_pad({0, 0, 0, 0});
  prim->set_group(1);
  prim->set_format(mindspore::Format::NHWC);

  MS_ASSERT(tflite_op != nullptr);
  MS_ASSERT(tflite_model != nullptr);
  const auto &tflite_subgraph = tflite_model->subgraphs.front();
  if (tflite_subgraph == nullptr) {
    MS_LOG(ERROR) << "tflite_subgraph is nullptr";
    return nullptr;
  }
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
  const auto &weight_tensor = tflite_subgraph->tensors.at(tflite_op->inputs[1]);
  if (weight_tensor == nullptr) {
    MS_LOG(ERROR) << "the weight tensor is null";
    return nullptr;
  }
  auto weight_shape = weight_tensor->shape;
  prim->set_in_channel(weight_shape[3]);
  prim->set_out_channel(weight_shape[0]);
  prim->set_kernel_size({weight_shape[1], weight_shape[2]});

  // calculate pad params
  const auto &dataTensor = tflite_subgraph->tensors.at(tflite_op->inputs[0]);
  std::vector<int64_t> params;
  int status = getPaddingParam(dataTensor, padMode, tflite_attr->stride_h, tflite_attr->stride_w, weight_shape[1],
                               weight_shape[2], &params);
  if (status != RET_OK && status != RET_NO_CHANGE) {
    MS_LOG(ERROR) << "get padding params failed";
    return nullptr;
  } else if (status == RET_OK) {
    prim->set_pad_list(params);
  }

  return prim.release();
}

ops::PrimitiveC *TfliteDepthwiseConv2DParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                                    const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::Conv2DFusion>();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "new Conv2DFusion failed";
    return nullptr;
  }

  prim->set_pad({0, 0, 0, 0});
  prim->set_format(mindspore::Format::NHWC);

  MS_ASSERT(tflite_op != nullptr);
  MS_ASSERT(tflite_model != nullptr);
  const auto &tflite_subgraph = tflite_model->subgraphs.front();
  if (tflite_subgraph == nullptr) {
    MS_LOG(ERROR) << "tflite_subgraph is nullptr";
    return nullptr;
  }
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
  const auto &weight_tensor = tflite_subgraph->tensors.at(tflite_op->inputs.at(1));
  if (weight_tensor == nullptr) {
    MS_LOG(ERROR) << "the weight tensor is null";
    return nullptr;
  }
  auto weight_shape = weight_tensor->shape;
  prim->set_kernel_size({weight_shape[1], weight_shape[2]});
  prim->set_in_channel(weight_shape[3]);
  prim->set_group(weight_shape[3] / tflite_attr->depth_multiplier);

  // get data tensor
  const auto &data_tensor = tflite_subgraph->tensors.at(tflite_op->inputs.at(0));
  if (data_tensor == nullptr) {
    MS_LOG(ERROR) << "data_tensor is nullptr";
    return nullptr;
  }
  auto data_shape = data_tensor->shape;
  if (!data_shape.empty()) {
    prim->set_out_channel(data_shape[3] * tflite_attr->depth_multiplier);
  }

  // calculate pad params
  std::vector<int64_t> params;
  int status = getPaddingParam(data_tensor, padMode, tflite_attr->stride_h, tflite_attr->stride_w, weight_shape[1],
                               weight_shape[2], &params);
  if (status != RET_OK && status != RET_NO_CHANGE) {
    MS_LOG(ERROR) << "get padding params failed";
    return nullptr;
  } else if (status == RET_OK) {
    prim->set_pad_list(params);
  }
  prim->AddAttr(ops::kIsDepthWise, MakeValue<bool>(true));

  return prim.release();
}

TfliteNodeRegister g_tfliteConv2DParser(tflite::BuiltinOperator_CONV_2D, new TfliteConvParser());
TfliteNodeRegister g_tfliteDepthwiseConv2DParser(tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
                                                 new TfliteDepthwiseConv2DParser());

}  // namespace lite
}  // namespace mindspore
