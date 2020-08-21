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

#include "mindspore/lite/tools/converter/parser/caffe/caffe_convolution_parser.h"
#include <memory>

namespace mindspore {
namespace lite {
STATUS CaffeConvolutionParser::ParseGroupConvolution(schema::CNodeT *op,
                                                   schema::Conv2DT *attr) {
  if (attr->group == 1) {
    return RET_OK;
  }
  std::unique_ptr<schema::DepthwiseConv2DT> depthwiseConv2DParam
    = std::make_unique<schema::DepthwiseConv2DT>();
  if (depthwiseConv2DParam == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return RET_ERROR;
  }

  depthwiseConv2DParam->format = attr->format;
  depthwiseConv2DParam->channelIn = attr->channelIn;
  depthwiseConv2DParam->channelMultiplier = attr->channelOut / attr->channelIn;
  depthwiseConv2DParam->kernelW = attr->kernelW;
  depthwiseConv2DParam->kernelH = attr->kernelH;
  depthwiseConv2DParam->strideW = attr->strideW;
  depthwiseConv2DParam->strideH = attr->strideH;
  depthwiseConv2DParam->padMode = attr->padMode;
  depthwiseConv2DParam->padUp = attr->padUp;
  depthwiseConv2DParam->padDown = attr->padDown;
  depthwiseConv2DParam->padLeft = attr->padLeft;
  depthwiseConv2DParam->padRight = attr->padRight;
  depthwiseConv2DParam->dilateW = attr->dilateW;
  depthwiseConv2DParam->dilateH = attr->dilateH;
  depthwiseConv2DParam->hasBias = attr->hasBias;
  depthwiseConv2DParam->activationType = attr->activationType;
  delete attr;
  op->primitive->value.type = schema::PrimitiveType_DepthwiseConv2D;
  op->primitive->value.value = depthwiseConv2DParam.release();
  return RET_OK;
}

STATUS CaffeConvolutionParser::Parse(const caffe::LayerParameter &proto,
                                     const caffe::LayerParameter &weight,
                                     schema::CNodeT *op,
                                     std::vector<schema::TensorT *> *weightVec) {
  MS_LOG(DEBUG) << "parse CaffeConvolutionParser";

  if (op == nullptr) {
    MS_LOG(ERROR) << "op is null";
    return RET_NULL_PTR;
  }
  op->primitive = std::make_unique<schema::PrimitiveT>();
  if (op->primitive == nullptr) {
    MS_LOG(ERROR) << "op->primitive is null";
    return RET_NULL_PTR;
  }

  std::unique_ptr<schema::Conv2DT> attr(new (std::nothrow) schema::Conv2DT());

  attr->format = schema::Format_NCHW;

  const caffe::ConvolutionParameter convParam = proto.convolution_param();
  CaffeConvBaseParser convParser;
  // parse pad
  std::vector<int64_t> pad(4, 0);
  auto status = convParser.ParsePads(convParam, &pad);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "ParsePads for " << proto.name().c_str() <<" failed";
    return RET_ERROR;
  }
  attr->padUp = pad[0];
  attr->padDown = pad[1];
  attr->padLeft = pad[2];
  attr->padRight = pad[3];

  // parse stride
  std::vector<int64_t> stride(2, 0);
  status = convParser.ParseStrides(convParam, &stride);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "ParseStrides for " << proto.name().c_str() << " failed";
    return RET_ERROR;
  }
  attr->strideH = stride[0];
  attr->strideW = stride[1];

  // parse dilation
  std::vector<int64_t> dilation(2, 0);
  status = convParser.ParseDilations(convParam, &dilation);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "ParseDilations for " << proto.name().c_str() << " failed";
    return RET_ERROR;
  }
  attr->dilateH = dilation[0];
  attr->dilateW = dilation[1];

  // parse kernel
  std::vector<int64_t> kernel(2, 0);
  status = convParser.ParseKernels(convParam, &kernel);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "ParseKernels for " << proto.name().c_str() << " failed";
    return RET_ERROR;
  }
  attr->kernelH = kernel[0];
  attr->kernelW = kernel[1];

  attr->hasBias = convParam.bias_term();
  attr->group = convParser.ParseGroup(convParam, proto.type());
  auto ret = convParser.ParseChannelOut(convParam, &(attr->channelOut));
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "conv channel out failed";
    return RET_ERROR;
  }
  auto &weightBlob = weight.blobs(0);
  if (weightBlob.has_shape()) {
    attr->channelIn = weightBlob.shape().dim(1) * attr->group;
  } else {
    // get shape information from Blob parameters(caffe proto v1)
    attr->channelIn = weightBlob.channels() * attr->group;
  }
  attr->padMode = schema::PadMode_CAFFE;

  op->name = proto.name();
  op->primitive->value.type = schema::PrimitiveType_Conv2D;
  op->primitive->value.value = attr.get();

  status = ParseGroupConvolution(op, attr.release());
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Parse group convolution failed";
    return RET_ERROR;
  }

  status = convParser.ParseWeight(weight, weightVec);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "ParseWeight for " << proto.name().c_str() << " failed";
    return RET_ERROR;
  }

  return status;
}

CaffeNodeRegistrar g_caffeConvolutionParser("Convolution", new CaffeConvolutionParser());
}  // namespace lite
}  // namespace mindspore

