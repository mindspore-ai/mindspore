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

#include "tools/converter/parser/caffe/caffe_convolution_parser.h"
#include <memory>

namespace mindspore {
namespace lite {
STATUS CaffeConvolutionParser::ParseDepthwiseConvolution(schema::PrimitiveT *primitiveT, schema::Conv2DT *attr) {
  if (attr->group == 1 || attr->group != attr->channelOut) {
    return RET_OK;
  }
  std::unique_ptr<schema::DepthwiseConv2DT> depthwiseConv2DParam = std::make_unique<schema::DepthwiseConv2DT>();
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
  depthwiseConv2DParam->activationType = attr->activationType;
  delete attr;
  primitiveT->value.type = schema::PrimitiveType_DepthwiseConv2D;
  primitiveT->value.value = depthwiseConv2DParam.release();
  return RET_OK;
}

PrimitiveC *CaffeConvolutionParser::ParseLitePrimitive(const caffe::LayerParameter &proto,
                                                       const caffe::LayerParameter &weight) {
  auto attr = std::make_unique<schema::Conv2DT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new attr failed";
    return nullptr;
  }

  attr->format = schema::Format_NCHW;

  const caffe::ConvolutionParameter &convParam = proto.convolution_param();
  // parse pad
  std::vector<int64_t> pad(4, 0);
  auto status = CaffeConvBaseParser::ParsePads(convParam, &pad);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "ParsePads for " << proto.name().c_str() << " failed";
    return nullptr;
  }
  attr->padUp = pad[0];
  attr->padDown = pad[1];
  attr->padLeft = pad[2];
  attr->padRight = pad[3];

  // parse stride
  std::vector<int64_t> stride(2, 0);
  status = CaffeConvBaseParser::ParseStrides(convParam, &stride);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "ParseStrides for " << proto.name().c_str() << " failed";
    return nullptr;
  }
  attr->strideH = stride[0];
  attr->strideW = stride[1];

  // parse dilation
  std::vector<int64_t> dilation(2, 0);
  status = CaffeConvBaseParser::ParseDilations(convParam, &dilation);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "ParseDilations for " << proto.name().c_str() << " failed";
    return nullptr;
  }
  attr->dilateH = dilation[0];
  attr->dilateW = dilation[1];

  // parse kernel
  std::vector<int64_t> kernel(2, 0);
  status = CaffeConvBaseParser::ParseKernels(convParam, &kernel);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "ParseKernels for " << proto.name().c_str() << " failed";
    return nullptr;
  }
  attr->kernelH = kernel[0];
  attr->kernelW = kernel[1];

  attr->group = CaffeConvBaseParser::ParseGroup(convParam, proto.type());
  auto ret = CaffeConvBaseParser::ParseChannelOut(convParam, &(attr->channelOut));
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "conv channel out failed";
    return nullptr;
  }
  if (weight.blobs_size() < 1) {
    MS_LOG(ERROR) << "conv weight blob is empty";
    return nullptr;
  }
  auto &weightBlob = weight.blobs(0);
  if (weightBlob.has_shape()) {
    attr->channelIn = weightBlob.shape().dim(1) * attr->group;
  } else {
    attr->channelIn = weightBlob.channels() * attr->group;
  }
  attr->padMode = schema::PadMode_CAFFE;

  auto primitive = std::make_unique<schema::PrimitiveT>();
  primitive->value.type = schema::PrimitiveType_Conv2D;
  primitive->value.value = attr.release();

  status = ParseDepthwiseConvolution(primitive.get(), static_cast<schema::Conv2DT *>(primitive->value.value));
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Parse depthwise convolution failed";
    return nullptr;
  }

  return PrimitiveC::Create(primitive.release());
}

CaffeNodeRegistrar g_caffeConvolutionParser("Convolution", new CaffeConvolutionParser());
}  // namespace lite
}  // namespace mindspore
