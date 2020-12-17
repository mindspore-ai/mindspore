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

#include "tools/converter/parser/caffe/caffe_deconvolution_parser.h"
#include <memory>

namespace mindspore {
namespace lite {
STATUS CaffeDeconvolutionParser::ParseGroupDeconvolution(schema::PrimitiveT *primitive, schema::DeConv2DT *attr) {
  if (attr->group == 1) {
    return RET_OK;
  }

  std::unique_ptr<schema::DeDepthwiseConv2DT> deDepthwiseConv2DParam = std::make_unique<schema::DeDepthwiseConv2DT>();
  if (deDepthwiseConv2DParam == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return RET_ERROR;
  }
  deDepthwiseConv2DParam->format = attr->format;
  deDepthwiseConv2DParam->channelIn = attr->channelOut;
  deDepthwiseConv2DParam->channelMultiplier = attr->channelIn / attr->channelOut;
  deDepthwiseConv2DParam->kernelW = attr->kernelW;
  deDepthwiseConv2DParam->kernelH = attr->kernelH;
  deDepthwiseConv2DParam->strideW = attr->strideW;
  deDepthwiseConv2DParam->strideH = attr->strideH;
  deDepthwiseConv2DParam->padMode = attr->padMode;
  deDepthwiseConv2DParam->padUp = attr->padUp;
  deDepthwiseConv2DParam->padDown = attr->padDown;
  deDepthwiseConv2DParam->padLeft = attr->padLeft;
  deDepthwiseConv2DParam->padRight = attr->padRight;
  deDepthwiseConv2DParam->dilateW = attr->dilateW;
  deDepthwiseConv2DParam->dilateH = attr->dilateH;
  deDepthwiseConv2DParam->activationType = attr->activationType;
  delete attr;
  primitive->value.type = schema::PrimitiveType_DeDepthwiseConv2D;
  primitive->value.value = deDepthwiseConv2DParam.release();
  return RET_OK;
}

PrimitiveC *CaffeDeconvolutionParser::ParseLitePrimitive(const caffe::LayerParameter &proto,
                                                         const caffe::LayerParameter &weight) {
  std::unique_ptr<schema::DeConv2DT> attr(new (std::nothrow) schema::DeConv2DT());

  attr->format = schema::Format::Format_NCHW;

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
    MS_LOG(ERROR) << "deconv channel get failed";
    return nullptr;
  }
  auto &weightBlob = weight.blobs(0);
  if (weightBlob.has_shape()) {
    if (attr->group == 1)
      attr->channelIn = weightBlob.shape().dim(0) * attr->group;
    else
      attr->channelIn = weightBlob.shape().dim(1) * attr->group;
  } else {
    attr->channelIn = weightBlob.num() * attr->group;
  }
  attr->padMode = schema::PadMode_CAFFE;
  auto primitive = std::make_unique<schema::PrimitiveT>();
  primitive->value.type = schema::PrimitiveType_DeConv2D;
  primitive->value.value = attr.release();

  status = ParseGroupDeconvolution(primitive.get(), primitive->value.AsDeConv2D());
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Parse group deconvolution failed";
    return nullptr;
  }
  return PrimitiveC::Create(primitive.release());
}

CaffeNodeRegistrar g_caffeDeconvolutionParser("Deconvolution", new CaffeDeconvolutionParser());
}  // namespace lite
}  // namespace mindspore
