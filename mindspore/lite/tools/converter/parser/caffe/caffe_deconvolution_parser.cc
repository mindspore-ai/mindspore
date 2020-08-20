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

#include <memory>
#include "mindspore/lite/tools/converter/parser/caffe/caffe_deconvolution_parser.h"

namespace mindspore {
namespace lite {
void CaffeDeconvolutionParser::ParseGroupDeconvolution(schema::CNodeT *op, schema::DeConv2DT *attr) {
  if (attr == nullptr || attr->group == 1) {
    return;
  }

  std::unique_ptr<schema::DeDepthwiseConv2DT> deDepthwiseConv2DParam
    = std::make_unique<schema::DeDepthwiseConv2DT>();
  if (deDepthwiseConv2DParam == nullptr) {
    MS_LOG(ERROR) << "new DeDepthwiseConv2DT failed";
    return;
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
  deDepthwiseConv2DParam->hasBias = attr->hasBias;
  deDepthwiseConv2DParam->activationType = attr->activationType;
  delete attr;
  op->primitive->value.type = schema::PrimitiveType_DeDepthwiseConv2D;
  op->primitive->value.value = deDepthwiseConv2DParam.release();
}
STATUS CaffeDeconvolutionParser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight,
                                       schema::CNodeT *op, std::vector<schema::TensorT *> *weightVec) {
  op->name = proto.name();
  auto *attr = new schema::DeConv2DT();
  attr->format = schema::Format_NCHW;
  const caffe::ConvolutionParameter convParam = proto.convolution_param();

  CaffeConvBaseParser convParser;
  // parse pad
  std::vector<int64_t> pad(4, 0);
  auto status = convParser.ParsePads(convParam, &pad);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "ParsePads for " << proto.name().c_str() << " failed";
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
    MS_LOG(ERROR) << "deconv channel get failed";
    return RET_ERROR;
  }
  auto &weightBlob = weight.blobs(0);
  if (weightBlob.has_shape()) {
    if (attr->group == 1)
      attr->channelIn = weightBlob.shape().dim(0) * attr->group;
    else
      attr->channelIn = weightBlob.shape().dim(1) * attr->group;
  } else {
    // get shape information from Blob parameters(caffe proto v1)
    attr->channelIn = weightBlob.num() * attr->group;
  }
  attr->padMode = schema::PadMode_CAFFE;
  op->primitive = std::make_unique<schema::PrimitiveT>();
  op->primitive->value.type = schema::PrimitiveType_DeConv2D;
  op->primitive->value.value = attr;
  ParseGroupDeconvolution(op, attr);
  status = convParser.ParseWeight(weight, weightVec);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "ParseWeight for " << proto.name().c_str() << " failed";
  }

  return status;
}

CaffeNodeRegistrar g_caffeDeconvolutionParser("Deconvolution", new CaffeDeconvolutionParser());
}  // namespace lite
}  // namespace mindspore

