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
#include "ops/fusion/conv2d_transpose_fusion.h"

namespace mindspore {
namespace lite {

ops::PrimitiveC *CaffeDeconvolutionParser::Parse(const caffe::LayerParameter &proto,
                                                 const caffe::LayerParameter &weight) {
  auto primitive_c = new (std::nothrow) ops::Conv2dTransposeFusion();
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "new Conv2dTransposeFusion failed";
    return nullptr;
  }

  primitive_c->set_pad({0, 0, 0, 0});
  primitive_c->set_format(mindspore::Format::NCHW);
  primitive_c->set_pad_mode(mindspore::PadMode::PAD);

  const caffe::ConvolutionParameter &convParam = proto.convolution_param();
  // parse pad
  std::vector<int64_t> pad(4, 0);
  if (CaffeConvBaseParser::ParsePads(convParam, &pad) != RET_OK) {
    MS_LOG(ERROR) << "ParsePads for " << proto.name().c_str() << " failed";
    return nullptr;
  }
  primitive_c->set_pad_list({pad[0], pad[1], pad[2], pad[3]});

  // parse stride
  std::vector<int64_t> stride(2, 0);
  if (CaffeConvBaseParser::ParseStrides(convParam, &stride) != RET_OK) {
    MS_LOG(ERROR) << "ParseStrides for " << proto.name().c_str() << " failed";
    return nullptr;
  }
  primitive_c->set_stride({stride[0], stride[1]});

  // parse dilation
  std::vector<int64_t> dilation(2, 0);
  if (CaffeConvBaseParser::ParseDilations(convParam, &dilation) != RET_OK) {
    MS_LOG(ERROR) << "ParseDilations for " << proto.name().c_str() << " failed";
    return nullptr;
  }
  primitive_c->set_dilation({dilation[0], dilation[1]});

  // parse kernel
  std::vector<int64_t> kernel(2, 0);
  if (CaffeConvBaseParser::ParseKernels(convParam, &kernel) != RET_OK) {
    MS_LOG(ERROR) << "ParseKernels for " << proto.name().c_str() << " failed";
    return nullptr;
  }
  primitive_c->set_kernel_size({kernel[0], kernel[1]});

  // parse group
  auto group = CaffeConvBaseParser::ParseGroup(convParam, proto.type());
  primitive_c->set_group(group);

  // parse channelOut
  int32_t channelOut;
  if (CaffeConvBaseParser::ParseChannelOut(convParam, &channelOut) != RET_OK) {
    MS_LOG(ERROR) << "deconv channel get failed";
    return nullptr;
  }
  primitive_c->set_out_channel((int64_t)channelOut);

  // parse channelIN
  auto &weightBlob = weight.blobs(0);
  if (weightBlob.has_shape()) {
    if (group == 1)
      primitive_c->set_in_channel(weightBlob.shape().dim(0) * group);
    else
      primitive_c->set_in_channel(weightBlob.shape().dim(1) * group);
  } else {
    primitive_c->set_in_channel(weightBlob.num() * group);
  }
  if (group != 1) {
    primitive_c->AddAttr(ops::kIsDepthWise, MakeValue<bool>(true));
  }
  return primitive_c;
}

CaffeNodeRegistrar g_caffeDeconvolutionParser("Deconvolution", new CaffeDeconvolutionParser());
}  // namespace lite
}  // namespace mindspore
