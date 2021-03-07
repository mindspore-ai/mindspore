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
#include "ops/fusion/conv2d_fusion.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *CaffeConvolutionParser::Parse(const caffe::LayerParameter &proto,
                                               const caffe::LayerParameter &weight) {
  auto prim = std::make_unique<ops::Conv2DFusion>();

  prim->set_pad({0, 0, 0, 0});
  prim->set_pad_mode(mindspore::PadMode::PAD);
  prim->set_format(mindspore::Format::NCHW);
  prim->set_activation_type(mindspore::NO_ACTIVATION);

  const caffe::ConvolutionParameter &convParam = proto.convolution_param();
  // parse kernel
  std::vector<int64_t> kernel(2, 0);
  if (CaffeConvBaseParser::ParseKernels(convParam, &kernel) != RET_OK) {
    return nullptr;
  }
  prim->set_kernel_size(kernel);

  // parse stride
  std::vector<int64_t> stride(2, 0);
  if (CaffeConvBaseParser::ParseStrides(convParam, &stride) != RET_OK) {
    return nullptr;
  }
  prim->set_stride(stride);

  // parse dilation
  std::vector<int64_t> dilation(2, 0);
  if (CaffeConvBaseParser::ParseDilations(convParam, &dilation) != RET_OK) {
    return nullptr;
  }
  prim->set_dilation(dilation);

  // parse pad
  std::vector<int64_t> pad(4, 0);
  if (CaffeConvBaseParser::ParsePads(convParam, &pad) != RET_OK) {
    return nullptr;
  }
  prim->set_pad_list(pad);

  // parse channelOut
  int channel_out = 0;
  if (CaffeConvBaseParser::ParseChannelOut(convParam, &channel_out) != RET_OK) {
    return nullptr;
  }
  prim->set_out_channel(channel_out);

  // parse group
  auto group = CaffeConvBaseParser::ParseGroup(convParam, proto.type());
  prim->set_group(group);

  // parse channelIn
  if (weight.blobs_size() < 1) {
    MS_LOG(ERROR) << "conv weight blob is empty";
    return nullptr;
  }
  auto &weightBlob = weight.blobs(0);
  auto channelIn = weightBlob.has_shape() ? weightBlob.shape().dim(1) * group : weightBlob.channels() * group;
  prim->set_in_channel(channelIn);

  if (group != 1 || group != channel_out) {
    prim->AddAttr(ops::kIsDepthWise, MakeValue<bool>(true));
  }

  return prim.release();
}

CaffeNodeRegistrar g_caffeConvolutionParser("Convolution", new CaffeConvolutionParser());
}  // namespace lite
}  // namespace mindspore
