/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "parser/caffe/caffe_convolution_parser.h"
#include <memory>
#include "common/check_base.h"
#include "common/op_attr.h"
#include "ops/fusion/conv2d_fusion.h"

namespace mindspore {
namespace lite {
BaseOperatorPtr CaffeConvolutionParser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) {
  auto prim = std::make_shared<ops::Conv2DFusion>();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "prim is nullptr.";
    return nullptr;
  }
  prim->set_pad({0, 0, 0, 0});
  prim->set_pad_mode(mindspore::PadMode::PAD);
  prim->set_format(mindspore::Format::NCHW);
  prim->set_activation_type(mindspore::NO_ACTIVATION);

  const caffe::ConvolutionParameter &convParam = proto.convolution_param();
  // parse kernel
  std::vector<int64_t> kernel(kNums2, 0);
  if (CaffeConvBaseParser::ParseKernels(convParam, &kernel) != RET_OK) {
    return nullptr;
  }
  prim->set_kernel_size(kernel);

  // parse stride
  std::vector<int64_t> stride(kNums2, 0);
  if (CaffeConvBaseParser::ParseStrides(convParam, &stride) != RET_OK) {
    return nullptr;
  }
  prim->set_stride(stride);

  // parse dilation
  std::vector<int64_t> dilation(kNums2, 0);
  if (CaffeConvBaseParser::ParseDilations(convParam, &dilation) != RET_OK) {
    return nullptr;
  }
  prim->set_dilation(dilation);

  // parse pad
  std::vector<int64_t> pad(kNums4, 0);
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
  int group = 1;
  if (convParam.has_group()) {
    group = convParam.group();
  } else {
    if (proto.type() == "DepthwiseConv") {
      group = static_cast<int>(convParam.num_output());
    }
  }
  prim->set_group(group);

  // parse channelIn
  if (weight.blobs_size() < 1) {
    MS_LOG(ERROR) << "conv weight blob is empty";
    return nullptr;
  }
  auto &weightBlob = weight.blobs(0);
  if (weightBlob.has_shape()) {
    if (weightBlob.shape().dim_size() != kNums4) {
      MS_LOG(ERROR) << "Dims of Kernel should be " << kNums4;
      return nullptr;
    }
    if (INT_MUL_OVERFLOW(weightBlob.shape().dim(1), group)) {
      MS_LOG(ERROR) << "int mul overflow.";
      return nullptr;
    }
    prim->set_in_channel(weightBlob.shape().dim(1) * group);
  } else {
    if (INT_MUL_OVERFLOW(weightBlob.channels(), group)) {
      MS_LOG(ERROR) << "int mul overflow.";
      return nullptr;
    }
    prim->set_in_channel(weightBlob.channels() * group);
  }

  if (convParam.has_bias_term()) {
    (void)prim->AddAttr(dpico::kBiasTerm, api::MakeValue<bool>(convParam.bias_term()));
  }

  return prim;
}

CaffeNodeRegistrar g_caffeConvolutionParser("Convolution", new CaffeConvolutionParser());
}  // namespace lite
}  // namespace mindspore
