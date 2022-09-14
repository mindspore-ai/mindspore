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
#include "include/registry/converter_context.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
PrimitiveCPtr CaffeDeconvolutionParser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) {
  auto prim = std::make_unique<ops::Conv2dTransposeFusion>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  auto prim_c = prim->GetPrim();
  MS_CHECK_TRUE_RET(prim_c != nullptr, nullptr);

  prim->set_pad({0, 0, 0, 0});
  auto value_ptr = MakeValue<int64_t>(mindspore::Format::NCHW);
  MS_CHECK_TRUE_RET(value_ptr != nullptr, nullptr);
  (void)prim_c->AddAttr(mindspore::ops::kOriginalFormat, value_ptr);
  prim->set_pad_mode(mindspore::PadMode::PAD);
  prim->set_output_paddings({0, 0});

  const caffe::ConvolutionParameter &convParam = proto.convolution_param();
  // parse pad
  std::vector<int64_t> pad(4, 0);
  if (CaffeConvBaseParser::ParsePads(convParam, &pad) != RET_OK) {
    return nullptr;
  }
  prim->set_pad_list({pad[0], pad[1], pad[2], pad[3]});

  // parse stride
  std::vector<int64_t> stride(2, 0);
  if (CaffeConvBaseParser::ParseStrides(convParam, &stride) != RET_OK) {
    return nullptr;
  }
  prim->set_stride({stride[0], stride[1]});

  // parse dilation
  std::vector<int64_t> dilation(2, 0);
  if (CaffeConvBaseParser::ParseDilations(convParam, &dilation) != RET_OK) {
    return nullptr;
  }
  prim->set_dilation({dilation[0], dilation[1]});

  // parse kernel
  std::vector<int64_t> kernel(2, 0);
  if (CaffeConvBaseParser::ParseKernels(convParam, &kernel) != RET_OK) {
    return nullptr;
  }
  prim->set_kernel_size({kernel[0], kernel[1]});

  // parse group
  auto group = CaffeConvBaseParser::ParseGroup(convParam, proto.type());
  prim->set_group(group);

  // parse channelOut
  int32_t channelOut;
  if (CaffeConvBaseParser::ParseChannelOut(convParam, &channelOut) != RET_OK) {
    return nullptr;
  }
  prim->set_out_channel((int64_t)channelOut);

  // parse channelIN
  auto &weightBlob = weight.blobs(0);
  if (weightBlob.has_shape()) {
    if (group == 1)
      prim->set_in_channel(weightBlob.shape().dim(0) * group);
    else
      prim->set_in_channel(weightBlob.shape().dim(1) * group);
  } else {
    prim->set_in_channel(weightBlob.num() * group);
  }
  if (group != 1) {
    auto bool_ptr = MakeValue<bool>(true);
    MS_CHECK_TRUE_RET(bool_ptr != nullptr, nullptr);
    (void)prim_c->AddAttr(ops::kIsDepthWise, bool_ptr);
  }

  int fmk_type = converter::FmkType::kFmkTypeCaffe;
  auto fmk_type_ptr = MakeValue(fmk_type);
  MS_CHECK_TRUE_RET(fmk_type_ptr != nullptr, nullptr);
  (void)prim_c->AddAttr(ops::kFmkType, fmk_type_ptr);
  return prim->GetPrim();
}

CaffeNodeRegistrar g_caffeDeconvolutionParser("Deconvolution", new CaffeDeconvolutionParser());
}  // namespace lite
}  // namespace mindspore
