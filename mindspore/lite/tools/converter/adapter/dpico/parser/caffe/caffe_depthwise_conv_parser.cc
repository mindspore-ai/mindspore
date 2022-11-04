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

#include "parser/caffe/caffe_depthwise_conv_parser.h"
#include <memory>
#include "ops/fusion/conv2d_fusion.h"
#include "ops/op_name.h"

namespace mindspore {
namespace lite {
BaseOperatorPtr CaffeDepthwiseConvolutionParser::Parse(const caffe::LayerParameter &proto,
                                                       const caffe::LayerParameter &weight) {
  auto prim = CaffeConvolutionParser::Parse(proto, weight);
  if (prim != nullptr) {
    (void)prim->AddAttr(ops::kIsDepthWise, api::MakeValue<bool>(true));
  }
  return prim;
}

CaffeNodeRegistrar g_caffeDepthwiseConvParser("DepthwiseConv", new CaffeDepthwiseConvolutionParser());
}  // namespace lite
}  // namespace mindspore
