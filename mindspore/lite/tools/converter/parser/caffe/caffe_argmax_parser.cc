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

#include "tools/converter/parser/caffe/caffe_argmax_parser.h"
#include <memory>
#include "ops/fusion/arg_max_fusion.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *CaffeArgMaxParser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) {
  auto prim = std::make_unique<ops::ArgMaxFusion>();

  prim->set_keep_dims(true);
  prim->set_out_max_value(false);
  prim->set_top_k(1);

  const caffe::ArgMaxParameter &argmaxParam = proto.argmax_param();
  if (argmaxParam.has_out_max_val()) {
    prim->set_out_max_value(argmaxParam.out_max_val());
  }
  if (argmaxParam.has_top_k()) {
    prim->set_top_k(argmaxParam.top_k());
  }
  if (argmaxParam.has_axis()) {
    prim->set_axis(argmaxParam.axis());
  }

  return prim.release();
}

CaffeNodeRegistrar g_caffeArgMaxParser("ArgMax", new CaffeArgMaxParser());
}  // namespace lite
}  // namespace mindspore
