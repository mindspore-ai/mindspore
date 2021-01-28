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

#include "tools/converter/parser/caffe/caffe_exp_parser.h"
#include <memory>
#include <vector>
#include "ops/fusion/exp_fusion.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *CaffeExpParser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) {
  auto prim = std::make_unique<ops::ExpFusion>();

  const caffe::ExpParameter &exp_param = proto.exp_param();
  if (exp_param.has_base()) {
    prim->set_base(exp_param.base());
  } else {
    prim->set_base(-1);  // -1 represent base = e
  }
  if (exp_param.has_scale()) {
    prim->set_scale(exp_param.scale());
  } else {
    prim->set_scale(1);
  }
  if (exp_param.has_shift()) {
    prim->set_shift(exp_param.shift());
  } else {
    prim->set_shift(0);
  }

  return prim.release();
}

CaffeNodeRegistrar g_caffeExpParser("Exp", new CaffeExpParser());
}  // namespace lite
}  // namespace mindspore
