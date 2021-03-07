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

#include "tools/converter/parser/caffe/caffe_power_parser.h"
#include <vector>
#include <memory>
#include "ops/fusion/pow_fusion.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *CaffePowerParser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) {
  auto prim = std::make_unique<ops::PowFusion>();

  const caffe::PowerParameter &powerParam = proto.power_param();
  float power = 1.0;
  float scale = 1.0;
  float shift = 0.0;
  if (proto.has_power_param()) {
    if (powerParam.has_power()) {
      power = powerParam.power();
    }
    if (powerParam.has_scale()) {
      scale = powerParam.scale();
    }
    if (powerParam.has_shift()) {
      shift = powerParam.shift();
    }
  }
  prim->AddAttr("power", MakeValue(power));
  prim->set_scale(scale);
  prim->set_shift(shift);

  return prim.release();
}

CaffeNodeRegistrar g_caffePowerParser("Power", new CaffePowerParser());
}  // namespace lite
}  // namespace mindspore
