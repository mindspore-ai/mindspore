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
#include "ops/fusion/pow_fusion.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *CaffePowerParser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) {
  auto primitive_c = new (std::nothrow) ops::PowFusion();
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "new PowFusion failed";
    return nullptr;
  }

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
  primitive_c->AddAttr("power", MakeValue(power));
  primitive_c->set_scale(scale);
  primitive_c->set_shift(shift);

  return primitive_c;
}

CaffeNodeRegistrar g_caffePowerParser("Power", new CaffePowerParser());
}  // namespace lite
}  // namespace mindspore
