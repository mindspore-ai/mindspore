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
#include <memory>
#include <vector>

namespace mindspore {
namespace lite {
PrimitiveC *CaffePowerParser::ParseLitePrimitive(const caffe::LayerParameter &proto,
                                                 const caffe::LayerParameter &weight) {
  std::unique_ptr<schema::PowerT> attr = std::make_unique<schema::PowerT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return nullptr;
  }

  const caffe::PowerParameter &powerParam = proto.power_param();
  if (proto.has_power_param()) {
    attr->power = powerParam.has_power() ? powerParam.power() : 1.0;
    attr->scale = powerParam.has_scale() ? powerParam.scale() : 1.0;
    attr->shift = powerParam.has_shift() ? powerParam.shift() : 0.0;
  } else {
    attr->power = 1.0;
    attr->scale = 1.0;
    attr->shift = 0.0;
  }

  auto primitive = std::make_unique<schema::PrimitiveT>();
  primitive->value.type = schema::PrimitiveType_Power;
  primitive->value.value = attr.release();
  return PrimitiveC::Create(primitive.release());
}

CaffeNodeRegistrar g_caffePowerParser("Power", new CaffePowerParser());
}  // namespace lite
}  // namespace mindspore
