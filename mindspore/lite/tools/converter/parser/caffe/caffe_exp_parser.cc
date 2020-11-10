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

namespace mindspore {
namespace lite {
PrimitiveC *CaffeExpParser::ParseLitePrimitive(const caffe::LayerParameter &proto,
                                               const caffe::LayerParameter &weight) {
  std::unique_ptr<schema::ExpT> attr = std::make_unique<schema::ExpT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return nullptr;
  }

  const caffe::ExpParameter &exp_param = proto.exp_param();
  if (exp_param.has_base()) {
    attr->base = exp_param.base();
  } else {
    attr->base = -1;  // -1 represent base = e
  }
  if (exp_param.has_scale()) {
    attr->scale = exp_param.scale();
  } else {
    attr->scale = 1;
  }
  if (exp_param.has_shift()) {
    attr->shift = exp_param.shift();
  } else {
    attr->shift = 0;
  }
  auto primitive = std::make_unique<schema::PrimitiveT>();
  primitive->value.type = schema::PrimitiveType_Exp;
  primitive->value.value = attr.release();
  return PrimitiveC::Create(primitive.release());
}

CaffeNodeRegistrar g_caffeExpParser("Exp", new CaffeExpParser());
}  // namespace lite
}  // namespace mindspore
