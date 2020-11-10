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

#include "tools/converter/parser/caffe/caffe_flatten_parser.h"
#include <memory>

namespace mindspore {
namespace lite {
PrimitiveC *CaffeFlattenParser::ParseLitePrimitive(const caffe::LayerParameter &proto,
                                                   const caffe::LayerParameter &weight) {
  std::unique_ptr<schema::FlattenT> attr = std::make_unique<schema::FlattenT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return nullptr;
  }
  auto primitive = std::make_unique<schema::PrimitiveT>();
  primitive->value.type = schema::PrimitiveType_Flatten;
  primitive->value.value = attr.release();
  return PrimitiveC::Create(primitive.release());
}

CaffeNodeRegistrar g_CaffeFlattenParser("Flatten", new CaffeFlattenParser());
}  // namespace lite
}  // namespace mindspore
