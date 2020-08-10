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
#include <memory>
#include "mindspore/lite/tools/converter/parser/caffe/caffe_flatten_parser.h"

namespace mindspore {
namespace lite {
STATUS CaffeFlattenParser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight,
                                       schema::CNodeT *op, std::vector<schema::TensorT *> *weightVec) {
  if (op == nullptr) {
    // MS_LOGE("null pointer dereferencing.");
    return RET_NULL_PTR;
  }
  std::unique_ptr<schema::ReshapeT> attr(new schema::ReshapeT());
  attr->format = schema::Format_NCHW;

  op->primitive = std::make_unique<schema::PrimitiveT>();
  op->primitive->value.type = schema::PrimitiveType_Flatten;
  op->primitive->value.value = attr.release();
  return RET_OK;
}

CaffeNodeRegistrar g_CaffeFlattenParser("Flatten", new CaffeFlattenParser());
}  // namespace lite
}  // namespace mindspore
