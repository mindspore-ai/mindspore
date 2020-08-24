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

#include "tools/converter/parser/caffe/caffe_tanh_parser.h"
#include <memory>
#include <vector>

namespace mindspore {
namespace lite {
STATUS CaffeTanhParser::Parse(const caffe::LayerParameter &proto,
                               const caffe::LayerParameter &weight,
                               schema::CNodeT *op,
                               std::vector<schema::TensorT *> *weightVec) {
  std::unique_ptr<schema::ActivationT> attr(new schema::ActivationT());
  attr->type = schema::ActivationType_TANH;
  op->primitive = std::make_unique<schema::PrimitiveT>();
  op->primitive->value.value = attr.release();
  op->primitive->value.type = schema::PrimitiveType_Activation;
  return RET_OK;
}

CaffeNodeRegistrar g_caffeTanhParser("TanH", new CaffeTanhParser());
}  // namespace lite
}  // namespace mindspore

