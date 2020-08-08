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
#include "mindspore/lite/tools/converter/parser/caffe/caffe_relu_parser.h"

namespace mindspore {
namespace lite {
STATUS CaffeReluParser::Parse(const caffe::LayerParameter &proto,
                              const caffe::LayerParameter &weight,
                              schema::CNodeT *op,
                              std::vector<schema::TensorT *> *weightVec) {
  std::unique_ptr<schema::ActivationT> attr(new schema::ActivationT());
  attr->type = schema::ActivationType_RELU;
  // relu: negative_slope = 0, no parameter;
  // leakyrelu: negative_slope != 0;
  if (proto.has_relu_param() && proto.relu_param().has_negative_slope()) {
    float negative_slope = proto.relu_param().negative_slope();
    if (0 != negative_slope) {
      attr->type = schema::ActivationType_LEAKY_RELU;
      attr->alpha = negative_slope;
    }
  }
  op->primitive = std::make_unique<schema::PrimitiveT>();
  op->primitive->value.value = attr.release();
  op->primitive->value.type = schema::PrimitiveType_Activation;
  return RET_OK;
}

CaffeNodeRegistrar g_caffeReluParser("ReLU", new CaffeReluParser());
}  // namespace lite
}  // namespace mindspore

