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
#include "mindspore/lite/tools/converter/parser/caffe/caffe_softmax_parser.h"
#include "utils/log_adapter.h"

static const int32_t CAFFE_SOFTMAX_DEFAULT_AXIS = 1;

namespace mindspore {
namespace lite {
STATUS CaffeSoftmaxParser::Parse(const caffe::LayerParameter &proto,
                                 const caffe::LayerParameter &weight,
                                 schema::CNodeT *op,
                                 std::vector<schema::TensorT *> *weightVec) {
  std::unique_ptr<schema::SoftMaxT> attr(new schema::SoftMaxT());
  if (proto.has_softmax_param() && proto.softmax_param().has_axis()) {
    if (proto.softmax_param().axis() == -1) {
      MS_LOG(ERROR) << "axis with -1 may lead to calculation errors when input less than 4 dims.";
    }
    attr->axis = proto.softmax_param().axis();
  } else {
    attr->axis = CAFFE_SOFTMAX_DEFAULT_AXIS;
  }
  op->primitive = std::make_unique<schema::PrimitiveT>();
  op->primitive->value.value = attr.release();
  op->primitive->value.type = schema::PrimitiveType_SoftMax;
  return RET_OK;
}

CaffeNodeRegistrar g_caffeSoftmaxParser("Softmax", new CaffeSoftmaxParser());
}  // namespace lite
}  // namespace mindspore

