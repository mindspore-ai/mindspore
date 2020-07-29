/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0f
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <memory>
#include "mindspore/lite/tools/converter/parser/caffe/caffe_prelu_parser.h"

namespace mindspore {
namespace lite {
STATUS CaffePReluParser::Parse(const caffe::LayerParameter &proto,
                               const caffe::LayerParameter &weight,
                               schema::CNodeT *op,
                               std::vector<schema::TensorT *> *weightVec) {
  std::unique_ptr<schema::CaffePReLUT> attr(new schema::CaffePReLUT());
  const caffe::PReLUParameter pReluParam = proto.prelu_param();

  if (pReluParam.has_channel_shared()) {
    attr->channelShared = pReluParam.channel_shared();
  } else {
    attr->channelShared = false;
  }

  if (weight.blobs_size() == 0) {
    // MS_LOGE("PRelu No blobs data in layer %s", proto.name().c_str());
    return RET_ERROR;
  }

  auto slope = ConvertWeight(weight.blobs(0));
  if (slope == nullptr) {
    // MS_LOGE("CaffePRelu convert slope for layer %s failed.", weight.name().c_str());
    return RET_ERROR;
  }
  weightVec->push_back(slope);
  op->primitive = std::make_unique<schema::PrimitiveT>();
  op->primitive->value.type = schema::PrimitiveType_CaffePReLU;
  op->primitive->value.value = attr.release();
  return RET_OK;
}

CaffeNodeRegistrar g_caffePReluParser("PReLU", new CaffePReluParser());
}  // namespace lite
}  // namespace mindspore

