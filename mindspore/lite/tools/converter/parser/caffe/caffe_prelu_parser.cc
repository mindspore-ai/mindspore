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

#include "tools/converter/parser/caffe/caffe_prelu_parser.h"
#include <memory>

namespace mindspore {
namespace lite {
STATUS CaffePReluParser::Parse(const caffe::LayerParameter &proto,
                               const caffe::LayerParameter &weight,
                               schema::CNodeT *op,
                               std::vector<schema::TensorT *> *weightVec) {
  MS_LOG(DEBUG) << "parse CaffePReluParser";
  if (op == nullptr) {
    MS_LOG(ERROR) << "op is null";
    return RET_NULL_PTR;
  }
  op->primitive = std::make_unique<schema::PrimitiveT>();
  if (op->primitive == nullptr) {
    MS_LOG(ERROR) << "op->primitive is null";
    return RET_NULL_PTR;
  }

  std::unique_ptr<schema::PReLUT> attr = std::make_unique<schema::PReLUT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return RET_NULL_PTR;
  }

  const caffe::PReLUParameter pReluParam = proto.prelu_param();
  if (pReluParam.has_channel_shared()) {
    attr->channelShared = pReluParam.channel_shared();
  } else {
    attr->channelShared = false;
  }

  if (weight.blobs_size() == 0) {
    MS_LOG(ERROR) << "PRelu No blobs data in layer " << proto.name().c_str();
    return RET_ERROR;
  }

  auto slope = ConvertWeight(weight.blobs(0));
  if (slope == nullptr) {
    MS_LOG(ERROR) << "CaffePRelu convert slope for layer " << weight.name().c_str() << " failed.";
    return RET_ERROR;
  }
  weightVec->push_back(slope);

  op->name = proto.name();
  op->primitive->value.type = schema::PrimitiveType_PReLU;
  op->primitive->value.value = attr.release();
  return RET_OK;
}

CaffeNodeRegistrar g_caffePReluParser("PReLU", new CaffePReluParser());
}  // namespace lite
}  // namespace mindspore

