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
#include "ops/fusion/prelu_fusion.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *CaffePReluParser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) {
  auto primitive_c = new (std::nothrow) ops::PReLUFusion();
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "new PReLUFusion failed";
    return nullptr;
  }

  const caffe::PReLUParameter &pReluParam = proto.prelu_param();
  if (pReluParam.has_channel_shared()) {
    primitive_c->set_channel_shared(pReluParam.channel_shared());
  } else {
    primitive_c->set_channel_shared(false);
  }

  return primitive_c;
}

CaffeNodeRegistrar g_caffePReluParser("PReLU", new CaffePReluParser());
}  // namespace lite
}  // namespace mindspore
