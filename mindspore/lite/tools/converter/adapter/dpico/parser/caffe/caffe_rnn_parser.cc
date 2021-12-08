/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "parser/caffe/caffe_rnn_parser.h"
#include <memory>
#include <vector>
#include <functional>
#include "ops/custom.h"
#include "common/op_attr.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *CaffeRnnParser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) {
  auto prim = std::make_unique<ops::Custom>();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "prim is nullptr.";
    return nullptr;
  }
  prim->set_type("Rnn");

  if (proto.has_recurrent_param()) {
    const auto &rnn_param = proto.recurrent_param();
    if (rnn_param.has_num_output()) {
      prim->AddAttr(dpico::kNumOutput, MakeValue<uint32_t>(rnn_param.num_output()));
    }
    if (rnn_param.has_expose_hidden()) {
      prim->AddAttr(dpico::kExposeHidden, MakeValue<bool>(rnn_param.expose_hidden()));
      prim->AddAttr(dpico::kOutputLastFrameFlag, MakeValue<bool>(rnn_param.expose_hidden()));
      prim->AddAttr(dpico::kInitialHOnlineFlag, MakeValue<bool>(rnn_param.expose_hidden()));
      prim->AddAttr(dpico::kUseDefaultInitialHFlag, MakeValue<bool>(!rnn_param.expose_hidden()));
    }
  }

  // set default value
  prim->AddAttr(dpico::kKeepDirectionDimFlag, MakeValue<bool>(false));
  prim->AddAttr(dpico::kHasOutputGateFlag, MakeValue<bool>(true));

  return prim.release();
}

CaffeNodeRegistrar g_caffeRnnParser("RNN", new CaffeRnnParser());
}  // namespace lite
}  // namespace mindspore
