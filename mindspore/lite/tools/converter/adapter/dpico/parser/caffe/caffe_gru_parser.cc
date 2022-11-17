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

#include "parser/caffe/caffe_gru_parser.h"
#include <memory>
#include <vector>
#include <functional>
#include "ops/custom.h"
#include "common/op_attr.h"

namespace mindspore {
namespace lite {
BaseOperatorPtr CaffeGruParser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) {
  auto prim = std::make_shared<ops::Custom>();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "prim is nullptr.";
    return nullptr;
  }
  prim->set_type("Gru");

  if (proto.has_recurrent_param()) {
    const auto &gru_param = proto.recurrent_param();
    if (gru_param.has_num_output()) {
      (void)prim->AddAttr(dpico::kNumOutput, api::MakeValue<int64_t>(gru_param.num_output()));
    }
    if (gru_param.has_expose_hidden()) {
      (void)prim->AddAttr(dpico::kExposeHidden, api::MakeValue<bool>(gru_param.expose_hidden()));
      (void)prim->AddAttr(dpico::kOutputLastFrameFlag, api::MakeValue<bool>(gru_param.expose_hidden()));
      (void)prim->AddAttr(dpico::kInitialHOnlineFlag, api::MakeValue<bool>(gru_param.expose_hidden()));
      (void)prim->AddAttr(dpico::kUseDefaultInitialHFlag, api::MakeValue<bool>(!gru_param.expose_hidden()));
    } else {
      (void)prim->AddAttr(dpico::kUseDefaultInitialHFlag, api::MakeValue<bool>(true));
    }
  }

  // set default value
  (void)prim->AddAttr(dpico::kHasSplitHWeightFlag, api::MakeValue<bool>(true));
  (void)prim->AddAttr(dpico::kHasSplitBiasFlag, api::MakeValue<bool>(false));
  (void)prim->AddAttr(dpico::kGruWeightOrderZrhFlag, api::MakeValue<bool>(false));
  (void)prim->AddAttr(dpico::kOnnxModeOutFlag, api::MakeValue<bool>(false));
  (void)prim->AddAttr(dpico::kKeepDirectionDimFlag, api::MakeValue<bool>(false));

  return prim;
}

CaffeNodeRegistrar g_caffeGruParser("GRU", new CaffeGruParser());
}  // namespace lite
}  // namespace mindspore
