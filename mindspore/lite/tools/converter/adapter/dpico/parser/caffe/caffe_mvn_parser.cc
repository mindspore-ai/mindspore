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

#include "parser/caffe/caffe_mvn_parser.h"
#include <memory>
#include <vector>
#include "common/op_attr.h"
#include "ops/custom.h"
#include "ops/op_name.h"

namespace mindspore {
namespace lite {
BaseOperatorPtr CaffeMvnParser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) {
  auto prim = std::make_shared<ops::Custom>();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "prim is nullptr.";
    return nullptr;
  }
  prim->set_type("Mvn");

  if (proto.has_mvn_param()) {
    const caffe::MVNParameter &mvn_parameter = proto.mvn_param();
    if (mvn_parameter.has_eps()) {
      (void)prim->AddAttr(ops::kEps, api::MakeValue<float>(mvn_parameter.eps()));
    }
    if (mvn_parameter.has_across_channels()) {
      (void)prim->AddAttr(dpico::kAcrossChannels, api::MakeValue<bool>(mvn_parameter.across_channels()));
    }
    if (mvn_parameter.has_normalize_variance()) {
      (void)prim->AddAttr(dpico::kNormalizeVariance, api::MakeValue<bool>(mvn_parameter.normalize_variance()));
    }
  }

  return prim;
}

CaffeNodeRegistrar g_caffeMvnParser("MVN", new CaffeMvnParser());
}  // namespace lite
}  // namespace mindspore
