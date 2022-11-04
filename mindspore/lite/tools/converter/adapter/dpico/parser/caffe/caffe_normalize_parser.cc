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

#include "parser/caffe/caffe_normalize_parser.h"
#include <memory>
#include <vector>
#include "common/op_attr.h"
#include "ops/custom.h"
#include "ops/op_name.h"

namespace mindspore {
namespace lite {
BaseOperatorPtr CaffeNormalizeParser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) {
  auto prim = std::make_shared<ops::Custom>();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "prim is nullptr.";
    return nullptr;
  }
  prim->set_type("Normalize");

  if (proto.has_norm_param()) {
    const caffe::NormalizeParameter &normalize_param = proto.norm_param();
    if (normalize_param.has_across_spatial()) {
      (void)prim->AddAttr(dpico::kAcrossSpatial, api::MakeValue<bool>(normalize_param.across_spatial()));
    }
    if (normalize_param.has_channel_shared()) {
      (void)prim->AddAttr(dpico::kChannelShared, api::MakeValue<bool>(normalize_param.channel_shared()));
    }
    if (normalize_param.has_sqrt_a()) {
      (void)prim->AddAttr(dpico::kSqrtA, api::MakeValue<float>(normalize_param.sqrt_a()));
    }
    if (normalize_param.has_eps()) {
      (void)prim->AddAttr(ops::kEps, api::MakeValue<float>(normalize_param.eps()));
    }
  }

  return prim;
}

CaffeNodeRegistrar g_caffeNormalizeParser("Normalize", new CaffeNormalizeParser());
}  // namespace lite
}  // namespace mindspore
