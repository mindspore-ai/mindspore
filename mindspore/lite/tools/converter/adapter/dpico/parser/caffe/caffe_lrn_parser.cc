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

#include "parser/caffe/caffe_lrn_parser.h"
#include <memory>
#include "common/op_attr.h"
#include "ops/lrn.h"
#include "ops/op_name.h"

namespace mindspore {
namespace lite {
BaseOperatorPtr CaffeLRNParser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) {
  auto prim = std::make_shared<ops::LRN>();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "prim is nullptr.";
    return nullptr;
  }
  int64_t size = 5;
  float alpha = 1;
  float beta = 0.75;
  float k = 1;

  if (proto.has_lrn_param()) {
    const caffe::LRNParameter &lrnParam = proto.lrn_param();
    if (lrnParam.has_local_size()) {
      size = lrnParam.local_size();
    }
    if (lrnParam.has_alpha()) {
      alpha = lrnParam.alpha();
    }
    if (lrnParam.has_beta()) {
      beta = lrnParam.beta();
    }
    if (lrnParam.has_k()) {
      k = lrnParam.k();
    }
    if (lrnParam.has_norm_region()) {
      if (lrnParam.norm_region() == caffe::LRNParameter_NormRegion::LRNParameter_NormRegion_WITHIN_CHANNEL) {
        (void)prim->AddAttr(ops::kNormRegion, api::MakeValue("WITHIN_CHANNEL"));
      } else if (lrnParam.norm_region() == caffe::LRNParameter_NormRegion::LRNParameter_NormRegion_ACROSS_CHANNELS) {
        (void)prim->AddAttr(ops::kNormRegion, api::MakeValue("ACROSS_CHANNELS"));
      } else {
        MS_LOG(ERROR) << "invalid norm region param. " << lrnParam.norm_region();
        return nullptr;
      }
    }
  }

  if (size == 0) {
    MS_LOG(ERROR) << "Divide-by-zero error.";
    return nullptr;
  }
  alpha /= size;

  prim->set_beta(beta);
  prim->set_alpha(alpha);
  int two_sides = 2;
  prim->set_depth_radius(size / two_sides);
  (void)prim->AddAttr(dpico::kLrnK, api::MakeValue<float>(k));
  return prim;
}

CaffeNodeRegistrar g_caffeLRNParser("LRN", new CaffeLRNParser());
}  // namespace lite
}  // namespace mindspore
