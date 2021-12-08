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

#include "parser/caffe/caffe_threshold_parser.h"
#include <memory>
#include <vector>
#include "common/op_attr.h"
#include "ops/custom.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *CaffeThresholdParser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) {
  auto prim = std::make_unique<ops::Custom>();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "prim is nullptr.";
    return nullptr;
  }
  prim->set_type("Threshold");

  if (proto.has_threshold_param()) {
    const caffe::ThresholdParameter &threshold_param = proto.threshold_param();
    if (threshold_param.has_threshold()) {
      prim->AddAttr(dpico::kThreshold, MakeValue<float>(threshold_param.threshold()));
    }
  }

  return prim.release();
}

CaffeNodeRegistrar g_caffeThresholdParser("Threshold", new CaffeThresholdParser());
}  // namespace lite
}  // namespace mindspore
