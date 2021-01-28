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

#include "tools/converter/parser/caffe/caffe_scale_parser.h"
#include <memory>
#include "ops/fusion/scale_fusion.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *CaffeScaleParser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) {
  auto prim = std::make_unique<ops::ScaleFusion>();

  if (weight.blobs_size() + weight.bottom_size() < 2) {
    MS_LOG(ERROR) << "Scale bottom size:" << weight.bottom_size() << ", blobs size:" << weight.blobs_size()
                  << " invalid in layer " << weight.name().c_str();
    return nullptr;
  }

  const caffe::ScaleParameter &scaleParam = weight.scale_param();
  if (scaleParam.has_axis()) {
    auto axis = scaleParam.axis();
    if (axis < -4 || axis >= 4) {
      MS_LOG(ERROR) << "Scale axis value(" << axis << ") is not correct";
      return nullptr;
    }
    if (axis == -1) {
      MS_LOG(WARNING) << "axis with -1 may lead to calculation errors when input less than 4 dims.";
    }
  }
  prim->set_axis(1);

  return prim.release();
}

CaffeNodeRegistrar g_caffeScaleParser("Scale", new CaffeScaleParser());
}  // namespace lite
}  // namespace mindspore
