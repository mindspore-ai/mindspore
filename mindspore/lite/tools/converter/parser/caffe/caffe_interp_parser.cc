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

#include "tools/converter/parser/caffe/caffe_interp_parser.h"
#include <memory>
#include "ops/resize.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *CaffeInterpParser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) {
  auto prim = std::make_unique<ops::Resize>();

  prim->set_method(mindspore::ResizeMethod::LINEAR);
  prim->set_coordinate_transform_mode(mindspore::CoordinateTransformMode::ALIGN_CORNERS);

  const caffe::InterpParameter &interp_param = proto.interp_param();
  if (interp_param.has_height()) {
    int64_t height = interp_param.height();
    if (height < 0) {
      MS_LOG(ERROR) << "Interp height must be > 0";
      return nullptr;
    }
    prim->set_new_height(height);
  }

  if (interp_param.has_width()) {
    int64_t width = interp_param.width();
    if (width < 0) {
      MS_LOG(ERROR) << "Interp width must be > 0";
      return nullptr;
    }
    prim->set_new_width(width);
  }

  if (interp_param.has_zoom_factor()) {
    prim->AddAttr("zoom_factor", MakeValue(interp_param.zoom_factor()));
  }
  return prim.release();
}

CaffeNodeRegistrar g_caffeInterpParser("Interp", new CaffeInterpParser());
}  // namespace lite
}  // namespace mindspore
