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

#include "parser/caffe/caffe_interp_parser.h"
#include <memory>
#include "include/registry/converter_context.h"
#include "common/op_attr.h"
#include "ops/op_utils.h"
#include "ops/resize.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *CaffeInterpParser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) {
  auto prim = std::make_unique<ops::Resize>();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "prim is nullptr.";
    return nullptr;
  }
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
    prim->AddAttr(dpico::kZoomFactor, MakeValue<int32_t>(interp_param.zoom_factor()));
  }

  if (interp_param.has_shrink_factor()) {
    prim->AddAttr(dpico::kShrinkFactor, MakeValue<int32_t>(interp_param.shrink_factor()));
  }

  if (interp_param.has_pad_beg()) {
    prim->AddAttr(dpico::kPadBeg, MakeValue<int32_t>(interp_param.pad_beg()));
  }

  if (interp_param.has_pad_end()) {
    prim->AddAttr(dpico::kPadEnd, MakeValue<int32_t>(interp_param.pad_end()));
  }
  int fmk_type = converter::FmkType::kFmkTypeCaffe;
  prim->AddAttr(ops::kFmkType, MakeValue(fmk_type));
  return prim.release();
}

CaffeNodeRegistrar g_caffeInterpParser("Interp", new CaffeInterpParser());
}  // namespace lite
}  // namespace mindspore
