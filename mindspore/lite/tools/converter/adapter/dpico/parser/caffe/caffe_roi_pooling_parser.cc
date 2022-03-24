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

#include "parser/caffe/caffe_roi_pooling_parser.h"
#include <memory>
#include "ops/roi_pooling.h"

namespace mindspore {
namespace lite {
BaseOperatorPtr CaffeROIPoolingParser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) {
  auto prim = std::make_shared<ops::ROIPooling>();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "prim is nullptr.";
    return nullptr;
  }
  if (proto.has_roi_pooling_param()) {
    const auto &roi_pooling_param = proto.roi_pooling_param();
    if (roi_pooling_param.has_pooled_h()) {
      prim->set_pooled_h(roi_pooling_param.pooled_h());
    }
    if (roi_pooling_param.has_pooled_w()) {
      prim->set_pooled_w(roi_pooling_param.pooled_w());
    }
    if (roi_pooling_param.has_spatial_scale()) {
      prim->set_scale(roi_pooling_param.spatial_scale());
    }
  }

  return prim;
}

CaffeNodeRegistrar g_caffeROIPoolingParser("ROIPooling", new CaffeROIPoolingParser());
}  // namespace lite
}  // namespace mindspore
