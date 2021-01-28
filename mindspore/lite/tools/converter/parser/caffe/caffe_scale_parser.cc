/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

namespace mindspore {
namespace lite {
PrimitiveC *CaffeScaleParser::ParseLitePrimitive(const caffe::LayerParameter &proto,
                                                 const caffe::LayerParameter &weight) {
  std::unique_ptr<schema::ScaleT> attr = std::make_unique<schema::ScaleT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return nullptr;
  }

  if (weight.blobs_size() + weight.bottom_size() < 2) {
    MS_LOG(ERROR) << "Scale bottom size:" << weight.bottom_size() << ", blobs size:" << weight.blobs_size()
                  << " invalid in layer " << weight.name().c_str();
    return nullptr;
  }

  const caffe::ScaleParameter &scaleParam = weight.scale_param();
  attr->axis = 1;
  if (scaleParam.has_axis()) {
    uint32_t axis_index = 1;
    if (GetAxisIndex(scaleParam.axis(), &axis_index)) {
      MS_LOG(ERROR) << "scale get axis failed for layer " << weight.name().c_str();
      return nullptr;
    }
    attr->axis = axis_index;
  }
  auto primitive = std::make_unique<schema::PrimitiveT>();
  primitive->value.type = schema::PrimitiveType_Scale;
  primitive->value.value = attr.release();
  return PrimitiveC::Create(primitive.release());
}

STATUS CaffeScaleParser::GetAxisIndex(const int32_t &axis, uint32_t *axis_index) {
  if (axis < -4 || axis >= 4) {
    MS_LOG(ERROR) << "Scale axis value(" << axis << ") is not correct";
    return RET_ERROR;
  }

  if (axis == -1) {
    MS_LOG(WARNING) << "axis with -1 may lead to calculation errors when input less than 4 dims.";
  }

  *axis_index = (axis + 4) % 4;
  return RET_OK;
}

CaffeNodeRegistrar g_caffeScaleParser("Scale", new CaffeScaleParser());
}  // namespace lite
}  // namespace mindspore
