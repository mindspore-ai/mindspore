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

#include <memory>
#include "mindspore/lite/tools/converter/parser/caffe/caffe_interp_parser.h"

namespace mindspore {
namespace lite {
STATUS CaffeInterpParser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight,
                                schema::CNodeT *op, std::vector<schema::TensorT *> *weightVec) {
  std::unique_ptr<schema::ResizeT> attr(new schema::ResizeT());
  const caffe::InterpParameter interpParam = proto.interp_param();

  if (interpParam.has_height()) {
    int64_t height = interpParam.height();
    if (height < 0) {
      // MS_LOGE("Interp height must be > 0");
      return RET_ERROR;
    }
    attr->newHeight = height;
  }

  if (interpParam.has_width()) {
    int64_t width = interpParam.width();
    if (width < 0) {
      // MS_LOGE("Interp width must be > 0");
      return RET_ERROR;
    }
    attr->newWidth = width;
  }

  attr->alignCorners = true;
  attr->method = schema::ResizeMethod_BILINEAR;

  op->name = proto.name();
  op->primitive = std::make_unique<schema::PrimitiveT>();
  op->primitive->value.type = schema::PrimitiveType_Resize;
  op->primitive->value.value = attr.release();
  return RET_OK;
}

CaffeNodeRegistrar g_caffeInterpParser("Interp", new CaffeInterpParser());
}  // namespace lite
}  // namespace mindspore

