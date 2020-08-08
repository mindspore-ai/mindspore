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
#include "mindspore/lite/tools/converter/parser/caffe/caffe_crop_parser.h"

const int32_t CROP_AXIS = 2;

namespace mindspore {
namespace lite {
STATUS CaffeCropParser::Parse(const caffe::LayerParameter &proto,
                              const caffe::LayerParameter &weight,
                              schema::CNodeT *op,
                              std::vector<schema::TensorT *> *weightVec) {
  std::unique_ptr<schema::CropT> attr(new schema::CropT());
  if (!proto.has_crop_param()) {
    attr->axis = CROP_AXIS;
    std::vector<int64_t> offsets(2, 0);
    attr->offsets = offsets;
  } else {
    const caffe::CropParameter cropParam = proto.crop_param();

    if (cropParam.has_axis()) {
      if (cropParam.axis() == -1) {
        // MS_LOGW("axis with -1 may lead to calculation errors when input less than 4 dims.");
      }
      attr->axis = cropParam.axis();
    } else {
      attr->axis = CROP_AXIS;
    }

    if (cropParam.offset_size() != 0) {
      std::vector<int64_t> offsets;
      for (int i = 0; i < cropParam.offset_size(); i++) {
        offsets.push_back(cropParam.offset(i));
      }
      attr->offsets = offsets;
    }
  }
  op->primitive = std::make_unique<schema::PrimitiveT>();
  op->primitive->value.value = attr.release();
  op->primitive->value.type = schema::PrimitiveType_Crop;
  return RET_OK;
}

CaffeNodeRegistrar g_caffeCropParser("Crop", new CaffeCropParser());
}  // namespace lite
}  // namespace mindspore

