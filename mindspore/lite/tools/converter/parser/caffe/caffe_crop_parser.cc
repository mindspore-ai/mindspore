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

#include "tools/converter/parser/caffe/caffe_crop_parser.h"
#include <memory>

namespace mindspore {
namespace lite {
PrimitiveC *CaffeCropParser::ParseLitePrimitive(const caffe::LayerParameter &proto,
                                                const caffe::LayerParameter &weight) {
  std::unique_ptr<schema::CropT> attr = std::make_unique<schema::CropT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return nullptr;
  }

  if (!proto.has_crop_param()) {
    attr->axis = 2;
    std::vector<int64_t> offsets(2, 0);
    attr->offsets = offsets;
  } else {
    const caffe::CropParameter &cropParam = proto.crop_param();
    if (cropParam.has_axis()) {
      if (cropParam.axis() == -1) {
        MS_LOG(WARNING) << "axis with -1 may lead to calculation errors when input less than 4 dims.";
      }
      attr->axis = cropParam.axis();
    } else {
      attr->axis = 2;
    }

    if (cropParam.offset_size() != 0) {
      std::vector<int64_t> offsets;
      offsets.reserve(cropParam.offset_size());
      for (int i = 0; i < cropParam.offset_size(); i++) {
        offsets.push_back(cropParam.offset(i));
      }
      attr->offsets = offsets;
    }
  }
  auto primitive = std::make_unique<schema::PrimitiveT>();
  primitive->value.type = schema::PrimitiveType_Crop;
  primitive->value.value = attr.release();
  return PrimitiveC::Create(primitive.release());
}

CaffeNodeRegistrar g_caffeCropParser("Crop", new CaffeCropParser());
}  // namespace lite
}  // namespace mindspore
