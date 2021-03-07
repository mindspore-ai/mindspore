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
#include "ops/crop.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *CaffeCropParser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) {
  auto prim = std::make_unique<ops::Crop>();

  if (!proto.has_crop_param()) {
    prim->set_axis(2);
    std::vector<int64_t> offsets(2, 0);
    prim->set_offsets(offsets);
  } else {
    const caffe::CropParameter &cropParam = proto.crop_param();
    if (cropParam.has_axis()) {
      if (cropParam.axis() == -1) {
        MS_LOG(WARNING) << "axis with -1 may lead to calculation errors when input less than 4 dims.";
      }
      prim->set_axis(cropParam.axis());
    } else {
      prim->set_axis(2);
    }

    if (cropParam.offset_size() != 0) {
      std::vector<int64_t> offsets;
      offsets.reserve(cropParam.offset_size());
      for (int i = 0; i < cropParam.offset_size(); i++) {
        offsets.push_back(cropParam.offset(i));
      }
      prim->set_offsets(offsets);
    }
  }

  return prim.release();
}

CaffeNodeRegistrar g_caffeCropParser("Crop", new CaffeCropParser());
}  // namespace lite
}  // namespace mindspore
