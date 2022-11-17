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

#include "parser/caffe/caffe_decbbox_parser.h"
#include <memory>
#include <vector>
#include <map>
#include <string>
#include "common/op_attr.h"
#include "ops/custom.h"
#include "third_party/securec/include/securec.h"
#include "parser/detection_output_param_helper.h"

namespace mindspore {
namespace lite {
BaseOperatorPtr CaffeDecBBoxParser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) {
  auto prim = std::make_shared<ops::Custom>();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "prim is nullptr.";
    return nullptr;
  }
  prim->set_type("DecBBox");

  if (proto.has_num_anchors()) {
    (void)prim->AddAttr(dpico::kNumAnchors, api::MakeValue<int64_t>(proto.num_anchors()));
  }
  if (proto.has_num_bboxes_per_grid()) {
    (void)prim->AddAttr(dpico::kNumBboxesPerGrid, api::MakeValue<int64_t>(proto.num_bboxes_per_grid()));
  }
  if (proto.has_num_coords()) {
    (void)prim->AddAttr(dpico::kNumCoords, api::MakeValue<int64_t>(proto.num_coords()));
  }
  if (proto.has_num_classes()) {
    uint32_t num_classes = proto.num_classes();
    (void)prim->AddAttr(dpico::kNumClasses, api::MakeValue<int64_t>(num_classes));
    std::map<std::string, std::vector<uint8_t>> custom_attrs;
    std::vector<uint8_t> num_classes_attr(sizeof(uint32_t));
    if (memcpy_s(num_classes_attr.data(), num_classes_attr.size() * sizeof(uint8_t), &num_classes, sizeof(uint32_t)) !=
        EOK) {
      MS_LOG(ERROR) << "memcpy_s failed.";
      return nullptr;
    }
    custom_attrs[dpico::kNumClasses] = num_classes_attr;
    prim->set_attr(custom_attrs);
  }
  if (proto.has_num_grids_height()) {
    (void)prim->AddAttr(dpico::kNumGridsHeight, api::MakeValue<int64_t>(proto.num_grids_height()));
  }
  if (proto.has_num_grids_width()) {
    (void)prim->AddAttr(dpico::kNumGridsWidth, api::MakeValue<int64_t>(proto.num_grids_width()));
  }

  if (dpico::SetAttrsByDecBboxParam(prim, proto) != RET_OK) {
    MS_LOG(ERROR) << "set attrs by dec bbox param";
    return nullptr;
  }
  return prim;
}

CaffeNodeRegistrar g_caffeDecBBoxParser("DecBBox", new CaffeDecBBoxParser());
}  // namespace lite
}  // namespace mindspore
