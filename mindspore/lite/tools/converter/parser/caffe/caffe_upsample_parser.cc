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

#include "tools/converter/parser/caffe/caffe_upsample_parser.h"
#include <memory>
#include <vector>
#include "ops/resize.h"
#include "ops/op_utils.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
PrimitiveCPtr CaffeUpsampleParser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) {
  auto prim = std::make_unique<ops::Resize>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  auto prim_c = prim->GetPrim();
  MS_CHECK_TRUE_RET(prim_c != nullptr, nullptr);
  prim->set_method(mindspore::ResizeMethod::NEAREST);
  prim->set_coordinate_transform_mode(mindspore::CoordinateTransformMode::ASYMMETRIC);
  const caffe::UpsampleParameter &upsample_param = proto.upsample_param();
  if (upsample_param.has_scale()) {
    float scale = static_cast<float>(upsample_param.scale());
    if (scale < 0) {
      MS_LOG(ERROR) << "The scale of upsample must be > 0";
      return nullptr;
    }
    std::vector<float> scales = {1, scale, scale, 1};
    (void)prim_c->AddAttr("scale", MakeValue(scales));
  }
  (void)prim_c->AddAttr(ops::kOriginalOpName, MakeValue("Upsample"));
  return prim->GetPrim();
}

CaffeNodeRegistrar g_caffeUpsampleParser("Upsample", new CaffeUpsampleParser());
}  // namespace lite
}  // namespace mindspore
