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

#include "parser/caffe/caffe_upsample_parser.h"
#include <memory>
#include <map>
#include <string>
#include "common/op_attr.h"
#include "ops/custom.h"
#include "ops/op_name.h"
#include "third_party/securec/include/securec.h"

namespace mindspore {
namespace lite {
namespace {
constexpr float kDefaultScaleVal = 2.0;
}  // namespace
BaseOperatorPtr CaffeUpsampleParser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) {
  auto prim = std::make_shared<ops::Custom>();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "prim is nullptr.";
    return nullptr;
  }
  prim->set_type("Upsample");

  float scale = kDefaultScaleVal;
  std::map<std::string, std::vector<uint8_t>> custom_attrs;
  if (proto.has_upsample_param()) {
    const auto &upsample_param = proto.upsample_param();
    if (upsample_param.has_scale()) {
      scale = upsample_param.scale();
    }
    if (upsample_param.upsample_h()) {
      uint32_t upsample_h = upsample_param.upsample_h();
      (void)prim->AddAttr(dpico::kUpsampleH, api::MakeValue<int64_t>(upsample_h));

      std::vector<uint8_t> upsample_h_attr(sizeof(uint32_t));
      if (memcpy_s(upsample_h_attr.data(), upsample_h_attr.size() * sizeof(uint8_t), &upsample_h, sizeof(uint32_t)) !=
          EOK) {
        MS_LOG(ERROR) << "memcpy_s failed.";
        return nullptr;
      }
      custom_attrs[dpico::kUpsampleH] = upsample_h_attr;
    }
    if (upsample_param.upsample_w()) {
      uint32_t upsample_w = upsample_param.upsample_w();
      (void)prim->AddAttr(dpico::kUpsampleH, api::MakeValue<int64_t>(upsample_w));

      std::vector<uint8_t> upsample_w_attr(sizeof(uint32_t));
      if (memcpy_s(upsample_w_attr.data(), upsample_w_attr.size() * sizeof(uint8_t), &upsample_w, sizeof(uint32_t)) !=
          EOK) {
        MS_LOG(ERROR) << "memcpy_s failed.";
        return nullptr;
      }
      custom_attrs[dpico::kUpsampleW] = upsample_w_attr;
    }
    if (upsample_param.has_interpolation_mode()) {
      auto mode = upsample_param.interpolation_mode();
      switch (mode) {
        case caffe::UpsampleParameter_InterpolationMode_NEAREST:
          (void)prim->AddAttr(dpico::kInterpolationMode, api::MakeValue<std::string>(dpico::kNearest));
          break;
        case caffe::UpsampleParameter_InterpolationMode_BILINEAR:
          (void)prim->AddAttr(dpico::kInterpolationMode, api::MakeValue<std::string>(dpico::kBilinear));
          break;
        default:
          MS_LOG(ERROR) << "current interpolation mode is not supported. " << mode;
          return nullptr;
      }
    }
  }

  (void)prim->AddAttr(ops::kScale, api::MakeValue<float>(scale));
  std::vector<uint8_t> scale_attr(sizeof(float));
  if (memcpy_s(scale_attr.data(), scale_attr.size() * sizeof(uint8_t), &scale, sizeof(float)) != EOK) {
    MS_LOG(ERROR) << "memcpy_s failed.";
    return nullptr;
  }
  custom_attrs[ops::kScale] = scale_attr;
  prim->set_attr(custom_attrs);
  return prim;
}

CaffeNodeRegistrar g_caffeUpsampleParser("Upsample", new CaffeUpsampleParser());
}  // namespace lite
}  // namespace mindspore
