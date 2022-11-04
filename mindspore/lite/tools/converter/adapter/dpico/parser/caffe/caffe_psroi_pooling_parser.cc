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

#include "parser/caffe/caffe_psroi_pooling_parser.h"
#include <memory>
#include <map>
#include <string>
#include "common/op_attr.h"
#include "ops/custom.h"
#include "third_party/securec/include/securec.h"

namespace mindspore {
namespace lite {
BaseOperatorPtr CaffePSROIPoolingParser::Parse(const caffe::LayerParameter &proto,
                                               const caffe::LayerParameter &weight) {
  auto prim = std::make_shared<ops::Custom>();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "prim is nullptr.";
    return nullptr;
  }
  prim->set_type("PsRoiPool");

  std::map<std::string, std::vector<uint8_t>> custom_attrs;
  if (proto.has_psroi_pooling_param()) {
    const auto &psroi_pooling_param = proto.psroi_pooling_param();
    if (psroi_pooling_param.has_spatial_scale()) {
      float spatial_scale = psroi_pooling_param.spatial_scale();
      (void)prim->AddAttr(dpico::kSpatialScale, api::MakeValue<float>(spatial_scale));

      std::vector<uint8_t> spatial_scale_attr(sizeof(float));
      if (memcpy_s(spatial_scale_attr.data(), spatial_scale_attr.size() * sizeof(uint8_t), &spatial_scale,
                   sizeof(float)) != EOK) {
        MS_LOG(ERROR) << "memcpy_s failed.";
        return nullptr;
      }
      custom_attrs[dpico::kSpatialScale] = spatial_scale_attr;
    }
    if (psroi_pooling_param.has_group_size()) {
      int32_t group_size = psroi_pooling_param.group_size();
      (void)prim->AddAttr(dpico::kGroupSize, api::MakeValue<int64_t>(group_size));

      std::vector<uint8_t> group_size_attr(sizeof(int32_t));
      if (memcpy_s(group_size_attr.data(), group_size_attr.size() * sizeof(uint8_t), &group_size, sizeof(int32_t)) !=
          EOK) {
        MS_LOG(ERROR) << "memcpy_s failed.";
        return nullptr;
      }
      custom_attrs[dpico::kGroupSize] = group_size_attr;
    }
    if (psroi_pooling_param.has_output_dim()) {
      int32_t output_dim = psroi_pooling_param.output_dim();
      (void)prim->AddAttr(dpico::kOutputDim, api::MakeValue<int64_t>(output_dim));

      std::vector<uint8_t> output_dim_attr(sizeof(int32_t));
      if (memcpy_s(output_dim_attr.data(), output_dim_attr.size() * sizeof(uint8_t), &output_dim, sizeof(int32_t)) !=
          EOK) {
        MS_LOG(ERROR) << "memcpy_s failed.";
        return nullptr;
      }
      custom_attrs[dpico::kOutputDim] = output_dim_attr;
    }
  }

  prim->set_attr(custom_attrs);
  return prim;
}

CaffeNodeRegistrar g_caffePSROIPoolingParser("PSROIPooling", new CaffePSROIPoolingParser());
}  // namespace lite
}  // namespace mindspore
