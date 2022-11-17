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

#include "parser/onnx/onnx_roi_align_parser.h"
#include <memory>
#include <map>
#include <string>
#include <vector>
#include "common/op_attr.h"
#include "ops/custom.h"
#include "./onnx.pb.h"
#include "include/registry/node_parser_registry.h"
#include "mindapi/base/logging.h"
#include "third_party/securec/include/securec.h"
#include "ops/op_name.h"

namespace mindspore {
namespace lite {
ops::BaseOperatorPtr OnnxRoiAlignParser::Parse(const onnx::GraphProto &onnx_proto, const onnx::NodeProto &onnx_node) {
  auto prim = api::MakeShared<ops::Custom>();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "new Custom prim failed.";
    return nullptr;
  }
  prim->set_type("RoiAlign");
  std::string pool_mode = "avg";
  uint32_t output_height = 1;
  uint32_t output_width = 1;
  uint32_t sampling_ratio = 0;
  float spatial_scale = 1.0;
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    if (onnx_node_attr.name() == "mode") {
      pool_mode = onnx_node_attr.s();
    } else if (onnx_node_attr.name() == "output_height") {
      output_height = static_cast<uint32_t>(onnx_node_attr.i());
    } else if (onnx_node_attr.name() == "output_width") {
      output_width = static_cast<uint32_t>(onnx_node_attr.i());
    } else if (onnx_node_attr.name() == "sampling_ratio") {
      sampling_ratio = static_cast<uint32_t>(onnx_node_attr.i());
    } else if (onnx_node_attr.name() == "spatial_scale") {
      spatial_scale = onnx_node_attr.f();
    }
  }
  // set attr for mapper
  (void)prim->AddAttr(ops::kMode, api::MakeValue(pool_mode));
  (void)prim->AddAttr(dpico::kOutputHeight, api::MakeValue(output_height));
  (void)prim->AddAttr(dpico::kOutputWidth, api::MakeValue(output_width));
  (void)prim->AddAttr(dpico::kSamplingRatio, api::MakeValue(sampling_ratio));
  (void)prim->AddAttr(dpico::kSpatialScale, api::MakeValue(spatial_scale));

  // set attr for infershape
  std::map<std::string, std::vector<uint8_t>> custom_attrs;
  std::vector<uint8_t> output_height_attr(sizeof(uint32_t));
  if (memcpy_s(output_height_attr.data(), output_height_attr.size() * sizeof(uint8_t), &output_height,
               sizeof(uint32_t)) != EOK) {
    MS_LOG(ERROR) << "memcpy_s failed.";
    return nullptr;
  }
  custom_attrs[dpico::kOutputHeight] = output_height_attr;

  std::vector<uint8_t> output_width_attr(sizeof(uint32_t));
  if (memcpy_s(output_width_attr.data(), output_width_attr.size() * sizeof(uint8_t), &output_width, sizeof(uint32_t)) !=
      EOK) {
    MS_LOG(ERROR) << "memcpy_s failed.";
    return nullptr;
  }
  custom_attrs[dpico::kOutputWidth] = output_width_attr;
  prim->set_attr(custom_attrs);

  return prim;
}
}  // namespace lite
}  // namespace mindspore
using mindspore::converter::kFmkTypeOnnx;
namespace mindspore::registry {
REG_NODE_PARSER(kFmkTypeOnnx, RoiAlign, std::make_shared<lite::OnnxRoiAlignParser>())
}  // namespace mindspore::registry
