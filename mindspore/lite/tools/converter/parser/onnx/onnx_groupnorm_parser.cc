/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "tools/converter/parser/onnx/onnx_groupnorm_parser.h"
#include <memory>
#include "ops/group_norm_silu.h"

namespace mindspore {
namespace lite {
namespace {
constexpr bool activate_silu = false;
}

PrimitiveCPtr OnnxGroupNormParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto groupnorm_silu_prim = std::make_shared<ops::GroupNormSilu>();
  if (groupnorm_silu_prim == nullptr) {
    MS_LOG(ERROR) << "new GroupNormSilu prim failed!";
    return nullptr;
  }

  groupnorm_silu_prim->AddAttr(kAttrActivateSilu, api::MakeValue(activate_silu));
  int num_groups = 32;
  float eps = 0.00001;
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    if (onnx_node_attr.name() == kAttrNumGroups) {
      num_groups = onnx_node_attr.i();
      MS_LOG(INFO) << "Attr num_groups from ONNX is: " << num_groups;
    } else if (onnx_node_attr.name() == kAttrEps) {
      eps = onnx_node_attr.f();
      MS_LOG(INFO) << "Attr eps from ONNX is: " << eps;
    }
  }
  groupnorm_silu_prim->AddAttr(kAttrNumGroups, api::MakeValue(num_groups));
  groupnorm_silu_prim->AddAttr(kAttrEps, api::MakeValue(eps));
  return groupnorm_silu_prim->GetPrim();
}
OnnxNodeRegistrar g_onnxGroupNormParser("GroupNorm", new OnnxGroupNormParser());
}  // namespace lite
}  // namespace mindspore
