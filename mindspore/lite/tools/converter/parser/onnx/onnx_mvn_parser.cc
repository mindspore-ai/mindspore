/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "tools/converter/parser/onnx/onnx_mvn_parser.h"
#include <memory>
#include <vector>
#include "tools/converter/ops/ops_def.h"
#include "ir/value.h"

namespace mindspore {
namespace lite {
PrimitiveCPtr OnnxMVNParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<lite::MVN>();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "new MVN prim failed.";
    return nullptr;
  }
  (void)prim->AddAttr(mindspore::ops::kOriginalFormat, MakeValue<int64_t>(mindspore::Format::NCHW));
  std::vector<int64_t> axes = {};
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    if (onnx_node_attr.name() == "axes") {
      const int &size = onnx_node_attr.ints_size();
      for (int i = 0; i < size; ++i) {
        axes.push_back(onnx_node_attr.ints(i));
      }
      (void)prim->AddAttr("axes", MakeValue(axes));
    }
  }
  return prim;
}
OnnxNodeRegistrar g_onnxMeanVarianceNormalizationParser("MeanVarianceNormalization", new OnnxMVNParser());
}  // namespace lite
}  // namespace mindspore
