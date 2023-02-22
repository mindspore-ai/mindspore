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
#include "tools/converter/parser/tf/tf_cast_parser.h"
#include <string>
#include <memory>
#include <map>
#include <vector>
#include "tools/converter/parser/tf/tf_node_parser_registry.h"
#include "ops/cast.h"

namespace mindspore {
namespace lite {
PrimitiveCPtr TFCastParser::Parse(const tensorflow::NodeDef &tf_op,
                                  const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                  std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::Cast>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  auto prim_c = prim->GetPrim();
  MS_CHECK_TRUE_RET(prim_c != nullptr, nullptr);
  auto dst_type = TensorFlowUtils::ParseAttrDataType(tf_op, "DstT");
  if (dst_type == kTypeUnknown) {
    MS_LOG(ERROR) << "Get attr DstT failed";
    return nullptr;
  }
  (void)prim_c->AddAttr("to", MakeValue(static_cast<int32_t>(dst_type)));

  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK) {
    MS_LOG(ERROR) << "add op input failed";
    return nullptr;
  }

  return prim->GetPrim();
}

TFNodeRegistrar g_tfCastParser("Cast", new TFCastParser());
}  // namespace lite
}  // namespace mindspore
