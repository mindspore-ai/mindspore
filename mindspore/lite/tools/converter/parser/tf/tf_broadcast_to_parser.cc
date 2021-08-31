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
#include "tools/converter/parser/tf/tf_broadcast_to_parser.h"
#include <string>
#include <memory>
#include <map>
#include <vector>
#include "tools/converter/parser/tf/tf_node_parser_registry.h"
#include "ops/broadcast_to.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *TFBroadcastToParser::Parse(const tensorflow::NodeDef &tf_op,
                                            const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                            std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::BroadcastTo>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  if (tf_op.input_size() == 1) {
    MS_LOG(ERROR) << "tf broadcast_to parser not support one input now";
    return nullptr;
  } else if (tf_op.input_size() == 2) {
    *output_size = 1;
    if (AddOpInput(tf_op, 0, inputs) != RET_OK || AddOpInput(tf_op, 1, inputs) != RET_OK) {
      MS_LOG(ERROR) << "add op input failed";
      return nullptr;
    }
    return prim.release();
  } else {
    MS_LOG(ERROR) << "broadcast_to has " << tf_op.input_size() << " inputs, invalid";
    return nullptr;
  }
}

TFNodeRegistrar g_tfBroadcastToParser("BroadcastTo", new TFBroadcastToParser());
}  // namespace lite
}  // namespace mindspore
