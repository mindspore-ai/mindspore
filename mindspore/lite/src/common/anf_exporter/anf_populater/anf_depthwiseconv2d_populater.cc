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
#include "src/common/anf_exporter/anf_populater/anf_depthwiseconv2d_populater.h"
#include <vector>
#include <string>
#include <memory>
#include "src/common/anf_exporter/anf_populater/anf_node_populater_registry.h"
#include "ir/func_graph.h"
#include "ir/primitive.h"

namespace mindspore::lite {
int mindspore::lite::AnfDepwiseconv2DPopulater::Parse(mindspore::CNodePtr cnodePtr, schema::CNodeT *node,
                                          std::vector<schema::TensorT *> *outputs) {
  auto attr = std::make_unique<schema::DepthwiseConv2DT>();
  node->nodeType = schema::NodeType_CNode;
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_DepthwiseConv2D;
  node->primitive->value.value = attr.release();
  return 0;
}
AnfNodePopulaterRegistrar anfdepthwise2dParser("DepthwiseConv2D", new AnfDepwiseconv2DPopulater());
AnfNodePopulaterRegistrar anfdepthwise2dnativeParser("DepthwiseConv2dNative", new AnfDepwiseconv2DPopulater());
}  // namespace mindspore::lite
