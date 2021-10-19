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

#include "api/ir/func_graph.h"
#include "common/common_test.h"
#include "include/registry/model_parser.h"
#include "include/registry/model_parser_registry.h"
#include "include/registry/node_parser_registry.h"
#include "ops/addn.h"
#include "proto/graph.pb.h"

using mindspore::converter::kFmkTypeTf;
namespace mindspore {
namespace converter {
class AddNodeParser : public NodeParser {
 public:
  ops::PrimitiveC *Parse(const tensorflow::NodeDef &tf_op,
                         const std::map<std::string, const tensorflow::NodeDef *> &tf_node_map,
                         std::vector<std::string> *inputs, int *output_size) override {
    auto prim = std::make_unique<ops::AddN>();
    if (prim == nullptr) {
      MS_LOG(ERROR) << "make a shared_ptr failed.";
      return nullptr;
    }
    *output_size = 1;
    for (int i = 0; i < tf_op.input_size(); ++i) {
      inputs->push_back(tf_op.input(i));
    }
    return prim.release();
  }
};
REG_NODE_PARSER(kFmkTypeTf, Add, std::make_shared<AddNodeParser>());
}  // namespace converter

class NodeParserRegistryTest : public CommonTest {
 public:
  NodeParserRegistryTest() = default;
  void SetUp() override {
    auto model_parser = registry::ModelParserRegistry::GetModelParser(kFmkTypeTf);
    if (model_parser == nullptr) {
      return;
    }
    converter::ConverterParameters converter_parameters;
    converter_parameters.fmk = kFmkTypeTf;
    converter_parameters.model_file = "./tf_add.pb";
    func_graph_ = model_parser->Parse(converter_parameters);
  }
  api::FuncGraphPtr func_graph_ = nullptr;
};

TEST_F(NodeParserRegistryTest, TestRegistry) {
  ASSERT_NE(func_graph_, nullptr);
  auto node_list = api::FuncGraph::TopoSort(func_graph_->get_return());
  std::vector<CNodePtr> cnodes;
  for (auto &node : node_list) {
    if (node->isa<CNode>()) {
      cnodes.push_back(node->cast<CNodePtr>());
    }
  }
  ASSERT_EQ(cnodes.size(), 2);
  auto cnode = cnodes.front();
  ASSERT_EQ(cnode->size(), 3);
  auto prim = GetValueNode<std::shared_ptr<ops::AddN>>(cnode->input(0));
  ASSERT_NE(prim, nullptr);
}
}  // namespace mindspore
