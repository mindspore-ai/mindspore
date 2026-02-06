/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "common/backend_common_test.h"
#include "common/py_func_graph_fetcher.h"

#include "frontend/jit/ps/resource.h"
#include "frontend/jit/ps/action.h"
#include "mindspore/ccsrc/utils/ir_dump/dump_proto.h"
#include "load_mindir/load_model.h"
#include "ir/anf.h"
#include "ir/tensor.h"
#include "include/frontend/jit/ps/action_interface.h"

namespace mindspore {
class TestLoadExport : public BackendCommon {
 public:
  TestLoadExport() : getPyFun("gtest_input.mindir.mindir_test") {}
  ~TestLoadExport() override = default;
  // Expectation: No Expectation
  UT::PyFuncGraphFetcher getPyFun;
};

/// Feature: MindIR node attribute export and load.
/// Description: Node attribute export and load.
/// Expectation: success.
TEST_F(TestLoadExport, DISABLED_test_export_attr) {
  auto func_graph = getPyFun.CallAndParseRet("export_test", "add_node_attr_test");
  tensor::TensorPtr t = std::make_shared<tensor::Tensor>(kFloat32->type_id(), std::vector<int64_t>{1, 2, 3});

  auto export_return_node = func_graph->output();
  auto export_relu = export_return_node->cast<CNodePtr>();
  export_relu->AddAttr("TestAttr", MakeValue(true));
  export_relu->AddPrimalAttr("TestPrimalAttr", MakeValue(true));
  if (func_graph->manager() == nullptr) {
    std::vector<FuncGraphPtr> graphs{func_graph};
    FuncGraphManagerPtr manager = std::make_shared<FuncGraphManager>(graphs);
    manager->AddFuncGraph(func_graph);
  }
  // Renormalize func_graph to infer and set shape and type information.
  pipeline::ResourcePtr resource_ = std::make_shared<pipeline::Resource>();
  auto graph = pipeline::Renormalize(resource_, func_graph, {t->ToAbstract()});
  auto model_string = GetBinaryProtoString(graph);
  MindIRLoader mindir_loader;

  FuncGraphPtr dstgraph_ptr = mindir_loader.LoadMindIR(model_string.c_str(), model_string.size());
  auto return_node = dstgraph_ptr->output();
  auto load_relu = return_node->cast<CNodePtr>();
  auto test_primal_attr = load_relu->GetPrimalAttr("TestPrimalAttr");
  auto test_attr = load_relu->GetAttr("TestAttr");
  ASSERT_TRUE(GetValue<bool>(test_attr));
  ASSERT_TRUE(GetValue<bool>(test_primal_attr));
}

/// Feature: MindIR node attribute export and load.
/// Description: Node attribute export and load.
/// Expectation: success.
TEST_F(TestLoadExport, test_export_attr_contain_dict) {
  auto func_graph = getPyFun.CallAndParseRet("export_test", "add_node_attr_test");
  tensor::TensorPtr t = std::make_shared<tensor::Tensor>(kFloat32->type_id(), std::vector<int64_t>{1, 2, 3});

  // Renormalize func_graph to infer and set shape and type information.
  std::vector<FuncGraphPtr> graphs{func_graph};
  FuncGraphManagerPtr manager = std::make_shared<FuncGraphManager>(graphs);
  manager->AddFuncGraph(func_graph);
  pipeline::ResourcePtr resource_ = std::make_shared<pipeline::Resource>();
  auto graph = pipeline::Renormalize(resource_, func_graph, {t->ToAbstract()});

  auto export_return_node = graph->output();
  auto export_relu = export_return_node->cast<CNodePtr>();
  std::vector<std::pair<ValuePtr, ValuePtr>> key_values;
  key_values.emplace_back(std::make_pair(MakeValue("a"), MakeValue<int64_t>(1)));
  key_values.emplace_back(std::make_pair(MakeValue("b"), MakeValue(std::vector<int64_t>{1, 2, 3})));
  auto dict = std::make_shared<ValueDictionary>(key_values);
  auto tuple = std::make_shared<ValueTuple>(std::vector<ValuePtr>{MakeValue<int64_t>(2), dict});
  export_relu->AddPrimalAttr("TestPrimalAttr", tuple);

  auto model_string = GetBinaryProtoString(graph);
  ASSERT_NE(model_string, "");
  MindIRLoader mindir_loader;
  FuncGraphPtr dstgraph_ptr = mindir_loader.LoadMindIR(model_string.c_str(), model_string.size());
  ASSERT_NE(dstgraph_ptr, nullptr);
  auto return_node = dstgraph_ptr->output();
  auto load_relu = return_node->cast<CNodePtr>();
  ASSERT_NE(load_relu, nullptr);
  ASSERT_TRUE(load_relu->HasPrimalAttr("TestPrimalAttr"));
  auto test_primal_attr = load_relu->GetPrimalAttr("TestPrimalAttr");
  auto value_tuple = test_primal_attr->cast<ValueTuplePtr>();
  ASSERT_NE(value_tuple, nullptr);
  ASSERT_EQ(value_tuple->size(), 2);
  ASSERT_EQ(GetValue<int64_t>(value_tuple->value()[0]), 2);
  auto value_dict = value_tuple->value()[1]->cast<ValueDictionaryPtr>();
  ASSERT_NE(value_dict, nullptr);
  auto dict_key_values = value_dict->value();
  ASSERT_EQ(dict_key_values.size(), 2);
  ASSERT_EQ(GetValue<std::string>(dict_key_values[0].first), "a");
  ASSERT_EQ(GetValue<int64_t>(dict_key_values[0].second), 1);
  ASSERT_EQ(GetValue<std::string>(dict_key_values[1].first), "b");
  std::vector<int64_t> expect{1, 2, 3};
  ASSERT_EQ(GetValue<std::vector<int64_t>>(dict_key_values[1].second), expect);
}

/// Feature: MindIR export abstract scalar.
/// Description: abstract scalar export and load.
/// Expectation: success.
TEST_F(TestLoadExport, DISABLED_test_export_abstract_scalar) {
  auto func_graph = getPyFun.CallAndParseRet("export_test_scalar", "node_scalar_out_test");

  // Renormalize func_graph to infer and set shape and type information.
  pipeline::ResourcePtr resource_ = std::make_shared<pipeline::Resource>();
  auto graph = pipeline::Renormalize(
    resource_, func_graph,
    {std::make_shared<abstract::AbstractScalar>(kInt64), std::make_shared<abstract::AbstractScalar>(kInt64)});

  auto model_string = GetBinaryProtoString(graph);
  MindIRLoader mindir_loader;

  FuncGraphPtr dstgraph_ptr = mindir_loader.LoadMindIR(model_string.c_str(), model_string.size());
  auto load_return_node = dstgraph_ptr->output();
  auto mindir_call_abs = load_return_node->abstract();

  auto return_node = graph->output();
  auto ori_abs = return_node->abstract();

  EXPECT_TRUE(CheckEqualGraph(dstgraph_ptr, graph));
  EXPECT_TRUE(*ori_abs == *mindir_call_abs);
}
}  // namespace mindspore