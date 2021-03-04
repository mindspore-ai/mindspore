/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "common/common_test.h"
#include "frontend/parallel/step_parallel.h"
#include "frontend/parallel/graph_util/generate_graph.h"
#include "common/py_func_graph_fetcher.h"
#include "debug/draw.h"
#include "frontend/operator/ops.h"
#include "pipeline/jit/static_analysis/static_analysis.h"
#include "utils/convert_utils_py.h"

namespace mindspore {
namespace parallel {
extern size_t TOTAL_OPS;
class TestStepParallel : public UT::Common {
 public:
  TestStepParallel() {}
  void SetUp();
  void TearDown() {}
};

void Init_Device_Manager() {
  RankList dev_list;

  for (int32_t i = 0; i < 20; i++) {
    dev_list.push_back(i);
  }

  RankList stage_map;
  stage_map.push_back(16);
  stage_map.push_back(4);

  int32_t local_dev = 0;

  // create a new g_device_manager
  g_device_manager = std::make_shared<DeviceManager>();
  g_device_manager->Init(dev_list, local_dev, stage_map, "hccl");
}

void TestStepParallel::SetUp() {
  UT::InitPythonPath();
  Init_Device_Manager();
}

CNodePtr Make_Node(Shape x, Shape y, Shape out, int64_t condition = 0) {
  FuncGraphPtr func_graph = std::make_shared<FuncGraph>();
  ParameterPtr param1 = func_graph->add_parameter();
  ParameterPtr param2 = func_graph->add_parameter();
  param1->set_name("x");
  param2->set_name("y");
  BaseShapePtr shape1 = std::make_shared<abstract::Shape>(x);
  BaseShapePtr shape2 = std::make_shared<abstract::Shape>(y);
  BaseShapePtr shape3 = std::make_shared<abstract::Shape>(out);
  std::shared_ptr<tensor::Tensor> inputs_x = std::make_shared<tensor::Tensor>(kNumberTypeInt32, x);
  std::shared_ptr<tensor::Tensor> inputs_y = std::make_shared<tensor::Tensor>(kNumberTypeInt32, y);
  std::shared_ptr<tensor::Tensor> inputs_out = std::make_shared<tensor::Tensor>(kNumberTypeInt32, out);
  AbstractBasePtr abstract1 = abstract::FromValue(inputs_x, true);
  AbstractBasePtr abstract2 = abstract::FromValue(inputs_y, true);
  AbstractBasePtr abstract3 = abstract::FromValue(inputs_out, true);
  switch (condition) {
    case 0: {
      abstract1->set_shape(shape1);
      abstract2->set_shape(shape2);
      abstract3->set_shape(shape3);
      param1->set_abstract(abstract1);
      param2->set_abstract(abstract2);
      break;
    }
    case 1: {
      abstract1->set_shape(nullptr);
      param1->set_abstract(abstract1);
      param2->set_abstract(abstract2);
      break;
    }
    case 2: {
      abstract1->set_shape(shape1);
      abstract2->set_shape(shape2);
      param1->set_abstract(abstract1);
      param2->set_abstract(abstract2);
      abstract3 = abstract::FromValue(static_cast<int64_t>(1), false);
      break;
    }
    case 3: {
      std::vector<BaseShapePtr> shape_o = {std::make_shared<abstract::Shape>(x), std::make_shared<abstract::Shape>(y)};
      BaseShapePtr shape4 = std::make_shared<abstract::TupleShape>(shape_o);
      abstract1->set_shape(shape1);
      abstract2->set_shape(shape2);
      abstract3->set_shape(shape4);
      param1->set_abstract(abstract1);
      param2->set_abstract(abstract2);
      break;
    }
    default:
      MS_LOG(INFO) << "Do Nothing!";
  }
  std::vector<AnfNodePtr> inputs;
  inputs.push_back(NewValueNode(prim::kPrimMatMul));
  inputs.push_back(param1);
  inputs.push_back(param2);
  CNodePtr node = func_graph->NewCNode(inputs);
  node->set_abstract(abstract3);
  return node;
}

FuncGraphManagerPtr Make_Manager(int64_t condition = 0) {
  std::vector<int64_t> inputs_x = {64, 32};
  std::vector<int64_t> inputs_y = {32, 64};
  std::vector<int64_t> inputs_z = {64, 128};
  std::vector<int64_t> outputs_1 = {64, 64};
  std::vector<int64_t> outputs_2 = {64, 128};
  FuncGraphPtr func_graph = std::make_shared<FuncGraph>();
  ParameterPtr param1 = func_graph->add_parameter();
  ParameterPtr param2 = func_graph->add_parameter();
  ParameterPtr param3 = func_graph->add_parameter();
  std::shared_ptr<tensor::Tensor> inputs_x_dim = std::make_shared<tensor::Tensor>(kNumberTypeInt32, inputs_x);
  std::shared_ptr<tensor::Tensor> inputs_y_dim = std::make_shared<tensor::Tensor>(kNumberTypeInt32, inputs_y);
  std::shared_ptr<tensor::Tensor> inputs_z_dim = std::make_shared<tensor::Tensor>(kNumberTypeInt32, inputs_z);
  std::shared_ptr<tensor::Tensor> inputs_out1_dim = std::make_shared<tensor::Tensor>(kNumberTypeInt32, outputs_1);
  std::shared_ptr<tensor::Tensor> inputs_out2_dim = std::make_shared<tensor::Tensor>(kNumberTypeInt32, outputs_2);
  AbstractBasePtr abstract_x = abstract::FromValue(inputs_x_dim, true);
  AbstractBasePtr abstract_y = abstract::FromValue(inputs_y_dim, true);
  AbstractBasePtr abstract_z = abstract::FromValue(inputs_z_dim, true);
  AbstractBasePtr abstract_out1 = abstract::FromValue(inputs_out1_dim, true);
  AbstractBasePtr abstract_out2 = abstract::FromValue(inputs_out2_dim, true);
  param1->set_abstract(abstract_x);
  param2->set_abstract(abstract_y);
  param3->set_abstract(abstract_z);
  Dimensions v1 = {2, 2};
  Dimensions v2 = {2, 4};
  std::vector<ValuePtr> elements = {MakeValue(v1), MakeValue(v2)};
  ValueTuplePtr var = std::make_shared<ValueTuple>(elements);
  std::vector<AnfNodePtr> inputs;
  inputs.push_back(NewValueNode(prim::kPrimMatMul));
  inputs.push_back(param1);
  inputs.push_back(param2);
  CNodePtr node1 = func_graph->NewCNode(inputs);
  node1->set_in_forward_flag(true);
  node1->set_abstract(abstract_out1);
  PrimitivePtr prim1 = node1->input(0)->cast<ValueNodePtr>()->value()->cast<PrimitivePtr>();
  ValuePtr transpose_a = MakeValue(false);
  ValuePtr transpose_b = MakeValue(false);
  prim1->AddAttr("transpose_a", transpose_a);
  prim1->AddAttr("transpose_b", transpose_b);
  prim1->AddAttr("instance_name", MakeValue("matmul1"));
  prim1->AddAttr("strategy", var);
  inputs.clear();
  Dimensions v3 = {2, 2};
  Dimensions v4 = {2, 4};
  std::vector<ValuePtr> elements2 = {MakeValue(v3), MakeValue(v4)};
  ValueTuplePtr var2 = std::make_shared<ValueTuple>(elements2);
  inputs.push_back(NewValueNode(prim::kPrimMatMul));
  inputs.push_back(node1);
  inputs.push_back(param3);
  CNodePtr node2 = func_graph->NewCNode(inputs);
  node2->set_in_forward_flag(true);
  node2->set_abstract(abstract_out2);
  inputs.clear();
  inputs.push_back(NewValueNode(prim::kPrimReturn));
  inputs.push_back(node2);
  CNodePtr cnode_return = func_graph->NewCNode(inputs);
  cnode_return->set_in_forward_flag(true);
  func_graph->set_return(cnode_return);
  PrimitivePtr prim2 = node2->input(0)->cast<ValueNodePtr>()->value()->cast<PrimitivePtr>();
  prim2->AddAttr("transpose_a", transpose_a);
  prim2->AddAttr("transpose_b", transpose_b);
  prim2->AddAttr("instance_name", MakeValue("matmul2"));
  prim2->AddAttr("strategy", var2);
  switch (condition) {
    case 1: {
      prim1->set_attr("strategy", MakeValue(static_cast<int64_t>(0)));
      break;
    }
    case 2: {
      std::vector<ValuePtr> elements_t = {MakeValue(static_cast<int64_t>(0))};
      ValueTuplePtr var_t = std::make_shared<ValueTuple>(elements_t);
      prim1->set_attr("strategy", var_t);
      break;
    }
    case 3: {
      Dimensions vt1 = {2, 4};
      Dimensions vt2 = {2, 4};
      std::vector<ValuePtr> elements_t2 = {MakeValue(vt1), MakeValue(vt2)};
      ValueTuplePtr var_t2 = std::make_shared<ValueTuple>(elements_t2);
      prim1->set_attr("strategy", var_t2);
      break;
    }
  }
  std::vector<FuncGraphPtr> func_graphs{func_graph};
  FuncGraphManagerPtr manager = std::make_shared<FuncGraphManager>(func_graphs, true);
  manager->Init();
  return manager;
}

TEST_F(TestStepParallel, GetPythonPath1) {
  OperatorName operator_name = "AllReduce";
  const std::string expect = "mindspore.ops.operations";
  auto temp = parallel::GetOpPythonPath(operator_name);
  ASSERT_EQ(temp, expect);
}

TEST_F(TestStepParallel, GetPythonPath2) {
  OperatorName operator_name = "Add";
  const std::string expect = "mindspore.ops.operations";
  auto temp = parallel::GetOpPythonPath(operator_name);
  ASSERT_EQ(temp, expect);
}

TEST_F(TestStepParallel, ExtractStrategy) {
  Dimensions v1 = {2, 2};
  Dimensions v2 = {4, 4};
  std::unordered_map<std::string, ValuePtr> attrs;
  // stage
  ValuePtr val1 = MakeValue(v1);
  ValuePtr val2 = MakeValue(v2);
  std::vector<ValuePtr> elements = {val1, val2};
  ValueTuplePtr strategy_tuple = std::make_shared<ValueTuple>(elements);
  attrs["strategy"] = strategy_tuple;
  Strategys strategy_expect = {v1, v2};
  StrategyPtr strategy = ExtractStrategy(attrs);
  Strategys strategy_test = strategy->GetInputDim();

  ASSERT_EQ(strategy_expect, strategy_test);
}

TEST_F(TestStepParallel, ExtractShape) {
  Shape inputs_x_dims = {64, 32};
  Shape inputs_y_dims = {32, 64};
  Shape outputs_dims = {64, 64};
  CNodePtr node = Make_Node(inputs_x_dims, inputs_y_dims, outputs_dims, 4);
  EXPECT_THROW({ ExtractShape(node); }, std::runtime_error);
}

TEST_F(TestStepParallel, ExtractShape1) {
  Shape inputs_x_dims = {64, 32};
  Shape inputs_y_dims = {32, 64};
  Shape outputs_dims = {64, 64};
  CNodePtr node = Make_Node(inputs_x_dims, inputs_y_dims, outputs_dims);
  std::vector<Shapes> shape_test = ExtractShape(node);
  Shapes inputs_shape = std::vector<Shape>{inputs_x_dims, inputs_y_dims};
  Shapes outputs_shape = std::vector<Shape>{outputs_dims};
  std::vector<Shapes> shape_expect = {inputs_shape, outputs_shape};
  ASSERT_EQ(shape_test, shape_expect);
}

TEST_F(TestStepParallel, ExtractShape2) {
  Shape inputs_x_dims = {64, 32};
  Shape inputs_y_dims = {32, 64};
  Shape outputs_dims = {64, 64};
  CNodePtr node = Make_Node(inputs_x_dims, inputs_y_dims, outputs_dims, 1);
  EXPECT_THROW({ ExtractShape(node); }, std::runtime_error);
}

TEST_F(TestStepParallel, ExtractShape3) {
  Shape inputs_x_dims = {64, 32};
  Shape inputs_y_dims = {32, 64};
  Shape outputs_dims = {64, 64};
  CNodePtr node = Make_Node(inputs_x_dims, inputs_y_dims, outputs_dims, 3);
  Shapes inputs_shape = std::vector<Shape>{inputs_x_dims, inputs_y_dims};
  std::vector<Shapes> shape_expect = {inputs_shape, inputs_shape};
  std::vector<Shapes> shape_test = ExtractShape(node);
  ASSERT_EQ(shape_test, shape_expect);
}

TEST_F(TestStepParallel, ExtractShape4) {
  Shape inputs_x_dims = {64, 32};
  Shape inputs_y_dims = {32, 64};
  Shape outputs_dims = {64, 64};
  CNodePtr node = Make_Node(inputs_x_dims, inputs_y_dims, outputs_dims, 2);
  Shapes inputs_shape = std::vector<Shape>{inputs_x_dims, inputs_y_dims};
  EXPECT_THROW({ ExtractShape(node); }, std::runtime_error);
}

TEST_F(TestStepParallel, CreatOpInstance) {
  ValuePtr attr0_value = MakeValue(REDUCE_OP_SUM);
  ValuePtr attr1_value = MakeValue("0-1-2");
  Attr attr0 = std::make_pair("op", attr0_value);
  Attr attr1 = std::make_pair("group", attr1_value);
  OperatorAttrs attrs = {attr0, attr1};
  OperatorName op_name = "AllReduce";
  OperatorParams operator_param;
  OperatorArgs args = std::make_pair(attrs, operator_param);
  auto op_instance = CreatOpInstance(args.first, op_name, "test");
  ASSERT_TRUE(op_instance);
  PrimitivePyPtr allreduce_ptr = dyn_cast<PrimitivePy>(op_instance);
  ASSERT_TRUE(allreduce_ptr);
  if (nullptr != allreduce_ptr) {
    MS_LOG(INFO) << "Get PrimitivePyPtr: " << allreduce_ptr->name();

    std::vector<py::object> arglist;
    (void)std::transform(attrs.begin(), attrs.end(), std::back_inserter(arglist),
                         [](Attr attr) { return ValuePtrToPyData(attr.second); });
    py::object allreduce_pyobj = parse::python_adapter::CallPyFn(
      "mindspore.parallel._utils", "_get_python_op", "AllReduce", "mindspore.ops.operations", "test", arglist);
    py::dict opAttr = py::getattr(allreduce_pyobj, "attrs");
    std::unordered_map<std::string, ValuePtr> attributes{};
    for (auto item : opAttr) {
      if (!py::isinstance<py::str>(item.first)) {
        MS_LOG(EXCEPTION) << "type error in py dict convert";
      }
      std::string name = py::cast<std::string>(item.first);
      MS_LOG(INFO) << "Attr name: " << name;

      ValuePtr converted_ret;
      if (name == "op") {
        parse::ConvertData(py::cast<py::object>(item.second), &converted_ret);
        ASSERT_EQ(converted_ret->ToString(), "sum");
      } else {
        if (name == "group") {
          parse::ConvertData(py::cast<py::object>(item.second), &converted_ret);
          ASSERT_EQ(converted_ret->ToString(), "0-1-2");
        } else if (name == "fusion") {
          parse::ConvertData(py::cast<py::object>(item.second), &converted_ret);
          ASSERT_EQ(converted_ret->ToString(), "0");
        } else if (name == "instance_name") {
          parse::ConvertData(py::cast<py::object>(item.second), &converted_ret);
          ASSERT_EQ(converted_ret->ToString(), "test");
        } else if (name == "index") {
          parse::ConvertData(py::cast<py::object>(item.second), &converted_ret);
          ASSERT_EQ(converted_ret->ToString(), "0");
        } else {
          MS_LOG(EXCEPTION) << "Test failed";
        }
      }
      attributes.emplace(name, converted_ret);
    }
  }
}

TEST_F(TestStepParallel, CreatOpInstance1) {
  OperatorAttrs attrs;
  OperatorName op_name = "ABC";
  OperatorParams operator_param;
  OperatorArgs args = std::make_pair(attrs, operator_param);
  EXPECT_THROW({ CreatOpInstance(args.first, op_name, "test"); }, std::runtime_error);
}

TEST_F(TestStepParallel, OperatorInstance) {
  // create  attrs and prim
  PrimitivePtr prim = NewValueNode(prim::kPrimMatMul)->value()->cast<PrimitivePtr>();
  ValuePtr transpose_a = MakeValue(false);
  ValuePtr transpose_b = MakeValue(false);
  prim->set_attr("transpose_a", transpose_a);
  prim->set_attr("transpose_b", transpose_b);
  auto attrs = prim->attrs();
  // create  strategy
  Strategys strategy = {{2, 2}, {2, 4}};
  StrategyPtr strategyPtr = parallel::NewStrategy(0, strategy);
  // create  shape
  Shapes inputs_shape = std::vector<Shape>{{64, 32}, {32, 64}};
  Shapes outputs_shape = std::vector<Shape>{{64, 64}};
  std::vector<Shapes> shape = {inputs_shape, outputs_shape};
  TOTAL_OPS = 0;
  OperatorInfoPtr matmul_info = OperatorInstance(prim, attrs, shape);
  matmul_info->Init(strategyPtr);
  std::string name_expect = "MatMulInfo00";
  std::string name_test = matmul_info->name();
  ASSERT_EQ(name_expect, name_test);
}

TEST_F(TestStepParallel, ExtractInformation) {
  FuncGraphManagerPtr manager = Make_Manager();
  FuncGraphSet graphs = manager->func_graphs();
  FuncGraphPtr graph = *graphs.begin();
  auto ret = graph->get_return();
  std::vector<AnfNodePtr> all_nodes = DeepScopedGraphSearch(ret);
  ExtractInformation(all_nodes);
}

TEST_F(TestStepParallel, ExtractInformation2) {
  FuncGraphManagerPtr manager = Make_Manager(2);
  FuncGraphSet graphs = manager->func_graphs();
  FuncGraphPtr graph = *graphs.begin();
  auto ret = graph->get_return();
  std::vector<AnfNodePtr> all_nodes = DeepScopedGraphSearch(ret);
  EXPECT_THROW({ ExtractInformation(all_nodes); }, std::runtime_error);
}

TEST_F(TestStepParallel, ExtractInformation3) {
  FuncGraphManagerPtr manager = Make_Manager(3);
  FuncGraphSet graphs = manager->func_graphs();
  FuncGraphPtr graph = *graphs.begin();
  auto ret = graph->get_return();
  std::vector<AnfNodePtr> all_nodes = DeepScopedGraphSearch(ret);
  EXPECT_THROW({ ExtractInformation(all_nodes); }, std::runtime_error);
}

TEST_F(TestStepParallel, ForwardCommunication1) {
  ValuePtr attr0_value = MakeValue(REDUCE_OP_SUM);
  ValuePtr attr1_value = MakeValue("0-1-2");
  Attr attr0 = std::make_pair("op", attr0_value);
  Attr attr1 = std::make_pair("group", attr1_value);
  OperatorAttrs attrs = {attr0, attr1};
  OperatorName op_name = "AllReduce";
  OperatorParams operator_param;
  OperatorArgs args = std::make_pair(attrs, operator_param);
  Operator op = std::make_pair(op_name, args);
  OperatorVector op_list = {op, op};
  FuncGraphManagerPtr manager = Make_Manager();
  FuncGraphSet graphs = manager->func_graphs();
  FuncGraphPtr graph = *graphs.begin();
  auto ret = graph->get_return();
  std::vector<AnfNodePtr> all_nodes = DeepScopedGraphSearch(ret);
  ExtractInformation(all_nodes);
  for (auto &node : all_nodes) {
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    FuncGraphPtr func_graph = node->func_graph();
    PrimitivePtr prim = cnode->input(0)->cast<ValueNodePtr>()->value()->cast<PrimitivePtr>();
    if (prim->name() == "MatMul") {
      ForwardCommunication(op_list, cnode);
      draw::Draw("forwardcommunication.dot", func_graph);
    }
  }
  AnfNodeSet after_nodes = manager->all_nodes();
  for (auto &node : after_nodes) {
    if (!node->isa<CNode>()) {
      continue;
    }
    auto &inputs = node->cast<CNodePtr>()->inputs();
    PrimitivePtr prim = inputs[0]->cast<ValueNodePtr>()->value()->cast<PrimitivePtr>();
    if (prim->name() == "Return" || prim->name() == "MatMul") {
      if (!inputs[1]->isa<Parameter>()) {
        CNodePtr pre_node = inputs[1]->cast<CNodePtr>();
        PrimitivePtr pre_prim = pre_node->input(0)->cast<ValueNodePtr>()->value()->cast<PrimitivePtr>();
        CNodePtr pre_node2 = pre_node->input(1)->cast<CNodePtr>();
        PrimitivePtr pre_prim2 = pre_node2->input(0)->cast<ValueNodePtr>()->value()->cast<PrimitivePtr>();
        ASSERT_EQ("AllReduce", pre_prim->name());
        ASSERT_EQ("AllReduce", pre_prim2->name());
      }
    }
  }
}

TEST_F(TestStepParallel, ForwardCommunication2) {
  OperatorVector op_list;
  FuncGraphManagerPtr manager = Make_Manager();
  FuncGraphSet graphs = manager->func_graphs();
  FuncGraphPtr graph = *graphs.begin();
  auto ret = graph->get_return();
  std::vector<AnfNodePtr> all_nodes = DeepScopedGraphSearch(ret);
  ExtractInformation(all_nodes);
  for (auto &node : all_nodes) {
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    FuncGraphPtr func_graph = node->func_graph();
    func_graph->set_manager(nullptr);
    PrimitivePtr prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    if (prim->name() == "MatMul") {
      EXPECT_THROW({ ForwardCommunication(op_list, cnode); }, std::runtime_error);
      break;
    }
  }
}

TEST_F(TestStepParallel, ForwardCommunication3) {
  OperatorVector op_list;
  FuncGraphManagerPtr manager = Make_Manager();
  FuncGraphSet graphs = manager->func_graphs();
  FuncGraphPtr graph = *graphs.begin();
  auto ret = graph->get_return();
  std::vector<AnfNodePtr> all_nodes = DeepScopedGraphSearch(ret);
  ExtractInformation(all_nodes);
  for (auto &node : all_nodes) {
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    FuncGraphPtr func_graph = node->func_graph();
    PrimitivePtr prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    if (prim->name() == "MatMul") {
      OperatorAttrs attrs;
      OperatorParams operator_param;
      OperatorArgs args = std::make_pair(attrs, operator_param);
      Operator op = std::make_pair("ABC", args);
      OperatorVector op_list = {op};
      EXPECT_THROW({ ForwardCommunication(op_list, cnode); }, std::runtime_error);
      break;
    }
  }
}

TEST_F(TestStepParallel, GetTensorInLayout) {
  // create  attrs and prim
  FuncGraphPtr func_graph = std::make_shared<FuncGraph>();
  Shape inputs_x_dims = {64, 32};
  Shape inputs_y_dims = {32, 64};
  Shape outputs_dims = {64, 64};
  CNodePtr node = Make_Node(inputs_x_dims, inputs_y_dims, outputs_dims);
  std::vector<AnfNodePtr> inputs(node->inputs());
  CNodePtr node1 = func_graph->NewCNode(inputs);
  PrimitivePtr prim = node1->input(0)->cast<ValueNodePtr>()->value()->cast<PrimitivePtr>();
  ValuePtr transpose_a = MakeValue(false);
  ValuePtr transpose_b = MakeValue(false);
  prim->set_attr("transpose_a", transpose_a);
  prim->set_attr("transpose_b", transpose_b);
  auto attrs = prim->attrs();
  // create  strategy
  Strategys strategy = {{2, 2}, {2, 4}};
  StrategyPtr strategyPtr = parallel::NewStrategy(0, strategy);
  // create  shape
  Shapes inputs_shape = std::vector<Shape>{{64, 32}, {32, 64}};
  Shapes outputs_shape = std::vector<Shape>{{64, 64}};
  std::vector<Shapes> shape = {inputs_shape, outputs_shape};
  OperatorInfoPtr matmul_info = OperatorInstance(prim, attrs, shape);
  matmul_info->Init(strategyPtr);
  node->set_user_data<OperatorInfo>(matmul_info);
  OperatorInfoPtr distribute_operator_pre = node->user_data<OperatorInfo>();
  TensorLayout tensorlayout_e;
  Shape array = {64, 64};
  TensorLayout tensorlayout = GetTensorInLayout(node1, prim, distribute_operator_pre);
  Shape tensor_shape_test = tensorlayout.tensor_shape().array();
  ASSERT_EQ(array, tensor_shape_test);
}

}  // namespace parallel
}  // namespace mindspore
