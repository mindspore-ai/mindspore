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
#include "parallel/step_parallel.h"
#include "parallel/step_auto_parallel.h"
#include "parallel/auto_parallel/edge_costmodel.h"
#include "parallel/ops_info/operator_info.h"
#include "operator/ops.h"
#include "pipeline/static_analysis/static_analysis.h"

namespace mindspore {
namespace parallel {

class TestStepAutoParallel : public UT::Common {
 public:
  TestStepAutoParallel() {}
  void SetUp();
  void TearDown() {}
};

void TestStepAutoParallel::SetUp() {
  std::list<int32_t> dev_list;

  for (int32_t i = 0; i < 20; i++) {
    dev_list.push_back(i);
  }

  std::list<int32_t> stage_map;
  stage_map.push_back(16);
  stage_map.push_back(4);

  int32_t local_dev = 0;

  // create a new g_device_manager
  g_device_manager = std::make_shared<DeviceManager>();
  g_device_manager->Init(dev_list, local_dev, stage_map, "hccl");
}

CNodePtr Create_Node(Shape x, Shape y, Shape out) {
  FuncGraphPtr func_graph = std::make_shared<FuncGraph>();
  ParameterPtr param1 = func_graph->add_parameter();
  ParameterPtr param2 = func_graph->add_parameter();
  param1->set_name("x");
  param2->set_name("y");
  BaseShapePtr shape1 = std::make_shared<abstract::Shape>(x);
  BaseShapePtr shape2 = std::make_shared<abstract::Shape>(y);
  BaseShapePtr shape3 = std::make_shared<abstract::Shape>(out);
  AbstractBasePtr abstract1 = abstract::FromValue(1, false);
  AbstractBasePtr abstract2 = abstract::FromValue(1, false);
  AbstractBasePtr abstract3 = abstract::FromValue(1, false);
  abstract1->set_shape(shape1);
  abstract2->set_shape(shape2);
  abstract3->set_shape(shape3);
  param1->set_abstract(abstract1);
  param2->set_abstract(abstract2);
  std::vector<AnfNodePtr> inputs;
  inputs.push_back(NewValueNode(prim::kPrimMatMul));
  inputs.push_back(param1);
  inputs.push_back(param2);
  CNodePtr node = func_graph->NewCNode(inputs);
  PrimitivePtr prim = node->input(0)->cast<ValueNodePtr>()->value()->cast<PrimitivePtr>();
  ValuePtr transpose_a = MakeValue(false);
  ValuePtr transpose_b = MakeValue(false);
  prim->set_attr("transpose_a", transpose_a);
  prim->set_attr("transpose_b", transpose_b);

  node->set_abstract(abstract3);
  return node;
}

CNodePtr Create_two_nodes(Shape x, Shape y, Shape z, Shape w, Shape out) {
  FuncGraphPtr func_graph = std::make_shared<FuncGraph>();
  ParameterPtr paramX = func_graph->add_parameter();
  ParameterPtr paramY = func_graph->add_parameter();
  ParameterPtr paramW = func_graph->add_parameter();
  paramX->set_name("x");
  paramY->set_name("y");
  paramW->set_name("w");
  BaseShapePtr shapeX = std::make_shared<abstract::Shape>(x);
  BaseShapePtr shapeY = std::make_shared<abstract::Shape>(y);
  BaseShapePtr shapeZ = std::make_shared<abstract::Shape>(z);
  BaseShapePtr shapeW = std::make_shared<abstract::Shape>(w);
  BaseShapePtr shapeOut = std::make_shared<abstract::Shape>(out);
  AbstractBasePtr abstractX = abstract::FromValue(1, false);
  AbstractBasePtr abstractY = abstract::FromValue(1, false);
  AbstractBasePtr abstractZ = abstract::FromValue(1, false);
  AbstractBasePtr abstractW = abstract::FromValue(1, false);
  AbstractBasePtr abstractOut = abstract::FromValue(1, false);
  abstractX->set_shape(shapeX);
  abstractY->set_shape(shapeY);
  abstractZ->set_shape(shapeZ);
  abstractW->set_shape(shapeW);
  abstractOut->set_shape(shapeOut);
  paramX->set_abstract(abstractX);
  paramY->set_abstract(abstractY);
  paramW->set_abstract(abstractW);

  std::vector<AnfNodePtr> MatMul_1_inputs;
  MatMul_1_inputs.push_back(NewValueNode(prim::kPrimMatMul));
  MatMul_1_inputs.push_back(paramX);
  MatMul_1_inputs.push_back(paramY);
  CNodePtr MatMul_1_node = func_graph->NewCNode(MatMul_1_inputs);
  PrimitivePtr prim = MatMul_1_node->input(0)->cast<ValueNodePtr>()->value()->cast<PrimitivePtr>();
  ValuePtr transpose_a = MakeValue(false);
  ValuePtr transpose_b = MakeValue(false);
  prim->set_attr("transpose_a", transpose_a);
  prim->set_attr("transpose_b", transpose_b);
  MatMul_1_node->set_abstract(abstractZ);

  std::vector<AnfNodePtr> MatMul_2_inputs;
  MatMul_2_inputs.push_back(NewValueNode(prim::kPrimMatMul));
  MatMul_2_inputs.push_back(MatMul_1_node);
  MatMul_2_inputs.push_back(paramW);
  CNodePtr MatMul_2_node = func_graph->NewCNode(MatMul_2_inputs);
  PrimitivePtr prim2 = MatMul_2_node->input(0)->cast<ValueNodePtr>()->value()->cast<PrimitivePtr>();
  ValuePtr transpose_a_2 = MakeValue(false);
  ValuePtr transpose_b_2 = MakeValue(false);
  prim2->set_attr("transpose_a", transpose_a);
  prim2->set_attr("transpose_b", transpose_b);
  MatMul_2_node->set_abstract(abstractOut);

  return MatMul_2_node;
}

TEST_F(TestStepAutoParallel, test_create_op_instance) {
  Shape inputs_x_dims = {64, 32};
  Shape inputs_y_dims = {32, 64};
  Shape outputs_dims = {64, 64};
  CNodePtr node = Create_Node(inputs_x_dims, inputs_y_dims, outputs_dims);
  bool result = node->input(0)->cast<ValueNodePtr>()->value()->isa<Primitive>();
  ASSERT_EQ(result, true);
  // creat prim and attrs
  PrimitivePtr prim = node->input(0)->cast<ValueNodePtr>()->value()->cast<PrimitivePtr>();
  auto attrs = prim->attrs();

  // creat shape
  Shapes inputs_shape = std::vector<Shape>{inputs_x_dims, inputs_y_dims};
  Shapes outputs_shape = std::vector<Shape>{outputs_dims};
  std::vector<Shapes> shape = {inputs_shape, outputs_shape};
  StrategyPtr strategyPtr;

  std::shared_ptr<OperatorInfo> matmul_info = NewOperatorInstance(prim, attrs, shape);
  node->set_operator_info(matmul_info);
  std::string name_expect = "MatMulInfo00";
  std::string name_test = matmul_info->name();
  ASSERT_EQ(name_expect, name_test);
}

TEST_F(TestStepAutoParallel, test_create_edge) {
  Shape inputs_x_dims = {64, 32};
  Shape inputs_y_dims = {32, 64};
  Shape outputs_z_dims = {64, 64};
  Shape inputs_w_dims = {64, 128};
  Shape outputs_dim = {64, 128};
  CNodePtr node = Create_two_nodes(inputs_x_dims, inputs_y_dims, outputs_z_dims, inputs_w_dims, outputs_dim);

  // u-->v
  PrimitivePtr v_prim = node->input(0)->cast<ValueNodePtr>()->value()->cast<PrimitivePtr>();
  auto v_attrs = v_prim->attrs();
  PrimitivePtr u_prim = node->input(1)->cast<CNodePtr>()->input(0)->cast<ValueNodePtr>()->value()->cast<PrimitivePtr>();
  auto u_attrs = u_prim->attrs();

  // creat v node
  Shapes v_inputs_shape = std::vector<Shape>{outputs_z_dims, inputs_w_dims};
  Shapes v_outputs_shape = std::vector<Shape>{outputs_dim};
  std::vector<Shapes> v_shape = {v_inputs_shape, v_outputs_shape};
  StrategyPtr v_strategyPtr;
  std::shared_ptr<OperatorInfo> v_matmul_info = NewOperatorInstance(v_prim, v_attrs, v_shape);

  // create u node
  Shapes u_inputs_shape = std::vector<Shape>{inputs_x_dims, inputs_y_dims};
  Shapes u_outputs_shape = std::vector<Shape>{outputs_z_dims};
  std::vector<Shapes> u_shape = {u_inputs_shape, u_outputs_shape};
  StrategyPtr u_strategyPtr;
  std::shared_ptr<OperatorInfo> u_matmul_info = NewOperatorInstance(u_prim, u_attrs, u_shape);

  std::string edge_name = u_prim->name() + "-" + v_prim->name();
  std::shared_ptr<Edge> edge_ptr = std::make_shared<Edge>(edge_name, u_matmul_info, v_matmul_info, 0, 0, false);
  std::string expected_name = "MatMul-MatMul";
  ASSERT_EQ(edge_ptr->edge_name(), expected_name);
}

}  // namespace parallel
}  // namespace mindspore
