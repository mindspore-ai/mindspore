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
#include <iostream>
#include <unordered_map>

#include "pybind11/pybind11.h"

#include "transform/transform_base_test.h"
#include "common/py_func_graph_fetcher.h"
#include "pipeline/jit/parse/parse.h"
#include "debug/draw.h"
#include "debug/anf_ir_dump.h"
#include "pipeline/jit/static_analysis/prim.h"
#include "frontend/operator/ops.h"
#include "common/common_test.h"

#define private public
#include "transform/graph_ir/types.h"
#include "transform/graph_ir/convert.h"
#include "securec/include/securec.h"
#include "utils/utils.h"
using std::cout;
using std::endl;
using std::string;
using std::unordered_map;

namespace mindspore {
namespace transform {
using AbstractScalar = abstract::AbstractScalar;
using mindspore::parse::ResolveAll;

class TestConvert : public UT::Common {
 public:
  TestConvert() {}
  virtual void SetUp();
  virtual void TearDown();
  static const std::shared_ptr<Float> kF32;
};

void TestConvert::SetUp() { UT::InitPythonPath(); }
void TestConvert::TearDown() {}

const std::shared_ptr<Float> TestConvert::kF32 = std::make_shared<Float>(32);

AnfGraphPtr createAnfGraph() { return std::make_shared<AnfGraph>(); }

TEST_F(TestConvert, TestConstruct) {
  AnfGraphPtr func_graph = std::make_shared<AnfGraph>();
  DfGraphConvertor converter(func_graph);
  converter.ConvertAllNode().GetComputeGraph();
  ASSERT_NE(converter.ErrCode(), SUCCESS);
}

#if (!defined ENABLE_GE)

namespace {

bool MakeDfGraph(PrimitivePtr prim, unsigned int nparam) {
  std::shared_ptr<FuncGraph> anf_graph = MakeFuncGraph(prim, nparam);
  std::shared_ptr<FuncGraphManager> graph_manager = MakeManager({anf_graph});

  draw::Draw("ut_prim_" + prim->name() + ".dot", anf_graph);
  DumpIR("ut_prim_" + prim->name() + ".ir", anf_graph);

  DfGraphConvertor converter(anf_graph);
  auto df_graph = converter.ConvertAllNode().BuildGraph().GetComputeGraph();
  converter.DrawComputeGraph(prim->name() + ".dot");
  if (converter.ErrCode() != 0) {
    MS_LOG(ERROR) << "DfGraphConvertor convert " << prim->name() << " error, error code is: " << converter.ErrCode();
    return false;
  }
  if (df_graph == nullptr) {
    MS_LOG(ERROR) << "DfGraphConvertor get " << prim->name() << " compute func_graph failed";
    return false;
  }
  return true;
}

}  // namespace

TEST_F(TestConvert, TestConvertConv2d) {
  PrimitivePtr conv2d = prim::kPrimConv2D;
  conv2d->AddAttr("stride", MakeValue(static_cast<int64_t>(2)));
  conv2d->AddAttr("pad", MakeValue(static_cast<int64_t>(0)));
  conv2d->AddAttr("dilation", MakeValue(static_cast<int64_t>(0)));

  FuncGraphPtr anf_graph = MakeFuncGraph(conv2d, 2);
  std::shared_ptr<FuncGraphManager> graph_manager = MakeManager({anf_graph});

  draw::Draw("ut_prim_conv2d1.dot", anf_graph);
  DumpIR("ut_prim_conv2d1.ir", anf_graph);

  DfGraphConvertor converter(anf_graph);
  auto df_graph = converter.ConvertAllNode().BuildGraph().GetComputeGraph();
  converter.DrawComputeGraph("conv2d.dot");
  ASSERT_EQ(converter.ErrCode(), 0);
  ASSERT_NE(df_graph, nullptr);
}

TEST_F(TestConvert, TestConvertMaxpooling) {
  auto prim = std::make_shared<Primitive>("MaxPool");
  FuncGraphPtr anf_graph = MakeFuncGraph(prim, 5);  // ary, ksize, stride, padding, data_format

  std::shared_ptr<FuncGraphManager> graph_manager = MakeManager({anf_graph});
  draw::Draw("ut_prim_maxpooling.dot", anf_graph);
  DumpIR("ut_prim_maxpooling.ir", anf_graph);

  DfGraphConvertor converter(anf_graph);
  auto df_graph = converter.ConvertAllNode().BuildGraph().GetComputeGraph();
  converter.DrawComputeGraph("maxpooling.dot");
  ASSERT_EQ(converter.ErrCode(), 0);
  ASSERT_NE(df_graph, nullptr);
}

TEST_F(TestConvert, TestReluOps) {
  auto prim = prim::kPrimRelu;
  prim->AddAttr("T", MakeValue(static_cast<int64_t>(0)));

  auto func_graph = MakeFuncGraph(prim, 1);
  ASSERT_TRUE(nullptr != func_graph);

  // save the func_graph to manager
  std::shared_ptr<FuncGraphManager> manager = Manage(func_graph);

  // call resolve
  bool ret_ = ResolveAll(manager);
  ASSERT_TRUE(ret_);

  // draw graph
  auto anfGraph = *(manager->func_graphs().begin());
  DfGraphConvertor converter(anfGraph);
  converter.ConvertAllNode().BuildGraph().GetComputeGraph();
  ASSERT_EQ(converter.ErrCode(), 0);
}

TEST_F(TestConvert, TestConvertBatchNorm) {
  PrimitivePtr batch_norm = prim::kPrimBatchNorm;
  batch_norm->AddAttr("epsilon", MakeValue(0.001f));
  batch_norm->AddAttr("momentum", MakeValue(0.1f));

  FuncGraphPtr anf_graph = std::make_shared<FuncGraph>();
  std::vector<AnfNodePtr> inputs;
  inputs.push_back(NewValueNode(batch_norm));
  for (unsigned int i = 0; i < 5; i++) {
    inputs.push_back(anf_graph->add_parameter());
  }
  CNodePtr cnode_prim = anf_graph->NewCNode(inputs);
  inputs.clear();

  inputs.push_back(NewValueNode(prim::kPrimTupleGetItem));
  inputs.push_back(cnode_prim);
  inputs.push_back(NewValueNode(static_cast<int64_t>(2)));
  CNodePtr cnode_getitem = anf_graph->NewCNode(inputs);
  inputs.clear();

  inputs.push_back(NewValueNode(prim::kPrimRelu));
  inputs.push_back(cnode_getitem);
  CNodePtr cnode_relu = anf_graph->NewCNode(inputs);
  inputs.clear();

  inputs.push_back(NewValueNode(std::make_shared<Primitive>("Return")));
  inputs.push_back(cnode_relu);
  CNodePtr cnode_return = anf_graph->NewCNode(inputs);
  anf_graph->set_return(cnode_return);

  std::shared_ptr<FuncGraphManager> graph_manager = MakeManager({anf_graph});
  draw::Draw("ut_prim_batchnorm.dot", anf_graph);
  DumpIR("ut_prim_batchnorm.ir", anf_graph);

  DfGraphConvertor converter(anf_graph);
  auto df_graph = converter.ConvertAllNode().BuildGraph().GetComputeGraph();
  converter.DrawComputeGraph("batchnrom.dot");
  ASSERT_EQ(converter.ErrCode(), 0);
  ASSERT_NE(df_graph, nullptr);
}

TEST_F(TestConvert, TestConvertConvBackpropInput) {
  auto prim = prim::kPrimConv2DBackpropInput;
  const std::vector<int64_t> list{1,1};
  prim->AddAttr("stride", MakeValue(list));
  prim->AddAttr("pad", MakeValue(static_cast<int64_t>(0)));
  prim->AddAttr("pad_mode", MakeValue(std::string("pad")));
  prim->AddAttr("dilation", MakeValue(static_cast<int64_t>(1)));
  prim->AddAttr("group", MakeValue(static_cast<int64_t>(1)));
  prim->AddAttr("mode", MakeValue(static_cast<int64_t>(1)));
  prim->AddAttr("dilation", MakeValue(static_cast<int64_t>(1)));

  auto func_graph = MakeFuncGraph(prim, 3);
  ASSERT_NE(func_graph, nullptr);
  // save the func_graph to manager
  std::shared_ptr<FuncGraphManager> manager = Manage(func_graph);

  // call resolve
  bool ret_ = ResolveAll(manager);
  ASSERT_TRUE(ret_);

  // draw graph
  auto anf_graph = *(manager->func_graphs().begin());
  DfGraphConvertor converter(anf_graph);
  auto df_graph = converter.ConvertAllNode().BuildGraph().GetComputeGraph();

  converter.DrawComputeGraph("Conv2DBackpropInput.dot");
  ASSERT_EQ(converter.ErrCode(), 0);
  ASSERT_NE(df_graph, nullptr);
}

TEST_F(TestConvert, TestConvertConvBackpropFilter) {
  auto prim = prim::kPrimConv2DBackpropFilter;
  const std::vector<int64_t> list{1,1};
  prim->AddAttr("stride", MakeValue(list));
  prim->AddAttr("pad", MakeValue(static_cast<int64_t>(0)));
  prim->AddAttr("pad_mode", MakeValue(std::string("pad")));
  prim->AddAttr("dilation", MakeValue(static_cast<int64_t>(1)));
  prim->AddAttr("group", MakeValue(static_cast<int64_t>(1)));
  prim->AddAttr("mode", MakeValue(static_cast<int64_t>(1)));
  prim->AddAttr("dilation", MakeValue(static_cast<int64_t>(1)));

  auto func_graph = MakeFuncGraph(prim, 3);
  ASSERT_NE(func_graph, nullptr);
  // save the func_graph to manager
  std::shared_ptr<FuncGraphManager> manager = Manage(func_graph);

  // call resolve
  bool ret_ = ResolveAll(manager);
  ASSERT_TRUE(ret_);

  // draw graph
  auto anf_graph = *(manager->func_graphs().begin());
  DfGraphConvertor converter(anf_graph);
  auto df_graph = converter.ConvertAllNode().BuildGraph().GetComputeGraph();

  converter.DrawComputeGraph("Conv2DBackpropFilter.dot");
  ASSERT_EQ(converter.ErrCode(), 0);
  ASSERT_NE(df_graph, nullptr);
}

TEST_F(TestConvert, TestConvertReluGrad) {
  auto prim = prim::kPrimReluGrad;
  prim->AddAttr("alpha", MakeValue(0.1f));
  prim->AddAttr("beta", MakeValue(0.1f));
  prim->AddAttr("mode", MakeValue(static_cast<int64_t>(1)));

  auto func_graph = MakeFuncGraph(prim, 2);
  ASSERT_NE(func_graph, nullptr);
  // save the func_graph to manager
  std::shared_ptr<FuncGraphManager> manager = Manage(func_graph);

  // call resolve
  bool ret_ = ResolveAll(manager);
  ASSERT_TRUE(ret_);

  // draw graph
  auto anf_graph = *(manager->func_graphs().begin());
  DfGraphConvertor converter(anf_graph);
  auto df_graph = converter.ConvertAllNode().BuildGraph().GetComputeGraph();

  converter.DrawComputeGraph("ReluGrad.dot");
  ASSERT_EQ(converter.ErrCode(), 0);
  ASSERT_NE(df_graph, nullptr);
}

TEST_F(TestConvert, TestConvertBiasAdd) {
  auto prim = std::make_shared<Primitive>("BiasAdd");
  prim->AddAttr("alpha", MakeValue(0.0f));
  prim->AddAttr("beta", MakeValue(1.0f));

  auto func_graph = MakeFuncGraph(prim, 2);
  ASSERT_NE(func_graph, nullptr);
  // save the func_graph to manager
  std::shared_ptr<FuncGraphManager> manager = Manage(func_graph);

  // call resolve
  bool ret_ = ResolveAll(manager);
  ASSERT_TRUE(ret_);

  // draw graph
  auto anf_graph = *(manager->func_graphs().begin());
  DfGraphConvertor converter(anf_graph);
  auto df_graph = converter.ConvertAllNode().BuildGraph().GetComputeGraph();

  converter.DrawComputeGraph("BiasAdd.dot");
  ASSERT_EQ(converter.ErrCode(), 0);
  ASSERT_NE(df_graph, nullptr);
}

TEST_F(TestConvert, TestConvertBiasAddGrad) {
  auto prim = prim::kPrimBiasAddGrad;
  prim->AddAttr("alpha", MakeValue(0.0f));
  prim->AddAttr("beta", MakeValue(1.0f));

  auto func_graph = MakeFuncGraph(prim, 2);
  ASSERT_NE(func_graph, nullptr);
  // save the func_graph to manager
  std::shared_ptr<FuncGraphManager> manager = Manage(func_graph);

  // call resolve
  bool ret_ = ResolveAll(manager);
  ASSERT_TRUE(ret_);

  // draw graph
  auto anf_graph = *(manager->func_graphs().begin());
  DfGraphConvertor converter(anf_graph);
  auto df_graph = converter.ConvertAllNode().BuildGraph().GetComputeGraph();

  converter.DrawComputeGraph("BiasAddGrad.dot");
  ASSERT_EQ(converter.ErrCode(), 0);
  ASSERT_NE(df_graph, nullptr);
}

TEST_F(TestConvert, TestConvertMaxPoolGradWithArgmax) {
  auto prim = std::make_shared<Primitive>("MaxPoolGradWithArgmax");
  prim->AddAttr("alpha", MakeValue(0.0f));
  prim->AddAttr("beta", MakeValue(1.0f));
  prim->AddAttr("window", MakeValue(static_cast<int64_t>(2)));
  prim->AddAttr("stride", MakeValue(static_cast<int64_t>(1)));
  prim->AddAttr("ceil_mode", MakeValue(static_cast<int64_t>(0)));
  prim->AddAttr("data_mode", MakeValue(static_cast<int64_t>(0)));
  prim->AddAttr("alpha", MakeValue(0.1f));
  prim->AddAttr("beta", MakeValue(1.0f));

  auto func_graph = MakeFuncGraph(prim, 2);
  ASSERT_NE(func_graph, nullptr);
  // save the func_graph to manager
  std::shared_ptr<FuncGraphManager> manager = Manage(func_graph);

  // call resolve
  bool ret_ = ResolveAll(manager);
  ASSERT_TRUE(ret_);

  // draw graph
  auto anf_graph = *(manager->func_graphs().begin());
  DfGraphConvertor converter(anf_graph);
  auto df_graph = converter.ConvertAllNode().BuildGraph().GetComputeGraph();

  converter.DrawComputeGraph("MaxPoolGradWithArgmax.dot");
  ASSERT_EQ(converter.ErrCode(), 0);
  ASSERT_NE(df_graph, nullptr);
}

TEST_F(TestConvert, TestConcat) {
  auto prim = prim::kPrimConcat;

  std::shared_ptr<FuncGraph> anf_graph = MakeFuncGraph(prim, 2);
  std::shared_ptr<FuncGraphManager> graph_manager = MakeManager({anf_graph});

  draw::Draw("ut_prim_concat.dot", anf_graph);
  DumpIR("ut_prim_concat.ir", anf_graph);

  DfGraphConvertor converter(anf_graph);
  auto df_graph = converter.ConvertAllNode().BuildGraph().GetComputeGraph();
  converter.DrawComputeGraph("concat.dot");
  ASSERT_EQ(converter.ErrCode(), 0);
  ASSERT_NE(df_graph, nullptr);
}

TEST_F(TestConvert, TestGatherV2) {
  auto prim = prim::kPrimGather;

  std::shared_ptr<FuncGraph> anf_graph = MakeFuncGraph(prim, 3);
  std::shared_ptr<FuncGraphManager> graph_manager = MakeManager({anf_graph});

  draw::Draw("ut_prim_gatherv2.dot", anf_graph);
  DumpIR("ut_prim_gatherv2.ir", anf_graph);

  DfGraphConvertor converter(anf_graph);
  auto df_graph = converter.ConvertAllNode().BuildGraph().GetComputeGraph();
  converter.DrawComputeGraph("gatherv2.dot");
  ASSERT_EQ(converter.ErrCode(), 0);
  ASSERT_NE(df_graph, nullptr);
}

TEST_F(TestConvert, TestCast) {
  auto prim = prim::kPrimCast;

  std::shared_ptr<FuncGraph> anf_graph = MakeFuncGraph(prim, 2);
  std::shared_ptr<FuncGraphManager> graph_manager = MakeManager({anf_graph});

  draw::Draw("ut_prim_cast.dot", anf_graph);
  DumpIR("ut_prim_cast.ir", anf_graph);

  DfGraphConvertor converter(anf_graph);
  auto df_graph = converter.ConvertAllNode().BuildGraph().GetComputeGraph();
  converter.DrawComputeGraph("cast.dot");
  ASSERT_EQ(converter.ErrCode(), 0);
  ASSERT_NE(df_graph, nullptr);
}

TEST_F(TestConvert, TestExp) {
  auto prim = std::make_shared<Primitive>("Exp");

  std::shared_ptr<FuncGraph> anf_graph = MakeFuncGraph(prim, 1);
  std::shared_ptr<FuncGraphManager> graph_manager = MakeManager({anf_graph});

  draw::Draw("ut_prim_exp.dot", anf_graph);
  DumpIR("ut_prim_exp.ir", anf_graph);

  DfGraphConvertor converter(anf_graph);
  auto df_graph = converter.ConvertAllNode().BuildGraph().GetComputeGraph();
  converter.DrawComputeGraph("exp.dot");
  ASSERT_EQ(converter.ErrCode(), 0);
  ASSERT_NE(df_graph, nullptr);
}

TEST_F(TestConvert, TestFloor) {
  auto prim = std::make_shared<Primitive>("Floor");

  std::shared_ptr<FuncGraph> anf_graph = MakeFuncGraph(prim, 1);
  std::shared_ptr<FuncGraphManager> graph_manager = MakeManager({anf_graph});

  draw::Draw("ut_prim_floor.dot", anf_graph);
  DumpIR("ut_prim_floor.ir", anf_graph);

  DfGraphConvertor converter(anf_graph);
  auto df_graph = converter.ConvertAllNode().BuildGraph().GetComputeGraph();
  converter.DrawComputeGraph("floor.dot");
  ASSERT_EQ(converter.ErrCode(), 0);
  ASSERT_NE(df_graph, nullptr);
}

TEST_F(TestConvert, TestGreaterEqual) {
  auto prim = std::make_shared<Primitive>("GreaterEqual");

  std::shared_ptr<FuncGraph> anf_graph = MakeFuncGraph(prim, 2);
  std::shared_ptr<FuncGraphManager> graph_manager = MakeManager({anf_graph});

  draw::Draw("ut_prim_greater_equal.dot", anf_graph);
  DumpIR("ut_prim_greater_equal.ir", anf_graph);

  DfGraphConvertor converter(anf_graph);
  auto df_graph = converter.ConvertAllNode().BuildGraph().GetComputeGraph();
  converter.DrawComputeGraph("greater_equal.dot");
  ASSERT_EQ(converter.ErrCode(), 0);
  ASSERT_NE(df_graph, nullptr);
}

TEST_F(TestConvert, TestLess) {
  auto prim = std::make_shared<Primitive>("Less");
  prim->AddAttr("T", MakeValue(kFloat32));

  std::shared_ptr<FuncGraph> anf_graph = MakeFuncGraph(prim, 2);
  std::shared_ptr<FuncGraphManager> graph_manager = MakeManager({anf_graph});

  draw::Draw("ut_prim_less.dot", anf_graph);
  DumpIR("ut_prim_less.ir", anf_graph);

  DfGraphConvertor converter(anf_graph);
  auto df_graph = converter.ConvertAllNode().BuildGraph().GetComputeGraph();
  converter.DrawComputeGraph("less.dot");
  ASSERT_EQ(converter.ErrCode(), 0);
  ASSERT_NE(df_graph, nullptr);
}

TEST_F(TestConvert, TestLessEqual) {
  auto prim = std::make_shared<Primitive>("LessEqual");

  std::shared_ptr<FuncGraph> anf_graph = MakeFuncGraph(prim, 2);
  std::shared_ptr<FuncGraphManager> graph_manager = MakeManager({anf_graph});

  draw::Draw("ut_prim_less_equal.dot", anf_graph);
  DumpIR("ut_prim_less_equal.ir", anf_graph);

  DfGraphConvertor converter(anf_graph);
  auto df_graph = converter.ConvertAllNode().BuildGraph().GetComputeGraph();
  converter.DrawComputeGraph("less_equal.dot");
  ASSERT_EQ(converter.ErrCode(), 0);
  ASSERT_NE(df_graph, nullptr);
}

TEST_F(TestConvert, TestLogicalNot) {
  auto prim = std::make_shared<Primitive>("LogicalNot");

  std::shared_ptr<FuncGraph> anf_graph = MakeFuncGraph(prim, 1);
  std::shared_ptr<FuncGraphManager> graph_manager = MakeManager({anf_graph});

  draw::Draw("ut_prim_logical_not.dot", anf_graph);
  DumpIR("ut_prim_logical_not.ir", anf_graph);

  DfGraphConvertor converter(anf_graph);
  auto df_graph = converter.ConvertAllNode().BuildGraph().GetComputeGraph();
  converter.DrawComputeGraph("logical_not.dot");
  ASSERT_EQ(converter.ErrCode(), 0);
  ASSERT_NE(df_graph, nullptr);
}

TEST_F(TestConvert, TestAssignAdd) {
  auto prim = prim::kPrimAssignAdd;
  prim->AddAttr("use_locking", MakeValue(true));

  std::shared_ptr<FuncGraph> anf_graph = MakeFuncGraph(prim, 2);
  std::shared_ptr<FuncGraphManager> graph_manager = MakeManager({anf_graph});

  draw::Draw("ut_prim_assign_add.dot", anf_graph);
  DumpIR("ut_prim_assign_add.ir", anf_graph);

  DfGraphConvertor converter(anf_graph);
  auto df_graph = converter.ConvertAllNode().BuildGraph().GetComputeGraph();
  converter.DrawComputeGraph("assign_add.dot");
  ASSERT_EQ(converter.ErrCode(), 0);
  ASSERT_NE(df_graph, nullptr);
}

TEST_F(TestConvert, LogSoftmax) {
  auto prim = prim::kPrimLogSoftmax;
  prim->AddAttr("axis", MakeValue(static_cast<int64_t>(0)));

  std::shared_ptr<FuncGraph> anf_graph = MakeFuncGraph(prim, 1);
  std::shared_ptr<FuncGraphManager> graph_manager = MakeManager({anf_graph});

  draw::Draw("ut_prim_log_softmax.dot", anf_graph);
  DumpIR("ut_prim_log_softmax.ir", anf_graph);

  DfGraphConvertor converter(anf_graph);
  auto df_graph = converter.ConvertAllNode().BuildGraph().GetComputeGraph();
  converter.DrawComputeGraph("log_softmax.dot");
  ASSERT_EQ(converter.ErrCode(), 0);
  ASSERT_NE(df_graph, nullptr);
}

TEST_F(TestConvert, TestMaximumOps) {
  auto prim = prim::kPrimMaximum;
  bool ret = MakeDfGraph(prim, 2);
  ASSERT_TRUE(ret);
}

TEST_F(TestConvert, TestReduceMeanOps) {
  auto prim = prim::kPrimReduceMean;
  prim->AddAttr("keepdims", MakeValue(true));
  bool ret = MakeDfGraph(prim, 2);
  ASSERT_TRUE(ret);
}

TEST_F(TestConvert, TestMinimumOps) {
  auto prim = prim::kPrimMinimum;
  bool ret = MakeDfGraph(prim, 2);
  ASSERT_TRUE(ret);
}

TEST_F(TestConvert, TestFusedMinOrMaxGradOps) {
  // Add infer step to this test case
  ASSERT_TRUE(true);
}

TEST_F(TestConvert, TestSqueezeOps) {
  auto prim = prim::kPrimSqueeze;
  bool ret = MakeDfGraph(prim, 2);
  ASSERT_TRUE(ret);
}

TEST_F(TestConvert, TestMulOps) {
  auto prim = prim::kPrimMul;
  bool ret = MakeDfGraph(prim, 2);
  ASSERT_TRUE(ret);
}

TEST_F(TestConvert, TestNegOps) {
  auto prim = prim::kPrimNeg;
  bool ret = MakeDfGraph(prim, 1);
  ASSERT_TRUE(ret);
}

TEST_F(TestConvert, TestOneHotOps) {
  auto prim = prim::kPrimOneHot;
  prim->AddAttr("axis", MakeValue(static_cast<int64_t>(0)));
  bool ret = MakeDfGraph(prim, 4);
  ASSERT_TRUE(ret);
}

TEST_F(TestConvert, TestPowOps) {
  auto prim = std::make_shared<Primitive>("Pow");
  bool ret = MakeDfGraph(prim, 2);
  ASSERT_TRUE(ret);
}

TEST_F(TestConvert, TestReciprocalOps) {
  auto prim = std::make_shared<Primitive>("Reciprocal");
  bool ret = MakeDfGraph(prim, 1);
  ASSERT_TRUE(ret);
}

TEST_F(TestConvert, TestSelectOps) {
  auto prim = prim::kPrimSelect;
  bool ret = MakeDfGraph(prim, 3);
  ASSERT_TRUE(ret);
}

TEST_F(TestConvert, TestSqrtOps) {
  auto prim = std::make_shared<Primitive>("Sqrt");
  bool ret = MakeDfGraph(prim, 1);
  ASSERT_TRUE(ret);
}

TEST_F(TestConvert, TestSquareOps) {
  auto prim = std::make_shared<Primitive>("Square");
  bool ret = MakeDfGraph(prim, 1);
  ASSERT_TRUE(ret);
}

TEST_F(TestConvert, TestScalarSummaryOps) {
  auto prim = prim::kPrimScalarSummary;
  // should have only 1 input.
  bool ret = MakeDfGraph(prim, 2);
  ASSERT_TRUE(ret);
}

TEST_F(TestConvert, TestTensorSummaryOps) {
  auto prim = prim::kPrimTensorSummary;
  bool ret = MakeDfGraph(prim, 2);
  ASSERT_TRUE(ret);
}

TEST_F(TestConvert, TestHistogramSummaryOps) {
  auto prim = prim::kPrimHistogramSummary;
  bool ret = MakeDfGraph(prim, 2);
  ASSERT_TRUE(ret);
}

TEST_F(TestConvert, TestGreaterOps) {
  auto prim = std::make_shared<Primitive>("Greater");
  bool ret = MakeDfGraph(prim, 2);
  ASSERT_TRUE(ret);
}

TEST_F(TestConvert, TestEqualOps) {
  auto prim = std::make_shared<Primitive>("Equal");
  bool ret = MakeDfGraph(prim, 2);
  ASSERT_TRUE(ret);
}

TEST_F(TestConvert, TestArgMaxiOps) {
  auto prim = std::make_shared<Primitive>("Argmax");
  bool ret = MakeDfGraph(prim, 2);
  ASSERT_TRUE(ret);
}

TEST_F(TestConvert, TestResizeNearestNeighborOps) {
  auto prim = std::make_shared<Primitive>("ResizeNearestNeighbor");
  bool ret = MakeDfGraph(prim, 1);
  ASSERT_TRUE(ret);
}

TEST_F(TestConvert, TestApplyMomentumOps) {
  auto prim = std::make_shared<Primitive>("ApplyMomentum");
  bool ret = MakeDfGraph(prim, 5);
  ASSERT_TRUE(ret);
}

TEST_F(TestConvert, TestNPUGetFloatStatusOps) {
  auto prim = std::make_shared<Primitive>("NPUGetFloatStatus");
  bool ret = MakeDfGraph(prim, 1);
  ASSERT_TRUE(ret);
}

TEST_F(TestConvert, TestNPUAllocFloatStatusOps) {
  auto prim = std::make_shared<Primitive>("NPUAllocFloatStatus");
  bool ret = MakeDfGraph(prim, 0);
  ASSERT_TRUE(ret);
}

TEST_F(TestConvert, TestNPUClearFloatStatusOps) {
  auto prim = std::make_shared<Primitive>("NPUClearFloatStatus");
  bool ret = MakeDfGraph(prim, 1);
  ASSERT_TRUE(ret);
}

#endif

TEST_F(TestConvert, TestAddOps) {
  auto prim = std::make_shared<Primitive>("Add");
  auto func_graph = MakeFuncGraph(prim, 2);
  ASSERT_TRUE(nullptr != func_graph);

  // save the func_graph to manager
  std::shared_ptr<FuncGraphManager> manager = Manage(func_graph);

  // call resolve
  bool ret_ = ResolveAll(manager);
  ASSERT_TRUE(ret_);

  // draw graph
  auto anfGraph = *(manager->func_graphs().begin());
  DfGraphConvertor converter(anfGraph);
  converter.ConvertAllNode().BuildGraph().GetComputeGraph();
  ASSERT_EQ(converter.ErrCode(), 0);
}

TEST_F(TestConvert, TestConvertTensor) {
  float data[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  // Create a tensor with wanted data type and shape
  std::vector<int64_t> dims{2, 2, 3};
  std::vector<int64_t> ge_dims{2, 2, 3};
  auto type_id = kNumberTypeFloat32;
  MeTensor me_tensor(type_id, dims);
  // Get the writable data pointer of the tensor and cast it to its data type
  uint8_t* me_data_ptr = reinterpret_cast<uint8_t*>(me_tensor.data_c());
  // Copy or use the writable data pointer of the ME tensor
  memcpy_s(me_data_ptr, me_tensor.data().nbytes(), data, 12 * sizeof(float));
  auto me_tensor_ptr = std::make_shared<MeTensor>(me_tensor);
  auto ge_tensor_ptr = TransformUtil::ConvertTensor(me_tensor_ptr, kOpFormat_NCHW);
  ASSERT_EQ(ge_tensor_ptr->GetTensorDesc().GetFormat(), GeFormat::FORMAT_NCHW);
  ASSERT_EQ(ge_tensor_ptr->GetTensorDesc().GetDataType(), GeDataType::DT_FLOAT);
  // ASSERT_EQ(ge_tensor_ptr->GetTensorDesc().array().GetDims(), ge_dims);
  int i = 0;
  for (i = 0; i < ge_dims.size(); i++) {
    ASSERT_EQ(ge_dims[i], ge_tensor_ptr->GetTensorDesc().GetShape().GetDims()[i]);
  }
  for (i = 0; i < ge_tensor_ptr->GetTensorDesc().GetShape().GetShapeSize(); i++) {
    ASSERT_EQ(data[i], (reinterpret_cast<float*>(ge_tensor_ptr->GetData()))[i]);
  }
}

TEST_F(TestConvert, TestConvertTensor0Dims) {
  // shape with 0 dims is also valid
  std::vector<int64_t> dims{};
  auto type_id = kNumberTypeFloat32;
  auto me_tensor_ptr = std::make_shared<MeTensor>(type_id, dims);
  ASSERT_NE(TransformUtil::ConvertTensor(me_tensor_ptr, kOpFormat_NCHW), nullptr);
}

TEST_F(TestConvert, TestConvertTensorError) {
  std::vector<int64_t> dims2{2, 3, 4};
  auto type_id_2 = kNumberTypeFloat32;
  auto me_tensor_ptr_2 = std::make_shared<MeTensor>(type_id_2, dims2);
  ASSERT_NE(TransformUtil::ConvertTensor(me_tensor_ptr_2, "xyz"), nullptr);
}

TEST_F(TestConvert, TestUtilsConvertDataType) {
  ASSERT_EQ(TransformUtil::ConvertDataType(MeDataType::kNumberTypeFloat16), GeDataType::DT_FLOAT16);
  ASSERT_EQ(TransformUtil::ConvertDataType(MeDataType::kNumberTypeFloat32), GeDataType::DT_FLOAT);
  ASSERT_EQ(TransformUtil::ConvertDataType(MeDataType::kNumberTypeFloat64), GeDataType::DT_DOUBLE);
  ASSERT_EQ(TransformUtil::ConvertDataType(MeDataType::kNumberTypeInt8), GeDataType::DT_INT8);
  ASSERT_EQ(TransformUtil::ConvertDataType(MeDataType::kNumberTypeInt16), GeDataType::DT_INT16);
  ASSERT_EQ(TransformUtil::ConvertDataType(MeDataType::kNumberTypeInt32), GeDataType::DT_INT32);
  ASSERT_EQ(TransformUtil::ConvertDataType(MeDataType::kNumberTypeInt64), GeDataType::DT_INT64);
  ASSERT_EQ(TransformUtil::ConvertDataType(MeDataType::kNumberTypeUInt32), GeDataType::DT_UINT32);
  ASSERT_EQ(TransformUtil::ConvertDataType(MeDataType::kNumberTypeBool), GeDataType::DT_BOOL);
}

TEST_F(TestConvert, TestUtilsConvertFormat) {
  ASSERT_EQ(TransformUtil::ConvertFormat(kOpFormat_NCHW), GeFormat::FORMAT_NCHW);
  ASSERT_EQ(TransformUtil::ConvertFormat(kOpFormat_NC1HWC0), GeFormat::FORMAT_NC1HWC0);
  ASSERT_EQ(TransformUtil::ConvertFormat(kOpFormat_NHWC), GeFormat::FORMAT_NHWC);
  ASSERT_EQ(TransformUtil::ConvertFormat("xyz"), GeFormat::FORMAT_ND);
}

TEST_F(TestConvert, TestUtilsDataSize) {
  ASSERT_EQ(TransformUtil::GetDataTypeSize(MeDataType::kNumberTypeFloat32), 4);
  ASSERT_EQ(TransformUtil::GetDataTypeSize(MeDataType::kNumberTypeFloat16), 2);
  ASSERT_EQ(TransformUtil::GetDataTypeSize(MeDataType::kNumberTypeFloat64), 8);
  ASSERT_EQ(TransformUtil::GetDataTypeSize(MeDataType::kNumberTypeInt8), 1);
  ASSERT_EQ(TransformUtil::GetDataTypeSize(MeDataType::kNumberTypeInt16), 2);
  ASSERT_EQ(TransformUtil::GetDataTypeSize(MeDataType::kNumberTypeInt32), 4);
  ASSERT_EQ(TransformUtil::GetDataTypeSize(MeDataType::kNumberTypeInt64), 8);
  ASSERT_EQ(TransformUtil::GetDataTypeSize(MeDataType::kNumberTypeUInt32), 4);
  ASSERT_EQ(TransformUtil::GetDataTypeSize(MeDataType::kNumberTypeBool), 1);
}

TEST_F(TestConvert, TestConvertGeTensor) {
#define DTYPE float
  ge::DataType dt = ge::DataType::DT_FLOAT;

  std::vector<float> data1 = {1.1, 2.2, 3.3, 4.4, 6.6, 7.7, 8.8, 9.9};
  std::vector<DTYPE> data2 = {1, 2, 3, 4, 6, 7, 8, 9};
  auto data = data1;
  ge::Shape shape({2, 2, 2});
  ge::Format format = ge::Format::FORMAT_NCHW;
  ge::TensorDesc desc(shape, format, dt);
  GeTensorPtr ge_tensor_ptr =
    std::make_shared<GeTensor>(desc, reinterpret_cast<uint8_t*>(data.data()), data.size() * sizeof(DTYPE));
  GeTensor& ge_tensor = *ge_tensor_ptr;
  const DTYPE* ge_data = reinterpret_cast<DTYPE*>(ge_tensor.GetData());

  // make sure GetData()'s return is a reference
  assert(ge_data == reinterpret_cast<DTYPE*>(ge_tensor.GetData()));

  cout << "ge data size is: " << std::dec << ge_tensor.GetSize() << " bytes" << endl;
  for (int i = 0; i < ge_tensor.GetSize() / sizeof(DTYPE); i++) {
    cout << "ge data is: " << static_cast<DTYPE>(*(ge_data + i)) << endl;
  }

  MeTensorPtr me_tensor_ptr = TransformUtil::ConvertGeTensor(ge_tensor_ptr);
  MeTensor& me_tensor = *me_tensor_ptr;
  cout << "after convert ge tensor to me tensor" << endl;
  DTYPE* me_data = reinterpret_cast<DTYPE*>(me_tensor.data_c());
  PrintMeTensor(&me_tensor);

  assert(ge_tensor.GetSize() == me_tensor.data().nbytes());
  assert(memcmp(ge_data, me_data, ge_tensor.GetSize()) == 0);
}

TEST_F(TestConvert, TestConvertMakeTuple) {
  FuncGraphPtr func_graph = std::make_shared<FuncGraph>();
  std::vector<AnfNodePtr> inputs;
  inputs.push_back(NewValueNode(std::make_shared<Primitive>("MakeTuple")));
  for (int i = 0; i < 3; i++) {
    auto input = func_graph->add_parameter();
    input->set_name("x" + std::to_string(i));
    inputs.push_back(input);
  }
  CNodePtr cnode_prim = func_graph->NewCNode(inputs);
  inputs.clear();
  inputs.push_back(NewValueNode(std::make_shared<Primitive>("Return")));
  inputs.push_back(cnode_prim);
  CNodePtr cnode_return = func_graph->NewCNode(inputs);
  func_graph->set_return(cnode_return);

  // save the func_graph to manager
  std::shared_ptr<FuncGraphManager> manager = Manage(func_graph);

  // call resolve
  bool ret_ = ResolveAll(manager);
  ASSERT_TRUE(ret_);

  // draw graph
  auto anfGraph = *(manager->func_graphs().begin());
  DfGraphConvertor converter(anfGraph);
  converter.ConvertAllNode().BuildGraph().GetComputeGraph();
  ASSERT_EQ(converter.ErrCode(), 0);
}

TEST_F(TestConvert, TestConvertInputTensors) {
#define DTYPE float
  std::initializer_list<int64_t> list0 = {1, 1, 4, 4};
  std::initializer_list<int64_t> list1 = {2, 3, 4, 5};
  std::initializer_list<int64_t> list2 = {9, 9, 1, 1};
  MeTensorPtr input_ptr1 = MakeTensor(kF32, list0);
  MeTensorPtr input_ptr2 = MakeTensor(kF32, list1);
  MeTensorPtr input_ptr3 = MakeTensor(kF32, list2);
  std::vector<MeTensorPtr> me_inputs;
  me_inputs.emplace_back(input_ptr1);
  me_inputs.emplace_back(input_ptr2);
  me_inputs.emplace_back(input_ptr3);

  std::vector<GeTensorPtr> ge_tensors = TransformUtil::ConvertInputTensors(me_inputs, kOpFormat_NCHW);

  for (int i = 0; i < ge_tensors.size(); i++) {
    DTYPE* me_data = reinterpret_cast<DTYPE*>(me_inputs[i]->data_c());
    const DTYPE* ge_data = reinterpret_cast<DTYPE*>(ge_tensors[i]->GetData());
    ASSERT_TRUE(ge_tensors[i]->GetSize() == me_inputs[i]->data().nbytes());
    ASSERT_EQ(memcmp(ge_data, me_data, ge_tensors[i]->GetSize()), 0);
    ASSERT_TRUE(ge_tensors[i]->GetTensorDesc().GetShape().GetDims() ==
                TransformUtil::ConvertMeShape(me_inputs[i]->shape_c()).GetDims());
  }
}

TEST_F(TestConvert, TestConvertGeTensors) {
#define DTYPE float
  ge::DataType dt = ge::DataType::DT_FLOAT;

  std::vector<float> data1(16);
  std::vector<float> data2(120);
  std::vector<float> data3(81);
  ge::Shape shape1({1, 1, 4, 4});
  ge::Shape shape2({2, 3, 4, 5});
  ge::Shape shape3({9, 9, 1, 1});
  ge::Format format = ge::Format::FORMAT_NCHW;
  ge::TensorDesc desc1(shape1, format, dt);
  ge::TensorDesc desc2(shape2, format, dt);
  ge::TensorDesc desc3(shape3, format, dt);
  GeTensorPtr ge_tensor_ptr1 =
    std::make_shared<GeTensor>(desc1, reinterpret_cast<uint8_t*>(data1.data()), data1.size() * sizeof(DTYPE));
  GeTensorPtr ge_tensor_ptr2 =
    std::make_shared<GeTensor>(desc2, reinterpret_cast<uint8_t*>(data2.data()), data2.size() * sizeof(DTYPE));
  GeTensorPtr ge_tensor_ptr3 =
    std::make_shared<GeTensor>(desc3, reinterpret_cast<uint8_t*>(data3.data()), data3.size() * sizeof(DTYPE));

  std::vector<GeTensorPtr> ge_tensors;
  ge_tensors.emplace_back(ge_tensor_ptr1);
  ge_tensors.emplace_back(ge_tensor_ptr2);
  ge_tensors.emplace_back(ge_tensor_ptr3);

  std::vector<std::vector<int64_t>> request_dims;
  std::vector<int64_t> dims1 = {1, 1, 4, 4};
  std::vector<int64_t> dims2 = {2, 3, 4, 5};
  std::vector<int64_t> dims3 = {9, 9, 1, 1};
  request_dims.emplace_back(dims1);
  request_dims.emplace_back(dims2);
  request_dims.emplace_back(dims3);

  std::vector<MeTensorPtr> me_outputs = TransformUtil::ConvertGeTensors(ge_tensors, request_dims);

  for (int i = 0; i < ge_tensors.size(); i++) {
    DTYPE* me_data = reinterpret_cast<DTYPE*>(me_outputs[i]->data_c());
    const DTYPE* ge_data = reinterpret_cast<DTYPE*>(ge_tensors[i]->GetData());
    ASSERT_TRUE(ge_tensors[i]->GetSize() == me_outputs[i]->data().nbytes());
    ASSERT_EQ(memcmp(ge_data, me_data, ge_tensors[i]->GetSize()), 0);
    ASSERT_TRUE(request_dims[i] == me_outputs[i]->shape_c());
  }
}

TEST_F(TestConvert, TestConvertGeShape1) {
  GeShape ge_shape({10, 1, 1, 1});
  std::vector<int64_t> request_dims{10};
  ASSERT_TRUE(TransformUtil::ConvertGeShape(ge_shape, request_dims) == request_dims);
}

TEST_F(TestConvert, TestConvertGeShape2) {
  GeShape ge_shape({10, 15, 1, 1});
  std::vector<int64_t> request_dims{10, 15};
  ASSERT_TRUE(TransformUtil::ConvertGeShape(ge_shape, request_dims) == request_dims);
}

TEST_F(TestConvert, TestConvertGeShape3) {
  GeShape ge_shape({10, 13, 18, 1});
  std::vector<int64_t> request_dims{10, 13, 18};
  ASSERT_TRUE(TransformUtil::ConvertGeShape(ge_shape, request_dims) == request_dims);
}

TEST_F(TestConvert, TestConvertGeShape4) {
  GeShape ge_shape({1, 10, 1, 1});
  std::vector<int64_t> request_dims{10};
  ASSERT_TRUE(TransformUtil::ConvertGeShape(ge_shape, request_dims) == request_dims);
}

TEST_F(TestConvert, TestConvertGeShape5) {
  GeShape ge_shape({10, 1, 1, 2});
  std::vector<int64_t> request_dims{10};
  ASSERT_TRUE(TransformUtil::ConvertGeShape(ge_shape, request_dims) == TransformUtil::ConvertGeShape(ge_shape));
}

TEST_F(TestConvert, TestConvertGeShape6) {
  GeShape ge_shape({5, 2, 1, 1});
  std::vector<int64_t> request_dims{10};
  ASSERT_TRUE(TransformUtil::ConvertGeShape(ge_shape, request_dims) == TransformUtil::ConvertGeShape(ge_shape));
}

TEST_F(TestConvert, TestConvertGeShape7) {
  GeShape ge_shape({10});
  std::vector<int64_t> request_dims{10, 1};
  ASSERT_TRUE(TransformUtil::ConvertGeShape(ge_shape, request_dims) == TransformUtil::ConvertGeShape(ge_shape));
}
}  // namespace transform
}  // namespace mindspore
