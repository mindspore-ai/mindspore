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
#include "common/backend_common_test.h"
#include "common/py_func_graph_fetcher.h"
#include "include/backend/kernel_info.h"
#include "include/common/debug/anf_ir_dump.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"

#define private public
#define protected public
#include "plugin/device/ascend/optimizer/ir_fusion/layer_norm_beta_gamma_backprop_fusion.h"
#undef private
#undef protected

namespace mindspore {
namespace opt {

class TestHWLayerNormBetaGammaBackpropFusion : public BackendCommon {
 public:
  TestHWLayerNormBetaGammaBackpropFusion()
      : get_py_fun_("gtest_input.pre_activate.layer_norm_beta_gamma_backprop_fusion_test", true) {}
  ~TestHWLayerNormBetaGammaBackpropFusion() override = default;

  UT::PyFuncGraphFetcher get_py_fun_;
};

class MockLayerNormBetaGammaBackpropFusionKernelQuery : public KernelQuery {
 public:
  MockLayerNormBetaGammaBackpropFusionKernelQuery() = default;
  ~MockLayerNormBetaGammaBackpropFusionKernelQuery() override = default;
  void Query(const CNodePtr &cnode, std::vector<std::shared_ptr<kernel::KernelBuildInfo>> *kernel_info_list) override {
    if (common::AnfAlgo::GetCNodeName(cnode) == kLayerNormBetaGammaBackpropOpName) {
      kernel::KernelBuildInfo::KernelBuildInfoBuilder builder;
      builder.SetInputsFormat({kOpFormat_DEFAULT, kOpFormat_DEFAULT, kOpFormat_DEFAULT, kOpFormat_DEFAULT});
      builder.SetOutputsFormat({kOpFormat_DEFAULT, kOpFormat_DEFAULT});
      builder.SetInputsDeviceType({kNumberTypeFloat16, kNumberTypeFloat16, kNumberTypeFloat16, kNumberTypeFloat16});
      builder.SetOutputsDeviceType({kNumberTypeFloat32, kNumberTypeFloat32});
      kernel_info_list->push_back(builder.Build());
    }
  }
};

TEST_F(TestHWLayerNormBetaGammaBackpropFusion, layernorm_beta_gamma_backprop_fusion_matched) {
  /*
   * def before(input0, input1, input2, input3):
   *     layer = LayerNormBetaGammaBackprop(input0, input1, input2, input3)
   *     output0 = Cast(tuple_getitem(layer, 0))
   *     output1 = Cast(tuple_getitem(layer, 1))
   *     add = Add(output0, output1)
   *     return add
   */
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_layer_norm_beta_gamma_backprop_fusion", "before");
  std::vector<int64_t> shp{2, 32, 224, 224};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  auto ret = g->get_return();
  EXPECT_NE(ret, nullptr);
  auto add = ret->input(1);
  EXPECT_NE(add, nullptr);
  auto cast0 = add->cast<CNodePtr>()->input(1);
  EXPECT_NE(cast0, nullptr);
  auto cast1 = add->cast<CNodePtr>()->input(2);
  EXPECT_NE(cast1, nullptr);
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder1;
  builder1.SetInputsFormat({kOpFormat_DEFAULT});
  builder1.SetOutputsFormat({kOpFormat_DEFAULT});
  builder1.SetInputsDeviceType({kNumberTypeFloat16});
  builder1.SetOutputsDeviceType({kNumberTypeFloat32});
  cast0->set_kernel_info(std::make_shared<device::KernelInfo>());
  cast1->set_kernel_info(std::make_shared<device::KernelInfo>());
  cast0->set_abstract(x_abstract);
  cast1->set_abstract(x_abstract);
  AnfAlgo::SetSelectKernelBuildInfo(builder1.Build(), cast0.get());
  AnfAlgo::SetSelectKernelBuildInfo(builder1.Build(), cast1.get());

  auto tuple_getitem0 = cast0->cast<CNodePtr>()->input(1);
  EXPECT_NE(tuple_getitem0, nullptr);
  auto layer = tuple_getitem0->cast<CNodePtr>()->input(1);
  EXPECT_NE(layer, nullptr);
  AbstractBasePtrList new_node_list{x_abstract, x_abstract};
  layer->set_abstract(std::make_shared<abstract::AbstractTuple>(new_node_list));

  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder2;
  builder2.SetInputsFormat({kOpFormat_DEFAULT, kOpFormat_DEFAULT, kOpFormat_DEFAULT, kOpFormat_DEFAULT});
  builder2.SetOutputsFormat({kOpFormat_DEFAULT, kOpFormat_DEFAULT});
  builder2.SetInputsDeviceType({kNumberTypeFloat16, kNumberTypeFloat16, kNumberTypeFloat16, kNumberTypeFloat16});
  builder2.SetOutputsDeviceType({kNumberTypeFloat16, kNumberTypeFloat16});
  layer->set_kernel_info(std::make_shared<device::KernelInfo>());
  AnfAlgo::SetSelectKernelBuildInfo(builder2.Build(), layer.get());
  common::AnfAlgo::SetNodeAttr("shape_gamma", MakeValue(""), layer);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  auto pass = std::make_shared<opt::LayerNormBetaGammaBackpropFusion>();
  pass->kernel_query_ = std::make_shared<MockLayerNormBetaGammaBackpropFusionKernelQuery>();
  pm->AddPass(pass);
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(g);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_layer_norm_beta_gamma_backprop_fusion", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}

TEST_F(TestHWLayerNormBetaGammaBackpropFusion, layernorm_beta_gamma_backprop_fusion_unmatched_inputs_size) {
  /*
   * def before(input0, input1, input2):
   *     layer = LayerNormBetaGammaBackprop(input0, input1, input2)
   *     output0 = Cast(tuple_getitem(layer, 0))
   *     output1 = Cast(tuple_getitem(layer, 1))
   *     add = Add(output0, output1)
   *     return add
   */
  FuncGraphPtr g =
    get_py_fun_.CallAndParseRet("test_layer_norm_beta_gamma_backprop_fusion", "before_unmatched_inputs_size");
  std::vector<int64_t> shp{2, 32, 224, 224};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  auto ret = g->get_return();
  EXPECT_NE(ret, nullptr);
  auto add = ret->input(1);
  EXPECT_NE(add, nullptr);
  auto cast0 = add->cast<CNodePtr>()->input(1);
  EXPECT_NE(cast0, nullptr);
  auto cast1 = add->cast<CNodePtr>()->input(2);
  EXPECT_NE(cast1, nullptr);
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder1;
  builder1.SetInputsFormat({kOpFormat_DEFAULT});
  builder1.SetOutputsFormat({kOpFormat_DEFAULT});
  builder1.SetInputsDeviceType({kNumberTypeFloat16});
  builder1.SetOutputsDeviceType({kNumberTypeFloat32});
  cast0->set_kernel_info(std::make_shared<device::KernelInfo>());
  cast1->set_kernel_info(std::make_shared<device::KernelInfo>());
  AnfAlgo::SetSelectKernelBuildInfo(builder1.Build(), cast0.get());
  AnfAlgo::SetSelectKernelBuildInfo(builder1.Build(), cast1.get());

  auto tuple_getitem0 = cast0->cast<CNodePtr>()->input(1);
  EXPECT_NE(tuple_getitem0, nullptr);
  auto layer = tuple_getitem0->cast<CNodePtr>()->input(1);
  EXPECT_NE(layer, nullptr);
  AbstractBasePtrList new_node_list{x_abstract, x_abstract};
  layer->set_abstract(std::make_shared<abstract::AbstractTuple>(new_node_list));

  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder2;
  builder2.SetInputsFormat({kOpFormat_DEFAULT, kOpFormat_DEFAULT, kOpFormat_DEFAULT, kOpFormat_DEFAULT});
  builder2.SetOutputsFormat({kOpFormat_DEFAULT, kOpFormat_DEFAULT});
  builder2.SetInputsDeviceType({kNumberTypeFloat16, kNumberTypeFloat16, kNumberTypeFloat16, kNumberTypeFloat16});
  builder2.SetOutputsDeviceType({kNumberTypeFloat16, kNumberTypeFloat16});
  layer->set_kernel_info(std::make_shared<device::KernelInfo>());
  AnfAlgo::SetSelectKernelBuildInfo(builder2.Build(), layer.get());
  common::AnfAlgo::SetNodeAttr("shape_gamma", MakeValue(""), layer);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  auto pass = std::make_shared<opt::LayerNormBetaGammaBackpropFusion>();
  pass->kernel_query_ = std::make_shared<MockLayerNormBetaGammaBackpropFusionKernelQuery>();
  pm->AddPass(pass);
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(g);

  EXPECT_TRUE(CheckEqualGraph(g, new_graph));
  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_layer_norm_beta_gamma_backprop_fusion", "after");
  EXPECT_FALSE(CheckEqualGraph(g_after, new_graph));
}

TEST_F(TestHWLayerNormBetaGammaBackpropFusion, layernorm_beta_gamma_backprop_fusion_unmatched_attr) {
  /*
   * def before(input0, input1, input2, input3):
   *     layer = LayerNormBetaGammaBackprop(input0, input1, input2, input3)
   *     output0 = Cast(tuple_getitem(layer, 0))
   *     output1 = Cast(tuple_getitem(layer, 1))
   *     add = Add(output0, output1)
   *     return add
   */
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_layer_norm_beta_gamma_backprop_fusion", "before");
  std::vector<int64_t> shp{2, 32, 224, 224};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  auto ret = g->get_return();
  EXPECT_NE(ret, nullptr);
  auto add = ret->input(1);
  EXPECT_NE(add, nullptr);
  auto cast0 = add->cast<CNodePtr>()->input(1);
  EXPECT_NE(cast0, nullptr);
  auto cast1 = add->cast<CNodePtr>()->input(2);
  EXPECT_NE(cast1, nullptr);
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder1;
  builder1.SetInputsFormat({kOpFormat_DEFAULT});
  builder1.SetOutputsFormat({kOpFormat_DEFAULT});
  builder1.SetInputsDeviceType({kNumberTypeFloat16});
  builder1.SetOutputsDeviceType({kNumberTypeFloat32});
  cast0->set_kernel_info(std::make_shared<device::KernelInfo>());
  cast1->set_kernel_info(std::make_shared<device::KernelInfo>());
  AnfAlgo::SetSelectKernelBuildInfo(builder1.Build(), cast0.get());
  AnfAlgo::SetSelectKernelBuildInfo(builder1.Build(), cast1.get());

  auto tuple_getitem0 = cast0->cast<CNodePtr>()->input(1);
  EXPECT_NE(tuple_getitem0, nullptr);
  auto layer = tuple_getitem0->cast<CNodePtr>()->input(1);
  EXPECT_NE(layer, nullptr);
  AbstractBasePtrList new_node_list{x_abstract, x_abstract};
  layer->set_abstract(std::make_shared<abstract::AbstractTuple>(new_node_list));

  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder2;
  builder2.SetInputsFormat({kOpFormat_DEFAULT, kOpFormat_DEFAULT, kOpFormat_DEFAULT, kOpFormat_DEFAULT});
  builder2.SetOutputsFormat({kOpFormat_DEFAULT, kOpFormat_DEFAULT});
  builder2.SetInputsDeviceType({kNumberTypeFloat16, kNumberTypeFloat16, kNumberTypeFloat16, kNumberTypeFloat16});
  builder2.SetOutputsDeviceType({kNumberTypeFloat16, kNumberTypeFloat16});
  layer->set_kernel_info(std::make_shared<device::KernelInfo>());
  AnfAlgo::SetSelectKernelBuildInfo(builder2.Build(), layer.get());
  common::AnfAlgo::EraseNodeAttr("shape_gamma", layer);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  auto pass = std::make_shared<opt::LayerNormBetaGammaBackpropFusion>();
  pass->kernel_query_ = std::make_shared<MockLayerNormBetaGammaBackpropFusionKernelQuery>();
  pm->AddPass(pass);
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(g);

  EXPECT_TRUE(CheckEqualGraph(g, new_graph));
  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_layer_norm_beta_gamma_backprop_fusion", "after");
  EXPECT_FALSE(CheckEqualGraph(g_after, new_graph));
}

TEST_F(TestHWLayerNormBetaGammaBackpropFusion, layernorm_beta_gamma_backprop_fusion_unmatched_outputs_size) {
  /*
   * def before(input0, input1, input2, input3):
   *     layer = LayerNormBetaGammaBackprop(input0, input1, input2, input3)
   *     output0 = Cast(layer)
   *     return output0
   */
  FuncGraphPtr g =
    get_py_fun_.CallAndParseRet("test_layer_norm_beta_gamma_backprop_fusion", "before_unmatched_outputs_size");
  std::vector<int64_t> shp{2, 32, 224, 224};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  auto ret = g->get_return();
  EXPECT_NE(ret, nullptr);
  auto cast = ret->input(1);
  EXPECT_NE(cast, nullptr);
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder1;
  builder1.SetInputsFormat({kOpFormat_DEFAULT});
  builder1.SetOutputsFormat({kOpFormat_DEFAULT});
  builder1.SetInputsDeviceType({kNumberTypeFloat16});
  builder1.SetOutputsDeviceType({kNumberTypeFloat32});
  cast->set_kernel_info(std::make_shared<device::KernelInfo>());
  AnfAlgo::SetSelectKernelBuildInfo(builder1.Build(), cast.get());

  auto layer = cast->cast<CNodePtr>()->input(1);
  EXPECT_NE(layer, nullptr);
  layer->set_abstract(x_abstract);

  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder2;
  builder2.SetInputsFormat({kOpFormat_DEFAULT, kOpFormat_DEFAULT, kOpFormat_DEFAULT, kOpFormat_DEFAULT});
  builder2.SetOutputsFormat({kOpFormat_DEFAULT, kOpFormat_DEFAULT});
  builder2.SetInputsDeviceType({kNumberTypeFloat16, kNumberTypeFloat16, kNumberTypeFloat16, kNumberTypeFloat16});
  builder2.SetOutputsDeviceType({kNumberTypeFloat16, kNumberTypeFloat16});
  layer->set_kernel_info(std::make_shared<device::KernelInfo>());
  AnfAlgo::SetSelectKernelBuildInfo(builder2.Build(), layer.get());
  common::AnfAlgo::SetNodeAttr("shape_gamma", MakeValue(""), layer);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  auto pass = std::make_shared<opt::LayerNormBetaGammaBackpropFusion>();
  pass->kernel_query_ = std::make_shared<MockLayerNormBetaGammaBackpropFusionKernelQuery>();
  pm->AddPass(pass);
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(g);

  EXPECT_TRUE(CheckEqualGraph(g, new_graph));
  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_layer_norm_beta_gamma_backprop_fusion", "after");
  EXPECT_FALSE(CheckEqualGraph(g_after, new_graph));
}

TEST_F(TestHWLayerNormBetaGammaBackpropFusion, layernorm_beta_gamma_backprop_fusion_unmatched_input_device_data_type) {
  /*
   * def before(input0, input1, input2, input3):
   *     layer = LayerNormBetaGammaBackprop(input0, input1, input2, input3)
   *     output0 = Cast(tuple_getitem(layer, 0))
   *     output1 = Cast(tuple_getitem(layer, 1))
   *     add = Add(output0, output1)
   *     return add
   */
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_layer_norm_beta_gamma_backprop_fusion", "before");
  std::vector<int64_t> shp{2, 32, 224, 224};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  auto ret = g->get_return();
  EXPECT_NE(ret, nullptr);
  auto add = ret->input(1);
  EXPECT_NE(add, nullptr);
  auto cast0 = add->cast<CNodePtr>()->input(1);
  EXPECT_NE(cast0, nullptr);
  auto cast1 = add->cast<CNodePtr>()->input(2);
  EXPECT_NE(cast1, nullptr);
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder1;
  builder1.SetInputsFormat({kOpFormat_DEFAULT});
  builder1.SetOutputsFormat({kOpFormat_DEFAULT});
  builder1.SetInputsDeviceType({kNumberTypeFloat16});
  builder1.SetOutputsDeviceType({kNumberTypeFloat32});
  cast0->set_kernel_info(std::make_shared<device::KernelInfo>());
  cast1->set_kernel_info(std::make_shared<device::KernelInfo>());
  AnfAlgo::SetSelectKernelBuildInfo(builder1.Build(), cast0.get());
  AnfAlgo::SetSelectKernelBuildInfo(builder1.Build(), cast1.get());

  auto tuple_getitem0 = cast0->cast<CNodePtr>()->input(1);
  EXPECT_NE(tuple_getitem0, nullptr);
  auto layer = tuple_getitem0->cast<CNodePtr>()->input(1);
  EXPECT_NE(layer, nullptr);
  AbstractBasePtrList new_node_list{x_abstract, x_abstract};
  layer->set_abstract(std::make_shared<abstract::AbstractTuple>(new_node_list));

  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder2;
  builder2.SetInputsFormat({kOpFormat_DEFAULT, kOpFormat_DEFAULT, kOpFormat_DEFAULT, kOpFormat_DEFAULT});
  builder2.SetOutputsFormat({kOpFormat_DEFAULT, kOpFormat_DEFAULT});
  builder2.SetInputsDeviceType({kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeFloat16, kNumberTypeFloat16});
  builder2.SetOutputsDeviceType({kNumberTypeFloat16, kNumberTypeFloat16});
  layer->set_kernel_info(std::make_shared<device::KernelInfo>());
  AnfAlgo::SetSelectKernelBuildInfo(builder2.Build(), layer.get());
  common::AnfAlgo::SetNodeAttr("shape_gamma", MakeValue(""), layer);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  auto pass = std::make_shared<opt::LayerNormBetaGammaBackpropFusion>();
  pass->kernel_query_ = std::make_shared<MockLayerNormBetaGammaBackpropFusionKernelQuery>();
  pm->AddPass(pass);
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(g);

  EXPECT_TRUE(CheckEqualGraph(g, new_graph));
  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_layer_norm_beta_gamma_backprop_fusion", "after");
  EXPECT_FALSE(CheckEqualGraph(g_after, new_graph));
}
}  // namespace opt
}  // namespace mindspore