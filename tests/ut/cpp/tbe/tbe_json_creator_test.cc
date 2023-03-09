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

#include "utils/ms_context.h"
#include "common/backend_common_test.h"
#include "common/py_func_graph_fetcher.h"
#include "include/common/debug/anf_ir_dump.h"
#include "kernel/kernel.h"
#include "include/backend/kernel_info.h"
#include "include/backend/optimizer/optimizer.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "plugin/device/ascend/optimizer/buffer_fusion/ub_pattern_fusion.h"
#include "plugin/device/ascend/kernel/tbe/tbe_json/single_tbe_json_creator.h"
#include "plugin/device/ascend/kernel/tbe/tbe_json/fusion_tbe_json_creator.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore::kernel {

using KernelBuildInfoBuilder = kernel::KernelBuildInfo::KernelBuildInfoBuilder;
constexpr int64_t kShape4D = 4;
class TestHWTBEJsonCreator : public BackendCommon {
 public:
  TestHWTBEJsonCreator() : get_py_fun_("gtest_input.tbe.tbe_json_creator_test", true) {}
  ~TestHWTBEJsonCreator() override = default;

  UT::PyFuncGraphFetcher get_py_fun_;
};

TEST_F(TestHWTBEJsonCreator, DISABLED_test_tbe_single_common) {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  context->set_param<std::string>(MS_CTX_DEVICE_TARGET, kAscendDevice);
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_tbe_json_creator", "func_relu_relu_cast");
  std::vector<int64_t> shp{2, 32, 224, 224};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  AbstractBasePtrList args_spec_list{x_abstract};
  auto kg = GetKernelGraph(g, args_spec_list);

  auto ret = kg->get_return();
  EXPECT_NE(ret->input(1), nullptr);
  auto tuple = ret->input(1);
  EXPECT_NE(tuple, nullptr);
  auto cast = tuple->cast<CNodePtr>()->input(1);
  EXPECT_NE(cast, nullptr);
  auto relu2 = cast->cast<CNodePtr>()->input(1);
  EXPECT_NE(relu2, nullptr);
  auto relu1 = relu2->cast<CNodePtr>()->input(1);

  KernelBuildInfoBuilder builder;
  builder.SetInputsFormat({"NC1HWC0"});
  builder.SetOutputsFormat({"NC1HWC0"});
  builder.SetInputsDeviceType({kFloat32->type_id()});
  builder.SetOutputsDeviceType({kFloat32->type_id()});
  builder.SetKernelType(KernelType::TBE_KERNEL);
  builder.SetFusionType(kernel::kPatternElemWise);
  builder.SetProcessor(kernel::Processor::AICORE);
  builder.SetKernelType(KernelType::TBE_KERNEL);

  relu1->set_kernel_info(std::make_shared<device::KernelInfo>());
  AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), relu1.get());

  auto tbe_json_creator_select = std::make_shared<SelectTbeJsonCreator>();
  auto tbe_json_creator_build = std::make_shared<BuildTbeJsonCreator>();
  nlohmann::json kernel_json;
  EXPECT_TRUE(tbe_json_creator_select->GenJson(relu1, &kernel_json));
  EXPECT_EQ(tbe_json_creator_select->GetJsonHash(), 12207851473833394607U)
    << "Error json is:" << kernel_json << ", for expected json, see file: tbe_single_common_select.json";
  EXPECT_TRUE(tbe_json_creator_build->GenJson(relu1, &kernel_json));
  EXPECT_EQ(tbe_json_creator_build->GetJsonHash(), 2389029245513168162U)
    << "Error json is:" << kernel_json << ", for expected json, see file: tbe_single_common_build.json";
}

TEST_F(TestHWTBEJsonCreator, DISABLED_test_tbe_single_conv2d_backprop_filter) {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  context->set_param<std::string>(MS_CTX_DEVICE_TARGET, kAscendDevice);
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_tbe_json_creator", "func_conv2d_backprop_filter");
  std::vector<int64_t> shp{2, 32, 224, 224};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  AbstractBasePtrList abstract_list = {std::make_shared<abstract::AbstractScalar>(kShape4D)};
  auto y_abstract = std::make_shared<abstract::AbstractTuple>(abstract_list);
  AbstractBasePtrList args_spec_list{x_abstract, x_abstract, y_abstract};
  auto kg = GetKernelGraph(g, args_spec_list);

  auto ret = kg->get_return();
  EXPECT_NE(ret->input(1), nullptr);
  auto tuple = ret->input(1);
  EXPECT_NE(tuple, nullptr);
  auto conv2d_backprop_filter = tuple->cast<CNodePtr>()->input(1);

  KernelBuildInfoBuilder builder;
  builder.SetInputsFormat({"NC1HWC0", "NC1HWC0"});
  builder.SetOutputsFormat({"NC1HWC0"});
  builder.SetInputsDeviceType({kFloat32->type_id(), kFloat32->type_id()});
  builder.SetOutputsDeviceType({kFloat32->type_id()});
  builder.SetKernelType(KernelType::TBE_KERNEL);
  builder.SetFusionType(kernel::kPatternElemWise);
  builder.SetProcessor(kernel::Processor::AICORE);
  builder.SetKernelType(KernelType::TBE_KERNEL);

  conv2d_backprop_filter->set_kernel_info(std::make_shared<device::KernelInfo>());
  AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), conv2d_backprop_filter.get());

  auto tbe_json_creator_select = std::make_shared<SelectTbeJsonCreator>();
  auto tbe_json_creator_build = std::make_shared<BuildTbeJsonCreator>();
  nlohmann::json kernel_json;
  EXPECT_TRUE(tbe_json_creator_select->GenJson(conv2d_backprop_filter, &kernel_json));
  EXPECT_EQ(tbe_json_creator_select->GetJsonHash(), 14683931476519216146U)
    << "Error json is:" << kernel_json
    << ", for expected json, see file: tbe_single_conv2d_backprop_filter_select.json";
  EXPECT_TRUE(tbe_json_creator_build->GenJson(conv2d_backprop_filter, &kernel_json));
  EXPECT_EQ(tbe_json_creator_build->GetJsonHash(), 6097606936200506174U)
    << "Error json is:" << kernel_json
    << ", for expected json, see file: tbe_single_conv2d_backprop_filter_build.json";
}

TEST_F(TestHWTBEJsonCreator, DISABLED_test_tbe_single_dynamic_rnn) {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  context->set_param<std::string>(MS_CTX_DEVICE_TARGET, kAscendDevice);
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_tbe_json_creator", "func_dynamic_rnn");
  std::vector<int64_t> x_shp{2, 16, 64};
  std::vector<int64_t> w_shp{96, 128};
  std::vector<int64_t> b_shp{128};
  std::vector<int64_t> init_h_shp{1, 16, 32};
  std::vector<int64_t> init_c_shp{1, 16, 32};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat16, x_shp);
  auto w_abstract = std::make_shared<abstract::AbstractTensor>(kFloat16, w_shp);
  auto b_abstract = std::make_shared<abstract::AbstractTensor>(kFloat16, b_shp);
  auto init_h_abstract = std::make_shared<abstract::AbstractTensor>(kFloat16, init_h_shp);
  auto init_c_abstract = std::make_shared<abstract::AbstractTensor>(kFloat16, init_c_shp);
  auto seq_length_abstract = std::make_shared<abstract::AbstractNone>();

  AbstractBasePtrList args_spec_list{x_abstract,          w_abstract,      b_abstract,
                                     seq_length_abstract, init_h_abstract, init_c_abstract};
  auto kg = GetKernelGraph(g, args_spec_list);

  auto ret = kg->get_return();
  EXPECT_NE(ret->input(1), nullptr);
  auto tuple = ret->input(1);
  EXPECT_NE(tuple, nullptr);
  auto make_tuple = tuple->cast<CNodePtr>()->input(1);
  EXPECT_NE(tuple, nullptr);
  auto tuple2 = make_tuple->cast<CNodePtr>()->input(1);
  EXPECT_NE(tuple2, nullptr);
  auto dynamic_rnn = tuple2->cast<CNodePtr>()->input(1);

  KernelBuildInfoBuilder builder;
  builder.SetInputsFormat({"NC1HWC0", "NC1HWC0", "NC1HWC0", "NC1HWC0", "NC1HWC0", "NC1HWC0"});
  builder.SetOutputsFormat({"NC1HWC0", "NC1HWC0", "NC1HWC0", "NC1HWC0", "NC1HWC0", "NC1HWC0", "NC1HWC0", "NC1HWC0"});
  builder.SetInputsDeviceType({kFloat16->type_id(), kFloat16->type_id(), kFloat16->type_id(), kFloat16->type_id(),
                               kFloat16->type_id(), kFloat16->type_id()});
  builder.SetOutputsDeviceType({kFloat16->type_id(), kFloat16->type_id(), kFloat16->type_id(), kFloat16->type_id(),
                                kFloat16->type_id(), kFloat16->type_id(), kFloat16->type_id(), kFloat16->type_id()});
  builder.SetKernelType(KernelType::TBE_KERNEL);
  builder.SetFusionType(kernel::kPatternElemWise);
  builder.SetProcessor(kernel::Processor::AICORE);
  builder.SetKernelType(KernelType::TBE_KERNEL);

  dynamic_rnn->set_kernel_info(std::make_shared<device::KernelInfo>());
  AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), dynamic_rnn.get());

  auto tbe_json_creator_select = std::make_shared<SelectTbeJsonCreator>();
  auto tbe_json_creator_build = std::make_shared<BuildTbeJsonCreator>();
  nlohmann::json kernel_json;
  EXPECT_TRUE(tbe_json_creator_select->GenJson(dynamic_rnn, &kernel_json));
  EXPECT_EQ(tbe_json_creator_select->GetJsonHash(), 16143536111232395651U)
    << "Error json is:" << kernel_json << ", for expected json, see file: tbe_single_dynamic_rnn_select.json";
  EXPECT_TRUE(tbe_json_creator_build->GenJson(dynamic_rnn, &kernel_json));
  EXPECT_EQ(tbe_json_creator_build->GetJsonHash(), 14916511955212123861U)
    << "Error json is:" << kernel_json << ", for expected json, see file: tbe_single_dynamic_rnn_build.json";
}

TEST_F(TestHWTBEJsonCreator, DISABLED_test_tbe_single_layer_norm) {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  context->set_param<std::string>(MS_CTX_DEVICE_TARGET, kAscendDevice);
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_tbe_json_creator", "func_layer_norm");
  std::vector<int64_t> x_shp{2, 3};
  std::vector<int64_t> gamma_shp{3};
  std::vector<int64_t> beta_shp{3};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, x_shp);
  auto gamma_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, gamma_shp);
  auto beta_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, beta_shp);

  AbstractBasePtrList args_spec_list{
    x_abstract,
    gamma_abstract,
    beta_abstract,
  };
  auto kg = GetKernelGraph(g, args_spec_list);

  auto ret = kg->get_return();
  EXPECT_NE(ret->input(1), nullptr);
  auto tuple = ret->input(1);
  EXPECT_NE(tuple, nullptr);
  auto make_tuple = tuple->cast<CNodePtr>()->input(1);
  EXPECT_NE(tuple, nullptr);
  auto tuple2 = make_tuple->cast<CNodePtr>()->input(1);
  EXPECT_NE(tuple2, nullptr);
  auto layer_norm = tuple2->cast<CNodePtr>()->input(1);

  KernelBuildInfoBuilder builder;
  builder.SetInputsFormat({"NC1HWC0", "NC1HWC0", "NC1HWC0"});
  builder.SetOutputsFormat({"NC1HWC0", "NC1HWC0", "NC1HWC0"});
  builder.SetInputsDeviceType({kFloat32->type_id(), kFloat32->type_id(), kFloat32->type_id()});
  builder.SetOutputsDeviceType({kFloat32->type_id(), kFloat32->type_id(), kFloat32->type_id()});
  builder.SetKernelType(KernelType::TBE_KERNEL);
  builder.SetFusionType(kernel::kPatternElemWise);
  builder.SetProcessor(kernel::Processor::AICORE);
  builder.SetKernelType(KernelType::TBE_KERNEL);

  layer_norm->set_kernel_info(std::make_shared<device::KernelInfo>());
  AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), layer_norm.get());

  auto tbe_json_creator_select = std::make_shared<SelectTbeJsonCreator>();
  auto tbe_json_creator_build = std::make_shared<BuildTbeJsonCreator>();
  nlohmann::json kernel_json;
  EXPECT_TRUE(tbe_json_creator_select->GenJson(layer_norm, &kernel_json));
  EXPECT_EQ(tbe_json_creator_select->GetJsonHash(), 1161191001728520611U)
    << "Error json is:" << kernel_json << ", for expected json, see file: tbe_single_layer_norm_select.json";
  EXPECT_TRUE(tbe_json_creator_build->GenJson(layer_norm, &kernel_json));
  EXPECT_EQ(tbe_json_creator_build->GetJsonHash(), 2848618249728529296U)
    << "Error json is:" << kernel_json << ", for expected json, see file: tbe_single_layer_norm_build.json";
}

TEST_F(TestHWTBEJsonCreator, test_tbe_fusion_common) {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  context->set_param<std::string>(MS_CTX_DEVICE_TARGET, kAscendDevice);
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_tbe_json_creator", "func_relu_relu_cast");
  std::vector<int64_t> shp{2, 32, 224, 224};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  AbstractBasePtrList args_spec_list{x_abstract};
  auto kg = GetKernelGraph(g, args_spec_list);

  auto ret = kg->get_return();
  EXPECT_NE(ret->input(1), nullptr);
  auto tuple = ret->input(1);
  EXPECT_NE(tuple, nullptr);
  auto cast = tuple->cast<CNodePtr>()->input(1);
  EXPECT_NE(cast, nullptr);
  auto relu2 = cast->cast<CNodePtr>()->input(1);
  EXPECT_NE(relu2, nullptr);
  auto relu1 = relu2->cast<CNodePtr>()->input(1);

  KernelBuildInfoBuilder builder;
  builder.SetInputsFormat({"NC1HWC0"});
  builder.SetOutputsFormat({"NC1HWC0"});
  builder.SetInputsDeviceType({kFloat32->type_id()});
  builder.SetOutputsDeviceType({kFloat32->type_id()});
  builder.SetKernelType(KernelType::TBE_KERNEL);
  builder.SetFusionType(kernel::kPatternElemWise);
  builder.SetProcessor(kernel::Processor::AICORE);
  builder.SetKernelType(KernelType::TBE_KERNEL);

  relu1->set_kernel_info(std::make_shared<device::KernelInfo>());
  relu2->set_kernel_info(std::make_shared<device::KernelInfo>());
  AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), relu1.get());
  AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), relu2.get());

  KernelBuildInfoBuilder builder1;
  builder1.SetInputsFormat({"NC1HWC0"});
  builder1.SetOutputsFormat({"NC1HWC0"});
  builder1.SetInputsDeviceType({kFloat32->type_id()});
  builder1.SetOutputsDeviceType({kFloat16->type_id()});
  builder1.SetKernelType(KernelType::TBE_KERNEL);
  builder1.SetFusionType(kernel::kPatternOpaque);
  builder1.SetProcessor(kernel::Processor::AICORE);
  builder1.SetKernelType(KernelType::TBE_KERNEL);

  cast->set_kernel_info(std::make_shared<device::KernelInfo>());
  AnfAlgo::SetSelectKernelBuildInfo(builder1.Build(), cast.get());

  std::vector<AnfNodePtr> input_nodes;
  std::vector<AnfNodePtr> compute_nodes = {relu1, relu2};
  std::string full_name =
    "FusionOp_" + common::AnfAlgo::GetCNodeName(relu1) + "_" + common::AnfAlgo::GetCNodeName(relu2);
  for (auto &node : compute_nodes) {
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    for (size_t idx = 1; idx < cnode->inputs().size(); ++idx) {
      auto real_input = common::AnfAlgo::VisitKernel(cnode->input(idx), 0);
      if (std::find(compute_nodes.begin(), compute_nodes.end(), real_input.first) == compute_nodes.end()) {
        if (auto in = cnode->input(idx); std::find(input_nodes.begin(), input_nodes.end(), in) == input_nodes.end()) {
          input_nodes.push_back(in);
        }
      }
    }
  }

  FusionScopeInfo fusion_scope_info(0, full_name, "", input_nodes, compute_nodes, {});
  nlohmann::json fusion_json;
  auto tbe_json_creator = std::make_shared<FusionBuildTbeJsonCreator>();
  EXPECT_TRUE(tbe_json_creator->GenJson(fusion_scope_info, &fusion_json));
  EXPECT_EQ(tbe_json_creator->GetJsonHash(), 18379117451241093022U)
    << "Error json is:" << fusion_json << ", for expected json, see file: tbe_fusion_common.json";
}

TEST_F(TestHWTBEJsonCreator, test_fusion_add_conv2d) {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  context->set_param<std::string>(MS_CTX_DEVICE_TARGET, kAscendDevice);
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_tbe_json_creator", "fusion_add_conv2d");
  std::vector<int64_t> x_shp{10, 32, 32, 32};
  std::vector<int64_t> z_shp{32, 32, 3, 3};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, x_shp);
  auto z_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, z_shp);
  AbstractBasePtrList args_spec_list{x_abstract, x_abstract, z_abstract};
  auto kg = GetKernelGraph(g, args_spec_list);

  auto ret = kg->get_return();
  EXPECT_NE(ret->input(1), nullptr);
  auto tuple = ret->input(1);
  EXPECT_NE(tuple, nullptr);
  auto conv2d = tuple->cast<CNodePtr>()->input(1);
  EXPECT_NE(conv2d, nullptr);
  auto add = conv2d->cast<CNodePtr>()->input(1);
  EXPECT_NE(add, nullptr);

  KernelBuildInfoBuilder builder;
  builder.SetInputsFormat({"NC1HWC0", "NC1HWC0"});
  builder.SetOutputsFormat({"NC1HWC0"});
  builder.SetInputsDeviceType({kFloat32->type_id(), kFloat32->type_id()});
  builder.SetOutputsDeviceType({kFloat32->type_id()});
  builder.SetKernelType(KernelType::TBE_KERNEL);
  builder.SetFusionType(kernel::kPatternElemWise);
  builder.SetProcessor(kernel::Processor::AICORE);
  builder.SetKernelType(KernelType::TBE_KERNEL);

  add->set_kernel_info(std::make_shared<device::KernelInfo>());
  AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), add.get());
  conv2d->set_kernel_info(std::make_shared<device::KernelInfo>());
  AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), conv2d.get());

  std::vector<AnfNodePtr> input_nodes;
  std::vector<AnfNodePtr> compute_nodes = {add, conv2d};
  std::string full_name =
    "FusionOp_" + common::AnfAlgo::GetCNodeName(add) + "_" + common::AnfAlgo::GetCNodeName(conv2d);
  for (auto &node : compute_nodes) {
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    for (size_t idx = 1; idx < cnode->inputs().size(); ++idx) {
      auto real_input = common::AnfAlgo::VisitKernel(cnode->input(idx), 0);
      if (std::find(compute_nodes.begin(), compute_nodes.end(), real_input.first) == compute_nodes.end()) {
        if (auto in = cnode->input(idx); std::find(input_nodes.begin(), input_nodes.end(), in) == input_nodes.end()) {
          input_nodes.push_back(in);
        }
      }
    }
  }

  FusionScopeInfo fusion_scope_info(0, full_name, "", input_nodes, compute_nodes, {});
  nlohmann::json fusion_json;
  auto tbe_json_creator = std::make_shared<FusionBuildTbeJsonCreator>();
  EXPECT_TRUE(tbe_json_creator->GenJson(fusion_scope_info, &fusion_json));
  EXPECT_EQ(tbe_json_creator->GetJsonHash(), 16132617067967162574U)
    << "Error json is:" << fusion_json << ", for expected json, see file: test_fusion_add_conv2d.json";
}

}  // namespace mindspore::kernel
