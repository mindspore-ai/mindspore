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
#include "debug/anf_ir_dump.h"
#include "kernel/kernel.h"
#include "device/kernel_info.h"
#include "pre_activate/common/optimizer.h"
#include "session/anf_runtime_algorithm.h"
#include "pre_activate/ascend/buffer_fusion/buffer_fusion.h"

namespace mindspore {
namespace opt {
using KernelBuildInfoBuilder = kernel::KernelBuildInfo::KernelBuildInfoBuilder;
class TestHWBufferFusion : public BackendCommon {
 public:
  TestHWBufferFusion() : get_py_fun_("gtest_input.pre_activate.buffer_fusion_test", true) {}
  ~TestHWBufferFusion() override = default;

  UT::PyFuncGraphFetcher get_py_fun_;
};

TEST_F(TestHWBufferFusion, test_tbe_eltwise_fusion_1) {
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_tbe_eltwise_fusion_1", "before");
  std::vector<int> shp{2, 32, 224, 224};
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
  builder.SetFusionType(kernel::FusionType::ELEMWISE);
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
  builder1.SetFusionType(kernel::FusionType::OPAQUE);
  builder1.SetProcessor(kernel::Processor::AICORE);
  builder1.SetKernelType(KernelType::TBE_KERNEL);

  cast->set_kernel_info(std::make_shared<device::KernelInfo>());
  AnfAlgo::SetSelectKernelBuildInfo(builder1.Build(), cast.get());

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  auto buffer_fusion_pass = std::make_shared<opt::BufferFusion>();
  pm->AddPass(buffer_fusion_pass);
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(kg);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_tbe_eltwise_fusion_1", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}

TEST_F(TestHWBufferFusion, test_tbe_eltwise_fusion_2) {
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_tbe_eltwise_fusion_2", "before");
  std::vector<int> shp{32, 10};
  std::vector<int> shp_bias{10};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  auto y_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_bias);
  AbstractBasePtrList args_spec_list{x_abstract, y_abstract};
  auto kg = GetKernelGraph(g, args_spec_list);

  auto ret = kg->get_return();
  EXPECT_NE(ret->input(1), nullptr);
  auto tuple = ret->input(1);
  EXPECT_NE(tuple, nullptr);
  auto cast = tuple->cast<CNodePtr>()->input(1);
  EXPECT_NE(cast, nullptr);
  auto relu6 = cast->cast<CNodePtr>()->input(1);
  EXPECT_NE(relu6, nullptr);
  auto relu5 = relu6->cast<CNodePtr>()->input(1);
  EXPECT_NE(relu5, nullptr);
  auto relu4 = relu5->cast<CNodePtr>()->input(1);
  EXPECT_NE(relu4, nullptr);
  auto biasadd = relu4->cast<CNodePtr>()->input(1);
  EXPECT_NE(biasadd, nullptr);
  auto relu3 = biasadd->cast<CNodePtr>()->input(1);
  EXPECT_NE(relu3, nullptr);
  auto relu2 = relu3->cast<CNodePtr>()->input(1);
  EXPECT_NE(relu2, nullptr);
  auto relu1 = relu2->cast<CNodePtr>()->input(1);

  KernelBuildInfoBuilder builder;
  builder.SetInputsFormat({"NC1HWC0"});
  builder.SetOutputsFormat({"NC1HWC0"});
  builder.SetInputsDeviceType({kFloat32->type_id()});
  builder.SetOutputsDeviceType({kFloat32->type_id()});
  builder.SetKernelType(KernelType::TBE_KERNEL);
  builder.SetFusionType(kernel::FusionType::ELEMWISE);
  builder.SetProcessor(kernel::Processor::AICORE);
  builder.SetKernelType(KernelType::TBE_KERNEL);

  relu1->set_kernel_info(std::make_shared<device::KernelInfo>());
  relu2->set_kernel_info(std::make_shared<device::KernelInfo>());
  relu3->set_kernel_info(std::make_shared<device::KernelInfo>());
  relu4->set_kernel_info(std::make_shared<device::KernelInfo>());
  relu5->set_kernel_info(std::make_shared<device::KernelInfo>());
  relu6->set_kernel_info(std::make_shared<device::KernelInfo>());
  AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), relu1.get());
  AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), relu2.get());
  AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), relu3.get());
  AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), relu4.get());
  AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), relu5.get());
  AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), relu6.get());

  KernelBuildInfoBuilder builder1;
  builder1.SetInputsFormat({"NC1HWC0"});
  builder1.SetOutputsFormat({"NC1HWC0"});
  builder1.SetInputsDeviceType({kFloat32->type_id()});
  builder1.SetOutputsDeviceType({kFloat16->type_id()});
  builder1.SetKernelType(KernelType::TBE_KERNEL);
  builder1.SetFusionType(kernel::FusionType::OPAQUE);
  builder1.SetProcessor(kernel::Processor::AICORE);
  builder1.SetKernelType(KernelType::TBE_KERNEL);

  cast->set_kernel_info(std::make_shared<device::KernelInfo>());
  AnfAlgo::SetSelectKernelBuildInfo(builder1.Build(), cast.get());

  KernelBuildInfoBuilder builder2;
  builder2.SetInputsFormat({"NC1HWC0", "NC1HWC0"});
  builder2.SetOutputsFormat({"NC1HWC0"});
  builder2.SetInputsDeviceType({kFloat32->type_id(), kFloat32->type_id()});
  builder2.SetOutputsDeviceType({kFloat32->type_id()});
  builder2.SetKernelType(KernelType::TBE_KERNEL);
  builder2.SetFusionType(kernel::FusionType::COMMREDUCE);
  builder2.SetProcessor(kernel::Processor::AICORE);
  builder2.SetKernelType(KernelType::TBE_KERNEL);

  biasadd->set_kernel_info(std::make_shared<device::KernelInfo>());
  AnfAlgo::SetSelectKernelBuildInfo(builder2.Build(), biasadd.get());

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  auto buffer_fusion_pass = std::make_shared<opt::BufferFusion>();
  pm->AddPass(buffer_fusion_pass);
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(kg);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_tbe_eltwise_fusion_2", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}

TEST_F(TestHWBufferFusion, test_tbe_reduce_eltwise_fusion) {
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_tbe_reduce_eltwise_fusion", "before");
  std::vector<int> shp{32, 10};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  AbstractBasePtrList args_spec_list{x_abstract};
  auto kg = GetKernelGraph(g, args_spec_list);

  auto ret = kg->get_return();
  EXPECT_NE(ret->input(1), nullptr);
  auto tuple = ret->input(1);
  EXPECT_NE(tuple, nullptr);
  auto cast = tuple->cast<CNodePtr>()->input(1);
  EXPECT_NE(cast, nullptr);
  auto relu6 = cast->cast<CNodePtr>()->input(1);
  EXPECT_NE(relu6, nullptr);
  auto relu5 = relu6->cast<CNodePtr>()->input(1);
  EXPECT_NE(relu5, nullptr);
  auto relu4 = relu5->cast<CNodePtr>()->input(1);
  EXPECT_NE(relu4, nullptr);
  auto biasaddgrad = relu4->cast<CNodePtr>()->input(1);
  EXPECT_NE(biasaddgrad, nullptr);
  auto relu3 = biasaddgrad->cast<CNodePtr>()->input(1);
  EXPECT_NE(relu3, nullptr);
  auto relu2 = relu3->cast<CNodePtr>()->input(1);
  EXPECT_NE(relu2, nullptr);
  auto relu1 = relu2->cast<CNodePtr>()->input(1);

  KernelBuildInfoBuilder builder;
  builder.SetInputsFormat({"NC1HWC0"});
  builder.SetOutputsFormat({"NC1HWC0"});
  builder.SetInputsDeviceType({kFloat32->type_id()});
  builder.SetOutputsDeviceType({kFloat32->type_id()});
  builder.SetKernelType(KernelType::TBE_KERNEL);
  builder.SetFusionType(kernel::FusionType::ELEMWISE);
  builder.SetProcessor(kernel::Processor::AICORE);
  builder.SetKernelType(KernelType::TBE_KERNEL);

  relu1->set_kernel_info(std::make_shared<device::KernelInfo>());
  relu2->set_kernel_info(std::make_shared<device::KernelInfo>());
  relu3->set_kernel_info(std::make_shared<device::KernelInfo>());
  relu4->set_kernel_info(std::make_shared<device::KernelInfo>());
  relu5->set_kernel_info(std::make_shared<device::KernelInfo>());
  relu6->set_kernel_info(std::make_shared<device::KernelInfo>());
  AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), relu1.get());
  AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), relu2.get());
  AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), relu3.get());
  AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), relu4.get());
  AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), relu5.get());
  AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), relu6.get());

  KernelBuildInfoBuilder builder1;
  builder1.SetInputsFormat({"NC1HWC0"});
  builder1.SetOutputsFormat({"NC1HWC0"});
  builder1.SetInputsDeviceType({kFloat32->type_id()});
  builder1.SetOutputsDeviceType({kFloat16->type_id()});
  builder1.SetKernelType(KernelType::TBE_KERNEL);
  builder1.SetFusionType(kernel::FusionType::OPAQUE);
  builder1.SetProcessor(kernel::Processor::AICORE);
  builder1.SetKernelType(KernelType::TBE_KERNEL);

  cast->set_kernel_info(std::make_shared<device::KernelInfo>());
  AnfAlgo::SetSelectKernelBuildInfo(builder1.Build(), cast.get());

  KernelBuildInfoBuilder builder2;
  builder2.SetInputsFormat({"NC1HWC0"});
  builder2.SetOutputsFormat({"NC1HWC0"});
  builder2.SetInputsDeviceType({kFloat32->type_id()});
  builder2.SetOutputsDeviceType({kFloat32->type_id()});
  builder2.SetKernelType(KernelType::TBE_KERNEL);
  builder2.SetFusionType(kernel::FusionType::COMMREDUCE);
  builder2.SetProcessor(kernel::Processor::AICORE);
  builder2.SetKernelType(KernelType::TBE_KERNEL);

  biasaddgrad->set_kernel_info(std::make_shared<device::KernelInfo>());
  AnfAlgo::SetSelectKernelBuildInfo(builder2.Build(), biasaddgrad.get());

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  auto buffer_fusion_pass = std::make_shared<opt::BufferFusion>();
  pm->AddPass(buffer_fusion_pass);
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(kg);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_tbe_reduce_eltwise_fusion", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}

TEST_F(TestHWBufferFusion, test_tbe_matmul_eltwise_fusion) {
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_tbe_matmul_eltwise_fusion", "before");
  std::vector<int> x_shp{2048, 768};
  std::vector<int> y_shp{768, 768};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, x_shp);
  auto y_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, y_shp);
  AbstractBasePtrList args_spec_list{x_abstract, y_abstract};
  auto kg = GetKernelGraph(g, args_spec_list);

  auto ret = kg->get_return();
  EXPECT_NE(ret->input(1), nullptr);
  auto tuple = ret->input(1);
  EXPECT_NE(tuple, nullptr);
  auto cast = tuple->cast<CNodePtr>()->input(1);
  EXPECT_NE(cast, nullptr);
  auto relu = cast->cast<CNodePtr>()->input(1);
  EXPECT_NE(relu, nullptr);
  auto matmul = relu->cast<CNodePtr>()->input(1);

  KernelBuildInfoBuilder builder;
  builder.SetInputsFormat({"NC1HWC0"});
  builder.SetOutputsFormat({"NC1HWC0"});
  builder.SetInputsDeviceType({kFloat32->type_id()});
  builder.SetOutputsDeviceType({kFloat32->type_id()});
  builder.SetKernelType(KernelType::TBE_KERNEL);
  builder.SetFusionType(kernel::FusionType::ELEMWISE);
  builder.SetProcessor(kernel::Processor::AICORE);
  builder.SetKernelType(KernelType::TBE_KERNEL);
  relu->set_kernel_info(std::make_shared<device::KernelInfo>());
  AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), relu.get());

  KernelBuildInfoBuilder builder2;
  builder2.SetInputsFormat({"NC1HWC0", "NC1HWC0"});
  builder2.SetOutputsFormat({"NC1HWC0"});
  builder2.SetInputsDeviceType({kFloat32->type_id(), kFloat32->type_id()});
  builder2.SetOutputsDeviceType({kFloat32->type_id()});
  builder2.SetKernelType(KernelType::TBE_KERNEL);
  builder2.SetFusionType(kernel::FusionType::OPAQUE);
  builder2.SetProcessor(kernel::Processor::AICORE);
  builder2.SetKernelType(KernelType::TBE_KERNEL);
  matmul->set_kernel_info(std::make_shared<device::KernelInfo>());
  AnfAlgo::SetSelectKernelBuildInfo(builder2.Build(), matmul.get());

  KernelBuildInfoBuilder builder1;
  builder1.SetInputsFormat({"NC1HWC0"});
  builder1.SetOutputsFormat({"NC1HWC0"});
  builder1.SetInputsDeviceType({kFloat32->type_id()});
  builder1.SetOutputsDeviceType({kFloat16->type_id()});
  builder1.SetKernelType(KernelType::TBE_KERNEL);
  builder1.SetFusionType(kernel::FusionType::OPAQUE);
  builder1.SetProcessor(kernel::Processor::AICORE);
  builder1.SetKernelType(KernelType::TBE_KERNEL);
  cast->set_kernel_info(std::make_shared<device::KernelInfo>());
  AnfAlgo::SetSelectKernelBuildInfo(builder1.Build(), cast.get());

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  auto buffer_fusion_pass = std::make_shared<opt::BufferFusion>();
  pm->AddPass(buffer_fusion_pass);
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(kg);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_tbe_matmul_eltwise_fusion", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}
}  // namespace opt
}  // namespace mindspore
