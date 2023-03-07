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
#include "tools/graph_kernel/converter/graph_kernel_optimization.h"

#include <vector>
#include <string>
#include <memory>
#include "ir/func_graph.h"
#include "backend/common/graph_kernel/graph_kernel_flags.h"
#include "backend/common/optimizer/graph_optimizer.h"

#include "backend/common/graph_kernel/core/arithmetic_simplify.h"
#include "backend/common/graph_kernel/core/eliminate_redundant_output.h"
#include "backend/common/graph_kernel/core/shape_ops_splitter.h"
#include "backend/common/graph_kernel/core/update_state_formatter.h"
#include "backend/common/graph_kernel/core/transform_op_optimizer.h"

#include "tools/graph_kernel/converter/akg/kernel_builder.h"
#include "tools/graph_kernel/converter/conv_tuning_expander.h"
#include "tools/graph_kernel/converter/format_recognition.h"
#include "tools/graph_kernel/converter/graph_kernel_cluster_lite.h"
#include "tools/graph_kernel/converter/graph_kernel_expander_lite.h"
#include "tools/graph_kernel/converter/graph_kernel_splitter_lite.h"
#include "tools/graph_kernel/converter/parameter_to_tensor.h"
#include "tools/graph_kernel/converter/eliminate_maketuple_getitem.h"

namespace mindspore::graphkernel {
using opt::GraphOptimizer;
constexpr size_t kStagePreProcess = 0;
constexpr size_t kStageCluster = 1;
constexpr size_t kStageHLO1 = 2;
constexpr size_t kStageSplit = 3;
constexpr size_t kStageBuildKernel = 4;

GkPassManagerPtr GraphKernelOptimizer::PreProcess() const {
  auto pm = std::make_shared<GraphKernelPassManagerLite>(kStagePreProcess, "preprocess");
  // Recognize the formats for all CNodes
  pm->Add(std::make_shared<FormatRecognition>(), OptLevel_1);

  // Convert the const parameters to const tensors
  pm->Add(std::make_shared<ParameterToTensor>(), OptLevel_1);
  return pm;
}

GkPassManagerPtr GraphKernelOptimizer::Cluster() const {
  auto pm = std::make_shared<GraphKernelPassManagerLite>(kStageCluster, "cluster");
  // Expand complex basic kernels to composite kernels
  pm->Add(std::make_shared<GraphKernelExpanderLite>(), OptLevel_1);
  pm->Add(std::make_shared<ConvTuningExpander>(), OptLevel_1);

  // Cluster basic kernels and composite kernels
  pm->Add(std::make_shared<GraphKernelClusterLite>(), OptLevel_1);

  // Eliminate the outputs without external user
  pm->Add(std::make_shared<EliminateRedundantOutput>(), OptLevel_1);
  return pm;
}

GkPassManagerPtr GraphKernelOptimizer::HighLevelOpt1() const {
  auto pm = std::make_shared<GraphKernelPassManagerLite>(kStageHLO1, "highlevelopt1");
  pm->Add(std::make_shared<ArithmeticSimplify>(), OptLevel_2);
  // Eliminate redundant transform ops
  pm->Add(std::make_shared<TransformOpOptimizer>(), OptLevel_2);
  return pm;
}

GkPassManagerPtr GraphKernelOptimizer::Split() const {
  auto pm = std::make_shared<GraphKernelPassManagerLite>(kStageSplit, "split");
  // Make certain nodes redundant so that they are used by only one user,
  // which can avoid unnecessary input-output and get better performance.
  // preprocess for ShapeOpsSplitter
  pm->Add(std::make_shared<ExtendOutputForUpdateState>(), OptLevel_1);
  std::vector<PrimitivePtr> duplicated_ops = {prim::kPrimReshape};
  pm->Add(std::make_shared<ShapeOpsSplitter>(duplicated_ops), OptLevel_1);

  // Split kernel according to costmodel
  pm->Add(std::make_shared<GraphKernelSplitterWithTuning>(), OptLevel_1);

  // After Simplify and Splitter, a lot of redundant getitem/maketuple
  // will be exposed, use ElimMaketupleGetitem Pass to delete them.
  pm->Add(std::make_shared<ElimMaketupleGetitem>(), OptLevel_1);

  // Eliminate the redundant node that is copied above but not handled by GraphKernelSplitter
  pm->Add(std::make_shared<MergeOutputForUpdateState>(), OptLevel_1);
  pm->Add(std::make_shared<EliminateRedundantOutput>(), OptLevel_1);
  return pm;
}

GkPassManagerPtr GraphKernelOptimizer::BuildKernel() const {
  auto pm = std::make_shared<GraphKernelPassManagerLite>(kStageBuildKernel, "buildkernel");
  // build akg and replace graph kernel nodes
  pm->Add(std::make_shared<KernelBuilder>(), OptLevel_1);
  return pm;
}

void GraphKernelOptimizer::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(converter_param_);
  auto optimizer = std::make_shared<GraphOptimizer>("graph_kernel_optimizer");
  std::vector<GkPassManagerPtr> pm_list;
  (void)pm_list.emplace_back(PreProcess());
  (void)pm_list.emplace_back(Cluster());
  (void)pm_list.emplace_back(HighLevelOpt1());
  (void)pm_list.emplace_back(Split());
  (void)pm_list.emplace_back(BuildKernel());

  for (auto &pm : pm_list) {
    pm->SetDumpIr(converter_param_->save_type);
    optimizer->AddPassManager(pm);
  }

  auto mng = func_graph->manager();
  if (mng == nullptr) {
    mng = Manage(func_graph, true);
    func_graph->set_manager(mng);
  }
  (void)optimizer->Optimize(func_graph);
}

void GraphKernelOptimize(const FuncGraphPtr &func_graph, const std::shared_ptr<ConverterPara> &param) {
  if (graphkernel::GraphKernelFlags::GetInstance().IsEnableGraphKernel()) {
    GraphKernelOptimizer(param).Run(func_graph);
  }
}
}  // namespace mindspore::graphkernel
