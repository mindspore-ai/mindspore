/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>
#include <map>
#include "include/backend/optimizer/graph_optimizer.h"
#include "ir/func_graph.h"
#include "mindspore/core/ops/array_ops.h"

#include "backend/common/graph_kernel/core/arithmetic_simplify.h"
#include "backend/common/graph_kernel/core/eliminate_redundant_output.h"
#include "backend/common/graph_kernel/core/graph_kernel_utils.h"
#include "backend/common/graph_kernel/core/shape_ops_splitter.h"
#include "backend/common/graph_kernel/core/transform_op_optimizer.h"
#include "backend/common/graph_kernel/core/update_state_formatter.h"
#include "backend/common/graph_kernel/graph_kernel_flags.h"
#include "backend/common/graph_kernel/core/graph_kernel_op_combiner.h"

#include "tools/graph_kernel/converter/akg/utils.h"
#include "tools/graph_kernel/converter/callback_impl.h"
#include "tools/graph_kernel/converter/conv_tuning_expander.h"
#include "tools/graph_kernel/converter/eliminate_maketuple_getitem.h"
#include "tools/graph_kernel/converter/format_recognition.h"
#include "tools/graph_kernel/converter/graph_kernel_cluster_lite.h"
#include "tools/graph_kernel/converter/graph_kernel_expander_lite.h"
#include "tools/graph_kernel/converter/graph_kernel_splitter_lite.h"
#include "tools/graph_kernel/converter/kernel_builder.h"
#include "tools/graph_kernel/converter/parameter_to_tensor.h"
#include "tools/graph_kernel/converter/basic_op_infer_shape.h"
#include "tools/graph_kernel/converter/rename_fullname_with_scope.h"

namespace mindspore {
namespace graphkernel {
using opt::GraphOptimizer;
constexpr size_t kStagePreProcess = 0;
constexpr size_t kStageCluster = 1;
constexpr size_t kStageHLO1 = 2;
constexpr size_t kStageSplit = 3;
constexpr size_t kStageBuildKernel = 4;
constexpr size_t kStagePostProcess = 5;

class EmptyPass : public opt::Pass {
 public:
  EmptyPass() : Pass("empty_pass") {}
  ~EmptyPass() override = default;
  bool Run(const FuncGraphPtr &func_graph) override { return false; }
};

GkPassManagerPtr GraphKernelOptimizer::PreProcess() const {
  auto pm = std::make_shared<GraphKernelPassManagerLite>(kStagePreProcess, "preprocess");

  // put an empty pass here to dump the ir before GraphKernel
  pm->Add(std::make_shared<EmptyPass>(), OptLevel_1);

  // Recognize the formats for all CNodes
  pm->Add(std::make_shared<FormatRecognition>(), OptLevel_1);

  // Dynamic infer shape
  pm->Add(std::make_shared<DynOpInferShape>(), OptLevel_1, is_ascend);

  // Convert the const parameters to const tensors
  pm->Add(std::make_shared<ParameterToTensor>(), OptLevel_1, is_cpu);
  return pm;
}

GkPassManagerPtr GraphKernelOptimizer::Cluster() const {
  auto pm = std::make_shared<GraphKernelPassManagerLite>(kStageCluster, "cluster");
  // Expand complex basic kernels to composite kernels
  pm->Add(std::make_shared<GraphKernelExpanderLite>(), OptLevel_1);

  // Combine supported parallel ops that with common inputs
  pm->Add(std::make_shared<GraphKernelOpCombiner>(), OptLevel_3);

  pm->Add(std::make_shared<ConvTuningExpander>(), OptLevel_1, is_cpu);

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
  pm->Add(std::make_shared<ExtendOutputForUpdateState>(), OptLevel_1, is_cpu);
  std::vector<PrimitivePtr> duplicated_ops = {prim::kPrimReshape};
  pm->Add(std::make_shared<ShapeOpsSplitter>(duplicated_ops), OptLevel_1);
  // Split kernel according to costmodel
  pm->Add(std::make_shared<GraphKernelSplitterWithTuning>(), OptLevel_1);

  // After Simplify and Splitter, a lot of redundant getitem/maketuple
  // will be exposed, use ElimMaketupleGetitem Pass to delete them.
  pm->Add(std::make_shared<ElimMaketupleGetitem>(), OptLevel_1);

  // Eliminate the redundant node that is copied above but not handled by GraphKernelSplitter
  pm->Add(std::make_shared<MergeOutputForUpdateState>(), OptLevel_1, is_cpu);
  pm->Add(std::make_shared<EliminateRedundantOutput>(), OptLevel_1);
  return pm;
}

GkPassManagerPtr GraphKernelOptimizer::BuildKernel() const {
  auto pm = std::make_shared<GraphKernelPassManagerLite>(kStageBuildKernel, "buildkernel");
  // build akg and replace graph kernel nodes
  pm->Add(std::make_shared<KernelBuilder>(), OptLevel_1);
  return pm;
}

GkPassManagerPtr GraphKernelOptimizer::PostProcess() const {
  auto pm = std::make_shared<GraphKernelPassManagerLite>(kStagePostProcess, "postprocess");
  pm->Add(std::make_shared<RenameFullnameWithScope>(), OptLevel_1);
  return pm;
}

std::unordered_set<std::string> CheckSupport() {
  std::unordered_set<std::string> support_backend;
#ifdef AKG_USE_LLVM
  (void)support_backend.emplace("CPU");
#endif
#if defined(AKG_USE_LLVM) && defined(AKG_USE_CUDA)
  (void)support_backend.emplace("GPU");
#endif
#ifdef AKG_ENABLE_D
  (void)support_backend.emplace("Ascend");
#endif
  return support_backend;
}

bool CheckAkg() {
  std::ostringstream py_cmd;
  py_cmd << kAddMSLiteAkg;
  py_cmd << "from akg.ms import compilewithjsonname\n";
  std::string cmd = "python -c \"" + py_cmd.str() + "\"";
  auto ret = std::system(cmd.c_str());
  if (WEXITSTATUS(ret) != 0) {
    MS_LOG(WARNING)
      << "Could not find akg in python, Graph Kernel fusion has been turned off. Please make sure you have "
         "installed akg. process content is as follows:\n"
      << cmd;
    return false;
  }
  return true;
}

void GraphKernelOptimizer::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  if (converter_param_ == nullptr) {
    converter_param_ = std::make_shared<ConverterPara>();
    converter_param_->device = "Ascend";
  }
  const CallbackImplRegister callback_reg(
    [this]() { return std::static_pointer_cast<Callback>(std::make_shared<CallbackImpl>(converter_param_)); });
  if (!CheckAkg()) {
    return;
  }
  auto akg_support_backend = CheckSupport();
  auto device = Callback::Instance()->GetTargetFromContext();
  if (akg_support_backend.find(device) == akg_support_backend.end()) {
    MS_LOG(WARNING)
      << "Graph Kernel fusion for" << device
      << " backend is not enabled, Graph Kernel fusion has been turned off. Please check if you have set the "
         "correct compilation options for "
      << device << " backend.";
    return;
  }
  is_cpu = (device == "CPU");
  is_ascend = (device == "Ascend");

  auto optimizer = std::make_shared<GraphOptimizer>("graph_kernel_optimizer");
  optimizer->AddPassManager(PreProcess());
  optimizer->AddPassManager(Cluster());
  optimizer->AddPassManager(HighLevelOpt1());
  optimizer->AddPassManager(Split());
  optimizer->AddPassManager(BuildKernel());
  optimizer->AddPassManager(PostProcess());

  auto mng = func_graph->manager();
  if (mng == nullptr) {
    mng = Manage(func_graph, true);
    func_graph->set_manager(mng);
  }
  GkUtils::UpdateFuncGraphManager(mng, func_graph);
  (void)optimizer->Optimize(func_graph);
}

std::string UpdateFlags(const std::string &device, const std::string &graph_kernel_flags) {
  std::string res = graph_kernel_flags;
  if (device.find("Ascend") != std::string::npos &&
      graph_kernel_flags.find("enable_dynamic_shape_fusion") == std::string::npos) {
    res += " --enable_dynamic_shape_fusion";
  }
  return res;
}
}  // namespace graphkernel

lite::STATUS GraphKernelOptimize(const FuncGraphPtr &func_graph, const std::shared_ptr<ConverterPara> &param) {
#ifndef Debug
  try {
#endif
    if (param == nullptr) {
      return lite::RET_OK;
    }
    if (param->graphKernelParam.graph_kernel_flags.empty()) {
      return lite::RET_OK;
    }
    std::map<std::string, std::string> jit_config;
    jit_config["graph_kernel_flags"] =
      graphkernel::UpdateFlags(param->device, param->graphKernelParam.graph_kernel_flags);
    graphkernel::GraphKernelFlags::SaveJitConfig(jit_config);

    if (graphkernel::GraphKernelFlags::GetInstance().IsEnableGraphKernel()) {
      MS_LOG(INFO) << "Run graphkernel optimization begin.";
      graphkernel::GraphKernelOptimizer(param).Run(func_graph);
      MS_LOG(INFO) << "Run graphkernel optimization end.";
    }
    return lite::RET_OK;
#ifndef Debug
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << e.what();
    return lite::RET_ERROR;
  }
#endif
}
}  // namespace mindspore
