/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/optimizer/buffer_fusion/ub_fusion_optimizer.h"
#include <memory>
#include <string>
#include "plugin/device/ascend/optimizer/buffer_fusion/ub_pattern_fusion.h"
#include "plugin/device/ascend/optimizer/buffer_fusion/eltwise_fusion_pass.h"
#include "plugin/device/ascend/optimizer/buffer_fusion/multi_output_fusion_pass.h"
#include "plugin/device/ascend/optimizer/buffer_fusion/conv2dbackprop_eltwise_eltwise_fusion_pass.h"
#include "plugin/device/ascend/optimizer/buffer_fusion/conv2dbackprop_eltwise_fusion_pass.h"
#include "plugin/device/ascend/optimizer/buffer_fusion/conv_single_in_fusion_pass.h"
#include "plugin/device/ascend/optimizer/buffer_fusion/conv_double_in_fusion_pass.h"
#include "plugin/device/ascend/optimizer/buffer_fusion/matmul_eltwise_fusion_pass.h"
#include "plugin/device/ascend/optimizer/buffer_fusion/matmul_confusiontranspose_fusion_pass.h"
#include "plugin/device/ascend/optimizer/buffer_fusion/matmul_dropoutdomaskv3_add_fusion_pass.h"
#include "plugin/device/ascend/optimizer/buffer_fusion/batchmatmul_dropoutdomaskv3_fusion_pass.h"
#include "plugin/device/ascend/optimizer/buffer_fusion/batchmatmul_reducesum_fusion_pass.h"
#include "plugin/device/ascend/optimizer/buffer_fusion/batchmatmul_eltwise_fusion_pass.h"
#include "plugin/device/ascend/optimizer/buffer_fusion/depthwiseconv_eltwise_fusion_pass.h"
#include "plugin/device/ascend/optimizer/buffer_fusion/bnupdate_eltwise_fusion_pass.h"
#include "plugin/device/ascend/optimizer/buffer_fusion/bnupdate_eltwise_eltwise_fusion_pass.h"
#include "plugin/device/ascend/optimizer/buffer_fusion/conv_bnreduce_fusion_pass.h"
#include "plugin/device/ascend/optimizer/buffer_fusion/reduce_eltwise_fusion_pass.h"
#include "plugin/device/ascend/optimizer/buffer_fusion/segment_eltwise_fusion_pass.h"

#include "utils/ms_context.h"
#include "include/common/debug/anf_ir_dump.h"
#include "include/common/debug/dump_proto.h"
#include "include/backend/debug/profiler/profiling.h"
#include "plugin/device/ascend/kernel/tbe/tbe_kernel_compile.h"

namespace mindspore {
namespace opt {
constexpr char kDisablePrebuildEnv[] = "MS_DEV_DISABLE_PREBUILD";

void AscendBackendUBFusionOptimization(const std::shared_ptr<session::KernelGraph> &kernel_graph) {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (!context_ptr->get_param<bool>(MS_CTX_IR_FUSION_FLAG)) {
    MS_LOG(INFO) << "UBFusion is not enable, skip";
    return;
  }
  if (kernel_graph->is_dynamic_shape()) {
    MS_LOG(INFO) << "Dynamic shape skip fusion";
    return;
  }
  profiler::CollectHostInfo("Ascend", "Graph Optimization", "BackendOptimization_UBFusionOptimization", 0, 0, 0);
  auto pre_build = common::GetEnv(kDisablePrebuildEnv);
  if (pre_build.empty() || (pre_build != "true" && pre_build != "True")) {
    auto &build_manager = kernel::ascend::TbeKernelCompileManager::GetInstance();
    build_manager.TbePreBuild(kernel_graph);
  }
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name = "hwopt_d_ub_fusion_before_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph);
  }
#endif
  auto fusion_id_allocator = std::make_shared<FusionIdAllocator>();
  MS_EXCEPTION_IF_NULL(fusion_id_allocator);
  fusion_id_allocator->Init();
  auto optimizer = std::make_shared<GraphOptimizer>();
  auto ub_fusion_pm = std::make_shared<PassManager>("ub_fusion_pm");
  ub_fusion_pm->AddPass(std::make_shared<Conv2DBackpropEltwiseEltwiseFusionPass>(fusion_id_allocator));
  ub_fusion_pm->AddPass(std::make_shared<Conv2DBackpropEltwiseFusionPass>(fusion_id_allocator));
  ub_fusion_pm->AddPass(std::make_shared<ConvBnReduceFusionPass>(fusion_id_allocator));
  ub_fusion_pm->AddPass(std::make_shared<ConvSingleInFusionPass>(fusion_id_allocator));
  ub_fusion_pm->AddPass(std::make_shared<BnupdateEltwiseFusionPass>(fusion_id_allocator));
  ub_fusion_pm->AddPass(std::make_shared<BnupdateEltwiseEltwiseFusionPass>(fusion_id_allocator));
  ub_fusion_pm->AddPass(std::make_shared<MatmulEltwiseFusionPass>(fusion_id_allocator));
  ub_fusion_pm->AddPass(std::make_shared<MatmulDropoutDoMaskV3AddFusionPass>(fusion_id_allocator));
  ub_fusion_pm->AddPass(std::make_shared<BatchMatmulReduceSumFusionPass>(fusion_id_allocator));
  ub_fusion_pm->AddPass(std::make_shared<ConvDoubleInFusionPass>(fusion_id_allocator));
  ub_fusion_pm->AddPass(std::make_shared<ReduceEltwiseFusionPass>(fusion_id_allocator));
  ub_fusion_pm->AddPass(std::make_shared<SegmentEltwiseFusionPass>(fusion_id_allocator));
  ub_fusion_pm->AddPass(std::make_shared<MultiOutputFusionPass>(fusion_id_allocator));
  ub_fusion_pm->AddPass(std::make_shared<EltwiseFusionPass>(fusion_id_allocator));
  ub_fusion_pm->AddPass(std::make_shared<DepthwiseConvEltwiseFusionPass>(fusion_id_allocator));
  ub_fusion_pm->AddPass(std::make_shared<MatmulConfusionTranposeFusionPass>(fusion_id_allocator));
  ub_fusion_pm->AddPass(std::make_shared<BatchMatmulEltwiseFusionPass>(fusion_id_allocator));
  ub_fusion_pm->AddPass(std::make_shared<BatchMatmulDropoutDoMaskV3FusionPass>(fusion_id_allocator));
  ub_fusion_pm->AddPass(std::make_shared<UbPatternFusion>());
  optimizer->AddPassManager(ub_fusion_pm);
  (void)optimizer->Optimize(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
#ifdef ENABLE_DUMP_IR
  if (context_ptr->CanDump(kIntroductory)) {
    std::string file_name = "hwopt_d_ub_fusion_after_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph);
  }
#endif
  profiler::CollectHostInfo("Ascend", "Graph Optimization", "BackendOptimization_UBFusionOptimization", 0, 0, 1);
}
}  // namespace opt
}  // namespace mindspore
