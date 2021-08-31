/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "tools/converter/graphdef_transform.h"
#include <string>
#include <algorithm>
#include "schema/model_generated.h"
#include "src/common/log_adapter.h"
#include "src/common/log_util.h"
#include "tools/converter/converter_flags.h"
#include "tools/converter/legacy_optimizer/graph/dtype_trans_pass.h"
#include "tools/converter/legacy_optimizer/fusion/quant_cast_fusion_pass.h"
#include "tools/converter/legacy_optimizer/fusion/mul_add_fusion_pass.h"
#include "tools/converter/legacy_optimizer/graph/infershape_pass.h"
#include "tools/converter/legacy_optimizer/graph/batchnorm_convert_scale_pass.h"
#include "tools/converter/legacy_optimizer/graph/isolated_node_remove_pass.h"
#include "tools/converter/legacy_optimizer/graph/dropout_node_remove_pass.h"
#include "tools/converter/legacy_optimizer/graph/topological_sort_pass.h"
#include "tools/converter/legacy_optimizer/graph/tensor_quant_pass.h"
#include "tools/converter/legacy_optimizer/graph/tensor_name_pass.h"
#include "tools/converter/legacy_optimizer/graph/infer_quant_param_pass.h"
#include "tools/converter/legacy_optimizer/graph/set_unused_quant_param_to_default_pass.h"
#include "tools/converter/legacy_optimizer/graph/convert_fp32_to_fp16_pass.h"
#include "tools/converter/legacy_optimizer/graph/subgraph_node_pass.h"
#include "tools/converter/legacy_optimizer/graph/subgraph_tensor_pass.h"

using std::string;
namespace mindspore::lite {
std::vector<schema::CNodeT *> GraphDefTransform::GetGraphNodes() {
  std::vector<schema::CNodeT *> old_nodes{};
  old_nodes.resize(graph_defT_->nodes.size());
  std::transform(graph_defT_->nodes.begin(), graph_defT_->nodes.end(), old_nodes.begin(),
                 [](const std::unique_ptr<schema::CNodeT> &node) { return node.get(); });
  return old_nodes;
}

GraphDefTransform::GraphDefTransform() = default;

GraphDefTransform::~GraphDefTransform() = default;

void GraphDefTransform::SetGraphDef(schema::MetaGraphT *dst_def) { graph_defT_ = dst_def; }

int GraphDefTransform::Transform(const converter::Flags &ctx) {
  STATUS status;
  {
    auto old_nodes = GetGraphNodes();
    Optimizer unused_op_remove_optimizer;
    if (!ctx.trainModel) {
      unused_op_remove_optimizer.AddPass(new DropoutNodeRemovePass());
    }
    unused_op_remove_optimizer.AddPass(new IsolatedNodeRemovePass());
    unused_op_remove_optimizer.AddPass(new SubgraphNodePass(old_nodes));
    status = unused_op_remove_optimizer.Run(graph_defT_);
    if (status != RET_OK && status != RET_NO_CHANGE) {
      MS_LOG(ERROR) << "Run unused_op_remove_optimizer graphPasses Failed";
      return status;
    }
  }

  // format transpose global optimize
  {
    // init old node indices
    auto old_nodes = GetGraphNodes();
    Optimizer format_trans_optimizer;
    if (!ctx.trainModel && ctx.fmk != converter::kFmkTypeOnnx) {
      format_trans_optimizer.AddPass(new (std::nothrow) IsolatedNodeRemovePass());
      format_trans_optimizer.AddPass(new (std::nothrow) SubgraphNodePass(old_nodes));
    }
    status = format_trans_optimizer.Run(graph_defT_);
    if (status != RET_OK && status != RET_NO_CHANGE && status != RET_INFER_INVALID) {
      MS_LOG(ERROR) << "Run format_trans_optimizer graphPasses Failed";
      return status;
    }
  }

  // node replace
  if (!ctx.trainModel) {
    // init old node indices
    auto old_nodes = GetGraphNodes();
    Optimizer replace_optimizer;
    replace_optimizer.AddPass(new (std::nothrow) InferShapePass(ctx.fmk));
    replace_optimizer.AddPass(new (std::nothrow) BatchNormConvertScalePass(ctx.fmk));
    replace_optimizer.AddPass(new (std::nothrow) IsolatedNodeRemovePass());
    replace_optimizer.AddPass(new SubgraphNodePass(old_nodes));
    status = replace_optimizer.Run(graph_defT_);
    if (status != RET_OK && status != RET_NO_CHANGE) {
      MS_LOG(ERROR) << "Run replace_optimizer BatchNormConvertScalePass Failed";
      return status;
    }
  }

  // node fusion
  {
    // init old node indices
    auto old_nodes = GetGraphNodes();
    Optimizer fusion_optimizer;
    fusion_optimizer.AddPass(new (std::nothrow) MulAddFusionPass());
    fusion_optimizer.AddPass(new (std::nothrow) IsolatedNodeRemovePass());
    fusion_optimizer.AddPass(new (std::nothrow) SubgraphNodePass(old_nodes));
    status = fusion_optimizer.Run(graph_defT_);
    if (status != RET_OK && status != RET_NO_CHANGE) {
      MS_LOG(ERROR) << "Run fusion_optimizer graphPasses Failed";
      return status;
    }
  }

  // quantization
  if (ctx.fmk != converter::kFmkTypeTf) {
    // init old node indices
    auto old_nodes = GetGraphNodes();
    Optimizer tensor_quant_optimizer;
    tensor_quant_optimizer.AddPass(new (std::nothrow) TopologicalSortPass());
    tensor_quant_optimizer.AddPass(new (std::nothrow) InferQuantParamPass());
    tensor_quant_optimizer.AddPass(new (std::nothrow) InferShapePass(ctx.fmk));
    tensor_quant_optimizer.AddPass(new (std::nothrow) TensorQuantPass());
    tensor_quant_optimizer.AddPass(new (std::nothrow) SubgraphNodePass(old_nodes));
    status = tensor_quant_optimizer.Run(graph_defT_);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "DoQuantize failed!";
      return status;
    }
  }

  // quantization
  if (ctx.fmk != converter::kFmkTypeTf) {
    // init old node indices
    Optimizer quant_node_optimizer;
    quant_node_optimizer.AddPass(new (std::nothrow) TopologicalSortPass());
    auto old_nodes = GetGraphNodes();
    quant_node_optimizer.AddPass(new (std::nothrow) InferShapePass(ctx.fmk));
    quant_node_optimizer.AddPass(new (std::nothrow) DTypeTransPass(ctx.inputDataType, ctx.outputDataType));
    quant_node_optimizer.AddPass(new (std::nothrow) QuantCastFusionPass());
    quant_node_optimizer.AddPass(new (std::nothrow) IsolatedNodeRemovePass());
    quant_node_optimizer.AddPass(new (std::nothrow) SubgraphNodePass(old_nodes));
    status = quant_node_optimizer.Run(graph_defT_);
    if (status != RET_OK && status != RET_NO_CHANGE) {
      MS_LOG(ERROR) << "Run quant_node_optimizer graphPasses Failed";
      return status;
    }
  }

  {
    // init old node indices
    auto old_nodes = GetGraphNodes();
    Optimizer switch_optimizer;
    switch_optimizer.AddPass(new (std::nothrow) IsolatedNodeRemovePass());
    switch_optimizer.AddPass(new (std::nothrow) SubgraphNodePass(old_nodes));
    switch_optimizer.AddPass(new (std::nothrow) SubgraphTensorPass());
    status = switch_optimizer.Run(graph_defT_);
    if (status != RET_OK && status != RET_NO_CHANGE) {
      MS_LOG(ERROR) << "Run switch_optimizer Failed";
      return status;
    }
  }

  {
    Optimizer nested_loop_optimizer;
    auto old_nodes = GetGraphNodes();
    nested_loop_optimizer.AddPass(new (std::nothrow) SubgraphNodePass(old_nodes));
    nested_loop_optimizer.AddPass(new (std::nothrow) TopologicalSortPass());
    status = nested_loop_optimizer.Run(graph_defT_);
    if (status != RET_OK && status != RET_NO_CHANGE) {
      MS_LOG(ERROR) << "Run nested_loop_optimizer graphPasses Failed";
      return status;
    }
  }

  {
    Optimizer forming_model_optimizer;
    forming_model_optimizer.AddPass(new (std::nothrow) InferShapePass(ctx.fmk));
    forming_model_optimizer.AddPass(new (std::nothrow) SetUnusedQuantParamToDefaultPass());
    forming_model_optimizer.AddPass(new (std::nothrow) TensorNamePass());
    forming_model_optimizer.AddPass(new (std::nothrow) ConvertFP32ToFP16Pass(ctx.saveFP16));
    status = forming_model_optimizer.Run(graph_defT_);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Run InferShapeOptimizer graphPasses Failed.";
      return status;
    }
  }
  return RET_OK;
}
}  // namespace mindspore::lite
