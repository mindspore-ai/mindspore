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

#include "tools/converter/graphdef_transform.h"
#include <string>
#include <algorithm>
#include "schema/model_generated.h"
#include "src/common/log_adapter.h"
#include "tools/converter/converter_flags.h"
#include "tools/converter/legacy_optimizer/graph/dtype_trans_pass.h"
#include "tools/converter/legacy_optimizer/fusion/format_trans_fusion_pass.h"
#include "tools/converter/legacy_optimizer/fusion/quant_cast_fusion_pass.h"
#include "tools/converter/legacy_optimizer/fusion/mul_add_fusion_pass.h"
#include "tools/converter/legacy_optimizer/graph/trans_format_remove_pass.h"
#include "tools/converter/legacy_optimizer/graph/infershape_pass.h"
#include "tools/converter/legacy_optimizer/graph/batchnorm_convert_scale_pass.h"
#include "tools/converter/legacy_optimizer/graph/format_trans_pass.h"
#include "tools/converter/legacy_optimizer/graph/trans_format_insert_pass.h"
#include "tools/converter/legacy_optimizer/graph/global_format_transform_pass.h"
#include "tools/converter/legacy_optimizer/graph/isolated_node_remove_pass.h"
#include "tools/converter/legacy_optimizer/graph/dropout_node_remove_pass.h"
#include "tools/converter/legacy_optimizer/graph/topological_sort_pass.h"
#include "tools/converter/legacy_optimizer/graph/tensor_quant_pass.h"
#include "tools/converter/legacy_optimizer/graph/tensor_name_pass.h"
#include "tools/converter/legacy_optimizer/graph/infer_quant_param_pass.h"
#include "tools/converter/legacy_optimizer/graph/set_unused_quant_param_to_default_pass.h"
#include "tools/converter/legacy_optimizer/graph/switch_pass.h"
#include "tools/converter/legacy_optimizer/graph/subgraph_node_pass.h"
#include "tools/converter/legacy_optimizer/graph/subgraph_tensor_pass.h"
#include "tools/converter/legacy_optimizer/graph/nested_loop_expand_pass.h"

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

  // generate and infer quant parameters
  {
    Optimizer infer_quant_param_pass;
    infer_quant_param_pass.AddPass(new (std::nothrow) TopologicalSortPass());
    infer_quant_param_pass.AddPass(new (std::nothrow) InferQuantParamPass());
    status = infer_quant_param_pass.Run(graph_defT_);
    if (status != RET_OK && status != RET_NO_CHANGE) {
      MS_LOG(ERROR) << "Run infer_quant_param_pass graphPasses Failed";
      return status;
    }
  }

  {
    // format transform
    // init old node indices
    auto old_nodes = GetGraphNodes();

    Optimizer format_trans_optimizer;
    auto format_trans_pass = new (std::nothrow) FormatTransPass();
    if (format_trans_pass == nullptr) {
      MS_LOG(ERROR) << "new formatTransPass failed";
      return RET_MEMORY_FAILED;
    }
    format_trans_pass->set_quant_type(ctx.quantType);
    format_trans_pass->set_fmk_type(ctx.fmk);
    format_trans_optimizer.AddPass(format_trans_pass);
    format_trans_optimizer.AddPass(new (std::nothrow) SubgraphNodePass(old_nodes));
    format_trans_optimizer.AddPass(new (std::nothrow) TopologicalSortPass());
    if (ctx.fmk != converter::FmkType_TF) {
      format_trans_optimizer.AddPass(new (std::nothrow) InferShapePass());
    }
    status = format_trans_optimizer.Run(graph_defT_);
    if (status != RET_OK && status != RET_NO_CHANGE && status != RET_INFER_INVALID) {
      MS_LOG(ERROR) << "Run format_trans_optimizer graphPasses Failed";
      return status;
    }
  }
  {
    // init old node indices
    auto old_nodes = GetGraphNodes();
    Optimizer format_trans_optimizer;
    format_trans_optimizer.AddPass(new (std::nothrow) FormatTransFusionPass());
    format_trans_optimizer.AddPass(new (std::nothrow) IsolatedNodeRemovePass());
    format_trans_optimizer.AddPass(new (std::nothrow) TransOpRemovePass());
    format_trans_optimizer.AddPass(new (std::nothrow) TransOpInsertPass());
    format_trans_optimizer.AddPass(new (std::nothrow) FormatTransFusionPass());
    format_trans_optimizer.AddPass(new (std::nothrow) IsolatedNodeRemovePass());
    format_trans_optimizer.AddPass(new (std::nothrow) SubgraphNodePass(old_nodes));
    status = format_trans_optimizer.Run(graph_defT_);
    if (status != RET_OK && status != RET_NO_CHANGE && status != RET_INFER_INVALID) {
      MS_LOG(ERROR) << "Run format_trans_optimizer graphPasses Failed";
      return status;
    }
  }

  {
    // init old node indices
    auto old_nodes = GetGraphNodes();
    Optimizer format_trans_optimizer;
    if (!ctx.trainModel && ctx.fmk != converter::FmkType_ONNX) {
      format_trans_optimizer.AddPass(new (std::nothrow) GlobalFormatTransformPass());
      format_trans_optimizer.AddPass(new (std::nothrow) IsolatedNodeRemovePass());
      format_trans_optimizer.AddPass(new (std::nothrow) SubgraphNodePass(old_nodes));
    }
    status = format_trans_optimizer.Run(graph_defT_);
    if (status != RET_OK && status != RET_NO_CHANGE && status != RET_INFER_INVALID) {
      MS_LOG(ERROR) << "Run format_trans_optimizer graphPasses Failed";
      return status;
    }
  }

  // postconvert pass
  {
    // init old node indices
    auto old_nodes = GetGraphNodes();
    Optimizer replace_optimizer;
    if (!ctx.trainModel) {
      auto batch_norm_scale_pass = new (std::nothrow) BatchNormConvertScalePass();
      if (batch_norm_scale_pass == nullptr) {
        MS_LOG(ERROR) << "new batch_norm_scale_pass failed.";
        return RET_ERROR;
      }
      batch_norm_scale_pass->SetFmk(ctx.fmk);
      replace_optimizer.AddPass(batch_norm_scale_pass);
    }
    replace_optimizer.AddPass(new (std::nothrow) IsolatedNodeRemovePass());
    replace_optimizer.AddPass(new SubgraphNodePass(old_nodes));
    status = replace_optimizer.Run(graph_defT_);
    if (status != RET_OK && status != RET_NO_CHANGE) {
      MS_LOG(ERROR) << "Run replace_optimizer BatchNormConvertScalePass Failed";
      return status;
    }
  }

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

  // do quantization
  if (ctx.fmk != converter::FmkType_TF) {
    // init old node indices
    auto old_nodes = GetGraphNodes();
    Optimizer tensor_quant_optimizer;
    tensor_quant_optimizer.AddPass(new (std::nothrow) TopologicalSortPass());
    tensor_quant_optimizer.AddPass(new (std::nothrow) InferShapePass());
    tensor_quant_optimizer.AddPass(new (std::nothrow) TensorQuantPass());
    tensor_quant_optimizer.AddPass(new (std::nothrow) SubgraphNodePass(old_nodes));
    status = tensor_quant_optimizer.Run(graph_defT_);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "DoQuantize failed!";
      return status;
    }
  }

  // insert quantNode and deQuantNode
  if (ctx.fmk != converter::FmkType_TF) {
    // init old node indices
    auto old_nodes = GetGraphNodes();
    Optimizer quant_node_optimizer;
    quant_node_optimizer.AddPass(new (std::nothrow) SubgraphNodePass(old_nodes));
    quant_node_optimizer.AddPass(new (std::nothrow) TopologicalSortPass());
    quant_node_optimizer.AddPass(new (std::nothrow) InferShapePass());
    status = quant_node_optimizer.Run(graph_defT_);
    if (status != RET_OK && status != RET_NO_CHANGE) {
      MS_LOG(ERROR) << "Run quant_node_optimizer graphPasses Failed";
      return status;
    }
    auto old_nodes2 = GetGraphNodes();
    quant_node_optimizer.AddPass(new (std::nothrow) InferQuantParamPass());
    auto dtype_trans_pass = new (std::nothrow) DTypeTransPass();
    if (dtype_trans_pass == nullptr) {
      MS_LOG(ERROR) << "new dtype_trans_pass failed";
      return RET_MEMORY_FAILED;
    }
    dtype_trans_pass->set_input_data_dtype(ctx.inputDataType);
    dtype_trans_pass->set_output_data_dtype(ctx.outputDataType);
    quant_node_optimizer.AddPass(dtype_trans_pass);
    quant_node_optimizer.AddPass(new (std::nothrow) QuantCastFusionPass());
    quant_node_optimizer.AddPass(new (std::nothrow) IsolatedNodeRemovePass());
    quant_node_optimizer.AddPass(new (std::nothrow) SubgraphNodePass(old_nodes2));
    status = quant_node_optimizer.Run(graph_defT_);
    if (status != RET_OK && status != RET_NO_CHANGE) {
      MS_LOG(ERROR) << "Run quant_node_optimizer graphPasses Failed";
      return status;
    }
  }

  // switch pass
  {
    // init old node indices
    auto old_nodes = GetGraphNodes();
    Optimizer switch_optimizer;
    switch_optimizer.AddPass(new (std::nothrow) SwitchPass());
    switch_optimizer.AddPass(new (std::nothrow) IsolatedNodeRemovePass());
    switch_optimizer.AddPass(new (std::nothrow) SubgraphNodePass(old_nodes));
    status = switch_optimizer.Run(graph_defT_);
    if (status != RET_OK && status != RET_NO_CHANGE) {
      MS_LOG(ERROR) << "Run switch_optimizer Failed";
      return status;
    }
  }

  // subgraph tensor pass
  {
    Optimizer subgraph_tensor_optimizer;
    subgraph_tensor_optimizer.AddPass(new (std::nothrow) SubgraphTensorPass());
    status = subgraph_tensor_optimizer.Run(graph_defT_);
    if (status != RET_OK && status != RET_NO_CHANGE) {
      MS_LOG(ERROR) << "Run subgraph tensor pass Failed";
      return status;
    }
  }

  // tensor name
  {
    // init old node indices
    auto old_nodes = GetGraphNodes();
    Optimizer name_optimizer;
    name_optimizer.AddPass(new (std::nothrow) SubgraphNodePass(old_nodes));
    name_optimizer.AddPass(new (std::nothrow) TopologicalSortPass());
    name_optimizer.AddPass(new (std::nothrow) TensorNamePass());
    status = name_optimizer.Run(graph_defT_);
    if (status != RET_OK && status != RET_NO_CHANGE) {
      MS_LOG(ERROR) << "Run name_optimizer graphPasses Failed";
      return status;
    }
  }

  {
    Optimizer nested_loop_optimizer;
    nested_loop_optimizer.AddPass(new (std::nothrow) NestedLoopExpandPass());
    status = nested_loop_optimizer.Run(graph_defT_);
    if (status != RET_OK && status != RET_NO_CHANGE) {
      MS_LOG(ERROR) << "Run nested_loop_optimizer graphPasses Failed";
      return status;
    }
  }

  {
    Optimizer quant_param_optimizer;
    quant_param_optimizer.AddPass(new (std::nothrow) SetUnusedQuantParamToDefaultPass());
    status = quant_param_optimizer.Run(graph_defT_);
    if (status != RET_OK && status != RET_NO_CHANGE) {
      MS_LOG(ERROR) << "Run quant_param_optimizer graphPasses Failed";
      return status;
    }
  }
  return RET_OK;
}  // namespace mindspore::lite
}  // namespace mindspore::lite
