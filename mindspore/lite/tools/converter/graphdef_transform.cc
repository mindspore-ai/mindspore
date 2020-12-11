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
#include "tools/converter/legacy_optimizer/graph/unused_node_remove_pass.h"
#include "tools/converter/legacy_optimizer/graph/dropout_node_remove_pass.h"
#include "tools/converter/legacy_optimizer/graph/topological_sort_pass.h"
#include "tools/converter/legacy_optimizer/graph/tensor_quant_pass.h"
#include "tools/converter/legacy_optimizer/graph/tensor_name_pass.h"
#include "tools/converter/legacy_optimizer/graph/infer_quant_param_pass.h"
#include "tools/converter/legacy_optimizer/graph/set_unused_quant_param_to_default_pass.h"

using std::string;
namespace mindspore::lite {
GraphDefTransform::GraphDefTransform() = default;

GraphDefTransform::~GraphDefTransform() = default;

void GraphDefTransform::SetGraphDef(schema::MetaGraphT *_dstDef) { graphDefT = _dstDef; }

int GraphDefTransform::Transform(const converter::Flags &ctx) {
  STATUS status;
  {
    Optimizer unusedOpRemoveOptimizer;
    unusedOpRemoveOptimizer.AddPass(new UnusedNodeRemovePass());
    if (!ctx.trainModel) {
      unusedOpRemoveOptimizer.AddPass(new DropoutNodeRemovePass());
    }
    unusedOpRemoveOptimizer.AddPass(new IsolatedNodeRemovePass());
    status = unusedOpRemoveOptimizer.Run(graphDefT);
    if (status != RET_OK && status != RET_NO_CHANGE) {
      MS_LOG(ERROR) << "Run unusedOpRemoveOptimizer graphPasses Failed";
      return status;
    }
  }
  // topological sorting
  {
    Optimizer topologicalOptimizer;
    topologicalOptimizer.AddPass(new (std::nothrow) TopologicalSortPass());
    status = topologicalOptimizer.Run(graphDefT);
    if (status != RET_OK && status != RET_NO_CHANGE) {
      MS_LOG(ERROR) << "Run topologicalOptimizer graphPasses Failed";
      return status;
    }
  }

  // generate and infer quant parameters
  {
    Optimizer inferQuantParamPass;
    inferQuantParamPass.AddPass(new (std::nothrow) TopologicalSortPass());
    inferQuantParamPass.AddPass(new (std::nothrow) InferQuantParamPass());
    status = inferQuantParamPass.Run(graphDefT);
    if (status != RET_OK && status != RET_NO_CHANGE) {
      MS_LOG(ERROR) << "Run topologicalOptimizer graphPasses Failed";
      return status;
    }
  }

  // postconvert pass
  {
    Optimizer fusionOptimizer;
    if (!ctx.trainModel) {
      auto batch_norm_scale_pass = new (std::nothrow) BatchNormConvertScalePass();
      if (batch_norm_scale_pass == nullptr) {
        MS_LOG(ERROR) << "new batch_norm_scale_pass failed.";
        return RET_ERROR;
      }
      batch_norm_scale_pass->SetFmk(ctx.fmk);
      fusionOptimizer.AddPass(batch_norm_scale_pass);
    }
    fusionOptimizer.AddPass(new (std::nothrow) IsolatedNodeRemovePass());
    status = fusionOptimizer.Run(graphDefT);
    if (status != RET_OK && status != RET_NO_CHANGE) {
      MS_LOG(ERROR) << "Run fusionOptimizer BatchNormConvertScalePass Failed";
      return status;
    }
  }
  // format transform
  {
    Optimizer formatTransOptimizer;
    auto formatTransPass = new (std::nothrow) FormatTransPass();
    if (formatTransPass == nullptr) {
      MS_LOG(ERROR) << "new formatTransPass failed";
      return RET_MEMORY_FAILED;
    }
    formatTransPass->SetQuantType(ctx.quantType);
    formatTransPass->SetFmk(ctx.fmk);
    formatTransOptimizer.AddPass(formatTransPass);
    formatTransOptimizer.AddPass(new (std::nothrow) TopologicalSortPass());
    formatTransOptimizer.AddPass(new (std::nothrow) InferShapePass());
    formatTransOptimizer.AddPass(new (std::nothrow) FormatTransFusionPass());
    formatTransOptimizer.AddPass(new (std::nothrow) IsolatedNodeRemovePass());
    formatTransOptimizer.AddPass(new (std::nothrow) TransOpRemovePass());
    formatTransOptimizer.AddPass(new (std::nothrow) TransOpInsertPass());
    formatTransOptimizer.AddPass(new (std::nothrow) FormatTransFusionPass());
    formatTransOptimizer.AddPass(new (std::nothrow) IsolatedNodeRemovePass());
    if (!ctx.trainModel && ctx.fmk != converter::FmkType_ONNX) {
      formatTransOptimizer.AddPass(new (std::nothrow) GlobalFormatTransformPass());
      formatTransOptimizer.AddPass(new (std::nothrow) IsolatedNodeRemovePass());
    }
    status = formatTransOptimizer.Run(graphDefT);
    if (status != RET_OK && status != RET_NO_CHANGE && status != RET_INFER_INVALID) {
      MS_LOG(ERROR) << "Run formatTransOptimizer graphPasses Failed";
      return status;
    }
  }

  {
    Optimizer fusionOptimizer;
    fusionOptimizer.AddPass(new (std::nothrow) MulAddFusionPass());
    fusionOptimizer.AddPass(new (std::nothrow) IsolatedNodeRemovePass());
    status = fusionOptimizer.Run(graphDefT);
    if (status != RET_OK && status != RET_NO_CHANGE) {
      MS_LOG(ERROR) << "Run fusionOptimizer graphPasses Failed";
      return status;
    }
  }

  // do quantization
  {
    Optimizer tensorQuantOptimizer;
    tensorQuantOptimizer.AddPass(new (std::nothrow) TopologicalSortPass());
    tensorQuantOptimizer.AddPass(new (std::nothrow) InferShapePass());
    tensorQuantOptimizer.AddPass(new (std::nothrow) TensorQuantPass());
    status = tensorQuantOptimizer.Run(graphDefT);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "DoQuantize failed!";
      return status;
    }
  }

  // insert quantNode and deQuantNode
  {
    Optimizer quantNodeOptimizer;
    auto dTypeTransPass = new (std::nothrow) DTypeTransPass();
    if (dTypeTransPass == nullptr) {
      MS_LOG(ERROR) << "new dTypeTransPass failed";
      return RET_MEMORY_FAILED;
    }
    dTypeTransPass->SetInputDataDType(ctx.inputDataType);
    dTypeTransPass->SetOutputDataDType(ctx.outputDataType);
    quantNodeOptimizer.AddPass(new (std::nothrow) TopologicalSortPass());
    quantNodeOptimizer.AddPass(new (std::nothrow) InferShapePass());
    quantNodeOptimizer.AddPass(dTypeTransPass);
    quantNodeOptimizer.AddPass(new (std::nothrow) QuantCastFusionPass());
    quantNodeOptimizer.AddPass(new (std::nothrow) IsolatedNodeRemovePass());
    quantNodeOptimizer.AddPass(new (std::nothrow) SetUnusedQuantParamToDefaultPass());
    status = quantNodeOptimizer.Run(graphDefT);
    if (status != RET_OK && status != RET_NO_CHANGE) {
      MS_LOG(ERROR) << "Run quantNodeOptimizer graphPasses Failed";
      return status;
    }
  }

  // tensor name
  {
    Optimizer nameOptimizer;
    nameOptimizer.AddPass(new (std::nothrow) TopologicalSortPass());
    nameOptimizer.AddPass(new (std::nothrow) TensorNamePass());
    status = nameOptimizer.Run(graphDefT);
    if (status != RET_OK && status != RET_NO_CHANGE) {
      MS_LOG(ERROR) << "Run nameOptimizer graphPasses Failed";
      return status;
    }
  }

  // topological sorting
  {
    Optimizer topologicalOptimizer;
    topologicalOptimizer.AddPass(new (std::nothrow) TopologicalSortPass());
    status = topologicalOptimizer.Run(graphDefT);
    if (status != RET_OK && status != RET_NO_CHANGE) {
      MS_LOG(ERROR) << "Run topologicalOptimizer graphPasses Failed";
      return status;
    }
  }
  return RET_OK;
}
}  // namespace mindspore::lite
