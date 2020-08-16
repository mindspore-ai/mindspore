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
#include <iostream>
#include <memory>
#include <string>
#include "schema/model_generated.h"
#include "utils/log_adapter.h"
#include "src/common/op_utils.h"
#include "tools/converter/converter_flags.h"
#include "tools/converter/legacy_optimizer/graph/dtype_trans_pass.h"
// #include "tools/converter/legacy_optimizer/fusion/matmul_biasadd_fusion_pass.h"
#include "tools/converter/legacy_optimizer/fusion/format_trans_fusion_pass.h"
#include "tools/converter/legacy_optimizer/fusion/format_trans_transpose_fusion_pass.h"
#include "tools/converter/legacy_optimizer/fusion/quant_cast_fusion_pass.h"
#include "tools/converter/legacy_optimizer/graph/weight_format_hardcode_pass.h"
#include "tools/converter/legacy_optimizer/graph/weight_format_transform_pass.h"
#include "tools/converter/legacy_optimizer/graph/format_trans_pass.h"
#include "tools/converter/legacy_optimizer/graph/eltwise_format_trans_pass.h"
#include "tools/converter/legacy_optimizer/graph/isolated_node_remove_pass.h"
#include "tools/converter/legacy_optimizer/graph/unused_node_remove_pass.h"
#include "tools/converter/legacy_optimizer/graph/topological_sort_pass.h"
#include "tools/converter/quantizer/aware_quantizer.h"
#include "tools/converter/converter.h"

using std::string;
namespace mindspore::lite {
GraphDefTransform::GraphDefTransform() = default;

GraphDefTransform::~GraphDefTransform() = default;

void GraphDefTransform::SetGraphDef(schema::MetaGraphT *_dstDef) { graphDefT = _dstDef; }

void GraphDefTransform::CreateQuantizer(const converter::Flags *flags) {
  auto type = flags->quantType;
  switch (type) {
    case QuantType::QuantType_AwareTraining: {
      MS_LOG(INFO) << "create AwareTrainingQuantizer!";
      fbQuantizer =
        std::make_unique<quant::AwareQuantizer>(graphDefT, flags->inputInferenceTypeIn, flags->stdDev, flags->mean);
      break;
    }
    default:
      //      MS_LOGI("will support quantizer type %s in the future!", flags->quantTypeIn.c_str());
      break;
  }
}

int GraphDefTransform::Transform(const converter::Flags &ctx) {
  STATUS status;
  {
    Optimizer weightFormatOptimizer;
    auto weightHardCodePass = new WeightFormatHardCodePass();
    auto weightFormatPass = new WeightFormatTransformPass();
    weightHardCodePass->SetQuantType(ctx.quantType);
    weightHardCodePass->SetFmkType(ctx.fmk);
    weightFormatPass->SetQuantType(ctx.quantType);
    weightFormatPass->SetFmkType(ctx.fmk);
//    weightFormatPass->SetDstFormat(Format_KHWC);
    weightFormatOptimizer.AddPass(weightHardCodePass);
    weightFormatOptimizer.AddPass(weightFormatPass);
    status = weightFormatOptimizer.Run(graphDefT);
    if (status != RET_OK && status != RET_NO_CHANGE) {
      MS_LOG(ERROR) << "Run weightFormatOptimizer graphPasses Failed";
      return status;
    }
  }

  {
    Optimizer unusedOpRemoveOptimizer;
    unusedOpRemoveOptimizer.AddPass(new UnusedNodeRemovePass());
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
    if (mQuantizer != nullptr) {
      Optimizer topologicalOptimizer;
      topologicalOptimizer.AddPass(new (std::nothrow) TopologicalSortPass());
      status = topologicalOptimizer.Run(graphDefT);
      if (status != RET_OK && status != RET_NO_CHANGE) {
        MS_LOG(ERROR) << "Run topologicalOptimizer graphPasses Failed";
        return status;
      }
      if (!(this->graphDefT->fmkType == converter::FmkType_TF &&
            this->graphDefT->nodes.front()->quantType == QuantType::QuantType_AwareTraining)) {
        status = mQuantizer->GenerateQuantParam();
        if (status != RET_OK) {
          MS_LOG(ERROR) << "GenerateQuantParam failed";
          return status;
        }
        status = mQuantizer->DetermineNodeQuantType();
        if (status != RET_OK) {
          MS_LOG(ERROR) << "DetermineNodeQuant failed";
        }
      }
    }
  }

  // format transform
  if (ctx.formatTrans) {
    Optimizer formatTransOptimizer;
    auto formatTransPass = new (std::nothrow) FormatTransPass();
    if (formatTransPass == nullptr) {
      MS_LOG(ERROR) << "new formatTransPass failed";
      return RET_ERROR;
    }
    formatTransPass->SetQuantType(ctx.quantType);
    formatTransPass->SetFmk(ctx.fmk);
    formatTransOptimizer.AddPass(formatTransPass);
    formatTransOptimizer.AddPass(new EltwiseFormatTransPass());
    formatTransOptimizer.AddPass(new (std::nothrow) FormatTransFusionPass());
    formatTransOptimizer.AddPass(new (std::nothrow) IsolatedNodeRemovePass());
    //    if (ctx.quantType == QuantType_AwareTraining) {
    //      formatTransOptimizer.AddPass(new (std::nothrow) FormatTransNodeQuantParamFillPass());
    //    }
    status = formatTransOptimizer.Run(graphDefT);
    if (status != RET_OK && status != RET_NO_CHANGE) {
      MS_LOG(ERROR) << "Run formatTransOptimizer graphPasses Failed";
      return status;
    }
  }
  {
    Optimizer fusionOptimizer;
    fusionOptimizer.AddPass(new (std::nothrow) FormatTransPermuteFusionPass());
    fusionOptimizer.AddPass(new (std::nothrow) IsolatedNodeRemovePass());
    status = fusionOptimizer.Run(graphDefT);
    if (status != RET_OK && status != RET_NO_CHANGE) {
      MS_LOG(ERROR) << "Run fusionOptimizer graphPasses Failed";
      return status;
    }
  }

  // do quantization
  if (fbQuantizer != nullptr) {
    status = fbQuantizer->DoQuantize();
    if (status != RET_OK) {
      MS_LOG(ERROR) << "DoQuantize failed!";
      return status;
    }
  }

  // insert quantNode and deQuantNode
  if (ctx.quantType == QuantType_AwareTraining) {
    Optimizer quantNodeOptimizer;
    auto dTypeTransPass = new (std::nothrow) DTypeTransPass();
    if (dTypeTransPass == nullptr) {
      MS_LOG(ERROR) << "new dTypeTransPass failed";
      return RET_ERROR;
    }
    dTypeTransPass->SetInputDataDType(ctx.inputInferenceType);
    quantNodeOptimizer.AddPass(dTypeTransPass);
    quantNodeOptimizer.AddPass(new (std::nothrow) QuantCastFusionPass());
    quantNodeOptimizer.AddPass(new (std::nothrow) IsolatedNodeRemovePass());
    status = quantNodeOptimizer.Run(graphDefT);
    if (status != RET_OK && status != RET_NO_CHANGE) {
      MS_LOG(ERROR) << "Run quantNodeOptimizer graphPasses Failed";
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
