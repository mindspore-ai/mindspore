/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "tools/converter/anf_transform.h"
#include <memory>
#include <string>
#include "utils/log_adapter.h"
#include "tools/optimizer/fusion/conv_biasadd_fusion.h"
#include "tools/optimizer/fusion/conv_activation_fusion.h"
#include "tools/optimizer/fusion/conv_tuple_activation_fusion.h"
#include "tools/optimizer/fusion/conv_scale_fusion.h"
#include "tools/optimizer/fusion/conv_bn_fusion.h"
#include "tools/optimizer/fusion/constant_folding_fusion.h"
#include "tools/converter/quantizer/post_training_quantizer.h"
#include "tools/converter/quantizer/quant_cast.h"
#include "tools/converter/quantizer/weight_quantizer.h"

using std::string;
namespace mindspore {
namespace lite {
AnfTransform::AnfTransform() = default;

AnfTransform::~AnfTransform() = default;

FuncGraphPtr AnfTransform::Transform(const FuncGraphPtr &old_graph, const converter::Flags *config) {
  MS_ASSERT(nullptr != old_graph);
  // fusion const_fold
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>("anf fusion pass manager", false);

  // for now - trainning is not supporting fuse operations
  if (config != nullptr && config->trainModel == false) {
    pm->AddPass(std::make_shared<opt::ConvBiasaddFusion>());
    pm->AddPass(std::make_shared<opt::ConvBatchNormFusion>());
    pm->AddPass(std::make_shared<opt::ConvScaleFusion>());
    pm->AddPass(std::make_shared<opt::ConvActivationFusion>(true, "conv_relu", schema::PrimitiveType_Activation,
                                                            schema::ActivationType_RELU));
    pm->AddPass(std::make_shared<opt::ConvActivationFusion>(true, "conv_relu6", schema::PrimitiveType_Activation,
                                                            schema::ActivationType_RELU6));
    pm->AddPass(std::make_shared<opt::ConvTupleActivationFusion>(true, "conv_tuple_relu",
                                                                schema::PrimitiveType_Activation,
                                                                schema::ActivationType_RELU));
    pm->AddPass(std::make_shared<opt::ConvTupleActivationFusion>(true, "conv_tuple_relu6",
                                                                schema::PrimitiveType_Activation,
                                                                schema::ActivationType_RELU6));
  }

  pm->AddPass(std::make_shared<opt::ConstFoldPass>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(old_graph);
  if (new_graph == nullptr) {
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_NULL_PTR);
    return nullptr;
  }
  // quant
  if (config != nullptr) {
    if (config->quantType == schema::QuantType_PostTraining) {
      this->mQuantizer = std::make_unique<quant::PostTrainingQuantizer>(new_graph, config->configFile, 8);
      if (mQuantizer == nullptr) {
        MS_LOG(ERROR) << "New PostTrainingQuantizer failed";
        ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_MEMORY_FAILED);
        return nullptr;
      }
    } else if (config->quantType == schema::QuantType_WeightQuant) {
      auto bitNum = static_cast<size_t>(std::stoull(config->bitNum));
      if (bitNum != quant::UINT8_QUANTIZATION) {
        MS_LOG(ERROR) << "Current Only Support 8 bit weight quant";
        ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_ERROR);
        return nullptr;
      }
      this->mQuantizer = std::make_unique<quant::WeightQuantizer>(
        new_graph, config->quantSize, config->convWeightQuantChannelThreshold, config->bitNum);
      if (mQuantizer == nullptr) {
        MS_LOG(ERROR) << "New WeightQuantizer failed";
        ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_MEMORY_FAILED);
        return nullptr;
      }
    }
  }
  if (mQuantizer != nullptr) {
    mQuantizer->flags = *config;
    auto status = mQuantizer->DoQuantize(new_graph);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Quant failed " << status;
      ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
      return nullptr;
    }
    if (config->quantType == schema::QuantType_PostTraining) {
      quant::QuantCast quant_cast;
      quant_cast.SetInputDataDType(kNumberTypeFloat32);
      status = quant_cast.Run(new_graph);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "add QuantCast error";
        ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
        return nullptr;
      }
    }
  }

  return new_graph;
}
}  // namespace lite
}  // namespace mindspore
