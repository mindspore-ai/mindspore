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
#include "tools/optimizer/fusion/conv_scale_fusion.h"
#include "tools/optimizer/fusion/conv_bn_fusion.h"
#include "tools/optimizer/fusion/constant_folding_fusion.h"

using std::string;
namespace mindspore {
namespace lite {
AnfTransform::AnfTransform() = default;

AnfTransform::~AnfTransform() = default;

void AnfTransform::SetGraphDef(schema::MetaGraphT *_dstDef) { graphDefT = _dstDef; }

FuncGraphPtr AnfTransform::Transform(const FuncGraphPtr &old_graph) {
  // return old_graph;
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::ConvBiasaddFusion>());
  pm->AddPass(std::make_shared<opt::ConvBatchNormFusion>());
  pm->AddPass(std::make_shared<opt::ConvScaleFusion>());
  pm->AddPass(std::make_shared<opt::ConvActivationFusion>(true, "conv_relu", schema::PrimitiveType_Activation,
                                                         schema::ActivationType_RELU));
  pm->AddPass(std::make_shared<opt::ConvActivationFusion>(true, "conv_relu6", schema::PrimitiveType_Activation,
                                                         schema::ActivationType_RELU6));
  pm->AddPass(std::make_shared<opt::ConstFoldPass>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(old_graph);
  return new_graph;
}
}  // namespace lite
}  // namespace mindspore

