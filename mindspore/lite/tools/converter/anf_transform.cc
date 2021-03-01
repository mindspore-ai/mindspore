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

#include "tools/converter/anf_transform.h"
#include <memory>
#include <string>
#include "src/common/log_adapter.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "mindspore/core/ir/primitive.h"
#include "tools/optimizer/fusion/conv_biasadd_fusion.h"
#include "tools/optimizer/fusion/conv_activation_fusion.h"
#include "tools/optimizer/fusion/conv_tuple_activation_fusion.h"
#include "tools/optimizer/fusion/conv_scale_fusion.h"
#include "tools/optimizer/fusion/conv_bn_fusion.h"
#include "tools/optimizer/fusion/conv_tuplegetitem_fusion.h"
#include "tools/optimizer/fusion/constant_folding_fusion.h"
#include "tools/optimizer/fusion/tf_norm_fusion.h"
#include "tools/optimizer/fusion/onnx_layer_norm_fusion.h"
#include "tools/optimizer/fusion/batchmatmul_fusion.h"
#include "tools/optimizer/fusion/sigmoid_mul_fusion.h"
#include "tools/optimizer/fusion/conv_conv_fusion.h"
#include "tools/optimizer/fusion/tflite_lstm_cell_fusion.h"
#include "tools/optimizer/fusion/tf_lstm_cell_fusion.h"
#include "tools/optimizer/fusion/tf_bidirection_gru_fusion.h"
#include "tools/optimizer/fusion/tf_bidirection_gru_cf_fusion.h"
#include "tools/optimizer/graph/primitive_adjust_pass.h"
#include "tools/optimizer/graph/mindir_adjust_pass.h"
#include "tools/optimizer/graph/redundant_op_remove_pass.h"
#include "tools/optimizer/graph/weight_format_hardcode_pass.h"
#include "tools/optimizer/graph/weight_format_transform_pass.h"
#include "tools/optimizer/graph/clip_convert_activation_pass.h"
#include "tools/optimizer/graph/group_depthwise_op_convert_pass.h"
#include "tools/optimizer/graph/tflite_inputs_adjust_pass.h"
#include "tools/optimizer/graph/onnx_inputs_adjust_pass.h"
#include "tools/optimizer/graph/onnx_pad_adjust_pass.h"
#include "tools/optimizer/graph/update_conv2d_param_pass.h"
#include "tools/optimizer/graph/unused_node_remove_pass.h"
#include "tools/optimizer/graph/unused_cast_node_remove_pass.h"
#include "tools/optimizer/graph/unused_transpose_node_remove_pass.h"
#include "tools/optimizer/graph/infershape_pass.h"
#include "tools/optimizer/graph/slice_prepose_pass.h"
#include "tools/optimizer/graph/while_pass.h"
#include "tools/optimizer/graph/if_pass.h"
#include "tools/optimizer/graph/functionalize_control_op_pass.h"
#include "tools/optimizer/graph/inputs_adjust_pass.h"
#include "tools/converter/quantizer/post_training_quantizer.h"
#include "tools/converter/quantizer/quant_cast.h"
#include "tools/converter/quantizer/weight_quantizer.h"

using std::string;
namespace mindspore::lite {
AnfTransform::AnfTransform() = default;

AnfTransform::~AnfTransform() = default;

int AnfTransform::AddFusionPass(const std::shared_ptr<opt::GraphOptimizer> &optimizer, const converter::Flags *config) {
  auto fusion_pm = std::make_shared<opt::PassManager>("anf fusion pass manager", false);

  // for now - training is not supporting fuse operations
  if (!config->trainModel) {
    // remove quantdtype when awaretraining
    fusion_pm->AddPass(std::make_shared<opt::ConvBiasaddFusion>());
    auto conv_bn_pass = std::make_shared<opt::ConvBatchNormFusion>();
    conv_bn_pass->SetFmkType(config->fmk);
    fusion_pm->AddPass(conv_bn_pass);
    auto conv_scale_pass = std::make_shared<opt::ConvScaleFusion>();
    conv_scale_pass->SetFmkType(config->fmk);
    fusion_pm->AddPass(conv_scale_pass);
    fusion_pm->AddPass(std::make_shared<opt::TfNormFusion>());
    fusion_pm->AddPass(std::make_shared<opt::OnnxLayerNormFusion>());
    fusion_pm->AddPass(std::make_shared<opt::BatchMatMulFusion>());
    fusion_pm->AddPass(std::make_shared<opt::SigmoidMulFusion>());
    fusion_pm->AddPass(std::make_shared<opt::ConvActivationFusion>());
    fusion_pm->AddPass(std::make_shared<opt::ConvTupleGetItemFusion>());
    fusion_pm->AddPass(std::make_shared<opt::ConvTupleActivationFusion>());
    fusion_pm->AddPass(std::make_shared<opt::TfliteLstmCellFusion>());
    fusion_pm->AddPass(std::make_shared<opt::TfLstmCellFusion>());
    fusion_pm->AddPass(std::make_shared<opt::TfBidirectionGruFusion>());
  }
  if (config->fmk == lite::converter::FmkType_MS) {
    auto remove_unused_cast_pass = std::make_shared<opt::RemoveUnusedCastOpPass>();
    if (remove_unused_cast_pass == nullptr) {
      MS_LOG(ERROR) << "RemoveUnusedCastOpPass should be specified";
      return RET_ERROR;
    }
    remove_unused_cast_pass->SetFmkType(config->fmk);
    fusion_pm->AddPass(remove_unused_cast_pass);
  }
  if (config->fmk == lite::converter::FmkType_ONNX) {
    auto remove_unused_transpose_pass = std::make_shared<opt::RemoveUnusedTransposeOpPass>();
    if (remove_unused_transpose_pass == nullptr) {
      MS_LOG(ERROR) << "RemoveUnusedTransposeOpPass should be specified";
      return RET_ERROR;
    }
    remove_unused_transpose_pass->SetFmkType(config->fmk);
    fusion_pm->AddPass(remove_unused_transpose_pass);
  }
  fusion_pm->AddPass(std::make_shared<opt::ConvConvFusion>());
  optimizer->AddPassManager(fusion_pm);
  return RET_OK;
}

int AnfTransform::AddGraphPass(const std::shared_ptr<opt::GraphOptimizer> &optimizer, const converter::Flags *config) {
  auto graph_pm = std::make_shared<opt::PassManager>("anf graph pass manager", true);
  if (config->fmk == lite::converter::FmkType_TFLITE || config->fmk == lite::converter::FmkType_TF ||
      config->fmk == lite::converter::FmkType_ONNX) {
    graph_pm->AddPass(std::make_shared<opt::WhilePass>());
    graph_pm->AddPass(std::make_shared<opt::IfPass>());
  }
  auto weight_format_hardcode_pass = std::make_shared<opt::WeightFormatHardCodePass>();
  weight_format_hardcode_pass->SetFmkType(config->fmk);
  weight_format_hardcode_pass->SetQuantType(config->quantType);
  graph_pm->AddPass(weight_format_hardcode_pass);
  auto weight_format_transform_pass = std::make_shared<opt::WeightFormatTransformPass>();
  weight_format_transform_pass->SetFmkType(config->fmk);
  weight_format_transform_pass->SetQuantType(config->quantType);
  graph_pm->AddPass(weight_format_transform_pass);
  auto slice_prepose_pass = std::make_shared<opt::SlicePreposePass>();
  slice_prepose_pass->SetFmkType(config->fmk);
  graph_pm->AddPass(slice_prepose_pass);
  optimizer->AddPassManager(graph_pm);
  return RET_OK;
}

int AnfTransform::AddConvertPass(const std::shared_ptr<opt::GraphOptimizer> &optimizer,
                                 const converter::Flags *config) {
  auto convert_pm = std::make_shared<opt::PassManager>("anf graph convert pass manager", true);
  convert_pm->AddPass(std::make_shared<opt::ClipConvertActivationPass>());
  if (config->fmk == lite::converter::FmkType_TFLITE) {
    convert_pm->AddPass(std::make_shared<opt::GroupDepthwiseOpConvertPass>());
    convert_pm->AddPass(std::make_shared<opt::TfliteInputsAdjustPass>());
  }
  optimizer->AddPassManager(convert_pm);
  return RET_OK;
}

int AnfTransform::AddConstFoldPass(const std::shared_ptr<opt::GraphOptimizer> &optimizer,
                                   const converter::Flags *config) {
  auto const_fold_pm = std::make_shared<opt::PassManager>("const fold fusion pass manager", false);
  const_fold_pm->AddPass(std::make_shared<opt::RemoveRedundantOpPass>());
  if (!config->trainModel) {
    auto inne_context_ptr = std::make_shared<lite::InnerContext>();
    inne_context_ptr->Init();
    const_fold_pm->AddPass(std::make_shared<opt::ConstFoldPass>(inne_context_ptr));
  }
  auto update_conv2d_param_pass = std::make_shared<opt::UpdateConv2DParamPass>();
  update_conv2d_param_pass->SetFmkType(config->fmk);
  const_fold_pm->AddPass(update_conv2d_param_pass);
  auto weight_format_hardcode_pass = std::make_shared<opt::WeightFormatHardCodePass>();
  weight_format_hardcode_pass->SetFmkType(config->fmk);
  weight_format_hardcode_pass->SetQuantType(config->quantType);
  const_fold_pm->AddPass(weight_format_hardcode_pass);
  auto infershape_pass = std::make_shared<opt::InferShapePass>();
  infershape_pass->SetFmkType(config->fmk);
  const_fold_pm->AddPass(infershape_pass);
  optimizer->AddPassManager(const_fold_pm);
  return RET_OK;
}

int AnfTransform::RunAdjustPass(const FuncGraphPtr &old_graph, const converter::Flags *config) {
  if (config->fmk == converter::FmkType_MS) {
    if (RunMindirAdjustPass(old_graph, config) != RET_OK) {
      return RET_ERROR;
    }
  }
  auto adjust_input = std::make_shared<opt::InputAdjustPass>();
  if (!adjust_input->Run(old_graph)) {
    MS_LOG(ERROR) << "adjust input failed.";
    return RET_ERROR;
  }
  switch (config->fmk) {
    case converter::FmkType_ONNX:
      return RunOnnxAdjustPass(old_graph, config);
    case converter::FmkType_TF:
      return RunTFAdjustPass(old_graph, config);
    default:
      return RET_OK;
  }
}

int AnfTransform::RunMindirAdjustPass(const FuncGraphPtr &old_graph, const converter::Flags *config) {
  auto primitive_adjust_pass = std::make_shared<opt::PrimitiveAdjustPass>();
  primitive_adjust_pass->SetFmkType(config->fmk);
  if (!primitive_adjust_pass->Run(old_graph)) {
    MS_LOG(ERROR) << "primitive adjust failed.";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_ERROR);
    return RET_ERROR;
  }
  auto mindir_adjust_pass = std::make_shared<opt::MindirAdjustPass>();
  mindir_adjust_pass->SetFmkType(config->fmk);
  mindir_adjust_pass->SetQuantType(config->quantType);
  mindir_adjust_pass->SetTrainFlag(config->trainModel);
  if (!mindir_adjust_pass->Run(old_graph)) {
    MS_LOG(ERROR) << "mindir adjust failed.";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_ERROR);
    return RET_ERROR;
  }
  return RET_OK;
}

int AnfTransform::RunOnnxAdjustPass(const FuncGraphPtr &old_graph, const converter::Flags *config) {
  // onnx pre adjustment
  auto onnx_adjust_pass = std::make_shared<opt::OnnxInputAdjustOpPass>();
  if (!onnx_adjust_pass->Run(old_graph)) {
    MS_LOG(ERROR) << "onnx adjust failed.";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_ERROR);
    return RET_ERROR;
  }
  auto onnx_pad_adjust_pass = std::make_shared<opt::OnnxPadAdjustPass>();
  if (!onnx_pad_adjust_pass->Run(old_graph)) {
    MS_LOG(ERROR) << "onnx pad adjust failed.";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_ERROR);
    return RET_ERROR;
  }
  return RET_OK;
}

int AnfTransform::RunTFAdjustPass(const FuncGraphPtr &old_graph, const converter::Flags *config) {
  auto functionalize_control_op_pass = std::make_shared<opt::FunctionalizeControlOpPass>();
  if (!functionalize_control_op_pass->Run(old_graph)) {
    MS_LOG(ERROR) << "functionalize control op pass failed.";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_ERROR);
    return RET_ERROR;
  }
  return RET_OK;
}

int AnfTransform::RunPrecedingPass(const FuncGraphPtr &old_graph, const converter::Flags &config) {
  MS_ASSERT(old_graph != nullptr);
  auto asylic_optimizer = std::make_shared<opt::GraphOptimizer>();
  auto asylic_pm = std::make_shared<opt::PassManager>("asylic pass manager", false);
  // fuse tf1.x bidirection_gru into GRU, must be placed here because graph is cyclic
  asylic_pm->AddPass(std::make_shared<opt::TfBidirectionGruCfFusion>());
  // remove remaining cyclic nodes
  asylic_pm->AddPass(std::make_shared<opt::UnusedNodeRemovePass>());
  asylic_optimizer->AddPassManager(asylic_pm);
  if (!asylic_optimizer->Optimize(old_graph)) {
    MS_LOG(ERROR) << "gru cf fusion pass failed.";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_ERROR);
    return RET_ERROR;
  }
  return RET_OK;
}

int AnfTransform::DoQuantize(const FuncGraphPtr &old_graph, const converter::Flags *config,
                             const FuncGraphPtr &new_graph) {
  // quant
  if (config->quantType == schema::QuantType_PostTraining) {
    this->m_quantizer_ = std::make_unique<quant::PostTrainingQuantizer>(new_graph, config->configFile, config->bitNum);
    if (m_quantizer_ == nullptr) {
      MS_LOG(ERROR) << "New PostTrainingQuantizer failed";
      ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_MEMORY_FAILED);
      return RET_ERROR;
    }
  } else if (config->quantType == schema::QuantType_WeightQuant) {
    this->m_quantizer_ = std::make_unique<quant::WeightQuantizer>(new_graph, *config);
    if (m_quantizer_ == nullptr) {
      MS_LOG(ERROR) << "New WeightQuantizer failed";
      ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_MEMORY_FAILED);
      return RET_ERROR;
    }
  }
  if (m_quantizer_ != nullptr) {
    m_quantizer_->flags = *config;
    auto status = m_quantizer_->DoQuantize(new_graph);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Quant failed " << status;
      ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
      return RET_ERROR;
    }
  }
  return RET_OK;
}

FuncGraphPtr AnfTransform::TransformSingleFuncGraph(const FuncGraphPtr &old_graph, const converter::Flags *config) {
  MS_ASSERT(nullptr != old_graph);
  if (config == nullptr) {
    MS_LOG(ERROR) << "config should be specified";
    return nullptr;
  }
  if (old_graph->has_flag("HasTransformed")) {
    old_graph->set_flag("HasTransformed", false);
    return old_graph;
  }

  auto status = RunPrecedingPass(old_graph, *config);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Run Preceding pass failed.";
    return nullptr;
  }

  status = RunAdjustPass(old_graph, config);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Run Adjust pass failed.";
    return nullptr;
  }

  auto optimizer = std::make_shared<opt::GraphOptimizer>();

  status = AddConstFoldPass(optimizer, config);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Add const fold pass failed.";
    return nullptr;
  }

  status = AddConvertPass(optimizer, config);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Add convert pass failed.";
    return nullptr;
  }

  status = AddFusionPass(optimizer, config);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Add fusion pass failed.";
    return nullptr;
  }

  status = AddGraphPass(optimizer, config);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Add graph pass failed.";
    return nullptr;
  }

  auto new_graph = optimizer->Optimize(old_graph);
  if (new_graph == nullptr) {
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_NULL_PTR);
    return nullptr;
  }

  status = DoQuantize(old_graph, config, new_graph);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Do Quantize failed.";
    return nullptr;
  }

  return new_graph;
}

STATUS AnfTransform::GetAllFuncGraph(const FuncGraphPtr &main_graph, FuncGraphVector *subgraphs,
                                     std::vector<ValueNodePtr> *vnodes) {
  auto nodes = TopoSort(main_graph->get_return());
  for (auto &node : nodes) {
    auto fg = GetValueNode<FuncGraphPtr>(node);
    if (fg) {
      vnodes->push_back(utils::cast<ValueNodePtr>(node));
      subgraphs->push_back(fg);
    }
  }
  return RET_OK;
}

FuncGraphPtr AnfTransform::Transform(const FuncGraphPtr &main_graph, const converter::Flags *config) {
  // transform main_graph
  auto new_main_graph = TransformSingleFuncGraph(main_graph, config);
  if (new_main_graph == nullptr) {
    MS_LOG(ERROR) << "TransformSingleFuncGraph failed.";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_ERROR);
    return nullptr;
  }

  // transform sub_graph
  FuncGraphVector subgraphs{};
  std::vector<ValueNodePtr> vnodes{};
  int ret = GetAllFuncGraph(main_graph, &subgraphs, &vnodes);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "GetAllFuncGraph failed " << ret;
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(ret);
    return nullptr;
  }
  for (size_t i = 0; i < subgraphs.size(); i++) {
    auto new_graph = Transform(subgraphs.at(i), config);
    new_graph->set_flag("HasTransformed", true);
    vnodes.at(i)->set_value(new_graph);
  }

  return new_main_graph;
}
}  // namespace mindspore::lite
