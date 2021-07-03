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
#include <unordered_map>
#include "src/common/log_adapter.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "mindspore/core/ir/primitive.h"
#include "tools/optimizer/fusion/affine_activation_fusion.h"
#include "tools/optimizer/fusion/affine_fusion.h"
#include "tools/optimizer/fusion/conv_biasadd_fusion.h"
#include "tools/optimizer/fusion/conv_activation_fusion.h"
#include "tools/optimizer/fusion/conv_tuple_activation_fusion.h"
#include "tools/optimizer/fusion/conv_scale_fusion.h"
#include "tools/optimizer/fusion/conv_bn_fusion.h"
#include "tools/optimizer/fusion/conv_tuplegetitem_fusion.h"
#include "tools/optimizer/fusion/constant_folding_fusion.h"
#include "tools/optimizer/fusion/norm_fusion.h"
#include "tools/optimizer/fusion/batchmatmul_fusion.h"
#include "tools/optimizer/fusion/sigmoid_mul_fusion.h"
#include "tools/optimizer/fusion/conv_conv_fusion.h"
#include "tools/optimizer/fusion/tflite_lstm_cell_fusion.h"
#include "tools/optimizer/fusion/tf_lstm_cell_fusion.h"
#include "tools/optimizer/fusion/tf_bidirection_gru_fusion.h"
#include "tools/optimizer/fusion/multi_head_attention_fusion.h"
#include "tools/optimizer/fusion/glu_fusion.h"
#include "tools/optimizer/fusion/tflite_rel_pos_multi_head_attention_fusion.h"
#include "tools/optimizer/fusion/matmul_add_fusion.h"
#include "tools/optimizer/fusion/tf_gelu_fusion.h"
#include "tools/optimizer/fusion/onnx_gelu_fusion.h"
#include "tools/optimizer/fusion/squeeze_fusion.h"
#include "tools/optimizer/fusion/reshape_reshape_fusion.h"
#include "tools/optimizer/graph/add_tensor_array.h"
#include "tools/optimizer/graph/redundant_op_remove_pass.h"
#include "tools/optimizer/graph/clip_convert_activation_pass.h"
#include "tools/optimizer/graph/update_conv2d_param_pass.h"
#include "tools/optimizer/graph/unused_cast_node_remove_pass.h"
#include "tools/optimizer/graph/infershape_pass.h"
#include "tools/optimizer/graph/slice_prepose_pass.h"
#include "tools/optimizer/graph/while_pass.h"
#include "tools/optimizer/graph/if_pass.h"
#include "tools/optimizer/graph/reduce_same_act_pass.h"
#include "tools/optimizer/graph/split_one_pass.h"
#include "tools/optimizer/graph/unify_format_pass.h"
#include "tools/converter/quantizer/post_training_quantizer.h"
#include "tools/converter/quantizer/quant_cast.h"
#include "tools/converter/quantizer/weight_quantizer.h"
#include "tools/optimizer/parallel/split_strategy.h"
#include "tools/optimizer/parallel/spliter.h"
#include "tools/optimizer/fisson/iter_node_outputs.h"
#include "tools/optimizer/fisson/node_out_shapes.h"
#include "tools/optimizer/parallel/parallel_pass.h"
#include "include/registry/pass_registry.h"
#include "tools/optimizer/fisson/multi_conv_split_pass.h"

using std::string;
namespace mindspore::lite {
AnfTransform::AnfTransform() = default;

AnfTransform::~AnfTransform() = default;

int AnfTransform::RunFusionPass(const FuncGraphPtr &old_graph, const converter::Flags *config) {
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto fusion_pm = std::make_shared<opt::PassManager>("anf fusion pass manager", false);

  // for now - training is not supporting fuse operations
  if (!config->trainModel) {
    // remove quantdtype when awaretraining
    fusion_pm->AddPass(std::make_shared<opt::SqueezeFusion>());
    fusion_pm->AddPass(std::make_shared<opt::ReshapeReshapeFusion>());
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
    fusion_pm->AddPass(std::make_shared<opt::TfGeLUFusion>());
    fusion_pm->AddPass(std::make_shared<opt::OnnxGeLUFusion>());
    fusion_pm->AddPass(std::make_shared<opt::TfliteRelPosMultiHeadAttentionFusion>());
    fusion_pm->AddPass(std::make_shared<opt::GLUFusion>());
    fusion_pm->AddPass(std::make_shared<opt::ConstFoldPass>(config->fmk));
    fusion_pm->AddPass(std::make_shared<opt::AffineFusion>());
    fusion_pm->AddPass(std::make_shared<opt::AffineActivationFusion>());
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
  fusion_pm->AddPass(std::make_shared<opt::ConvConvFusion>());
  if (!config->trainModel) {
    fusion_pm->AddPass(std::make_shared<opt::MatMulAddFusion>());
  }
  optimizer->AddPassManager(fusion_pm);
  if (optimizer->Optimize(old_graph) == nullptr) {
    MS_LOG(ERROR) << "run op fusion failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int AnfTransform::RunParallelPass(const FuncGraphPtr &old_graph, const converter::Flags *config) {
  MS_LOG(DEBUG) << "Run ParallelPass start";
  if (config->trainModel || config->parallel_split_config_.parallel_split_type_ == converter::SplitNo) {
    return RET_OK;
  }
  if (config->parallel_split_config_.parallel_split_type_ == converter::SplitByUserRatio) {
    auto optimizer = std::make_shared<opt::GraphOptimizer>();
    auto graph_inputs = old_graph->get_inputs();
    opt::SplitMode split_mode = opt::NoSplit;
    for (const auto &graph_input : graph_inputs) {
      if (utils::isa<Parameter>(graph_input)) {
        auto input_parameter = dyn_cast<Parameter>(graph_input);
        MSLITE_CHECK_PTR(input_parameter->Shape());
        auto shape_ptr = input_parameter->Shape()->cast<abstract::ShapePtr>();
        MSLITE_CHECK_PTR(shape_ptr);
        auto batch = shape_ptr->shape().front();
        if (batch > opt::kDefaultBatch) {
          split_mode = opt::SplitN;
        } else {
          split_mode = opt::SplitH;
        }
        break;
      }
    }
    // 1. deal with split strategy
    std::unordered_map<std::string, opt::SplitStrategy> split_strategys =
      opt::ParserSplitStrategy(config->parallel_split_config_.parallel_compute_rates_,
                               config->parallel_split_config_.parallel_devices_, split_mode);
    if (split_strategys.empty()) {
      MS_LOG(ERROR) << "parse split_strategy error.";
      return RET_OK;
    }
    opt::Spliter::GetInstance()->RecordGraphInfo(old_graph);
    auto parallel_pm = std::make_shared<opt::PassManager>("anf parallel pass manager", true);
    // 2. preceding parallel pass
    parallel_pm->AddPass(std::make_shared<opt::IterNodeOutputs>());
    parallel_pm->AddPass(std::make_shared<opt::NodeOutShapes>());
    std::set<int, opt::IntCompare> match_multi_numbers = opt::Spliter::GetInstance()->graph_match_multi_numbers();
    int max_match_number = *match_multi_numbers.begin();
    // we do not deal with single conv node
    for (int match_number = max_match_number; match_number > opt::kDefaultBatch; --match_number) {
      // 3. multi_conv parallel pass
      parallel_pm->AddPass(std::make_shared<opt::MultiConvSplitPass>(split_strategys, config->fmk, match_number));
      parallel_pm->AddPass(std::make_shared<opt::IterNodeOutputs>());
      parallel_pm->AddPass(std::make_shared<opt::NodeOutShapes>());
    }
    optimizer->AddPassManager(parallel_pm);
    if (optimizer->Optimize(old_graph) == nullptr) {
      MS_LOG(ERROR) << "run const fold failed.";
      return RET_ERROR;
    }
  }
  MS_LOG(DEBUG) << "Run ParallelPass end";
  return RET_OK;
}

int AnfTransform::RunGraphPass(const FuncGraphPtr &old_graph, const converter::Flags *config) {
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto graph_pm = std::make_shared<opt::PassManager>("anf graph pass manager", true);
  if (config->fmk == lite::converter::FmkType_TFLITE || config->fmk == lite::converter::FmkType_TF ||
      config->fmk == lite::converter::FmkType_ONNX) {
    graph_pm->AddPass(std::make_shared<opt::WhilePass>());
    graph_pm->AddPass(std::make_shared<opt::IfPass>());
  }
  auto slice_prepose_pass = std::make_shared<opt::SlicePreposePass>();
  slice_prepose_pass->SetFmkType(config->fmk);
  graph_pm->AddPass(slice_prepose_pass);
  graph_pm->AddPass(std::make_shared<opt::AddTensorArray>());
  optimizer->AddPassManager(graph_pm);
  if (optimizer->Optimize(old_graph) == nullptr) {
    MS_LOG(ERROR) << "run  graph pass failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int AnfTransform::RunConvertPass(const FuncGraphPtr &old_graph, const converter::Flags *config) {
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto convert_pm = std::make_shared<opt::PassManager>("anf graph convert pass manager", true);
  convert_pm->AddPass(std::make_shared<opt::ClipConvertActivationPass>());
  optimizer->AddPassManager(convert_pm);
  if (optimizer->Optimize(old_graph) == nullptr) {
    MS_LOG(ERROR) << "run graph convert pass failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int AnfTransform::RunConstFoldPass(const FuncGraphPtr &old_graph, const converter::Flags *config) {
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto const_fold_pm = std::make_shared<opt::PassManager>("const fold fusion pass manager", false);
  const_fold_pm->AddPass(std::make_shared<opt::RemoveRedundantOpPass>());
  if (!config->trainModel) {
    const_fold_pm->AddPass(std::make_shared<opt::ConstFoldPass>(config->fmk));
  }
  auto update_conv2d_param_pass = std::make_shared<opt::UpdateConv2DParamPass>();
  update_conv2d_param_pass->SetFmkType(config->fmk);
  const_fold_pm->AddPass(update_conv2d_param_pass);
  auto infershape_pass = std::make_shared<opt::InferShapePass>();
  infershape_pass->SetFmkType(config->fmk);
  const_fold_pm->AddPass(infershape_pass);
  optimizer->AddPassManager(const_fold_pm);
  if (optimizer->Optimize(old_graph) == nullptr) {
    MS_LOG(ERROR) << "run const fold failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

STATUS AnfTransform::RunPluginPass(const FuncGraphPtr &old_graph, int position) {
  auto instance = opt::PassRegistry::GetInstance();
  auto plugin_passes = instance->GetPasses();
  if (plugin_passes.find(position) == plugin_passes.end()) {
    MS_LOG(DEBUG) << "there is no plugin pass in current position.";
    return RET_OK;
  }

  auto plugin_pass = plugin_passes.at(position);
  if (!plugin_pass->Run(old_graph)) {
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_ERROR);
    return RET_ERROR;
  }
  return RET_OK;
}

int AnfTransform::DoQuantize(const FuncGraphPtr &old_graph, const converter::Flags *config) {
  // quant
  if (config->quantType == schema::QuantType_PostTraining) {
    this->m_quantizer_ = std::make_unique<quant::PostTrainingQuantizer>(old_graph, config->configFile, config->bitNum);
    if (m_quantizer_ == nullptr) {
      MS_LOG(ERROR) << "New PostTrainingQuantizer failed";
      ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_MEMORY_FAILED);
      return RET_ERROR;
    }
  } else if (config->quantType == schema::QuantType_WeightQuant) {
    this->m_quantizer_ = std::make_unique<quant::WeightQuantizer>(old_graph, *config);
    if (m_quantizer_ == nullptr) {
      MS_LOG(ERROR) << "New WeightQuantizer failed";
      ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_MEMORY_FAILED);
      return RET_ERROR;
    }
  }
  if (m_quantizer_ != nullptr) {
    m_quantizer_->flags = *config;
    auto status = m_quantizer_->DoQuantize(old_graph);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Quant failed " << status;
      ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
      return RET_ERROR;
    }
  }
  return RET_OK;
}

FuncGraphPtr AnfTransform::TransformFuncGraph(const FuncGraphPtr &old_graph, const converter::Flags *config) {
  MS_ASSERT(old_graph != nullptr);
  if (config == nullptr) {
    MS_LOG(ERROR) << "config should be specified";
    return nullptr;
  }
  int status;
  for (auto &fg : func_graphs_) {
    status = RunConstFoldPass(fg, config);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Run const fold pass failed.";
      return nullptr;
    }

    status = RunConvertPass(fg, config);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Run convert pass failed.";
      return nullptr;
    }
  }

  auto format_pass = std::make_shared<opt::UnifyFormatPass>();
  format_pass->Init(config->fmk, config->trainModel);
  if (!format_pass->RunOnlyForShape(old_graph)) {
    MS_LOG(ERROR) << "Run format pass failed.";
    return nullptr;
  }

  status = RunPluginPass(old_graph, opt::POSITION_BEGIN);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Run plugin pass failed.";
    return nullptr;
  }

  auto reduce_act_pass = std::make_shared<opt::ReduceSameActPass>();
  for (auto &fg : func_graphs_) {
    if (!reduce_act_pass->Run(fg)) {
      MS_LOG(ERROR) << "Run reduce same act pass failed.";
      return nullptr;
    }
  }

  auto split_one_pass = std::make_shared<opt::SplitOnePass>();
  for (auto &fg : func_graphs_) {
    if (!split_one_pass->Run(fg)) {
      MS_LOG(ERROR) << "Run split one pass failed.";
      return nullptr;
    }
  }

  for (auto &fg : func_graphs_) {
    if (!config->disableFusion) {
      status = RunFusionPass(fg, config);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "Run fusion pass failed.";
        return nullptr;
      }
    }
  }

  format_pass = std::make_shared<opt::UnifyFormatPass>();
  format_pass->Init(config->fmk, config->trainModel);
  if (!format_pass->Run(old_graph)) {
    MS_LOG(ERROR) << "Run format pass failed.";
    return nullptr;
  }

  status = RunPluginPass(old_graph, opt::POSITION_END);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Run plugin pass failed.";
    return nullptr;
  }

  for (auto &fg : func_graphs_) {
    status = RunGraphPass(fg, config);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Run convert pass failed.";
      return nullptr;
    }

    status = RunParallelPass(fg, config);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Run convert pass failed.";
      return nullptr;
    }

    status = DoQuantize(fg, config);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Do Quantize failed.";
      return nullptr;
    }
  }
  return old_graph;
}

void AnfTransform::GetAllFuncGraph(const FuncGraphPtr &func_graph) {
  if (func_graphs_.find(func_graph) == func_graphs_.end()) {
    func_graphs_.insert(func_graph);
  } else {
    return;
  }

  auto nodes = func_graph->nodes();
  for (auto &node : nodes) {
    if (IsValueNode<FuncGraph>(node)) {
      auto new_fg = (node->cast<ValueNodePtr>()->value())->cast<FuncGraphPtr>();
      GetAllFuncGraph(new_fg);
    }
    if (utils::isa<CNodePtr>(node)) {
      auto cnode = node->cast<CNodePtr>();
      for (auto &input : cnode->inputs()) {
        if (input->isa<ValueNode>()) {
          if (IsValueNode<FuncGraph>(input)) {
            auto new_fg = (input->cast<ValueNodePtr>()->value())->cast<FuncGraphPtr>();
            GetAllFuncGraph(new_fg);
          }
        }
      }
    }
  }
}

FuncGraphPtr AnfTransform::Transform(const FuncGraphPtr &main_graph, const converter::Flags *config) {
  GetAllFuncGraph(main_graph);
  auto new_graph = TransformFuncGraph(main_graph, config);
  if (new_graph == nullptr) {
    MS_LOG(ERROR) << "optimizer failed.";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_NULL_PTR);
  }
  return new_graph;
}
}  // namespace mindspore::lite
