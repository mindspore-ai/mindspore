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
#include <deque>
#include "nnacl/op_base.h"
#include "src/common/log_adapter.h"
#include "tools/converter/optimizer_manager.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "ir/primitive.h"
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
#include "tools/optimizer/fusion/conv_pad_fusion.h"
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
#include "tools/optimizer/graph/control_flow_pass.h"
#include "tools/optimizer/graph/reduce_same_act_pass.h"
#include "tools/optimizer/graph/split_one_pass.h"
#include "tools/optimizer/graph/decrease_transpose_algo.h"
#include "tools/optimizer/graph/specify_graph_input_format.h"
#include "tools/optimizer/graph/dump_graph.h"
#include "tools/converter/quantizer/full_quant_quantizer.h"
#include "tools/converter/quantizer/quant_cast.h"
#include "tools/converter/quantizer/weight_quantizer.h"
#include "tools/optimizer/parallel/split_strategy.h"
#include "tools/optimizer/parallel/spliter.h"
#include "tools/optimizer/fisson/iter_node_outputs.h"
#include "tools/optimizer/fisson/node_out_shapes.h"
#include "tools/optimizer/parallel/parallel_pass.h"
#include "include/registry/pass_registry.h"
#include "tools/optimizer/fisson/multi_conv_split_pass.h"
#include "tools/optimizer/fusion/transpose_fusion.h"
#include "tools/optimizer/format/to_nchw_format.h"
#include "tools/optimizer/format/to_nhwc_format.h"
#include "tools/converter/acl/acl_pass.h"

using std::string;
namespace mindspore::lite {
AnfTransform::AnfTransform() = default;

AnfTransform::~AnfTransform() = default;

STATUS AnfTransform::MarkTrainInputOp(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  for (size_t i = 1; i < cnode->inputs().size(); i++) {
    auto input_node = cnode->input(i);
    if (!utils::isa<CNodePtr>(input_node)) {
      continue;
    }
    auto input_cnode = utils::cast<CNodePtr>(input_node);
    MS_CHECK_TRUE_RET(input_cnode != nullptr, RET_ERROR);
    auto prim = GetValueNode<PrimitivePtr>(input_cnode->input(0));
    if (prim == nullptr) {
      MS_LOG(DEBUG) << "Primitive is nullptr.";
      continue;
    }
    prim->AddAttr("trainOp", MakeValue(true));
  }
  return RET_OK;
}

STATUS AnfTransform::MarkTrainWeightSharingOp(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  auto node_list = TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    auto graph_cnode = utils::cast<CNodePtr>(node);
    MS_CHECK_TRUE_RET(graph_cnode != nullptr, RET_ERROR);
    auto graph_prim = GetValueNode<PrimitivePtr>(graph_cnode->input(0));
    if (graph_prim == nullptr) {
      MS_LOG(DEBUG) << "Primitive is nullptr.";
      continue;
    }
    for (size_t i = 1; i < graph_cnode->inputs().size(); i++) {
      for (size_t j = 1; j < cnode->inputs().size(); j++) {
        if ((graph_cnode->input(i) == cnode->input(j)) && utils::isa<Parameter>(cnode->input(j))) {
          graph_prim->AddAttr("trainOp", MakeValue(true));
        }
      }
    }
  }
  return RET_OK;
}

STATUS AnfTransform::MarkTrainOp(const FuncGraphPtr &func_graph) {
  auto node_list = TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    auto cnode = utils::cast<CNodePtr>(node);
    MS_CHECK_TRUE_RET(cnode != nullptr, RET_ERROR);
    auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    if (prim == nullptr) {
      MS_LOG(DEBUG) << "Primitive is nullptr.";
      continue;
    }
    if (opt::IsTrainOp(cnode)) {
      prim->AddAttr("trainOp", MakeValue(true));
      auto status = MarkTrainInputOp(func_graph, cnode);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "MarkTrainInputOp failed.";
        return RET_ERROR;
      }
      status = MarkTrainWeightSharingOp(func_graph, cnode);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "MarkTrainWeightSharingOp failed.";
        return RET_ERROR;
      }
    }
  }
  return RET_OK;
}

int AnfTransform::RunFusionPass(const FuncGraphPtr &old_graph, const converter::Flags *config) {
  auto status = MarkTrainOp(old_graph);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "MarkTrainOp failed.";
    return RET_ERROR;
  }
  CHECK_NULL_RETURN(config);
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  CHECK_NULL_RETURN(optimizer);
  auto fusion_pm = std::make_shared<opt::PassManager>("anf fusion pass manager", false);
  CHECK_NULL_RETURN(fusion_pm);

  // The training model only does the fusion of the inference part
  // remove quantdtype when awaretraining
  fusion_pm->AddPass(std::make_shared<opt::SqueezeFusion>());
  fusion_pm->AddPass(std::make_shared<opt::TransposeFusion>());
  fusion_pm->AddPass(std::make_shared<opt::ReshapeReshapeFusion>());
  fusion_pm->AddPass(std::make_shared<opt::ConvBiasaddFusion>());
  fusion_pm->AddPass(std::make_shared<opt::ConvBatchNormFusion>(config->fmk));
  fusion_pm->AddPass(std::make_shared<opt::ConvScaleFusion>(config->fmk));
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
  if (config->fmk == converter::kFmkTypeMs && !config->trainModel) {
    auto remove_unused_cast_pass = std::make_shared<opt::RemoveUnusedCastOpPass>();
    if (remove_unused_cast_pass == nullptr) {
      MS_LOG(ERROR) << "RemoveUnusedCastOpPass should be specified";
      return RET_ERROR;
    }
    remove_unused_cast_pass->SetFmkType(config->fmk);
    fusion_pm->AddPass(remove_unused_cast_pass);
  }
  fusion_pm->AddPass(std::make_shared<opt::ConvConvFusion>());
  fusion_pm->AddPass(std::make_shared<opt::ConvPadFusion>());
  fusion_pm->AddPass(std::make_shared<opt::MatMulAddFusion>());
  optimizer->AddPassManager(fusion_pm);
  if (optimizer->Optimize(old_graph) == nullptr) {
    MS_LOG(ERROR) << "run op fusion failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int AnfTransform::RunParallelPass(const FuncGraphPtr &old_graph, const converter::Flags *config) {
  CHECK_NULL_RETURN(old_graph);
  CHECK_NULL_RETURN(config);
  MS_LOG(DEBUG) << "Run ParallelPass start";
  if (config->trainModel || config->parallel_split_config_.parallel_split_type_ == converter::SplitNo) {
    return RET_OK;
  }
  if (config->parallel_split_config_.parallel_split_type_ == converter::SplitByUserRatio) {
    auto optimizer = std::make_shared<opt::GraphOptimizer>();
    CHECK_NULL_RETURN(optimizer);
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
    CHECK_NULL_RETURN(parallel_pm);
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
  CHECK_NULL_RETURN(old_graph);
  CHECK_NULL_RETURN(config);
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  CHECK_NULL_RETURN(optimizer);
  auto graph_pm = std::make_shared<opt::PassManager>("anf graph pass manager", true);
  CHECK_NULL_RETURN(graph_pm);
  if (config->fmk == converter::kFmkTypeTflite || config->fmk == converter::kFmkTypeTf ||
      config->fmk == converter::kFmkTypeOnnx) {
    graph_pm->AddPass(std::make_shared<opt::ControlFlowPass>());
  }
  auto slice_prepose_pass = std::make_shared<opt::SlicePreposePass>();
  CHECK_NULL_RETURN(slice_prepose_pass);
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
#ifdef ENABLE_LITE_ACL
  auto acl_pass = std::make_shared<opt::AclPass>(config->fmk);
  if (!acl_pass->Run(old_graph)) {
    MS_LOG(ERROR) << "Acl pass failed.";
    return RET_ERROR;
  }
#endif
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  CHECK_NULL_RETURN(optimizer);
  auto convert_pm = std::make_shared<opt::PassManager>("anf graph convert pass manager", true);
  CHECK_NULL_RETURN(convert_pm);
  convert_pm->AddPass(std::make_shared<opt::RemoveRedundantOpPass>(config->trainModel));
  auto infershape_pass = std::make_shared<opt::InferShapePass>(config->fmk, config->trainModel);
  CHECK_NULL_RETURN(infershape_pass);
  convert_pm->AddPass(infershape_pass);
  auto update_conv2d_param_pass = std::make_shared<opt::UpdateConv2DParamPass>();
  convert_pm->AddPass(update_conv2d_param_pass);
  optimizer->AddPassManager(convert_pm);
  if (optimizer->Optimize(old_graph) == nullptr) {
    MS_LOG(ERROR) << "run graph convert pass failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int AnfTransform::RunConstFoldPass(const FuncGraphPtr &old_graph, const converter::Flags *config) {
  CHECK_NULL_RETURN(config);
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto const_fold_pm = std::make_shared<opt::PassManager>("const fold fusion pass manager", false);
  CHECK_NULL_RETURN(optimizer);
  CHECK_NULL_RETURN(const_fold_pm);
  if (!config->trainModel) {
    const_fold_pm->AddPass(std::make_shared<opt::ConstFoldPass>(config->fmk));
  }
  const_fold_pm->AddPass(std::make_shared<opt::InferShapePass>(config->fmk, config->trainModel));
  const_fold_pm->AddPass(std::make_shared<opt::UpdateConv2DParamPass>());
  const_fold_pm->AddPass(std::make_shared<opt::ClipConvertActivationPass>());
  optimizer->AddPassManager(const_fold_pm);
  if (optimizer->Optimize(old_graph) == nullptr) {
    MS_LOG(ERROR) << "run const fold failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

void AnfTransform::GetFuncGraphs(const FuncGraphPtr &func_graph, std::set<FuncGraphPtr> *all_func_graphs) {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(all_func_graphs != nullptr);
  all_func_graphs->insert(func_graph);
  auto nodes = func_graph->GetOrderedCnodes();
  std::deque<CNodePtr> to_process{};
  to_process.insert(to_process.end(), nodes.begin(), nodes.end());
  while (!to_process.empty()) {
    auto &cur_cnode = to_process.front();
    for (auto &input : cur_cnode->inputs()) {
      if (!IsValueNode<FuncGraph>(input)) {
        continue;
      }
      auto new_fg = GetValueNode<FuncGraphPtr>(input);
      if (all_func_graphs->find(new_fg) != all_func_graphs->end()) {
        continue;
      }
      all_func_graphs->insert(new_fg);
      auto new_nodes = new_fg->GetOrderedCnodes();
      to_process.insert(to_process.end(), new_nodes.begin(), new_nodes.end());
    }
    to_process.pop_front();
  }
}

int AnfTransform::DoSingleGraphQuantize(const FuncGraphPtr &old_graph, const converter::Flags *config) {
  // quant
  if (config->commonQuantParam.quant_type == schema::QuantType_QUANT_ALL) {
    this->m_quantizer_ = std::make_unique<quant::FullQuantQuantizer>(old_graph, config->commonQuantParam.bit_num);
    if (m_quantizer_ == nullptr) {
      MS_LOG(ERROR) << "New FullQuantQuantizer failed";
      ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_MEMORY_FAILED);
      return RET_ERROR;
    }
  } else if (config->commonQuantParam.quant_type == schema::QuantType_QUANT_WEIGHT) {
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
      MS_LOG(ERROR) << "DoQuantization failed " << status;
      ReturnCode::GetSingleReturnCode()->UpdateReturnCode(status);
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int AnfTransform::DoQuantize(const FuncGraphPtr &old_graph, const converter::Flags *config) {
  std::set<FuncGraphPtr> all_func_graphs{};
  GetFuncGraphs(old_graph, &all_func_graphs);
  for (auto &item : all_func_graphs) {
    auto status = DoSingleGraphQuantize(item, config);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Do Quantize failed.";
      return status;
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

  auto status = RunConvertPass(old_graph, config);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Run convert pass failed.";
    return nullptr;
  }

  if (!RunExternalPass(old_graph, registry::POSITION_BEGIN)) {
    MS_LOG(ERROR) << "Run external pass failed, place is BEGIN";
    return nullptr;
  }

  status = RunConstFoldPass(old_graph, config);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Run const fold pass failed.";
    return nullptr;
  }

  if (!RunOptimizerPass(old_graph, {"InferShapePass"})) {
    MS_LOG(WARNING) << "Run infershape opt pass failed.";
  } else {
    if (!RunOptimizerPass(old_graph, {"DecreaseTransposeAlgo"})) {
      MS_LOG(ERROR) << "Run transpose opt pass failed.";
      return nullptr;
    }
  }

  auto reduce_act_pass = std::make_shared<opt::ReduceSameActPass>();
  MS_CHECK_TRUE_RET(reduce_act_pass != nullptr, nullptr);
  if (!reduce_act_pass->Run(old_graph)) {
    MS_LOG(ERROR) << "Run reduce same act pass failed.";
    return nullptr;
  }

  auto split_one_pass = std::make_shared<opt::SplitOnePass>();
  MS_CHECK_TRUE_RET(split_one_pass != nullptr, nullptr);
  if (!split_one_pass->Run(old_graph)) {
    MS_LOG(ERROR) << "Run split one pass failed.";
    return nullptr;
  }

  if (!config->disableFusion) {
    status = RunFusionPass(old_graph, config);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Run fusion pass failed.";
      return nullptr;
    }
  }

  if (!RunExternalPass(old_graph, registry::POSITION_END)) {
    MS_LOG(ERROR) << "Run external pass failed, place is END";
    return nullptr;
  }

  if (!RunOptimizerPass(old_graph, {"InferShapePass"})) {
    MS_LOG(WARNING) << "Run infershape opt pass failed.";
    if (!RunOptimizerPass(old_graph, {"SpecifyGraphInputFormat"})) {
      MS_LOG(ERROR) << "specify the input format of exported model failed.";
      return nullptr;
    }
  } else {
    if (!RunOptimizerPass(old_graph, {"SpecifyGraphInputFormat", "DecreaseTransposeAlgo"})) {
      MS_LOG(ERROR) << "Run transpose opt pass failed.";
      return nullptr;
    }
  }

  status = RunGraphPass(old_graph, config);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Run convert pass failed.";
    return nullptr;
  }

  status = RunParallelPass(old_graph, config);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Run convert pass failed.";
    return nullptr;
  }

  status = DoQuantize(old_graph, config);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Do Quantize failed.";
    return nullptr;
  }

  return old_graph;
}

bool AnfTransform::StoreBuiltinPass(const converter::Flags *config) {
  if (config == nullptr) {
    MS_LOG(ERROR) << "config is nullptr";
    return false;
  }
  auto fmk = config->fmk;
  auto is_train = config->trainModel;
  std::unordered_map<std::string, opt::PassPtr> passes = {
    {"DumpGraph", std::make_shared<opt::DumpGraph>(config)},
    {"ToNCHWFormat", std::make_shared<opt::ToNCHWFormat>(fmk, is_train)},
    {"ToNHWCFormat", std::make_shared<opt::ToNHWCFormat>(fmk, is_train)},
    {"InferShapePass", std::make_shared<opt::InferShapePass>(fmk, is_train)},
    {"DecreaseTransposeAlgo", std::make_shared<opt::DecreaseTransposeAlgo>(fmk, is_train)},
    {"SpecifyGraphInputFormat", std::make_shared<opt::SpecifyGraphInputFormat>(config->graphInputFormat)}};
  bool succeed_store = true;
  for (auto iter = passes.begin(); iter != passes.end(); ++iter) {
    if (PassStorage::StorePass(iter->first, iter->second) != RET_OK) {
      MS_LOG(ERROR) << "external pass name conflicts with that of internal pass, the pass name is " << iter->first
                    << ", please edit external pass name.";
      succeed_store = false;
    }
  }
  return succeed_store;
}

FuncGraphPtr AnfTransform::Transform(const FuncGraphPtr &main_graph, const converter::Flags *config) {
  if (!StoreBuiltinPass(config)) {
    MS_LOG(ERROR) << "store pass failed.";
    return nullptr;
  }
  auto new_graph = TransformFuncGraph(main_graph, config);
  if (new_graph == nullptr) {
    MS_LOG(ERROR) << "optimizer failed.";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_NULL_PTR);
  }
  return new_graph;
}
}  // namespace mindspore::lite
