/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#define USE_DEPRECATED_API
#include "tools/converter/anf_transform.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <deque>
#include <map>
#include <tuple>
#include "nnacl/op_base.h"
#include "src/common/log_adapter.h"
#include "tools/converter/optimizer_manager.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/optimizer/common/pass_manager_extends.h"
#include "ir/primitive.h"
#include "tools/optimizer/fusion/add_activation_fusion.h"
#include "tools/optimizer/fusion/affine_activation_fusion.h"
#include "tools/optimizer/fusion/affine_fusion.h"
#include "tools/optimizer/fusion/conv_biasadd_fusion.h"
#include "tools/optimizer/fusion/conv_activation_fusion.h"
#include "tools/optimizer/fusion/conv_tuple_activation_fusion.h"
#include "tools/optimizer/fusion/conv_scale_fusion.h"
#include "tools/optimizer/fusion/conv_bn_fusion.h"
#include "tools/optimizer/fusion/conv_tuplegetitem_fusion.h"
#include "tools/optimizer/const_fold/constant_folding_fusion.h"
#include "tools/optimizer/fusion/hard_swish_fusion.h"
#include "tools/optimizer/fusion/norm_fusion.h"
#include "tools/optimizer/fusion/prelu_fusion.h"
#include "tools/optimizer/fusion/batchmatmul_fusion.h"
#include "tools/optimizer/fusion/batchnorm_to_scale_fusion.h"
#include "tools/optimizer/fusion/sigmoid_mul_fusion.h"
#include "tools/optimizer/fusion/conv_conv_fusion.h"
#include "tools/optimizer/fusion/conv_pad_fusion.h"
#include "tools/optimizer/fusion/tflite_lstm_cell_fusion.h"
#include "tools/optimizer/fusion/tf_lstm_cell_fusion.h"
#include "tools/optimizer/fusion/tf_bidirection_gru_fusion.h"
#include "tools/optimizer/fusion/tensor_dot_fusion.h"
#include "tools/optimizer/fusion/multi_head_attention_fusion.h"
#include "tools/optimizer/fusion/encoder_layer_fusion.h"
#include "tools/optimizer/fusion/glu_fusion.h"
#include "tools/optimizer/fusion/tflite_rel_pos_multi_head_attention_fusion.h"
#include "tools/optimizer/fusion/matmul_add_fusion.h"
#include "tools/optimizer/fusion/matmul_mul_fusion.h"
#include "tools/optimizer/fusion/mul_add_fusion.h"
#include "tools/optimizer/fusion/tf_gelu_fusion.h"
#include "tools/optimizer/fusion/onnx_gelu_fusion.h"
#include "tools/optimizer/fusion/squeeze_fusion.h"
#include "tools/optimizer/fusion/reshape_reshape_fusion.h"
#include "tools/optimizer/fusion/reshape_transpose_fusion.h"
#include "tools/optimizer/fusion/transpose_matmul_fusion.h"
#include "tools/optimizer/fusion/scale_activation_fusion.h"
#include "tools/optimizer/fusion/scale_scale_fusion.h"
#include "tools/optimizer/fusion/resize_fusion.h"
#include "tools/optimizer/fusion/fullconnected_fusion.h"
#include "tools/optimizer/fusion/fullconnected_add_fusion.h"
#include "tools/optimizer/fusion/add_concat_activation_fusion.h"
#include "tools/optimizer/fusion/matmul_activation_fusion.h"
#include "tools/optimizer/fusion/mul_activation_fusion.h"
#include "tools/optimizer/fusion/activation_fusion.h"
#include "tools/optimizer/fusion/reshape_reduce_fusion.h"
#include "tools/optimizer/graph/add_tensor_array.h"
#include "tools/optimizer/graph/redundant_op_remove_pass.h"
#include "tools/optimizer/graph/clip_convert_activation_pass.h"
#include "tools/optimizer/graph/mul_constant_pass.h"
#include "tools/optimizer/graph/update_conv2d_param_pass.h"
#include "tools/optimizer/graph/infershape_pass.h"
#include "tools/optimizer/graph/slice_prepose_pass.h"
#include "tools/optimizer/graph/control_flow_pass.h"
#include "tools/optimizer/graph/reduce_same_act_pass.h"
#include "tools/optimizer/graph/split_one_pass.h"
#include "tools/optimizer/graph/decrease_transpose_algo.h"
#include "tools/optimizer/graph/special_node_postprocess.h"
#include "tools/optimizer/graph/specify_graph_input_format.h"
#include "tools/optimizer/graph/dump_graph.h"
#include "tools/optimizer/graph/eliminate_redundant_cast_pass.h"
#include "tools/converter/quantizer/quantization_optimizer.h"
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
#include "tools/optimizer/fusion/expanddims_reshape_fusion.h"
#include "tools/optimizer/fusion/reduce_same_op_in_horizon.h"
#include "tools/optimizer/fusion/reshape_shape_fusion.h"
#include "tools/optimizer/fusion/transpose_gather_fusion.h"
#ifndef ENABLE_CLOUD_FUSION_INFERENCE
#include "tools/converter/adapter/acl/acl_pass.h"
#endif
#include "src/common/log_util.h"
#include "src/common/string_utils.h"
#include "src/common/config_infos.h"
#include "tools/optimizer/fusion/groupnorm_fusion.h"
#include "tools/optimizer/fusion/mul_reduce_fusion.h"
#include "tools/optimizer/fusion/reshape_like_operator_ablation.h"
#include "tools/optimizer/fusion/concat_concat_fusion.h"
#include "tools/optimizer/fusion/strided_slice_fusion.h"
#include "tools/optimizer/fusion/reduce_stack_fusion.h"
#include "tools/optimizer/fusion/remove_transitivity_op.h"
#include "tools/converter/import/cast_op_adjust.h"
#include "tools/converter/adapter/acl/plugin/acl_pass_plugin.h"
#include "tools/converter/quantizer/quant_helper/qat_transform.h"
#include "tools/converter/parser/conv2d_transpose_input_adjust.h"
#include "tools/converter/parser/parser_utils.h"
#include "tools/converter/parser/unify_format.h"
#include "backend/common/optimizer/graph_optimizer.h"
#include "tools/optimizer/fusion/squeeze_expanddims_fusion.h"
#include "mindspore/core/ops/op_name.h"
#include "tools/common/string_util.h"
#include "src/common/common.h"

using std::string;
namespace mindspore::lite {
namespace {
constexpr auto kOriginalFmkType = "original_fmk_type";
constexpr auto kConverterInputShape = "converter_input_shape";

std::string TransInputShapesToString(const std::map<std::string, std::vector<int64_t>> &shapes) {
  std::stringstream str_stream;
  size_t shape_index = 0;
  for (auto &item : shapes) {
    str_stream << item.first << ":";
    auto &shape = item.second;
    for (size_t d = 0; d < shape.size(); d++) {
      str_stream << shape[d];
      if (d + 1 != shape.size()) {
        str_stream << ",";
      }
    }
    if (shape_index + 1 != shapes.size()) {
      str_stream << ";";
    }
    shape_index++;
  }
  return str_stream.str();
}

std::map<std::string, std::vector<int64_t>> TransStringToInputShapes(const std::string &shapes_str) {
  std::map<std::string, std::vector<int64_t>> shapes;
  auto shapes_pairs = lite::SplitStringToVector(shapes_str, ';');
  for (auto &kv_str : shapes_pairs) {
    auto pos = kv_str.rfind(':');
    if (pos == std::string::npos || pos + 1 == kv_str.size()) {
      MS_LOG_ERROR << "Invalid input shapes string: " << shapes_str;
      return {};
    }
    auto name = kv_str.substr(0, pos);
    auto shape_str = kv_str.substr(pos + 1);
    auto shape_dims_str = lite::SplitStringToVector(shape_str, ',');
    std::vector<int64_t> shape;
    shape.reserve(shape_dims_str.size());
    for (auto &dim_str : shape_dims_str) {
      int dim = 0;
      if (!lite::ConvertIntNum(dim_str, &dim)) {
        MS_LOG_ERROR << "Invalid input shapes string: " << shapes_str;
        return {};
      }
      shape.push_back(dim);
    }
    shapes[name] = shape;
  }
  return shapes;
}
}  // namespace

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
    (void)prim->AddAttr("trainOp", MakeValue(true));
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
          (void)graph_prim->AddAttr("trainOp", MakeValue(true));
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
      (void)prim->AddAttr("trainOp", MakeValue(true));
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

int AnfTransform::RunFusionPass(const FuncGraphPtr &old_graph, const std::shared_ptr<ConverterPara> &param) {
  auto status = MarkTrainOp(old_graph);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "MarkTrainOp failed.";
    return RET_ERROR;
  }
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  CHECK_NULL_RETURN(optimizer);
  auto fusion_pm = std::make_shared<opt::LitePassManager>("anf fusion pass manager", false);
  CHECK_NULL_RETURN(fusion_pm);

  // The training model only does the fusion of the inference part
  // remove quantdtype when awaretraining
  std::vector<opt::PassPtr> fusions{std::make_shared<opt::AddConcatActivationFusion>(),
                                    std::make_shared<opt::HardSwishFusion>(),
                                    std::make_shared<opt::PReluFusion>(),
                                    std::make_shared<opt::SqueezeFusion>(),
                                    std::make_shared<opt::TransposeFusion>(),
                                    std::make_shared<opt::ReshapeReshapeFusion>(),
                                    std::make_shared<opt::ReshapeTransposeFusion>(),
                                    std::make_shared<opt::ConvBiasaddFusion>(),
                                    std::make_shared<opt::ConvBatchNormFusion>(param->fmk_type),
                                    std::make_shared<opt::ConvScaleFusion>(param->fmk_type),
                                    std::make_shared<opt::GroupNormFusion>(),
                                    std::make_shared<opt::TfNormFusion>(),
                                    std::make_shared<opt::OnnxLayerNormFusion>(),
                                    std::make_shared<opt::OnnxLayerNormFusion2>(),
                                    std::make_shared<opt::BatchMatMulFusion>(),
                                    std::make_shared<opt::BatchNormToScaleFusion>(),
                                    std::make_shared<opt::SigmoidMulFusion>(),
                                    std::make_shared<opt::ActivationFusion>(),
                                    std::make_shared<opt::ConvActivationFusion>(param),
                                    std::make_shared<opt::ConvTupleGetItemFusion>(),
                                    std::make_shared<opt::ConvTupleActivationFusion>(),
                                    std::make_shared<opt::TfliteLstmCellFusion>(),
                                    std::make_shared<opt::TfLstmCellFusion>(),
                                    std::make_shared<opt::TfBidirectionGruFusion>(),
                                    std::make_shared<opt::TfGeLUFusion>(),
                                    std::make_shared<opt::OnnxGeLUFusion>(),
                                    std::make_shared<opt::TfliteRelPosMultiHeadAttentionFusion>(),
                                    std::make_shared<opt::GLUFusion>(),
                                    std::make_shared<opt::ResizeFusion1>(),
                                    std::make_shared<opt::ResizeFusion2>(),
                                    std::make_shared<opt::ConstFoldPass>(param->fmk_type, param->train_model),
                                    std::make_shared<opt::AffineFusion>(),
                                    std::make_shared<opt::AffineActivationFusion>(),
                                    std::make_shared<opt::ConvConvFusion>(),
                                    std::make_shared<opt::ConvPadFusion>(),
                                    std::make_shared<opt::MatMulAddFusion>(),
                                    std::make_shared<opt::MatMulMulFusion>(),
                                    std::make_shared<opt::TransposeMatMulFusion>(),
                                    std::make_shared<opt::MulAddFusion>(),
                                    std::make_shared<opt::ScaleActivationFusion>(),
                                    std::make_shared<opt::ScaleScaleFusion>(),
                                    std::make_shared<opt::FullConnectedFusion>(),
                                    std::make_shared<opt::FullconnectedAddFusion>(),
                                    std::make_shared<opt::TensorDotFusion>(),
                                    std::make_shared<opt::MatMulActivationFusion>(param),
                                    std::make_shared<opt::MulActivationFusion>(),
                                    std::make_shared<opt::AddActivationFusion>(),
                                    std::make_shared<opt::ExpandDimsReshapeFusion>(),
                                    std::make_shared<opt::SqueezeExpandDimsFusion>()};
  if (param->optimize_transformer) {
    fusions.push_back(std::make_shared<opt::MultiHeadAttentionFusion>());
    fusions.push_back(std::make_shared<opt::EncoderLayerFusion>());
  }
  for (size_t index = 0; index < fusions.size(); index++) {
    auto pass_ptr = fusions.at(index);
    MS_CHECK_TRUE_RET(pass_ptr != nullptr, RET_ERROR);
    auto pass_name = pass_ptr->name();
    if (param->fusion_blacklists.find(pass_name) != param->fusion_blacklists.end()) {
      MS_LOG(INFO) << "Disable fusion: " << pass_name;
      continue;
    }
    fusion_pm->AddPass(pass_ptr);
  }
  optimizer->AddPassManager(fusion_pm);
  if (optimizer->Optimize(old_graph) == nullptr) {
    MS_LOG(ERROR) << "run op fusion failed.";
    return RET_ERROR;
  }

  // the following pass needs to check the return value.
  fusions = {std::make_shared<opt::ReduceSameOpInHorizon>(param), std::make_shared<opt::ReshapeReduceFusion>(),
             std::make_shared<opt::AblateReshapeLikeOp>(),        std::make_shared<opt::MulReduceFusion>(),
             std::make_shared<opt::ConcatConcatFusion>(),         std::make_shared<opt::ReduceStackFusion>(),
             std::make_shared<opt::RemoveTransitivityOp>(),       std::make_shared<opt::StridedSliceFusion>(),
             std::make_shared<opt::RemoveTransitivityOp>(),       std::make_shared<opt::ReshapeShapeFusion>(),
             std::make_shared<opt::TransposeGatherFusion>()};
  for (auto &pass : fusions) {
    MS_CHECK_TRUE_MSG(pass != nullptr, RET_ERROR, "pass is a nullptr.");
    if (param->fusion_blacklists.find(pass->name()) != param->fusion_blacklists.end()) {
      MS_LOG(INFO) << "Disable fusion: " << pass->name();
      continue;
    }
    if (!pass->Run(old_graph)) {
      MS_LOG(ERROR) << pass->name() << " running failed.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int AnfTransform::RunParallelPass(const FuncGraphPtr &old_graph, const std::shared_ptr<ConverterPara> &param) {
  MS_LOG(DEBUG) << "Run ParallelPass start";
  if (param->train_model || param->parallel_split_config.parallel_split_type_ == SplitNo) {
    return RET_OK;
  }
  if (param->parallel_split_config.parallel_split_type_ == SplitByUserRatio) {
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
    std::unordered_map<std::string, opt::SplitStrategy> split_strategys = opt::ParserSplitStrategy(
      param->parallel_split_config.parallel_compute_rates_, param->parallel_split_config.parallel_devices_, split_mode);
    if (split_strategys.empty()) {
      MS_LOG(WARNING) << "No valid split_strategy. Run convert without split";
      return RET_OK;
    }
    opt::Spliter::GetInstance()->RecordGraphInfo(old_graph);
    auto parallel_pm = std::make_shared<opt::LitePassManager>("anf parallel pass manager", true);
    CHECK_NULL_RETURN(parallel_pm);
    // 2. preceding parallel pass
    parallel_pm->AddPass(std::make_shared<opt::IterNodeOutputs>());
    parallel_pm->AddPass(std::make_shared<opt::NodeOutShapes>());
    std::set<int, opt::IntCompare> match_multi_numbers = opt::Spliter::GetInstance()->graph_match_multi_numbers();
    int max_match_number = *match_multi_numbers.begin();
    // we do not deal with single conv node
    for (int match_number = max_match_number; match_number > opt::kDefaultBatch; --match_number) {
      // 3. multi_conv parallel pass
      parallel_pm->AddPass(std::make_shared<opt::MultiConvSplitPass>(split_strategys, param->fmk_type, match_number));
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

int AnfTransform::RunGraphPass(const FuncGraphPtr &old_graph, const std::shared_ptr<ConverterPara> &param) {
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  CHECK_NULL_RETURN(optimizer);
  auto graph_pm = std::make_shared<opt::LitePassManager>("anf graph pass manager", true);
  CHECK_NULL_RETURN(graph_pm);
  if (param->fmk_type == converter::kFmkTypeTflite || param->fmk_type == converter::kFmkTypeTf ||
      param->fmk_type == converter::kFmkTypeOnnx) {
    graph_pm->AddPass(std::make_shared<opt::ControlFlowPass>());
  }
  auto slice_prepose_pass = std::make_shared<opt::SlicePreposePass>();
  CHECK_NULL_RETURN(slice_prepose_pass);
  slice_prepose_pass->SetFmkType(param->fmk_type);
  graph_pm->AddPass(slice_prepose_pass);
  optimizer->AddPassManager(graph_pm);
  if (optimizer->Optimize(old_graph) == nullptr) {
    MS_LOG(ERROR) << "run  graph pass failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int AnfTransform::RunConvertPass(const FuncGraphPtr &old_graph, const std::shared_ptr<ConverterPara> &param) {
  if (param->device.find("Ascend") != std::string::npos) {
    if (opt::AclPassPlugin::GetInstance().HasPluginSo()) {
      auto acl_pass_ptr = opt::AclPassPlugin::GetInstance().CreateAclPass(param);
      if (acl_pass_ptr == nullptr) {
        MS_LOG(ERROR) << "Acl pass ptr is nullptr.";
        return RET_ERROR;
      }

      if (!acl_pass_ptr->Run(old_graph)) {
        MS_LOG(ERROR) << "Acl pass failed.";
        opt::AclPassPlugin::GetInstance().DestroyAclPass(acl_pass_ptr);
        return RET_ERROR;
      }
      opt::AclPassPlugin::GetInstance().DestroyAclPass(acl_pass_ptr);
    }
  }
  // adjust for conv2d_transpose
  if (!(param->no_fusion && param->export_mindir == kMindIR)) {
    std::set<FuncGraphPtr> all_func_graphs = {};
    GetAllFuncGraph(old_graph, &all_func_graphs);
    auto conv2d_transpose_adjust = std::make_shared<Conv2DTransposeInputAdjust>();
    MS_CHECK_TRUE_MSG(conv2d_transpose_adjust != nullptr, RET_NULL_PTR, "conv2d_transpose_adjust is nullptr.");
    for (auto sub_graph : all_func_graphs) {
      if (!conv2d_transpose_adjust->Run(old_graph)) {
        MS_LOG(ERROR) << "adjust conv2d_transpose failed";
        return RET_ERROR;
      }
    }
  }
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  CHECK_NULL_RETURN(optimizer);
  auto convert_pm = std::make_shared<opt::LitePassManager>("anf graph convert pass manager", true);
  CHECK_NULL_RETURN(convert_pm);
  convert_pm->AddPass(std::make_shared<opt::RemoveRedundantOpPass>(param->train_model));
  convert_pm->AddPass(std::make_shared<opt::InferShapePass>(param->fmk_type, param->train_model));
  convert_pm->AddPass(std::make_shared<opt::CastOpAdjust>());
  convert_pm->AddPass(std::make_shared<opt::UpdateConv2DParamPass>());
  optimizer->AddPassManager(convert_pm);
  if (optimizer->Optimize(old_graph) == nullptr) {
    MS_LOG(ERROR) << "run graph convert pass failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int AnfTransform::RunConstFoldPass(const FuncGraphPtr &old_graph, const std::shared_ptr<ConverterPara> &param) {
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto const_fold_pm = std::make_shared<opt::LitePassManager>("const fold fusion pass manager", false);
  CHECK_NULL_RETURN(optimizer);
  CHECK_NULL_RETURN(const_fold_pm);
  const_fold_pm->AddPass(std::make_shared<opt::InferShapePass>(param->fmk_type, param->train_model));
  if (!param->train_model) {
    const_fold_pm->AddPass(std::make_shared<opt::ConstFoldPass>(param->fmk_type, param->train_model));
  }
  const_fold_pm->AddPass(std::make_shared<opt::UpdateConv2DParamPass>());
  const_fold_pm->AddPass(std::make_shared<opt::ClipConvertActivationPass>());
  optimizer->AddPassManager(const_fold_pm);
  if (optimizer->Optimize(old_graph) == nullptr) {
    MS_LOG(ERROR) << "run const fold failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int RunDecreaseTransposePass(const FuncGraphPtr &old_graph, const std::shared_ptr<ConverterPara> &param) {
  MS_ASSERT(old_graph != nullptr && param != nullptr);
  auto pass = std::make_shared<opt::DecreaseTransposeAlgo>(param->fmk_type, param->train_model, false);
  MS_CHECK_TRUE_RET(pass != nullptr, RET_ERROR);
  if (!pass->Run(old_graph)) {
    MS_LOG(ERROR) << "Run DecreaseTransposeAlgo pass failed";
    return RET_ERROR;
  }

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto decrease_trans_pm = std::make_shared<opt::LitePassManager>("decrease transpose fusion pass manager", false);
  CHECK_NULL_RETURN(optimizer);
  CHECK_NULL_RETURN(decrease_trans_pm);
  decrease_trans_pm->AddPass(std::make_shared<opt::ReshapeTransposeFusion>());
  decrease_trans_pm->AddPass(std::make_shared<opt::TransposeFusion>());
  optimizer->AddPassManager(decrease_trans_pm);
  if (optimizer->Optimize(old_graph) == nullptr) {
    MS_LOG(ERROR) << "run decrease transpose failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

bool AnfTransform::CheckExternalExtension(const std::shared_ptr<ConverterPara> &param) {
  return (!param->plugins_path.empty() && param->commonQuantParam.quant_type != quant::QUANT_NONE);
}

int AnfTransform::DoQuantize(const FuncGraphPtr &old_graph, const std::shared_ptr<ConverterPara> &param) {
  quant::QuantizationOptimizer quantization_optimizer(param);
  auto ret = quantization_optimizer.Run(old_graph);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Post training quantization failed.";
    return ret;
  }
  return RET_OK;
}

int AnfTransform::DoFormatForMindIR(const FuncGraphPtr &old_graph, const std::shared_ptr<ConverterPara> &param) {
  if (param->export_mindir != kMindIR) {
    return RET_OK;
  }
  if (param->no_fusion || param->device.find("Ascend") == std::string::npos) {
    MS_LOG(INFO) << "export MindIR, run pass ToNCHWFormat";
    if (!RunOptimizerPass(old_graph, {"ToNCHWFormat", "DecreaseTransposeAlgo"})) {
      MS_LOG(ERROR) << "Run ToNCHWFormat pass failed";
      return RET_ERROR;
    }
  }
  old_graph->set_attr(kOriginalFmkType, MakeValue(static_cast<int32_t>(param->fmk_type)));

  return RET_OK;
}

int AnfTransform::RunFormatTrans(const FuncGraphPtr &old_graph) {
  auto value = old_graph->get_attr(ops::kFormat);
  if (value != nullptr && GetValue<int64_t>(value) == mindspore::NHWC) {
    return RET_OK;
  }
  if (!RunOptimizerPass(old_graph, {"ToNHWCFormat", "DecreaseTransposeAlgo"})) {
    MS_LOG(ERROR) << "Run ToNHWCFormat pass failed";
    return RET_ERROR;
  }
  return RET_OK;
}

bool RunEliminateRedundantPass(const FuncGraphPtr &old_graph, const std::shared_ptr<ConverterPara> &param) {
  if (!RunOptimizerPass(old_graph, {"InferShapePass"})) {
    MS_LOG(WARNING) << "Run infershape opt pass failed.";
  } else if (!RunOptimizerPass(old_graph, {"DecreaseTransposeAlgo"})) {
    MS_LOG(ERROR) << "Run transpose opt pass failed.";
    return false;
  }

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  MS_CHECK_TRUE_RET(optimizer != nullptr, false);
  auto eliminate_pm = std::make_shared<opt::LitePassManager>("anf graph eliminate redundant pass manager", true);
  MS_CHECK_TRUE_RET(eliminate_pm != nullptr, false);
  eliminate_pm->AddPass(std::make_shared<opt::RemoveRedundantOpPass>(param->train_model));
  eliminate_pm->AddPass(std::make_shared<opt::EliminateRedundantCastPass>(param->fmk_type, param->train_model));
  eliminate_pm->AddPass(std::make_shared<opt::ReduceSameActPass>());
  eliminate_pm->AddPass(std::make_shared<opt::SplitOnePass>());
  eliminate_pm->AddPass(std::make_shared<opt::MulConstantPass>());
  optimizer->AddPassManager(eliminate_pm);
  if (optimizer->Optimize(old_graph) == nullptr) {
    MS_LOG(ERROR) << "run graph convert pass failed.";
    return false;
  }
  return true;
}

STATUS AnfTransform::ProcOnlineTransform(const FuncGraphPtr &old_graph, const std::shared_ptr<ConverterPara> &param) {
  if (!RunOptimizerPass(old_graph, {"RemoveRedundantOpPass", "InferShapePass", "ConstFoldPass"})) {
    MS_LOG(WARNING) << "Run infershape opt pass failed.";
  }
  auto status = DoFormatForMindIR(old_graph, param);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Do format for mindir failed.";
    return lite::RET_ERROR;
  }
  if (!param->input_shape.empty()) {
    auto graph_inputs = old_graph->get_inputs();
    std::map<std::string, std::vector<int64_t>> input_shape;
    for (auto &input : graph_inputs) {
      auto abstract = input->abstract();
      if (abstract) {
        input_shape[abstract->name()] = opt::GetAnfNodeOutputShape(input, 0);
      }
    }
    old_graph->set_attr(kConverterInputShape, MakeValue(TransInputShapesToString(input_shape)));
  }
  return lite::RET_OK;
}

int AnfTransform::RunPass(const FuncGraphPtr &old_graph, const std::shared_ptr<ConverterPara> &param) {
  auto status = RunConvertPass(old_graph, param);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Run convert pass failed.";
    return RET_ERROR;
  }

  if (!RunExternalPass(old_graph, registry::POSITION_BEGIN)) {
    MS_LOG(ERROR) << "Run external pass failed, place is BEGIN";
    return RET_ERROR;
  }

  status = RunConstFoldPass(old_graph, param);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Run const fold pass failed.";
    return RET_ERROR;
  }

  if (!RunEliminateRedundantPass(old_graph, param)) {
    MS_LOG(ERROR) << "Run elimination of redundant pass failed.";
    return RET_ERROR;
  }

  if (!param->no_fusion) {
    status = RunFusionPass(old_graph, param);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Run fusion pass failed.";
      return RET_ERROR;
    }
  }

  if (!RunExternalPass(old_graph, registry::POSITION_END)) {
    MS_LOG(ERROR) << "Run external pass failed, place is END";
    return RET_ERROR;
  }

  if (!RunOptimizerPass(old_graph, {"InferShapePass"})) {
    MS_LOG(WARNING) << "Run infershape opt pass failed.";
    status = RunOptimizerPass(old_graph, {"SpecialNodePostProcess"}) ? RET_OK : RET_ERROR;
  } else {
    status =
      RunOptimizerPass(old_graph, {"SpecialNodePostProcess"}) ? RunDecreaseTransposePass(old_graph, param) : RET_ERROR;
  }
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Run transpose opt pass failed.";
    return RET_ERROR;
  }

  status = RunGraphPass(old_graph, param);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Run convert pass failed.";
    return RET_ERROR;
  }

  status = RunParallelPass(old_graph, param);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Run convert pass failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

STATUS AnfTransform::TransformFuncGraph(const FuncGraphPtr &old_graph, const std::shared_ptr<ConverterPara> &param) {
  MS_ASSERT(old_graph != nullptr);
  MS_ASSERT(param != nullptr);
  if (param->no_fusion && param->export_mindir == kMindIR) {  // converter, online
    if (ProcOnlineTransform(old_graph, param) != lite::RET_OK) {
      MS_LOG(ERROR) << "Proc online transform failed.";
      return RET_ERROR;
    }
    auto status = DoQuantize(old_graph, param);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Do Quantize failed.";
      return RET_ERROR;
    }
    return RET_OK;
  }
  auto value = old_graph->get_attr(kIsOptimized);
  if (param->is_runtime_converter) {  // load online
    if (value != nullptr) {           // other models converted to MindIR
      auto status = RunFormatTrans(old_graph);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "Run format trans failed";
        return status;
      }
    }
  }

  if (RunPass(old_graph, param) != RET_OK) {
    MS_LOG(ERROR) << "Proc online transform failed.";
    return RET_ERROR;
  }

  if (CheckExternalExtension(param)) {
    MS_LOG(ERROR) << "Unsupported external extension with quantization.";
    return RET_ERROR;
  }
  auto qat_transform = quant::QATTransform(old_graph, param);
  auto status = qat_transform.Transform();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Do QATTransform failed.";
    return RET_ERROR;
  }

  status = DoQuantize(old_graph, param);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Do Quantize failed.";
    return RET_ERROR;
  }
  status = DoFormatForMindIR(old_graph, param);
  if (status != RET_OK) {
    return RET_ERROR;
  }
  return RET_OK;
}

bool AnfTransform::StoreBuiltinPass(const std::shared_ptr<ConverterPara> &param) {
  if (param == nullptr) {
    MS_LOG(ERROR) << "param is nullptr";
    return false;
  }
  auto fmk = param->fmk_type;
  auto is_train = param->train_model;

  // pass_name, pass and boolean value to indicate whether can be called by external extension,
  std::vector<std::tuple<std::string, opt::PassPtr, bool>> pass_infos = {
    {"DumpGraph", std::make_shared<opt::DumpGraph>(param), true},
    {"RemoveRedundantOpPass", std::make_shared<opt::RemoveRedundantOpPass>(param->train_model), false},
    {"ToNCHWFormat", std::make_shared<opt::ToNCHWFormat>(fmk, is_train), true},
    {"ToNHWCFormat", std::make_shared<opt::ToNHWCFormat>(fmk, is_train), true},
    {"ConstFoldPass", std::make_shared<opt::ConstFoldPass>(fmk, is_train), true},
    {"InferShapePass", std::make_shared<opt::InferShapePass>(fmk, is_train), false},
    {"DeleteRedundantTranspose", std::make_shared<opt::DeleteRedundantTranspose>(), false},
    {"SpecialNodePostProcess", std::make_shared<opt::SpecialNodePostProcess>(), false},
    {"DecreaseTransposeAlgo", std::make_shared<opt::DecreaseTransposeAlgo>(fmk, is_train), true}};
  for (const auto &pass_info : pass_infos) {
    MS_CHECK_TRUE_RET(std::get<1>(pass_info) != nullptr, false);
    PassStorage::StorePass(std::get<0>(pass_info), std::get<1>(pass_info), std::get<opt::kInputIndexTwo>(pass_info));
  }
  auto dump_graph_outer = std::make_shared<opt::DumpGraph>(param);
  MS_CHECK_TRUE_MSG(dump_graph_outer != nullptr, false, "dumpGraph object is a nullptr.");
  registry::PassRegistry("DumpGraph", dump_graph_outer);
  return true;
}

STATUS AnfTransform::Transform(const FuncGraphPtr &main_graph, const std::shared_ptr<ConverterPara> &param) {
  MS_CHECK_TRUE_MSG(main_graph != nullptr, RET_NULL_PTR, "Input func_graph is nullptr");
  MS_CHECK_TRUE_MSG(param != nullptr, RET_NULL_PTR, "Input converter param is nullptr");
  manager_ = Manage(main_graph, true);

  if (main_graph->has_attr(kOriginalFmkType)) {
    auto val_ptr = main_graph->get_attr(kOriginalFmkType);
    MS_CHECK_TRUE_MSG(val_ptr != nullptr, RET_NULL_PTR, "Val ptr is nullptr.");
    param->fmk_type = static_cast<converter::FmkType>(GetValue<int32_t>(val_ptr));
  }
  if (main_graph->has_attr(kConverterInputShape)) {
    auto val_ptr = main_graph->get_attr(kConverterInputShape);
    MS_CHECK_TRUE_MSG(val_ptr != nullptr, RET_NULL_PTR, "Val ptr is nullptr.");
    param->input_shape = TransStringToInputShapes(GetValue<std::string>(val_ptr));
    for (auto &kv : param->input_shape) {
      lite::ConverterInnerContext::GetInstance()->UpdateGraphInputTensorShape(kv.first, kv.second);
    }
  }
  if (!StoreBuiltinPass(param)) {
    MS_LOG(ERROR) << "store pass failed.";
    return RET_ERROR;
  }

  auto status = TransformFuncGraph(main_graph, param);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "optimizer failed.";
    return RET_NULL_PTR;
  }
  return RET_OK;
}
}  // namespace mindspore::lite
