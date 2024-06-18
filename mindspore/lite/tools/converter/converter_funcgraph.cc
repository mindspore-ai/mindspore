/**
 * Copyright 2020-2024 Huawei Technologies Co., Ltd
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
#include "tools/converter/converter_funcgraph.h"
#include <memory>
#include <vector>
#include <set>
#include <tuple>
#include <algorithm>
#include <utility>
#include "src/common/log_adapter.h"
#include "tools/common/meta_graph_serializer.h"
#include "tools/lite_exporter/anf_exporter.h"
#include "tools/graph_kernel/converter/graph_kernel_optimization.h"
#ifdef SUPPORT_TRAIN
#include "src/train/train_populate_parameter.h"
#endif
#include "include/registry/model_parser_registry.h"
#include "src/common/dynamic_library_loader.h"
#include "src/common/log_util.h"
#include "tools/converter/parser/parser_utils.h"
#include "tools/converter/import/mindspore_importer.h"
#include "nnacl/op_base.h"
#include "tools/converter/micro/coder/coder.h"
#include "src/common/prim_util.h"
#include "src/common/version_manager.h"
#include "tools/common/tensor_util.h"
#include "include/api/model.h"
#include "tools/mindir_exporter/mindir_serializer.h"
#include "src/common/primitive_t_utils.h"
#include "tools/converter/config_parser/acl_option_param_parser.h"
#include "tools/converter/config_parser/micro_param_parser.h"
#include "tools/converter/config_parser/preprocess_parser.h"
#include "tools/converter/config_parser/quant_param_parser.h"
#include "tools/common/string_util.h"
#include "src/common/file_utils.h"
#include "ops/dynamic_shape.h"
#include "tools/common/func_graph_utils.h"
#include "tools/converter/import/remove_public_primitive.h"
#include "tools/optimizer/graph/redundant_op_remove_pass.h"
#include "tools/optimizer/graph/input_data_type_trans_pass.h"
#include "tools/converter/parser/unify_format.h"
#include "tools/optimizer/graph/specify_graph_input_format.h"
#include "tools/optimizer/graph/specify_graph_output_format.h"
#include "tools/optimizer/graph/decrease_transpose_algo.h"
#include "tools/converter/anf_transform.h"
#include "tools/converter/offline_packing_optimizer.h"
#include "tools/converter/adapter/acl/plugin/acl_pass_plugin.h"
#include "tools/optimizer/format/to_nhwc_format.h"
#include "tools/optimizer/format/to_nchw_format.h"
#include "tools/converter/quantizer/quantization_optimizer.h"
#include "tools/converter/anf_transform_for_ge.h"
#include "tools/converter/adapter/acl/common/acl_types_utils.h"
#include "src/extendrt/delegate/plugin/ascend_ge_executor_plugin.h"
#include "tools/optimizer/graph/input_and_output_variable_pass.h"
#include "tools/optimizer/graph/output_variable_pass.h"
#include "tools/optimizer/graph/args_to_attr_pass.h"
#include "tools/optimizer/fusion/ffn_antiquant_fusion.h"
#include "tools/optimizer/common/pass_manager_extends.h"
#include "load_mindir/infer_mindir.h"
#include "tools/optimizer/fusion/matmul_allreduce_fusion.h"

namespace mindspore {
namespace lite {
FuncGraphPtr ConvertGraph(const api::FuncGraphPtr &func_graph) {
  auto impl = func_graph->impl();
  return std::dynamic_pointer_cast<FuncGraph>(impl);
}

FuncGraphPtr ConverterFuncGraph::Load3rdModelToFuncgraph(const std::shared_ptr<ConverterPara> &param) {
  api::FuncGraphPtr func_graph_base = nullptr;
  auto model_parser = registry::ModelParserRegistry::GetModelParser(param->fmk_type);
  if (model_parser == nullptr) {
    MS_LOG(ERROR) << "Unsupported to converter models with fmk: " << param->fmk_type;
    return nullptr;
  }
  if (!param->decrypt_key.empty()) {
    MS_LOG(ERROR) << "The 3rd model do not support decrypt.";
    return nullptr;
  }
  converter::ConverterParameters converter_parameters;
  converter_parameters.fmk = param->fmk_type;
  converter_parameters.save_type = param->save_type;
  converter_parameters.model_file = param->model_file;
  converter_parameters.weight_file = param->weight_file;
  if (param->config_infos.find(kOMConverterOptionsSection) != param->config_infos.end()) {
    converter_parameters.attrs = param->config_infos[kOMConverterOptionsSection];
  }
  func_graph_base = model_parser->Parse(converter_parameters);
  if (func_graph_base == nullptr) {
    delete model_parser;
    MS_LOG(ERROR) << "Get funcGraph failed for fmk: " << param->fmk_type;
    return nullptr;
  }
  delete model_parser;
  auto func_graph = ConvertGraph(func_graph_base);
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "convert graph failed for fmk: " << param->fmk_type;
    return nullptr;
  }
  func_graph->set_attr(kIsOptimized, MakeValue(false));
  return func_graph;
}

FuncGraphPtr ConverterFuncGraph::Load(const std::shared_ptr<ConverterPara> &param) {
  FuncGraphPtr func_graph;
  if (!param->decrypt_key.empty()) {
    unsigned char key[32];
    const size_t key_len = Hex2ByteArray(param->decrypt_key, key, 32);
    if (key_len == 0) {
      return nullptr;
    }
    MindIRLoader mindir_loader(false, key, key_len, param->decrypt_mode, false);
    func_graph = mindir_loader.LoadMindIR(param->model_file);
    auto ret = memset_s(key, sizeof(key), 0, key_len);
    if (ret != 0) {
      MS_LOG(EXCEPTION) << "memset_s error";
    }
  } else {
    MindIRLoader mindir_loader;
    func_graph = mindir_loader.LoadMindIR(param->model_file);
  }
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "Load MindIR file failed. Please check model file and decrypt key.";
    return nullptr;
  }
  auto manager = func_graph->manager();
  if (manager == nullptr) {
    manager = MakeManager();
    manager->AddFuncGraph(func_graph, true);
  }
  InferFuncGraphLoaded(func_graph);
  bool is_original = IsOriginalFuncGraph(func_graph);
  if (is_original) {
    func_graph->set_attr("graph_name", MakeValue("main_graph"));
    func_graph->set_attr("fmk", MakeValue(static_cast<int>(converter::kFmkTypeMs)));
  }

  return func_graph;
}

void SetIsGraphDynamicShapeAttr(const FuncGraphPtr &func_graph) {
  bool dyn_shape_value = false;
  for (auto input : func_graph->get_inputs()) {
    if (input->Shape() == nullptr) {
      MS_LOG(WARNING) << "input shape is nullptr!";
      continue;
    }
    if (input->Shape()->IsDynamic()) {
      dyn_shape_value = true;
    }
  }
  if (func_graph->has_attr(lite::kIsDynamicShape)) {
    func_graph->set_attr(lite::kIsDynamicShape, MakeValue(dyn_shape_value));
  } else {
    func_graph->attrs().emplace(lite::kIsDynamicShape, MakeValue(dyn_shape_value));
  }
}

FuncGraphPtr ConverterFuncGraph::Build(const std::shared_ptr<ConverterPara> &param) {
  FuncGraphPtr func_graph = nullptr;
  ConverterInnerContext::GetInstance()->SetTargetDevice(param->device);
  if (param->fmk_type == converter::FmkType::kFmkTypeMs) {
#ifdef SUPPORT_TRAIN
    kernel::PopulateTrainParameters();
#endif
    func_graph = Load(param);
  } else {
    func_graph = Load3rdModelToFuncgraph(param);
  }
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "Load model file failed!";
    return nullptr;
  }

  // Add attribute "isDynamicShape" to the func_graph to mark if the graph has dynamic input shapes.
  SetIsGraphDynamicShapeAttr(func_graph);

  return func_graph;
}

// get output_names must be between CommonAnfAdjust and Mindir2AnfAdjust;
STATUS ConverterFuncGraph::UnifyFuncGraphForInfer(const std::shared_ptr<ConverterPara> &param, FuncGraphPtr func_graph,
                                                  std::vector<std::string> *output_names) {
  bool is_original = IsOriginalFuncGraph(func_graph);
  if (!is_original) {
    return RET_OK;
  }
  if (!param->weightQuantParam.update_mindir) {
    MS_LOG(INFO) << "It will not unify funcgraph.";
    return RET_OK;
  }
  auto fmk = func_graph->get_attr("fmk");
  if (fmk == nullptr) {
    func_graph->set_attr("graph_name", MakeValue("main_graph"));
    func_graph->set_attr("fmk", MakeValue(static_cast<int>(converter::kFmkTypeMs)));
  }

  auto remove_public_primitive = std::make_shared<RemovePublicPrimitiveInterference>();
  MS_CHECK_TRUE_MSG(remove_public_primitive != nullptr, RET_NULL_PTR,
                    "RemovePublicPrimitiveInterference is a nullptr.");
  if (!remove_public_primitive->Run(func_graph)) {
    MS_LOG(ERROR) << "remove interference due to public-pirmitive failed!";
    return RET_ERROR;
  }
  MindsporeImporter::RemoveUnusedGraphInput(func_graph);

  auto status = CommonAnfAdjust(func_graph);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "CommonAnfAdjust failed!ret = " << status;
    return status;
  }

  // get output_names must be between CommonAnfAdjust and Mindir2AnfAdjust;
  *output_names = FuncGraphUtils::GetFuncGraphOutputNames(func_graph);
  if (output_names->empty()) {
    MS_LOG(ERROR) << "GetFuncGraphOutputNames failed!";
    return RET_ERROR;
  }

  // Ascend quant still need to use lite op.
  if (param->device.find("Ascend") != std::string::npos && param->fullQuantParam.target_device != quant::ASCEND) {
    MS_LOG(INFO) << "There is no need to adjust and pass graph when in Ascend.";
    return RET_OK;
  }
  status = MindsporeImporter::Mindir2AnfAdjust(func_graph, param);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Mindir2AnfAdjust failed!ret = " << status;
    return status;
  }

  if (!param->train_model) {
    auto redundant_op_remove_pass = std::make_shared<mindspore::opt::RemoveRedundantOpPass>(param->train_model, true);
    MS_CHECK_TRUE_MSG(redundant_op_remove_pass != nullptr, RET_NULL_PTR, "redundant_op_remove_pass is nullptr.");
    if (!redundant_op_remove_pass->Run(func_graph)) {
      MS_LOG(ERROR) << "Run remove redundant op failed!";
      return RET_ERROR;
    }
  }

  auto unify_format = std::make_shared<UnifyFormatToNHWC>(converter::kFmkTypeMs, param->train_model, param->save_type);
  MS_CHECK_TRUE_MSG(unify_format != nullptr, RET_NULL_PTR, "unify_format is nullptr.");
  if (!unify_format->Run(func_graph)) {
    MS_LOG(ERROR) << "Run insert transpose failed!";
    return RET_ERROR;
  }
  func_graph->set_attr(kIsOptimized, MakeValue(false));

  return status;
}

STATUS ConverterFuncGraph::UnifyFuncGraphInputFormat(const std::shared_ptr<ConverterPara> &param,
                                                     FuncGraphPtr func_graph) {
  mindspore::Format cur_input_format = DEFAULT_FORMAT;
  auto status = opt::SpecifyGraphInputFormat::GetCurGraphInputFormat(func_graph, param->fmk_type, &cur_input_format);
  if (!status) {
    MS_LOG(ERROR) << "Failed to get current format of graph input!";
    return RET_ERROR;
  }

  auto spec_input_format = param->spec_input_format;
  if (spec_input_format == DEFAULT_FORMAT) {
    if (param->save_type == kMindIR || param->fmk_type != converter::kFmkTypeMs) {
      // if it saves to mindir, the input format must be the same as the original model
      // if it saves to mindir lite, the input format must be the same as the original model for 3rd model
      func_graph->set_attr(kInputFormat, MakeValue(static_cast<int>(cur_input_format)));
      return RET_OK;
    }
    spec_input_format = NHWC;
  }
  opt::SpecifyGraphInputFormat pass(spec_input_format, cur_input_format);
  status = pass.Run(func_graph);
  if (!status) {
    MS_LOG(ERROR) << "Failed to Specify graph input format to " << spec_input_format;
    return RET_ERROR;
  }
  func_graph->set_attr(kInputFormat, MakeValue(static_cast<int>(spec_input_format)));
  return RET_OK;
}

STATUS ConverterFuncGraph::UnifyFuncGraphInOutDataType(const std::shared_ptr<ConverterPara> &param,
                                                       FuncGraphPtr func_graph) {
  opt::InOutDTypeTransPass pass(param->input_data_type, param->output_data_type);
  auto status = pass.Run(func_graph);
  if (!status) {
    MS_LOG(ERROR) << "Failed to Specify graph input data type to " << param->input_data_type
                  << " and output data type to " << param->output_data_type;
    return RET_ERROR;
  }
  return RET_OK;
}

void SetInputParameterName(const FuncGraphPtr &func_graph) {
  for (auto &input : func_graph->get_inputs()) {
    auto parameter = input->cast<ParameterPtr>();
    if (!parameter->has_default()) {
      auto abstract = parameter->abstract();
      if (abstract != nullptr && !abstract->name().empty()) {
        parameter->set_name(abstract->name());
      }
    }
  }
}

void SetInputParameterAbstractName(const FuncGraphPtr &func_graph) {
  for (auto &input : func_graph->get_inputs()) {
    auto parameter = input->cast<ParameterPtr>();
    if (!parameter->has_default()) {
      auto abstract = parameter->abstract();
      if (abstract != nullptr && abstract->name().empty()) {
        abstract->set_name(parameter->name());
      }
    }
  }
}

bool CheckNeedQuant(const std::shared_ptr<ConverterPara> &param, const FuncGraphPtr &func_graph) {
  bool is_ptq_quant = (param->commonQuantParam.quant_type == lite::quant::QUANT_ALL &&
                       param->fullQuantParam.target_device == lite::quant::ASCEND) ||
                      (param->commonQuantParam.quant_type == lite::quant::QUANT_WEIGHT &&
                       param->weightQuantParam.dequant_strategy == lite::quant::ON_THE_FLY);
  if (is_ptq_quant) {
    return true;
  }
  // Check if the model contains fakequant nodes
  const std::set<PrimitivePtr> fake_quant_types = {prim::kPrimFakeQuantPerLayer, prim::kPrimFakeQuantPerChannel};
  for (auto &cnode : func_graph->GetOrderedCnodes()) {
    auto op_name = cnode->fullname_with_scope();
    auto primitive = GetValueNode<PrimitivePtr>(cnode->input(0));
    if (primitive == nullptr) {
      return false;
    }
    for (const auto &type : fake_quant_types) {
      if (opt::CheckPrimitiveType(cnode, type)) {
        return true;
      }
    }
  }
  return false;
}

STATUS ConverterFuncGraph::QuantizationOptimizeForGE(const std::shared_ptr<ConverterPara> &param,
                                                     FuncGraphPtr func_graph) {
  CHECK_NULL_RETURN(param);
  CHECK_NULL_RETURN(func_graph);
  MS_LOG(INFO) << "It will run quant optimize";
  auto acl_pass_ptr = opt::AclPassPlugin::CreateAclPass(param);
  if (acl_pass_ptr == nullptr) {
    MS_LOG(ERROR) << "Failed to create acl pass";
    return RET_ERROR;
  }
  if (!acl_pass_ptr->Run(func_graph)) {
    MS_LOG(ERROR) << "Acl pass failed.";
    return RET_ERROR;
  }
  std::vector<opt::PassPtr> quant_fusions{std::make_shared<opt::FFNAntiquantFusion>()};
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  CHECK_NULL_RETURN(optimizer);
  auto fusion_pm = std::make_shared<opt::LitePassManager>("anf fusion pass manager", false);
  CHECK_NULL_RETURN(fusion_pm);
  for (size_t index = 0; index < quant_fusions.size(); index++) {
    auto pass_ptr = quant_fusions.at(index);
    MS_CHECK_TRUE_RET(pass_ptr != nullptr, RET_ERROR);
    fusion_pm->AddPass(pass_ptr);
  }
  optimizer->AddPassManager(fusion_pm);
  if (optimizer->Optimize(func_graph) == nullptr) {
    MS_LOG(ERROR) << "run op fusion failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

STATUS ConverterFuncGraph::OptimizeForGE(const std::shared_ptr<ConverterPara> &param, FuncGraphPtr func_graph) {
  AnfTransformForGe transform;
  auto status = transform.Transform(func_graph, param);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Transform anf graph for ge failed.";
    return status;
  }
  if (CheckNeedQuant(param, func_graph)) {
    status = QuantizationOptimizeForGE(param, func_graph);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Failed to quantization optimize for GE";
      return status;
    }
  }
  auto ret = RunGeOfflineConvert(param, func_graph);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Failed to Run GE Aoe Optimize or GE offline convert";
    return ret;
  }
  ret = RunVariableOptimize(param, func_graph);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Failed to Run variable op optimize";
    return ret;
  }

  return RET_OK;
}

STATUS ConverterFuncGraph::RunGeOfflineConvert(const std::shared_ptr<ConverterPara> &param, FuncGraphPtr func_graph) {
  if (param->is_runtime_converter) {
    MS_LOG(INFO) << "Call from Model::Build, skip AOE optimize and GE offline convert";
    return RET_OK;
  }
  if (param->device.find("Ascend") == std::string::npos) {
    MS_LOG(ERROR) << "Converter optimize should be ascend_oriented when provider is ge";
    return RET_ERROR;
  }
  if (!AscendGeExecutorPlugin::GetInstance().Register()) {
    MS_LOG(ERROR) << "Failed to register ge pass plugin";
    return RET_ERROR;
  }
  auto context = lite::acl::AsModelContext(param->aclModelOptionCfgParam, param->provider);
  if (context == nullptr) {
    MS_LOG(ERROR) << "Failed to converter ascend options to Model Context";
    return RET_ERROR;
  }
  bool run_aoe = !param->aclModelOptionCfgParam.aoe_mode.empty();
  if (!run_aoe) {
    auto sec_it = param->config_infos.find(kAoeGlobalOptionsSection);
    if (sec_it != param->config_infos.end()) {
      auto &options = sec_it->second;
      auto option_it = options.find("job_type");
      if (option_it != options.end()) {
        run_aoe = true;
      }
    }
  }
  param->config_infos[lite::kConverterParams][lite::kConverterOutputFile] = param->output_file;
  if (!run_aoe) {
    if (param->config_infos.find(kAscendContextSection) == param->config_infos.end() ||
        param->config_infos[kAscendContextSection].find(kParameterAsRefData) ==
          param->config_infos[kAscendContextSection].end()) {
      MS_LOG(INFO) << "Not find parameter_as_refdata in ascend_context, skip offline build graph";
      return RET_OK;
    }
    MS_LOG(INFO) << "GE offline model conversion begin";
    if (!AscendGeExecutorPlugin::GetInstance().OfflineBuildGraph(func_graph, context, param->config_infos)) {
      MS_LOG(ERROR) << "Failed to call GE offline model conversion";
      return RET_ERROR;
    }
    return RET_OK;
  } else {
    MS_LOG(ERROR) << "AOE tuning begin";
    if (!AscendGeExecutorPlugin::GetInstance().AoeTuning(func_graph, context, param->config_infos)) {
      MS_LOG(ERROR) << "Failed to call AOE Tuning";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

STATUS ConverterFuncGraph::CheckFuncGraph(const std::shared_ptr<ConverterPara> &param, FuncGraphPtr func_graph) {
  auto args_to_attr_pass = std::make_shared<opt::ArgsToAttrPass>();
  if (args_to_attr_pass == nullptr) {
    MS_LOG(ERROR) << "create pass failed";
    return RET_NULL_PTR;
  }
  if (!args_to_attr_pass->Run(func_graph)) {
    MS_LOG(ERROR) << "convert args to attr pass failed";
    return RET_ERROR;
  }
  return RET_OK;
}

STATUS ConverterFuncGraph::OptmizedConvert(const std::shared_ptr<ConverterPara> &param, FuncGraphPtr func_graph) {
  auto status = Quantize(param, func_graph);
  ClearBuiltinPass();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Quantize failed.";
    return status;
  }
  return RET_OK;
}

STATUS ConverterFuncGraph::Optimize(const std::shared_ptr<ConverterPara> &param, FuncGraphPtr func_graph) {
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "funcGraph is nullptr";
    return RET_ERROR;
  }

  if (CheckFuncGraph(param, func_graph) != RET_OK) {
    MS_LOG(ERROR) << "args to attr failed";
    return RET_ERROR;
  }

  bool is_optimized = IsOptimizedFuncGraph(func_graph);
  if (is_optimized) {
    auto status = OptmizedConvert(param, func_graph);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "optimized convert failed with " << status;
      return status;
    }
    return RET_OK;
  }

  if (param->provider == "ge") {
    MS_LOG(INFO) << "It will run ge optimize";
    return OptimizeForGE(param, func_graph);
  }
  std::vector<std::string> output_names;
  auto status = UnifyFuncGraphForInfer(param, func_graph, &output_names);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "UnifyFuncGraphForInfer failed.";
    return status;
  }
  // the original funcgraph's output_names is empty, if get by UnifyFuncGraphForInfer
  if (output_names.empty()) {
    output_names = FuncGraphUtils::GetFuncGraphOutputNames(func_graph);
    if (output_names.empty()) {
      MS_LOG(ERROR) << "GetFuncGraphOutputNames failed.";
      return RET_ERROR;
    }
  }

  status = UnifyFuncGraphInOutDataType(param, func_graph);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "UnifyFuncGraphForInfer failed.";
    return status;
  }

  // For converted MindIR model, update input name to the name of abstract.
  SetInputParameterName(func_graph);
  status = UpdateFuncGraphInputsAndOutputsDtype(func_graph);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Update graph inputs and outputs dtype failed.";
    return status;
  }

  status = UnifyFuncGraphInputFormat(param, func_graph);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "UnifyFuncGraphForInfer failed.";
    return status;
  }

  AnfTransform funcgraph_transform;
  status = funcgraph_transform.Transform(func_graph, param);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Transform anf graph failed.";
    return status;
  }

  status = UnifyFuncGraphOutputFormat(param, func_graph);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "UnifyFuncGraphOutputFormat failed.";
    return status;
  }

  FuncGraphUtils::SetFuncGraphOutputNames(func_graph, output_names);
  // Save input names to abstract, input names will be changed after load next time.
  SetInputParameterAbstractName(func_graph);
  if (!param->no_fusion) {
    func_graph->set_attr(kIsOptimized, MakeValue(true));
  }

  if (!param->cpuOptionCfgParam.architecture.empty()) {
    // Do offline pack.
    if (OfflinePackingOptimizer().Optimize(func_graph, "ANDROID_ARM_CPU") != RET_OK) {
      MS_LOG(ERROR) << "Do offline packing failed.";
      return status;
    }
  }

  return RET_OK;
}

int ConverterFuncGraph::Save(const std::shared_ptr<ConverterPara> &param, const FuncGraphPtr &func_graph, void **buff,
                             size_t *size) {
  mindspore::lite::MindIRSerializer serializer;
  if (param->provider == "ge") {
    serializer.SetRemoveVariableDir(false);
  }
  auto ret = serializer.Save(param, func_graph);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "MindIR serialize fail";
    return ret;
  }
  if (buff == nullptr || size == nullptr) {
    return RET_OK;
  }
  return serializer.GetBuffAndSize(buff, size);
}
STATUS ConverterFuncGraph::UnifyFuncGraphOutputFormat(const std::shared_ptr<ConverterPara> &param,
                                                      FuncGraphPtr func_graph) {
  auto spec_output_format = param->spec_output_format;
  if (spec_output_format == DEFAULT_FORMAT) {
    return RET_OK;
  }
  opt::SpecifyGraphOutputFormat pass(spec_output_format);
  auto status = pass.Run(func_graph);
  if (!status) {
    MS_LOG(ERROR) << "Failed to Specify graph output format to " << spec_output_format;
    return RET_ERROR;
  }
  opt::DecreaseTransposeAlgo transpose_pass;
  if (!transpose_pass.Run(func_graph)) {
    MS_LOG(ERROR) << "Failed to decrease transpose ";
    return RET_ERROR;
  }

  func_graph->set_attr(kOutputFormat, MakeValue(static_cast<int>(spec_output_format)));
  return RET_OK;
}
STATUS ConverterFuncGraph::Quantize(const std::shared_ptr<ConverterPara> &param, FuncGraphPtr func_graph) {
  if (!StoreBuiltinPass(param)) {
    MS_LOG(ERROR) << "store pass failed.";
    return RET_ERROR;
  }
  if (!RunOptimizerPass(func_graph, {"ToNHWCFormat", "DecreaseTransposeAlgo"})) {
    MS_LOG(ERROR) << "Run ToNHWCFormat pass failed";
    return RET_ERROR;
  }
  quant::QuantizationOptimizer quantization_optimizer(param);
  auto status = quantization_optimizer.Run(func_graph);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Post training quantization failed.";
    return status;
  }
  return RET_OK;
}
bool ConverterFuncGraph::StoreBuiltinPass(const std::shared_ptr<ConverterPara> &param) {
  if (param == nullptr) {
    MS_LOG(ERROR) << "param is nullptr";
    return false;
  }
  auto fmk = param->fmk_type;
  auto is_train = param->train_model;

  // pass_name, pass and boolean value to indicate whether can be called by external extension,
  std::vector<std::tuple<std::string, opt::PassPtr, bool>> pass_infos = {
    {"ToNCHWFormat", std::make_shared<opt::ToNCHWFormat>(fmk, is_train), true},
    {"ToNHWCFormat", std::make_shared<opt::ToNHWCFormat>(fmk, is_train), true},
    {"DecreaseTransposeAlgo", std::make_shared<opt::DecreaseTransposeAlgo>(fmk, is_train), true}};
  for (const auto &pass_info : pass_infos) {
    MS_CHECK_TRUE_RET(std::get<1>(pass_info) != nullptr, false);
    PassStorage::StorePass(std::get<0>(pass_info), std::get<1>(pass_info), std::get<opt::kInputIndexTwo>(pass_info));
  }
  return true;
}
void ConverterFuncGraph::ClearBuiltinPass() { PassStorage::ClearPass(); }

STATUS ConverterFuncGraph::RunVariableOptimize(const std::shared_ptr<ConverterPara> &param, FuncGraphPtr func_graph) {
  if (!param->ascendGeOptionCfg.inputs_to_variable.empty() && !param->ascendGeOptionCfg.outputs_to_variable.empty()) {
    auto input_and_output_variable = std::make_shared<opt::InputAndOutputVariablePass>(
      param->ascendGeOptionCfg.inputs_to_variable, param->ascendGeOptionCfg.outputs_to_variable);
    if (input_and_output_variable == nullptr) {
      MS_LOG(ERROR) << "input_and_output_variable is nullptr";
      return RET_ERROR;
    }
    if (!input_and_output_variable->Run(func_graph)) {
      MS_LOG(ERROR) << "Run input and output variable pass failed";
      return RET_ERROR;
    }
    return RET_OK;
  }
  if (!param->ascendGeOptionCfg.outputs_to_variable.empty()) {
    auto output_variable = std::make_shared<opt::OutputVariablePass>(param->ascendGeOptionCfg.outputs_to_variable);
    if (output_variable == nullptr) {
      MS_LOG(ERROR) << "output_variable is nullptr";
      return RET_ERROR;
    }
    if (!output_variable->Run(func_graph)) {
      MS_LOG(ERROR) << "Run output variable pass failed";
      return RET_ERROR;
    }
  }
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
