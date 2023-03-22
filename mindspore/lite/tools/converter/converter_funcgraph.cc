/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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
#include "tools/converter/anf_transform.h"
#include "tools/converter/offline_packing_optimizer.h"

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
    MS_LOG(ERROR) << "remove interference due to public-pirmitive failed.";
    return RET_ERROR;
  }
  MindsporeImporter::RemoveUnusedGraphInput(func_graph);

  auto status = CommonAnfAdjust(func_graph);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "CommonAnfAdjust failed.";
    return status;
  }

  // get output_names must be between CommonAnfAdjust and Mindir2AnfAdjust;
  *output_names = FuncGraphUtils::GetFuncGraphOutputNames(func_graph);
  if (output_names->empty()) {
    MS_LOG(ERROR) << "GetFuncGraphOutputNames failed.";
    return RET_ERROR;
  }
  if (param->device.find("Ascend") != std::string::npos) {
    MS_LOG(INFO) << "There is no need to adjust and pass graph when in Ascend.";
    return RET_OK;
  }
  status = MindsporeImporter::Mindir2AnfAdjust(func_graph, param);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Mindir2AnfAdjust failed.";
    return status;
  }

  if (!param->train_model) {
    auto redundant_op_remove_pass = std::make_shared<mindspore::opt::RemoveRedundantOpPass>(param->train_model, true);
    MS_CHECK_TRUE_MSG(redundant_op_remove_pass != nullptr, RET_NULL_PTR, "redundant_op_remove_pass is nullptr.");
    if (!redundant_op_remove_pass->Run(func_graph)) {
      MS_LOG(ERROR) << "Run remove redundant op failed";
      return RET_ERROR;
    }
  }

  auto unify_format = std::make_shared<UnifyFormatToNHWC>(converter::kFmkTypeMs, param->train_model, param->save_type);
  MS_CHECK_TRUE_MSG(unify_format != nullptr, RET_NULL_PTR, "unify_format is nullptr.");
  if (!unify_format->Run(func_graph)) {
    MS_LOG(ERROR) << "Run insert transpose failed.";
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
    MS_LOG(ERROR) << "Failed to get current format of graph input";
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

STATUS ConverterFuncGraph::UnifyFuncGraphInputDataType(const std::shared_ptr<ConverterPara> &param,
                                                       FuncGraphPtr func_graph) {
  if (param->input_data_type == DataType::kNumberTypeInt64) {
    if (param->fmk_type != FmkType::kFmkTypeTf) {
      MS_LOG(WARNING) << "In the current version, only TF model setting int64 input data type is supported.";
    }
    return RET_OK;
  }

  opt::InputDTypeTransPass pass(DataType::kNumberTypeInt32, DataType::kNumberTypeInt64);
  auto status = pass.Run(func_graph);
  if (!status) {
    MS_LOG(ERROR) << "Failed to Specify graph input data type to " << param->input_data_type;
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

STATUS ConverterFuncGraph::Optimize(const std::shared_ptr<ConverterPara> &param, FuncGraphPtr func_graph) {
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "funcGraph is nullptr";
    return RET_ERROR;
  }

  bool is_optimized = IsOptimizedFuncGraph(func_graph);
  if (is_optimized) {
    return RET_OK;
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

  status = UnifyFuncGraphInputDataType(param, func_graph);
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
}  // namespace lite
}  // namespace mindspore
