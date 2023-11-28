/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#include "extendrt/delegate/ascend_ge/ge_graph_executor.h"
#include <tuple>
#include <algorithm>
#include <utility>
#include "mindspore/core/ops/framework_ops.h"
#include "extendrt/delegate/factory.h"
#include "include/common/utils/scoped_long_running.h"
#include "include/api/context.h"
#include "include/api/status.h"
#include "include/transform/graph_ir/utils.h"
#include "include/backend/device_type.h"
#include "runtime/device/ms_device_shape_transfer.h"
#include "src/common/common.h"
#include "src/common/file_utils.h"
#include "cxx_api/acl_utils.h"
#include "mindspore/core/utils/ms_utils_secure.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/optimizer/graph/remove_load_pass.h"
#include "src/extendrt/utils/func_graph_utils.h"
#include "transform/graph_ir/transform_util.h"
#include "flow_graph/data_flow.h"
#ifdef MSLITE_ENABLE_GRAPH_KERNEL
#include "tools/graph_kernel/converter/graph_kernel_optimization.h"
#endif
#include "src/extendrt/utils/tensor_utils.h"
#include "framework/common/ge_inner_error_codes.h"
#include "src/extendrt/delegate/ascend_ge/aoe_api_tune_process.h"
#include "extendrt/delegate/ascend_ge/ge_utils.h"
#include "extendrt/delegate/ascend_ge/ge_dynamic_utils.h"
#include "mindspore/core/ops/lite_ops.h"
#include "mindspore/core/ops/nn_optimizer_ops.h"
#include "mindspore/lite/tools/common/string_util.h"
#include "mindspore/lite/src/extendrt/cxx_api/file_utils.h"
#include "mindspore/core/ops/custom.h"
#include "mindspore/lite/src/common/common.h"
#include "mindspore/lite/tools/common/custom_ascend_utils.h"
#include "inc/ops/array_ops.h"
#include "inc/ops/elewise_calculation_ops.h"

namespace mindspore {
namespace {
constexpr auto kProviderGe = "ge";
constexpr auto kDump = "dump";
constexpr auto kDumpMode = "dump_mode";
constexpr auto kProfiling = "profiler";
constexpr auto kDataFlowGraphType = "data_flow";
constexpr auto kCustomInputSize = 2;
constexpr auto kGraphKernelParam = "graph_kernel_param";
constexpr auto kUnkonwnSessionId = -1;
constexpr auto kRefModeNone = "none";
constexpr auto kRefModeVariable = "variable";
constexpr auto kRefModeAll = "all";
constexpr float kNumMicrosecondToMillisecond = 1000.0;
constexpr size_t kAlignRefData = 32;

size_t ALIGN_UP_REF_DATA(size_t size) {
  return ((size + kMemAlignSize + kAlignRefData - 1) / kMemAlignSize) * kMemAlignSize;
}

#ifdef MSLITE_ENABLE_GRAPH_KERNEL
std::shared_ptr<ConverterPara> ParseGraphKernelConfigs(const ConfigInfos &maps) {
  if (maps.find(kGraphKernelParam) == maps.end()) {
    return nullptr;
  }
  auto param = std::make_shared<ConverterPara>();
  const auto &gk_map = maps.at(kGraphKernelParam);
  std::stringstream oss;
  for (const auto &item : gk_map) {
    oss << "--" << item.first << "=" << item.second << " ";
  }
  param->device = GetSocVersion();
  param->graphKernelParam.graph_kernel_flags = oss.str();
  return param;
}
#endif
}  // namespace

std::atomic_uint32_t GeGraphExecutor::global_graph_idx_ = 0;
uint32_t GeGraphExecutor::GetNextGraphIdx() { return global_graph_idx_++; }
transform::DfGraphPtr GetDataFlowGraph(const FuncGraphPtr &anf_graph,
                                       const std::map<std::string, std::string> &ge_options) {
  MS_EXCEPTION_IF_NULL(anf_graph);
  auto return_node = anf_graph->get_return();
  MS_EXCEPTION_IF_NULL(return_node);
  auto nodes = anf_graph->TopoSort(return_node);
  auto itr = std::find_if(nodes.begin(), nodes.end(), [&](const AnfNodePtr &node) {
    return node != nullptr && node->isa<CNode>() && opt::CheckPrimitiveType(node, prim::kPrimCustom);
  });
  if (itr == nodes.end()) {
    MS_LOG(ERROR) << "The dataflow graph is invalid.";
    return nullptr;
  }
  auto custom_cnode = (*itr)->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(custom_cnode);
  if (custom_cnode->size() != kCustomInputSize) {
    MS_LOG(ERROR) << "The input of dataflow custom node is not 2.";
    return nullptr;
  }
  auto tensor = FuncGraphUtils::GetConstNodeValue(custom_cnode->input(1));
  MS_EXCEPTION_IF_NULL(tensor);
  auto data = tensor->data_c();
  MS_EXCEPTION_IF_NULL(data);
  auto flow_graph = reinterpret_cast<ge::dflow::FlowGraph *>(data);
  MS_EXCEPTION_IF_NULL(flow_graph);
  auto df_graph = std::make_shared<transform::DfGraph>(flow_graph->ToGeGraph());
  return df_graph;
}

GeGraphExecutor::~GeGraphExecutor() {
  if (ge_session_) {
    for (auto graph_id : init_graph_id_list_) {
      ge_session_->RemoveGraph(graph_id);
    }
    for (auto graph_id : compute_graph_id_list_) {
      ge_session_->RemoveGraph(graph_id);
      auto session_context = GeSessionManager::GetGeSessionContext(session_id_);
      if (session_context != nullptr) {
        (void)session_context->feature_graph_ids.erase(graph_id);
      }
    }
    ge_session_ = nullptr;
    GeSessionManager::TryReleaseGeSessionContext(session_id_);
    enable_update_weight_ = false;
    update_weight_ptr_ = nullptr;
  }
}

bool GeGraphExecutor::SetGeTensorShape(GeTensor *ge_tensor, ShapeVector shape) {
  if (!dyn_kv_cache_info_.dynamic_kv_cache) {
    return true;
  }
  auto ge_desc = ge_tensor->GetTensorDesc();
  ge::Shape ori_ge_shape(shape);
  ge_desc.Update(ori_ge_shape);
  ge_tensor->SetTensorDesc(ge_desc);
  MS_LOG(INFO) << "In SetGeTensorShape update ge shape to :" << shape;
  return true;
}

bool GeGraphExecutor::InitInputDeviceTensor(const FuncGraphPtr &anf_graph) {
  MS_LOG(INFO) << "Call InitInputDeviceTensor start.";
  auto inputs = anf_graph->get_inputs();
  inputs_buffer_infos_.resize(inputs.size());
  for (size_t i = 0; i < inputs.size(); i++) {
    auto &input_info = inputs_buffer_infos_[i];
    auto shape = FuncGraphUtils::GetTensorShape({inputs[i], 0});

    /* set max_batch size and max_seq_len for dyn shape */
    std::vector<int64_t> new_shape;
    for (size_t j = 0; j < shape.size(); j++) {
      if (shape[j] == abstract::Shape::kShapeDimAny) {
        new_shape.push_back(dyn_kv_cache_info_.max_seq_len_size);
      } else {
        new_shape.push_back(shape[j]);
      }
    }

    MS_LOG(INFO) << "Init input_" << i << " buffer for ge, change shape: " << shape << " -> " << new_shape;
    auto dtype = static_cast<TypeId>(FuncGraphUtils::GetTensorDataType({inputs[i], 0}));
    if (!InitInOutDeviceBuffer("Input " + std::to_string(i), new_shape, dtype, &input_info)) {
      return false;
    }
  }
  return true;
}

bool GeGraphExecutor::UpdateDynamicDimsOption(
  const std::vector<std::pair<std::string, tensor::TensorPtr>> &ref_data_tensors,
  std::map<std::string, std::string> *ge_options_ptr) {
  if (!dyn_kv_cache_info_.dynamic_kv_cache) {
    MS_LOG(INFO) << "Static ref data.";
    return true;
  }

  if (ge_options_ptr == nullptr) {
    MS_LOG(ERROR) << "Input argument ge_options_ptr cannot be nullptr";
    return false;
  }
  std::string new_dynamic_dims_str;
  std::vector<std::vector<int64_t>> config_dynamic_dims;
  if (!GeDynamicUtils::GetDynamicDims(context_, config_infos_, &config_dynamic_dims)) {
    MS_LOG(ERROR) << "Failed to get dynamic dims from AscendDeviceInfo or config file";
    return false;
  }
  if (config_dynamic_dims.empty()) {
    MS_LOG(INFO) << "Not found input shape in AscendDeviceInfo or config file";
    return true;
  }

  if (config_dynamic_dims.size() != dyn_kv_cache_info_.dynamic_kv_cache_dims.size()) {
    MS_LOG(ERROR) << "ge.dynamicDims num:" << config_dynamic_dims.size()
                  << ", dynamic_kv_cache_dims num: " << dyn_kv_cache_info_.dynamic_kv_cache_dims.size();
    return true;
  }

  int i = 0;
  for (auto input_dims : config_dynamic_dims) {
    /* set input dynamicDims */
    for (auto dim : input_dims) {
      new_dynamic_dims_str += std::to_string(dim) + ",";
    }

    /* set ref data dynamicDims */
    auto dyn_ref_dims = dyn_kv_cache_info_.dynamic_kv_cache_dims[i++];
    int64_t level_count = 0;
    while (level_count < dyn_kv_cache_info_.dyn_ref_num) {
      level_count++;
      for (auto dyn_ref_dim : dyn_ref_dims) {
        new_dynamic_dims_str += std::to_string(dyn_ref_dim) + ",";
      }
    }

    new_dynamic_dims_str.erase(new_dynamic_dims_str.end() - 1);
    new_dynamic_dims_str += ";";
  }
  new_dynamic_dims_str.erase(new_dynamic_dims_str.end() - 1);

  GeDynamicUtils::UpdateGraphDynamicDims(context_, &config_infos_, new_dynamic_dims_str);
  (*ge_options_ptr)["ge.dynamicDims"] = new_dynamic_dims_str;
  MS_LOG(INFO) << "Update ge.dynamicDims to " << new_dynamic_dims_str;
  return true;
}

void GeGraphExecutor::SetRefShape(std::vector<int64_t> *ref_shape, bool dyn, std::string tensor_name) {
  if (!dyn_kv_cache_info_.dynamic_kv_cache) {
    return;
  }

  if (dyn) {
    if (dyn_kv_cache_info_.batch_size_dyn) {
      (*ref_shape)[Index0] = abstract::Shape::kShapeDimAny;
      MS_LOG(INFO) << "for " << tensor_name << " update batch size to dyn(-1) for ge_option.";
    }
    if (dyn_kv_cache_info_.seq_length_dyn) {
      (*ref_shape)[Index2] = abstract::Shape::kShapeDimAny;
      MS_LOG(INFO) << "for " << tensor_name << " update seq length size to dyn(-1) for ge_option.";
    }
  } else {
    if (dyn_kv_cache_info_.batch_size_dyn) {
      (*ref_shape)[Index0] = dyn_kv_cache_info_.real_batch_size;
      MS_LOG(INFO) << "for " << tensor_name << " update batch size to " << dyn_kv_cache_info_.real_batch_size
                   << " for ge_option.";
    }
    if (dyn_kv_cache_info_.seq_length_dyn) {
      (*ref_shape)[Index2] = dyn_kv_cache_info_.real_seq_len_size;
      MS_LOG(INFO) << "for " << tensor_name << " update seq length size to " << dyn_kv_cache_info_.real_seq_len_size
                   << " for ge_option.";
    }
  }
}

void GeGraphExecutor::UpdateOutputShapeInfo() {
  if (!dyn_kv_cache_info_.dynamic_kv_cache) {
    return;
  }
  MS_LOG(INFO) << "Update output dtype and shape.";
  for (size_t i = 0; i < outputs_buffer_infos_.size(); i++) {
    auto &output_info = outputs_buffer_infos_[i];
    auto ge_tensor_desc = output_info.ge_tensor.GetTensorDesc();
    output_info.shape = transform::TransformUtil::ConvertGeShape(ge_tensor_desc.GetShape());
    output_info.max_size = SizeOf(output_info.shape) * GetDataTypeSize(output_info.dtype);
    output_info.dtype = transform::TransformUtil::ConvertGeDataType(ge_tensor_desc.GetDataType());
    output_info.device_addr = output_info.ge_tensor.GetData();
    MS_LOG(INFO) << "Update output_" << i << " dtype: " << output_info.dtype << ", shape: " << output_info.shape;
  }
  return;
}

void GeGraphExecutor::SetDynamicKVCache() {
  auto section_it = config_infos_.find(lite::kAscendContextSection);
  if (section_it == config_infos_.end()) {
    MS_LOG(INFO) << "no ascend_context info";
    return;
  }

  MS_LOG(INFO) << "dynamic_kv_cache info set.";
  auto &options = section_it->second;
  auto option_it = options.find("dynamic_kv_cache");
  if (option_it != options.end()) {
    MS_LOG(INFO) << "Find [ascend_context] dynamic_kv_cache : " << option_it->second;
    if (option_it->second == "seq") {
      dyn_kv_cache_info_.dynamic_kv_cache = true;
      dyn_kv_cache_info_.seq_length_dyn = true;
    } else if (option_it->second == "bs") {
      dyn_kv_cache_info_.dynamic_kv_cache = true;
      dyn_kv_cache_info_.batch_size_dyn = true;
    } else if (option_it->second == "both") {
      dyn_kv_cache_info_.dynamic_kv_cache = true;
      dyn_kv_cache_info_.seq_length_dyn = true;
      dyn_kv_cache_info_.batch_size_dyn = true;
    }
  } else {
    dyn_kv_cache_info_.dynamic_kv_cache = false;
    MS_LOG(INFO) << "no dynamic_kv_cache info";
    return;
  }

  MS_LOG(INFO) << "set dyn_kv_cache_info_.dynamic_kv_cache : " << dyn_kv_cache_info_.dynamic_kv_cache;
  MS_LOG(INFO) << "set dyn_kv_cache_info_.batch_size_dyn  : " << dyn_kv_cache_info_.batch_size_dyn;
  MS_LOG(INFO) << "set dyn_kv_cache_info_.seq_length_dyn : " << dyn_kv_cache_info_.seq_length_dyn;

  option_it = options.find("dynamic_kv_cache_dims");
  dyn_kv_cache_info_.dynamic_kv_cache_dims.clear();
  if (option_it != options.end()) {
    std::string dyn_ref_dims_str = option_it->second;
    MS_LOG(INFO) << "Find [ascend_context] dynamic_kv_cache_dims : " << dyn_ref_dims_str;
    auto dyn_ref_dims = lite::StrSplit(dyn_ref_dims_str, ";");
    for (auto &item : dyn_ref_dims) {
      std::vector<int64_t> real_dims;
      if (!lite::ParseShapeStr(item, &real_dims)) {
        MS_LOG(ERROR) << "Invalid dyn_ref_dims " << dyn_ref_dims;
        return;
      }
      dyn_kv_cache_info_.dynamic_kv_cache_dims.push_back(real_dims);
    }
  }
  MS_LOG(INFO) << "set   dyn_kv_cache_info_.dynamic_kv_cache_dims : " << dyn_kv_cache_info_.dynamic_kv_cache_dims;

  option_it = options.find("dynamic_kv_cache_num");
  if (option_it != options.end()) {
    std::string dynamic_kv_cache_num_str = option_it->second;
    MS_LOG(INFO) << "Find [ascend_context] dynamic_kv_cache_num : " << dynamic_kv_cache_num_str;
    dyn_kv_cache_info_.dyn_ref_num = std::stoi(dynamic_kv_cache_num_str);
  }
  MS_LOG(INFO) << "set dyn_kv_cache_info_.dyn_ref_num : " << dyn_kv_cache_info_.dyn_ref_num;
  return;
}

void GeGraphExecutor::InitRealShapeParam(const std::vector<tensor::Tensor> &inputs) {
  auto &input_0 = inputs[0];
  dyn_kv_cache_info_.real_batch_size = input_0.shape_c().at(Index0);
  MS_LOG(INFO) << "Real batch size : " << dyn_kv_cache_info_.real_batch_size;
  dyn_kv_cache_info_.real_seq_len_size = input_0.shape_c().at(Index1);
  MS_LOG(INFO) << "Real seq length size : " << dyn_kv_cache_info_.real_seq_len_size;
  return;
}

bool GeGraphExecutor::GetConfigOption(const std::string &section_name, const std::string &option_name,
                                      std::string *option_val) {
  if (option_val == nullptr) {
    MS_LOG(ERROR) << "Input argument option_val is nullptr";
    return false;
  }
  auto config_it = config_infos_.find(section_name);
  if (config_it == config_infos_.end()) {
    return false;
  }
  auto &options = config_it->second;
  auto option_it = options.find(option_name);
  if (option_it == options.end()) {
    return false;
  }
  *option_val = option_it->second;
  return true;
}

uint32_t GeGraphExecutor::GetRankID() const {
  auto ascend_info = GeUtils::GetAscendDeviceInfo(context_);
  if (ascend_info == nullptr) {
    MS_LOG(ERROR) << "Can not find ascend device context.";
    return 0;
  }
  return ascend_info->GetRankID();
}

uint32_t GeGraphExecutor::GetDeviceID() const {
  auto ascend_info = GeUtils::GetAscendDeviceInfo(context_);
  if (ascend_info == nullptr) {
    MS_LOG(ERROR) << "Can not find ascend device context.";
    return 0;
  }
  return ascend_info->GetDeviceID();
}

bool GeGraphExecutor::Init() {
  ge_global_context_ = GeDeviceContext::InitGlobalContext(context_, config_infos_);
  if (ge_global_context_ == nullptr) {
    MS_LOG(ERROR) << "Failed to Init global context";
    return false;
  }
  std::string ref_mode;
  (void)GetConfigOption(lite::kAscendContextSection, lite::kParameterAsRefData, &ref_mode);
  if (!ref_mode.empty()) {
    ref_mode = lite::StringTolower(ref_mode);
    if (ref_mode != kRefModeNone && ref_mode != kRefModeVariable && ref_mode != kRefModeAll) {
      MS_LOG(ERROR) << "Only " << kRefModeNone << ", " << kRefModeVariable << " or " << kRefModeAll
                    << " is supported for " << lite::kParameterAsRefData << ", but got " << ref_mode;
      return false;
    }
    if (ref_mode == kRefModeAll) {
      ref_mode_flag_ = transform::RefModeFlag::kRefModeAll;
    } else if (ref_mode == kRefModeVariable) {
      ref_mode_flag_ = transform::RefModeFlag::kRefModeVariable;
    } else {
      ref_mode_flag_ = transform::RefModeFlag::kRefModeNone;
    }
    MS_LOG(INFO) << "Set parameter ref mode " << ref_mode;
  } else {
    ref_mode_flag_ = transform::RefModeFlag::kRefModeNone;
  }
  std::string model_cache_mode;
  (void)GetConfigOption(lite::kAscendContextSection, lite::kModelCacheMode, &model_cache_mode);
  if (!model_cache_mode.empty()) {
    cache_mode_ = true;
    MS_LOG(INFO) << "Set set variable ref mode";
  }
  std::string variable_weights_list;
  (void)GetConfigOption(lite::kAscendContextSection, "variable_weights_list", &variable_weights_list);
  if (!variable_weights_list.empty()) {
    update_weight_ptr_ = std::make_shared<UpdateWeight>();
    if (update_weight_ptr_ == nullptr) {
      MS_LOG(ERROR) << "init update weight ptr failed.";
      return false;
    }
    if (!update_weight_ptr_->ParseUpdateWeightConfig(variable_weights_list)) {
      MS_LOG(ERROR) << "ParseUpdateWeightConfig failed.";
      update_weight_ptr_ = nullptr;
      return false;
    }
    enable_update_weight_ = true;
  }
  return true;
}

void GeGraphExecutor::GetGeSessionOptions(std::map<std::string, std::string> *ge_options_ptr) {
  MS_EXCEPTION_IF_NULL(ge_options_ptr);
  auto &ge_options = *ge_options_ptr;
  ge_options["ge.trainFlag"] = "0";
  ge_options["ge.enablePrintOpPass"] = "0";
  ge_options["ge.exec.device_id"] = std::to_string(GetDeviceID());
  ge_options["ge.exec.staticMemoryPolicy"] = "2";
  if (ref_mode_flag_ != transform::RefModeFlag::kRefModeNone) {
    if (!offline_mode_) {
      ge_options["ge.featureBaseRefreshable"] = "1";
    } else {
      ge_options["ge.featureBaseRefreshable"] = "0";
    }
    ge_options["ge.constLifecycle"] = "graph";
  }
  auto config_it = config_infos_.find(lite::kGeSessionOptionsSection);
  if (config_it != config_infos_.end()) {
    for (auto &item : config_it->second) {
      ge_options[item.first] = item.second;
      MS_LOG(INFO) << "Set ge session option " << item.first << " to " << item.second;
    }
  }
  config_it = config_infos_.find(lite::kAscendContextSection);
  if (config_it != config_infos_.end()) {
    GetGeSessionOptionsFromAscendContext(config_it->second, ge_options_ptr);
  }
}

bool GeGraphExecutor::SetModelCacheDir(std::map<std::string, std::string> *session_options_ptr) {
  if (!offline_mode_ && !cache_mode_) {
    return true;
  }
  auto &ge_options = *session_options_ptr;
  auto option_id = ge_options.find(kGeGraphCompilerCacheDir);
  if (option_id != ge_options.end()) {
    build_cache_dir_ = option_id->second;
    return true;
  }
  if (!offline_mode_) {  // cache_mode = True
    build_cache_dir_ = "model_build_cache_" + std::to_string(GetRankID());
    if (lite::CreateDir(build_cache_dir_) != RET_OK) {
      MS_LOG(ERROR) << "Failed to create build cache dir " << build_cache_dir_;
      return false;
    }
    ge_options[kGeGraphCompilerCacheDir] = build_cache_dir_;
    MS_LOG(INFO) << "Update session attr " << kGeGraphCompilerCacheDir << " to " << build_cache_dir_;
    return true;
  }
  bool build_cache_enabled = false;
  std::string output_file;
  (void)GetConfigOption(lite::kConverterParams, lite::kConverterOutputFile, &output_file);
  std::string output_dir = "./";
  if (output_file.find("/") != std::string::npos) {
    output_dir = output_file.substr(0, output_file.rfind("/") + 1);
  }
  auto ge_session_context = GeSessionManager::GetGeSessionContext(session_id_);
  if (ge_session_context) {
    const auto &last_ge_options = ge_session_context->session_options;
    if (auto it = last_ge_options.find(kGeGraphCompilerCacheDir); it != last_ge_options.end()) {
      build_cache_dir_ = it->second;
      build_cache_enabled = true;
    }
  }
  if (!build_cache_enabled) {
    std::string mindir_postfix = ".mindir";
    auto ops = output_file.find(mindir_postfix);
    if (ops != std::string::npos && ops == output_file.size() - mindir_postfix.size()) {
      output_file = output_file.substr(0, output_file.size() - mindir_postfix.size());
    }
    if (output_file.empty()) {
      MS_LOG(ERROR) << "Converter output file cannot be empty";
      return false;
    }
    build_cache_dir_ = output_file + "_variables";
  }
  if (lite::CreateDir(build_cache_dir_) != RET_OK) {
    MS_LOG(ERROR) << "Failed to create build cache dir " << build_cache_dir_;
    return false;
  }
  ge_options[kGeGraphCompilerCacheDir] = build_cache_dir_;
  MS_LOG(INFO) << "Update session attr " << kGeGraphCompilerCacheDir << " to " << build_cache_dir_;
  if (build_cache_dir_.find(output_dir) == 0) {
    build_cache_relative_dir_ = "./" + build_cache_dir_.substr(output_dir.size());
  }
  return true;
}

void GeGraphExecutor::GetGeSessionOptionsFromAscendContext(const std::map<std::string, std::string> &config,
                                                           std::map<std::string, std::string> *ge_options_ptr) {
  MS_EXCEPTION_IF_NULL(ge_options_ptr);
  auto &ge_options = *ge_options_ptr;
  auto option_id = config.find(lite::kDumpPathKey);
  if (option_id != config.end()) {
    auto dump_path = option_id->second;
    auto real_path = lite::RealPath(dump_path.c_str());
    std::ifstream ifs(real_path);
    if (!ifs.good() || !ifs.is_open()) {
      MS_LOG(EXCEPTION) << "The dump config file: " << real_path << " is not exit or open failed.";
    }
    nlohmann::json dump_cfg_json;
    try {
      dump_cfg_json = nlohmann::json::parse(ifs);
    } catch (const nlohmann::json::parse_error &error) {
      MS_LOG(EXCEPTION) << "parse json failed, please check the file: " << real_path;
    }
    if (dump_cfg_json[kDump] != nullptr && dump_cfg_json[kDump][kDumpMode] != nullptr) {
      ge_options["ge.exec.enableDump"] = "1";
      ge_options["ge.exec.dumpMode"] = dump_cfg_json[kDump][kDumpMode].get<std::string>();
    }
  }
  option_id = config.find(lite::kProfilingPathKey);
  if (option_id != config.end()) {
    auto profiling_path = option_id->second;
    auto real_path = lite::RealPath(profiling_path.c_str());
    std::ifstream ifs(real_path);
    if (!ifs.good() || !ifs.is_open()) {
      MS_LOG(EXCEPTION) << "The profiling_path config file: " << real_path << " is not exit or open failed.";
    }
    nlohmann::json profiling_cfg_json;
    try {
      profiling_cfg_json = nlohmann::json::parse(ifs);
    } catch (const nlohmann::json::parse_error &error) {
      MS_LOG(EXCEPTION) << "parse json failed, please check the file: " << real_path;
    }
    if (profiling_cfg_json[kProfiling] != nullptr) {
      ge_options["ge.exec.profilingMode"] = "1";
      ge_options["ge.exec.profilingOptions"] = profiling_cfg_json[kProfiling].dump();
    }
  }
  option_id = config.find(lite::kGeVariableMemoryMaxSize);
  if (option_id != config.end()) {
    ge_options["ge.variableMemoryMaxSize"] = option_id->second;
  }
  option_id = config.find(lite::kGeGraphMemoryMaxSize);
  if (option_id != config.end()) {
    ge_options["ge.graphMemoryMaxSize"] = option_id->second;
  }
  option_id = config.find(lite::kGraphCompilerCacheDirKey);
  if (option_id != config.end()) {
    ge_options[kGeGraphCompilerCacheDir] = option_id->second;
  }
}

void GeGraphExecutor::GetGeGraphOptions(const FuncGraphPtr &anf_graph,
                                        std::map<std::string, std::string> *ge_options_ptr) {
  MS_EXCEPTION_IF_NULL(anf_graph);
  MS_EXCEPTION_IF_NULL(ge_options_ptr);
  auto &ge_options = *ge_options_ptr;
  auto ascend_device_info = GeUtils::GetAscendDeviceInfo(context_);
  if (ascend_device_info == nullptr) {
    MS_LOG(EXCEPTION) << "Failed to get graph session options, can not find ascend device context.";
  }
  uint32_t rank_id = ascend_device_info->GetRankID();
  graph_name_ = std::to_string(rank_id) + "_" + std::to_string(global_graph_idx_) + "_" + anf_graph->ToString();
  for (auto &c : graph_name_) {
    if (c == '.') {
      c = '_';
    }
  }
  ge_options[kGeGraphKey] = graph_name_;
  auto config_it = config_infos_.find(lite::kGeGraphOptionsSection);
  if (config_it != config_infos_.end()) {
    for (auto &item : config_it->second) {
      ge_options[item.first] = item.second;
      MS_LOG(INFO) << "Set ge graph option " << item.first << " to " << item.second;
    }
  }

  auto precision_mode = ascend_device_info->GetPrecisionMode();
  if (!precision_mode.empty()) {
    ge_options["ge.exec.precision_mode"] = TransforPrecisionToAcl(precision_mode);
  }
  config_it = config_infos_.find(lite::kAscendContextSection);
  if (config_it == config_infos_.end()) {
    return;
  }
  auto config = config_it->second;
  auto option_id = config.find(lite::kModifyMixList);
  if (option_id != config.end()) {
    ge_options["ge.exec.modify_mixlist"] = option_id->second;
  }
}

int64_t GeGraphExecutor::GetSessionId() {
  std::string inner_group_id;
  (void)GetConfigOption(lite::kLiteInnerGroupSection, lite::kLiteInnerGroupId, &inner_group_id);
  if (inner_group_id.empty()) {
    return kUnkonwnSessionId;
  }
  int64_t session_id = kUnkonwnSessionId;
  if (!lite::ConvertStrToInt(inner_group_id, &session_id)) {
    MS_LOG_WARNING << "Failed to parse session_id " << inner_group_id << " to int64_t";
    return kUnkonwnSessionId;
  }
  return session_id;
}

bool GeGraphExecutor::CreateSession() {
  if (ge_session_ != nullptr) {
    MS_LOG(INFO) << "Ge session has already been created";
    return true;
  }
  session_id_ = GetSessionId();
  (void)setenv("GE_TRAIN", "0", 1);
  std::map<std::string, std::string> session_options;
  GetGeSessionOptions(&session_options);
  if (!SetModelCacheDir(&session_options)) {
    return false;
  }
  ge_session_ = GeSessionManager::CreateGeSession(session_id_, session_options);
  if (ge_session_ == nullptr) {
    MS_LOG(ERROR) << "Failed to create ge session";
    return false;
  }
  return true;
}

bool GeGraphExecutor::AddGraph(const transform::DfGraphPtr &graph, const std::map<std::string, std::string> &options,
                               uint32_t *graph_id_ret) {
  if (ge_session_ == nullptr) {
    MS_LOG(ERROR) << "Failed to add graph, ge session cannot be nullptr";
    return false;
  }
  auto graph_id = GetNextGraphIdx();
  for (auto &option : options) {
    MS_LOG(INFO) << "GE Graph " << graph_id << " option " << option.first << " = " << option.second;
  }
  auto ge_status = ge_session_->AddGraph(static_cast<uint32_t>(graph_id), *(graph), options);
  if (ge_status != ge::GRAPH_SUCCESS) {
    MS_LOG(ERROR) << "Call GE AddGraph Failed: " << ge::GEGetErrorMsg();
    return false;
  }
  *graph_id_ret = graph_id;
  return true;
}

void GeGraphExecutor::GetParams(const FuncGraphPtr &anf_graph, transform::TensorOrderMap *param_tensors) {
  MS_EXCEPTION_IF_NULL(anf_graph);

  transform::TensorOrderMap res;
  for (auto &anf_node : anf_graph->parameters()) {
    MS_EXCEPTION_IF_NULL(anf_node);
    auto para = anf_node->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(para);
    if (para->has_default()) {
      auto value = para->default_param();
      MS_EXCEPTION_IF_NULL(value);
      auto tensor = value->cast<std::shared_ptr<tensor::Tensor>>();
      MS_EXCEPTION_IF_NULL(tensor);
      auto para_name = para->name();
      res.emplace(para_name, tensor);
    }
  }
  if (session_id_ != kUnkonwnSessionId) {
    std::vector<std::string> graph_params;
    std::transform(res.begin(), res.end(), std::back_inserter(graph_params),
                   [](const auto &item) { return item.first; });
    auto new_params_set = GeSessionManager::UpdateSessionVariables(session_id_, graph_params);
    for (auto &item : res) {
      // parameters not in new_params_set has been init by other graph
      if (new_params_set.find(item.first) == new_params_set.end()) {
        item.second->set_init_flag(true);
      }
    }
  }
  *param_tensors = res;
}

bool GeGraphExecutor::UpdateGraphInputs(const FuncGraphPtr &graph) {
  std::string input_shape_str;
  std::vector<GeDynamicShapeInfo> input_shapes;
  if (!GeDynamicUtils::GetGraphInputShapes(context_, config_infos_, &input_shapes, &input_shape_str)) {
    MS_LOG(ERROR) << "Failed to get input shape from AscendDeviceInfo or config file";
    return false;
  }
  if (input_shapes.empty()) {
    MS_LOG(INFO) << "Not found input shape in AscendDeviceInfo or config file";
    return true;
  }
  auto inputs = graph->get_inputs();
  if (inputs.size() != input_shapes.size()) {
    MS_LOG(ERROR) << "FuncGraph input size " << inputs.size() << " != input size " << input_shapes.size()
                  << " in AscendDeviceInfo or config file " << input_shapes.size();
    return false;
  }
  for (size_t i = 0; i < input_shapes.size(); i++) {
    auto node = inputs[i];
    MS_CHECK_TRUE_RET(node != nullptr, false);
    auto input_shape = input_shapes[i];
    auto para = node->cast<ParameterPtr>();
    if (para == nullptr) {
      MS_LOG(ERROR) << "Cast input to Parameter failed";
      return false;
    }
    auto it = std::find_if(input_shapes.begin(), input_shapes.end(),
                           [&para](const auto &item) { return item.name == para->name(); });
    if (it == input_shapes.end()) {
      MS_LOG(ERROR) << "Failed to find input " << para->name() << " in input_shape " << input_shape_str;
      return false;
    }
    auto abstract = para->abstract();
    if (abstract == nullptr) {
      MS_LOG(ERROR) << "Get input abstract failed";
      return false;
    }
    ShapeVector shape;
    std::transform(it->shape.begin(), it->shape.end(), std::back_inserter(shape), [](auto &dim) { return dim.dim; });
    MS_LOG(INFO) << "Update shape of input_" << i << " " << para->name() << " to " << shape;
    abstract->set_shape(std::make_shared<abstract::Shape>(shape));
  }
  return true;
}

bool GeGraphExecutor::InitRefDataList(const std::vector<std::pair<std::string, tensor::TensorPtr>> &ref_data_tensors) {
  for (auto &item : ref_data_tensors) {
    auto para_name = item.first;
    auto &tensor = item.second;
    RefDataInfo ref_data_info;
    ref_data_info.name = para_name;
    ref_data_info.shape = tensor->shape_c();
    ref_data_info.dtype = tensor->data_type();
    ref_data_info.host_data = item.second;
    ref_data_infos_.push_back(ref_data_info);
  }
  return true;
}

bool GeGraphExecutor::InitMemoryContextManager() {
  auto session_context = GeSessionManager::GetGeSessionContext(session_id_);
  if (session_context != nullptr) {
    memory_manager_ = session_context->memory_manager.lock();
    context_manager_ = session_context->context_manager.lock();
  }
  if (memory_manager_ == nullptr) {
    memory_manager_ = std::make_shared<GeMemoryManager>();
    if (memory_manager_ == nullptr) {
      MS_LOG(ERROR) << "Failed to create memory manager";
      return false;
    }
    if (session_context != nullptr) {
      session_context->memory_manager = memory_manager_;
    }
  }
  if (context_manager_ == nullptr) {
    context_manager_ = std::make_shared<GeContextManager>();
    if (context_manager_ == nullptr) {
      MS_LOG(ERROR) << "Failed to create context manager";
      return false;
    }
    if (!context_manager_->InitContext(GetDeviceID())) {
      MS_LOG(ERROR) << "Failed to init device";
      return false;
    }
    if (session_context != nullptr) {
      session_context->context_manager = context_manager_;
    }
  }
  if (!context_manager_->SetContext()) {
    MS_LOG(ERROR) << "Failed to set ge context";
    return false;
  }
  return true;
}

bool GeGraphExecutor::InitRefDataDeviceTensor() {
  MS_LOG(INFO) << "InitRefDataDeviceTensor start.";
  if (ref_data_infos_.empty()) {
    MS_LOG(INFO) << "There is not ref data, no need to init ref data device data";
    return true;
  }
  std::map<std::string, RefDataInfo> session_ref_data_map;
  auto session_context = GeSessionManager::GetGeSessionContext(session_id_);
  if (session_context != nullptr) {
    session_ref_data_map = session_context->ref_data_map_;
  }

  size_t ref_data_total_size = 0;
  std::map<std::string, tensor::TensorPtr> new_param_tensor_map;
  for (size_t i = 0; i < ref_data_infos_.size(); i++) {
    auto &item = ref_data_infos_[i];
    auto tensor = item.host_data;
    item.host_data = nullptr;  // release host memory
    if (auto ref_it = session_ref_data_map.find(item.name); ref_it != session_ref_data_map.end()) {
      item.ge_tensor = ref_it->second.ge_tensor;
      continue;
    }
    ShapeVector ref_data_shape = tensor->shape_c();
    SetRefShape(&ref_data_shape, true, item.name);
    auto desc = transform::TransformUtil::GetGeTensorDesc(ref_data_shape, tensor->data_type(), kOpFormat_NCHW);
    if (desc == nullptr) {
      MS_LOG(ERROR) << "Failed to get Tensor Desc";
      return false;
    }
    desc->SetPlacement(::ge::kPlacementDevice);
    auto ret = item.ge_tensor.SetTensorDesc(*desc);
    if (ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "Failed to call ge::Tensor::SetTensorDesc, ret " << ret;
      return false;
    }
    item.size = tensor->Size();
    item.offset = ref_data_total_size;
    ref_data_total_size += ALIGN_UP_REF_DATA(tensor->Size());
    new_param_tensor_map[item.name] = tensor;
  }
  if (ref_data_total_size != 0) {
    auto device_memory = memory_manager_->MallocDeviceMemory("RefData input", ref_data_total_size);
    if (device_memory == nullptr) {
      return false;
    }
    for (auto &item : ref_data_infos_) {
      auto it = new_param_tensor_map.find(item.name);
      if (it == new_param_tensor_map.end()) {
        continue;
      }
      auto &tensor_val = it->second;
      auto dst_addr = device_memory + item.offset;
      if (!memory_manager_->MemcpyHost2Device(dst_addr, item.size, tensor_val->data_c(), tensor_val->Size())) {
        MS_LOG(ERROR) << "Failed to memory copy host data to device";
        return false;
      }
      auto ret = item.ge_tensor.SetData(dst_addr, item.size, [](uint8_t *) -> void {});
      if (ret != ge::GRAPH_SUCCESS) {
        MS_LOG(ERROR) << "Failed to call ge::Tensor SetData(uint8_t*, size, DeleteFunc), data size " << item.size;
        return false;
      }
      if (session_context != nullptr) {
        session_context->ref_data_map_[item.name] = item;
      }
    }
  }
  return true;
}

bool GeGraphExecutor::InitConstantFeatureDeviceMemory(uint32_t graph_id) {
  MS_LOG(INFO) << "Call GE GetCompiledGraphSummary start, graph id " << graph_id;
  auto graph_summary = ge_session_->GetCompiledGraphSummary(graph_id);
  if (graph_summary == nullptr) {
    MS_LOG(ERROR) << "Failed to call GE GetCompiledGraphSummary, graph id " << graph_id
                  << ", error: " << ge::GEGetErrorMsg();
    return false;
  }
  MS_LOG(INFO) << "Call GE GetCompiledGraphSummary end, graph id " << graph_id;

  if (!ParseGeCompiledGraphSummary(graph_summary)) {
    return false;
  }
  if (memory_manager_ == nullptr) {
    MS_LOG(ERROR) << "Memory manager or context manager is nullptr";
    return false;
  }
  // for constant memory
  MS_LOG(INFO) << "Const memory size " << runtime_info_.const_size << ", ge graph id: " << graph_id;
  if (runtime_info_.const_size > 0) {
    runtime_info_.const_addr = memory_manager_->MallocDeviceMemory(
      "Graph " + std::to_string(graph_id) + " constant memory", runtime_info_.const_size);
    if (runtime_info_.const_addr == nullptr) {
      MS_LOG(ERROR) << "Failed to malloc device memory for const memory, memory size " << runtime_info_.const_size;
      return false;
    }
    if (!SetConstMemory(graph_id, runtime_info_.const_addr, runtime_info_.const_size)) {
      MS_LOG(ERROR) << "Failed to SetConstMemory for graph " << graph_id << ", constant size "
                    << runtime_info_.const_size;
      return false;
    }
  }
  // for feature memory
  MS_LOG(INFO) << "Feature memory size " << runtime_info_.feature_size << ", ge graph id: " << graph_id;
  if (runtime_info_.feature_size > 0) {
    auto session_context = GeSessionManager::GetGeSessionContext(session_id_);
    if (session_context == nullptr || session_context->feature_size < runtime_info_.feature_size) {
      if (session_context != nullptr && session_context->feature_memory != nullptr) {
        memory_manager_->FreeDeviceMemory(session_context->feature_memory);
        session_context->feature_memory = nullptr;
        session_context->feature_size = 0;
      }
      runtime_info_.feature_addr = memory_manager_->MallocDeviceMemory(
        "Graph " + std::to_string(graph_id) + " feature memory", runtime_info_.feature_size);
      if (runtime_info_.feature_addr == nullptr) {
        MS_LOG(ERROR) << "Failed to malloc feature memory for const memory, memory size " << runtime_info_.feature_size;
        return false;
      }
      if (session_context != nullptr) {
        session_context->feature_memory = runtime_info_.feature_addr;
        session_context->feature_size = runtime_info_.feature_size;
        for (auto &item : session_context->feature_graph_ids) {
          if (!UpdateFeatureMemory(item.first, runtime_info_.feature_addr, item.second)) {
            MS_LOG(ERROR) << "Failed to UpdateFeatureMemory for graph " << item.first << ", org feature size "
                          << item.second << ", new malloc feature memory size " << runtime_info_.feature_size;
            return false;
          }
          MS_LOG(INFO) << "UpdateFeatureMemory for graph " << item.first << ", org feature size " << item.second
                       << ", new malloc feature memory size " << runtime_info_.feature_size;
        }
        session_context->feature_graph_ids[graph_id] = runtime_info_.feature_size;
      }
    } else {
      runtime_info_.feature_addr = session_context->feature_memory;
    }
    if (!UpdateFeatureMemory(graph_id, runtime_info_.feature_addr, runtime_info_.feature_size)) {
      MS_LOG(ERROR) << "Failed to UpdateFeatureMemory for graph " << graph_id << ", feature size "
                    << runtime_info_.feature_size;
      return false;
    }
    MS_LOG(INFO) << "UpdateFeatureMemory for graph " << graph_id << ", feature size " << runtime_info_.feature_size;
  }
  return true;
}

bool GeGraphExecutor::InitInOutDeviceBuffer(const std::string &name, const ShapeVector &shape, TypeId dtype,
                                            InOutBufferInfo *buffer_info) {
  auto &info = *buffer_info;
  auto desc = transform::TransformUtil::GetGeTensorDesc(shape, dtype, kOpFormat_NCHW);
  if (desc == nullptr) {
    MS_LOG(ERROR) << "Failed to get Tensor Desc";
    return false;
  }
  auto tensor_size = SizeOf(shape) * GetDataTypeSize(dtype);
  if (tensor_size <= 0) {
    MS_LOG(INFO) << "Failed to calculate " << name << " tensor size, shape " << ShapeVectorToStr(shape)
                 << ", date type " << dtype;
    return false;
  }
  desc->SetPlacement(::ge::kPlacementDevice);
  auto ret = info.ge_tensor.SetTensorDesc(*desc);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Failed to call ge::Tensor::SetTensorDesc, ret " << ret;
    return false;
  }
  info.device_addr = memory_manager_->MallocDeviceMemory(name, tensor_size);
  if (info.device_addr == nullptr) {
    MS_LOG(ERROR) << "Failed to malloc device memory for " << name << ", memory size " << tensor_size
                  << ", tensor shape " << shape;
    return false;
  }
  ret = info.ge_tensor.SetData(reinterpret_cast<uint8_t *>(info.device_addr), tensor_size, [](uint8_t *) -> void {});
  if (ret != ge::GRAPH_SUCCESS) {
    MS_LOG(ERROR) << "Failed to call ge::Tensor SetData(uint8_t*, size, DeleteFunc), data size " << tensor_size;
    return false;
  }
  info.max_size = tensor_size;
  info.shape = shape;
  info.dtype = dtype;
  return true;
}

bool GeGraphExecutor::UpdateInputShapeOption(
  const std::vector<std::pair<std::string, tensor::TensorPtr>> &ref_data_tensors,
  std::map<std::string, std::string> *ge_options_ptr) {
  if (ge_options_ptr == nullptr) {
    MS_LOG(ERROR) << "Input argument ge_options_ptr cannot be nullptr";
    return false;
  }
  std::string new_input_shape_str;
  std::string input_shape_str;
  std::vector<GeDynamicShapeInfo> input_shapes;
  if (!GeDynamicUtils::GetGraphInputShapes(context_, config_infos_, &input_shapes, &input_shape_str)) {
    MS_LOG(ERROR) << "Failed to get input shape from AscendDeviceInfo or config file";
    return false;
  }
  if (input_shapes.empty()) {
    MS_LOG(INFO) << "Not found input shape in AscendDeviceInfo or config file";
    return true;
  }
  auto shape_str = [](const ShapeVector &shape) -> std::string {
    std::string str;
    if (shape.empty()) {
      return str;
    }
    for (auto &dim : shape) {
      str += std::to_string(dim) + ",";
    }
    return str.substr(0, str.size() - 1);
  };
  std::map<std::string, std::string> shape_map;
  for (auto &item : input_shapes) {
    shape_map[item.name] = item.shape_str;
  }
  for (auto &item : ref_data_tensors) {
    ShapeVector ref_dyn_shape = item.second->shape_c();
    SetRefShape(&ref_dyn_shape, true, item.first);
    shape_map[item.first] = shape_str(ref_dyn_shape);
  }
  size_t index = 0;
  for (auto &item : shape_map) {
    new_input_shape_str += item.first + ":" + item.second;
    if (index + 1 < shape_map.size()) {
      new_input_shape_str += ";";
    }
    index++;
  }
  GeDynamicUtils::UpdateGraphInputShapes(context_, &config_infos_, new_input_shape_str);
  (*ge_options_ptr)["ge.inputShape"] = new_input_shape_str;
  MS_LOG(INFO) << "Update ge.inputShape to " << new_input_shape_str;
  return true;
}

bool GeGraphExecutor::InitRefDataContext(const std::vector<std::pair<std::string, tensor::TensorPtr>> &ref_data_tensors,
                                         std::map<std::string, std::string> *ge_options_ptr) {
  if (!UpdateInputShapeOption(ref_data_tensors, ge_options_ptr)) {
    MS_LOG(ERROR) << "Failed to update input shape option";
    return false;
  }
  if (!UpdateDynamicDimsOption(ref_data_tensors, ge_options_ptr)) {
    MS_LOG(ERROR) << "Failed to update dynamic dims option";
    return false;
  }
  if (!InitRefDataList(ref_data_tensors)) {
    MS_LOG(ERROR) << "Failed to init ref data list";
    return false;
  }
  return true;
}

bool GeGraphExecutor::UpdateWeights(const std::vector<std::vector<std::shared_ptr<tensor::Tensor>>> &weights) {
  auto time1 = lite::GetTimeUs();
  uint32_t init_graph_id = init_graph_id_list_[0];
  MS_LOG(INFO) << "init_graph_id: " << init_graph_id;
  if (update_weight_ptr_ == nullptr) {
    MS_LOG(ERROR) << "please init update weight class by build model.";
    return false;
  }
  std::vector<std::vector<std::shared_ptr<tensor::Tensor>>> new_weight_tensors;
  auto ret = update_weight_ptr_->UpdateConstantTensorData(weights, &new_weight_tensors);
  if (!ret) {
    MS_LOG(ERROR) << "update weight failed.";
    return false;
  }
  MS_LOG(DEBUG) << "ExecInitGraph start.";
  auto time2 = lite::GetTimeUs();
  MS_LOG(INFO) << "update weight prepare time: " << (time2 - time1) / kNumMicrosecondToMillisecond << " ms";

  // cppcheck-suppress cppcheckError
  for (size_t i = 0; i < new_weight_tensors.size(); i++) {
    std::vector<::ge::Tensor> ge_inputs;
    // cppcheck-suppress cppcheckError
    for (size_t j = 0; j < new_weight_tensors[i].size(); j++) {
      auto &input = new_weight_tensors[i][j];
      auto ge_tensor = transform::TransformUtil::ConvertTensor(input, kOpFormat_NCHW, false);
      if (ge_tensor == nullptr) {
        MS_LOG(ERROR) << "Failed to converter input " << i << " ME Tensor to GE Tensor";
        return false;
      }
      ge_inputs.emplace_back(*ge_tensor);
    }
    std::vector<::ge::Tensor> ge_outputs;
    auto ge_status = ge_session_->RunGraph(init_graph_id, ge_inputs, ge_outputs);
    if (ge_status != ge::GRAPH_SUCCESS) {
      MS_LOG(ERROR) << "Exec init graph failed, graph id " << init_graph_id;
      return false;
    }
  }
  auto time3 = lite::GetTimeUs();
  MS_LOG(INFO) << "update weight run init graph time: " << (time3 - time2) / kNumMicrosecondToMillisecond << " ms";
  return true;
}

transform::DfGraphPtr GeGraphExecutor::CreateGeGraphOnline(const FuncGraphPtr &anf_graph,
                                                           std::map<std::string, std::string> *ge_options_ptr) {
  std::vector<std::string> extra_variables_names = {};
  if (enable_update_weight_ && update_weight_ptr_ != nullptr) {
    auto ret = update_weight_ptr_->CreateAddOpNodeForGraph(anf_graph);
    if (!ret) {
      MS_LOG(ERROR) << "CreateAddOpNodeForGraph failed.";
      return nullptr;
    }
    extra_variables_names = update_weight_ptr_->GetVariableParamsName(anf_graph);
    if (extra_variables_names.empty()) {
      MS_LOG(WARNING) << "GetVariableParamsName failed.";
      return nullptr;
    }
  }
  transform::TensorOrderMap params_vals;
  GetParams(anf_graph, &params_vals);

  MS_LOG(INFO) << "extra_variables_names size: " << extra_variables_names.size();
  auto converter = std::make_shared<transform::DfGraphConvertor>(anf_graph, "", ref_mode_flag_, extra_variables_names);
  transform::BuildGraph(graph_name_, converter, params_vals);
  auto err_code = transform::ErrCode(converter);
  if (err_code != 0) {
    transform::ClearGraph();
    MS_LOG(ERROR) << "Convert df graph failed, err:" << err_code;
    return nullptr;
  }
  auto init_graph = transform::GetInitGraph(converter);
  if (init_graph != nullptr) {
    uint32_t init_graph_id = 0;
    if (!AddGraph(init_graph, {}, &init_graph_id)) {
      MS_LOG(ERROR) << "Failed to add init graph, graph name " << anf_graph->ToString();
      return nullptr;
    }
    if (enable_update_weight_ && update_weight_ptr_ != nullptr) {
      init_graph_id_list_.push_back(init_graph_id);
    }
    auto init_data_names = converter->GetInitDataNames();
    if (enable_update_weight_ && update_weight_ptr_ != nullptr) {
      if (!update_weight_ptr_->SetInitDataNames(init_data_names)) {
        MS_LOG(ERROR) << "set init data name failed.";
        return nullptr;
      }
    }
    // copy init weight to device
    if (!RunGeInitGraph(init_graph_id, init_data_names, params_vals)) {
      MS_LOG(ERROR) << "Failed to run init graph for " << anf_graph->ToString();
      return nullptr;
    }
    if (!enable_update_weight_) {
      ge_session_->RemoveGraph(init_graph_id);
    }
  } else {
    MS_LOG(INFO) << "There is no init graph for graph " << anf_graph->ToString();
  }
  if (ref_mode_flag_ != transform::RefModeFlag::kRefModeNone) {
    auto ref_data_names = converter->GetRefDataNames();
    std::vector<std::pair<std::string, tensor::TensorPtr>> ref_datas;
    std::transform(ref_data_names.begin(), ref_data_names.end(), std::back_inserter(ref_datas),
                   [&params_vals](auto &item) { return std::make_pair(item, params_vals.at(item)); });
    if (!InitRefDataContext(ref_datas, ge_options_ptr)) {
      MS_LOG(ERROR) << "Failed to init refdata context";
      return nullptr;
    }
  }
  auto df_graph = transform::GetComputeGraph(converter);
  return df_graph;
}

transform::DfGraphPtr GenExampleGraph(const std::string &name) {
  MS_LOG(INFO) << "Gen fake graph name is " << name;
  auto graph = std::make_shared<transform::DfGraph>(name);
  auto shape_data = std::vector<int64_t>({1, 1, 1, 1});
  transform::GeTensorDesc desc_data(ge::Shape(shape_data), ge::FORMAT_ND, ge::DT_FLOAT16);
  auto data = ge::op::Data("data");
  data.set_attr_index(0);
  data.update_input_desc_x(desc_data);
  data.update_output_desc_y(desc_data);
  auto add = ge::op::Add("add").set_input_x1(data).set_input_x2(data);
  std::vector<transform::Operator> inputs{data};
  std::vector<transform::Operator> outputs{add};
  graph->SetInputs(inputs);
  graph->SetOutputs(outputs);
  return graph;
}

bool GeGraphExecutor::CreateGeGraphOffline(const FuncGraphPtr &anf_graph,
                                           std::map<std::string, std::string> *ge_options_ptr, uint32_t *graph_id) {
  ref_mode_flag_ = transform::RefModeFlag::kRefModeVariable;
  if (!CreateSession()) {
    MS_LOG(ERROR) << "Failed to create ge session";
    return false;
  }
  tensor::TensorPtr model_cache = nullptr;
  std::string graph_name;
  std::map<std::string, ValuePtr> attr_map;
  std::vector<std::pair<std::string, tensor::TensorPtr>> ref_datas;
  if (!CustomAscendUtils::ParseCustomFuncGraph(anf_graph, &model_cache, &graph_name, &attr_map, &ref_datas)) {
    MS_LOG(ERROR) << "Failed to parse custom func graph";
    return false;
  }
  auto option_it = ge_options_ptr->find(kGeGraphCompilerCacheDir);
  if (option_it == ge_options_ptr->end()) {
    if (auto it = attr_map.find(lite::kNameAttrWeightDir); it != attr_map.end()) {
      auto weight_dir = GetValue<std::string>(it->second);
      std::string mindir_path;
      (void)GetConfigOption(lite::kConfigModelFileSection, lite::kConfigMindIRPathKey, &mindir_path);
      // relative weight path
      if (!mindir_path.empty() && !weight_dir.empty() && weight_dir[0] != '/') {
        if (mindir_path.find("/") != std::string::npos) {
          weight_dir = mindir_path.substr(0, mindir_path.rfind("/") + 1) + weight_dir;
        }
      }
      MS_LOG(INFO) << "Update ge options " << kGeGraphCompilerCacheDir << " to " << weight_dir;
      (*ge_options_ptr)[kGeGraphCompilerCacheDir] = weight_dir;
    }
  }
  if (graph_name.empty()) {
    MS_LOG(ERROR) << "Graph compiler cache dir or graph name is empty";
    return false;
  }
  graph_name_ = graph_name;
  MS_LOG(INFO) << "Update ge options " << kGeGraphKey << " to " << graph_name;
  (*ge_options_ptr)[kGeGraphKey] = graph_name;
  if (!ref_datas.empty()) {
    if (!InitRefDataContext(ref_datas, ge_options_ptr)) {
      MS_LOG(ERROR) << "Failed to init refdata context";
      return false;
    }
  }
  auto df_graph = GenExampleGraph(graph_name);
  if (!AddGraph(df_graph, *ge_options_ptr, graph_id)) {
    MS_LOG(ERROR) << "Failed to add compute graph, graph name " << anf_graph->ToString();
    return false;
  }
  return true;
}

transform::DfGraphPtr GeGraphExecutor::CompileGraphCommon(const FuncGraphPtr &anf_graph,
                                                          std::map<std::string, std::string> *ge_options_ptr) {
  if (!CreateSession()) {
    MS_LOG(ERROR) << "Failed to create ge session";
    return nullptr;
  }
  if (anf_graph == nullptr) {
    MS_LOG(ERROR) << "Input param graph is nullptr.";
    return nullptr;
  }
  if (ge_options_ptr == nullptr) {
    MS_LOG(ERROR) << "Input param ge_options_ptr is nullptr.";
    return nullptr;
  }

#ifdef MSLITE_ENABLE_GRAPH_KERNEL
  auto param = ParseGraphKernelConfigs(config_infos_);
  if (param != nullptr) {
    auto rank_id = common::GetEnv("RANK_ID");
    if (rank_id.empty()) {
      auto ascend_device_info = GeUtils::GetAscendDeviceInfo(context_);
      if (ascend_device_info != nullptr) {
        auto rank_id_value = ascend_device_info->GetRankID();
        common::SetEnv("RANK_ID", std::to_string(rank_id_value).c_str());
      }
    }
    if (GraphKernelOptimize(anf_graph, param) != lite::RET_OK) {
      MS_LOG(ERROR) << "Run graphkernel optimization failed.";
      return nullptr;
    }
  }
#endif

  // auto remove_load_pass = std::make_shared<opt::RemoveLoadPass>();
  // remove_load_pass->Run(anf_graph);

  if (!UpdateGraphInputs(anf_graph)) {
    MS_LOG(ERROR) << "Failed to update graph inputs";
    return nullptr;
  }

  opt::UpdateManager(anf_graph);

  transform::DfGraphPtr df_graph = nullptr;
  auto func_type = anf_graph->get_attr(kAttrFuncType);
  is_data_flow_graph_ = func_type != nullptr && GetValue<std::string>(func_type) == kDataFlowGraphType;
  if (!is_data_flow_graph_) {
    df_graph = CreateGeGraphOnline(anf_graph, ge_options_ptr);
  } else {
    df_graph = GetDataFlowGraph(anf_graph, *ge_options_ptr);
  }
  return df_graph;
}

bool GeGraphExecutor::CompileGraph(const FuncGraphPtr &anf_graph, const std::map<string, string> &,
                                   uint32_t *graph_id) {
  MS_CHECK_TRUE_RET(graph_id != nullptr, false);
  std::map<std::string, std::string> ge_options;
  SetDynamicKVCache();
  GetGeGraphOptions(anf_graph, &ge_options);

  uint32_t compute_graph_id = 0;
  if (CustomAscendUtils::IsCustomFuncGraph(anf_graph)) {
    MS_LOG(ERROR) << "Offline converted MindIR is not supported currently";
    return false;
  } else {
    auto df_graph = CompileGraphCommon(anf_graph, &ge_options);
    if (df_graph == nullptr) {
      MS_LOG(ERROR) << "Input param graph is nullptr.";
      return false;
    }
    if (!AddGraph(df_graph, ge_options, &compute_graph_id)) {
      MS_LOG(ERROR) << "Failed to add compute graph, graph name " << anf_graph->ToString();
      return false;
    }
  }
  compute_graph_id_list_.push_back(compute_graph_id);
  *graph_id = compute_graph_id;
  if (ref_mode_flag_ != transform::RefModeFlag::kRefModeNone) {
    if (!BuildGraphRefMode(anf_graph, compute_graph_id)) {
      MS_LOG(ERROR) << "Failed to build ge graph with refdata";
      return false;
    }
  }
  std::vector<tensor::TensorPtr> orig_output;
  std::vector<std::string> output_names;
  FuncGraphUtils::GetFuncGraphOutputsInfo(anf_graph, &orig_output, &output_names);
  original_graph_outputs_[*graph_id] = orig_output;
  return true;
}

bool GeGraphExecutor::Warmup(const FuncGraphPtr &func_graph, uint32_t graph_id) {
  std::vector<::ge::Tensor> ge_inputs;
  if (!GetOneRealInputs(func_graph, &ge_inputs)) {
    MS_LOG(ERROR) << "Failed to get one real inputs, graph id " << graph_id;
    return false;
  }
  std::vector<::ge::Tensor> ge_outputs;
  auto ret = RunGeGraphAsync(graph_id, ge_inputs, &ge_outputs);
  if (!ret) {
    MS_LOG(ERROR) << "Exec compute graph failed, graph id " << graph_id;
    return false;
  }
  return true;
}

bool GeGraphExecutor::GetOneRealInputs(const FuncGraphPtr &anf_graph, std::vector<ge::Tensor> *ge_tensors_ptr) {
  std::vector<std::pair<std::string, ShapeVector>> input_shapes_configs;
  std::string input_shape_str;
  if (!GeDynamicUtils::GetGraphOneRealShapes(context_, config_infos_, &input_shapes_configs, &input_shape_str)) {
    MS_LOG(ERROR) << "Failed to get one real input shape";
    return false;
  }
  std::vector<tensor::TensorPtr> inputs;
  std::vector<std::string> input_names;
  FuncGraphUtils::GetFuncGraphInputsInfo(anf_graph, &inputs, &input_names);
  if (!input_shapes_configs.empty() && input_shapes_configs.size() != inputs.size()) {
    MS_LOG(ERROR) << "Input count " << input_shapes_configs.size()
                  << " get from input_shape of AscendDeviceInfo or config file != input count " << inputs.size()
                  << " got from graph";
    return false;
  }
  std::vector<::ge::Tensor> ge_inputs;
  // cppcheck-suppress cppcheckError
  for (size_t i = 0; i < inputs.size(); i++) {
    auto &input = inputs[i];
    auto input_name = input_names[i];
    if (!input_shapes_configs.empty()) {
      auto it = std::find_if(input_shapes_configs.begin(), input_shapes_configs.end(),
                             [&input_name](const auto &item) { return input_name == item.first; });
      if (it == input_shapes_configs.end()) {
        MS_LOG(ERROR) << "Cannot find input " << input_name << " in input_shape " << input_shape_str;
        return false;
      }
      input = std::make_shared<tensor::Tensor>(input->data_type(), it->second);
    } else if (GeDynamicUtils::IsDynamicInputShapes({input->shape_c()})) {
      MS_LOG(ERROR) << "Input " << i << " is dynamic shape " << input->shape_c()
                    << ", but there is no input shape specified in AscendDeviceInfo or config file";
      return false;
    }
    MS_LOG(INFO) << "Input " << i << " shape " << input->shape_c() << ", datatype " << input->data_type();
    auto ge_tensor = transform::TransformUtil::ConvertTensor(input, kOpFormat_NCHW);
    if (ge_tensor == nullptr) {
      MS_LOG(ERROR) << "Failed to converter input " << i << " ME Tensor to GE Tensor";
      return false;
    }
    ge_inputs.emplace_back(*ge_tensor);
  }
  *ge_tensors_ptr = ge_inputs;
  return true;
}

bool GeGraphExecutor::AoeTuning(const FuncGraphPtr &anf_graph) {
  std::map<std::string, std::string> ge_options;
  GetGeGraphOptions(anf_graph, &ge_options);
  auto df_graph = CompileGraphCommon(anf_graph, &ge_options);
  if (df_graph == nullptr) {
    MS_LOG(ERROR) << "Input param graph is nullptr.";
    return false;
  }
  std::vector<::ge::Tensor> ge_inputs;
  if (!GetOneRealInputs(anf_graph, &ge_inputs)) {
    MS_LOG(ERROR) << "Failed to get one real inputs";
    return false;
  }
  AoeApiTuning tuning;
  auto status = tuning.AoeTurningGraph(ge_session_, df_graph, ge_inputs, context_, config_infos_);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "Failed to call AoeTurningGraph";
    return false;
  }
  return true;
}

bool GeGraphExecutor::ParseGeCompiledGraphSummary(const ::ge::CompiledGraphSummaryPtr &ge_graph_summary) {
  if (ge_graph_summary == nullptr) {
    MS_LOG(ERROR) << "CompiledGraphSummaryPtr cannot be nullptr";
    return false;
  }
  auto ge_status = ge_graph_summary->GetConstMemorySize(runtime_info_.const_size);
  if (ge_status != ge::GRAPH_SUCCESS) {
    MS_LOG(ERROR) << "Failed to call GetConstMemorySize, status: " << ge_status;
    return false;
  }
  ge_status = ge_graph_summary->GetFeatureMemorySize(runtime_info_.feature_size);
  if (ge_status != ge::GRAPH_SUCCESS) {
    MS_LOG(ERROR) << "Failed to call GetFeatureMemorySize, status: " << ge_status;
    return false;
  }
  std::vector<::ge::Shape> ge_shapes;
  ge_status = ge_graph_summary->GetOutputShapes(ge_shapes);
  if (ge_status != ge::GRAPH_SUCCESS) {
    MS_LOG(ERROR) << "Failed to call GetOutputShapes, status: " << ge_status;
    return false;
  }
  (void)std::transform(ge_shapes.begin(), ge_shapes.end(), std::back_inserter(runtime_info_.output_shapes),
                       [](const ::ge::Shape &ge_shape) -> ShapeVector { return ge_shape.GetDims(); });
  return true;
}

bool GeGraphExecutor::RunGeInitGraph(uint32_t init_graph_id, const std::vector<std::string> &init_data_names,
                                     const transform::TensorOrderMap &params_vals) {
  std::vector<tensor::TensorPtr> init_data_tensors;
  for (auto &item : init_data_names) {
    auto it = params_vals.find(item);
    if (it == params_vals.end()) {
      MS_LOG(ERROR) << "Cannot find parameter " << item << " in parameter map";
      return false;
    }
    init_data_tensors.push_back(it->second);
  }
  MS_LOG(DEBUG) << "ExecInitGraph start.";
  std::vector<::ge::Tensor> ge_inputs;
  for (size_t i = 0; i < init_data_tensors.size(); i++) {
    auto &input = init_data_tensors[i];
    auto ge_tensor = transform::TransformUtil::ConvertTensor(input, kOpFormat_NCHW, false);
    if (ge_tensor == nullptr) {
      MS_LOG(ERROR) << "Failed to converter input " << i << " ME Tensor to GE Tensor";
      return false;
    }
    ge_inputs.emplace_back(*ge_tensor);
  }
  std::vector<::ge::Tensor> ge_outputs;
  auto ge_status = ge_session_->RunGraph(init_graph_id, ge_inputs, ge_outputs);
  if (ge_status != ge::GRAPH_SUCCESS) {
    MS_LOG(ERROR) << "Exec init graph failed, graph id " << init_graph_id;
    return false;
  }
  MS_LOG(INFO) << "Exec init graph success, graph id " << init_graph_id;
  return true;
}

bool GeGraphExecutor::RunGeGraphAsync(uint32_t graph_id, const std::vector<::ge::Tensor> &inputs,
                                      std::vector<::ge::Tensor> *outputs) {
  bool is_finished = false;
  bool end_of_sequence = false;
  std::promise<void> promise;
  auto call_back = [outputs, &is_finished, &end_of_sequence, &promise](ge::Status ge_status,
                                                                       const std::vector<ge::Tensor> &ge_outputs) {
    if (ge_status == ge::GRAPH_SUCCESS) {
      *outputs = ge_outputs;
      is_finished = true;
    } else if (ge_status == ge::END_OF_SEQUENCE) {
      MS_LOG(ERROR) << "RunAsync out of range: End of sequence.";
      end_of_sequence = true;
    } else {
      MS_LOG(ERROR) << "RunAsync failed.";
    }
    promise.set_value();
    return;
  };
  if (ge_session_ == nullptr) {
    MS_LOG(ERROR) << "The GE session is null, can't run the graph!";
    return false;
  }
  ge::Status ret = ge_session_->RunGraphAsync(graph_id, inputs, call_back);
  if (ret != ge::GRAPH_SUCCESS) {
    MS_LOG(ERROR) << "Call GE RunGraphAsync Failed: " << ge::GEGetErrorMsg();
    return false;
  }
  auto future = promise.get_future();
  future.wait();
  if (end_of_sequence) {
    MS_LOG(ERROR) << "Failed to call GE RunGraphAsync: End of sequence";
    return false;
  }
  return is_finished;
}

bool GeGraphExecutor::RunDataFlowGraphAsync(uint32_t graph_id, const std::vector<::ge::Tensor> &inputs,
                                            std::vector<::ge::Tensor> *outputs) {
  ge::DataFlowInfo data_flow_info;
  int time_out = 3000;  // set the timeout to 3000s.
  auto ret = ge_session_->FeedDataFlowGraph(graph_id, inputs, data_flow_info, time_out);
  if (ret != ge::SUCCESS) {
    MS_LOG(ERROR) << "Feed input data failed.";
    return false;
  }
  ret = ge_session_->FetchDataFlowGraph(graph_id, *outputs, data_flow_info, time_out);
  if (ret != ge::SUCCESS) {
    MS_LOG(ERROR) << "Fetch output data failed.";
    return false;
  }
  return true;
}

bool GeGraphExecutor::InitInputDataTensor(const std::vector<tensor::Tensor> &inputs,
                                          std::vector<::ge::Tensor> *ge_inputs, std::vector<::ge::Tensor> *ge_outputs) {
  if (ge_inputs == nullptr || ge_outputs == nullptr) {
    MS_LOG(ERROR) << "Input argument ge_inputs or ge_outputs is nullptr";
    return false;
  }
  if (inputs_buffer_infos_.size() != inputs.size()) {
    MS_LOG(ERROR) << "Input data info size " << inputs_buffer_infos_.size() << " != inputs size " << inputs.size();
    return false;
  }
  if (memory_manager_ == nullptr) {
    MS_LOG(ERROR) << "Memory manager or context manager is nullptr";
    return false;
  }
  for (size_t i = 0; i < inputs.size(); i++) {
    auto &input = inputs[i];
    MS_LOG(INFO) << "Input " << i << " shape " << tensor::ShapeToString(input.shape_c()) << ", datatype "
                 << input.data_type();
    auto tensor_size = input.Size();
    auto &input_info = inputs_buffer_infos_[i];
    if (input_info.max_size < tensor_size) {
      MS_LOG(ERROR) << "Input " << i << " data size invalid, graph size " << input_info.max_size << ", given size "
                    << tensor_size;
      return false;
    }
    if (!memory_manager_->MemcpyHost2Device(input_info.device_addr, input_info.max_size, input.data_c(), tensor_size)) {
      return false;
    }
    SetGeTensorShape(&input_info.ge_tensor, input.shape_c());
    ge_inputs->push_back(input_info.ge_tensor);
  }
  for (auto &item : ref_data_infos_) {
    ShapeVector ref_real_shape = transform::TransformUtil::ConvertGeShape(item.ge_tensor.GetTensorDesc().GetShape());
    SetRefShape(&ref_real_shape, false, item.name);
    SetGeTensorShape(&item.ge_tensor, ref_real_shape);
    ge_inputs->push_back(item.ge_tensor);
  }
  for (auto &output : outputs_buffer_infos_) {
    ge_outputs->push_back(output.ge_tensor);
  }
  return true;
}

bool GeGraphExecutor::BuildGraphRefMode(const FuncGraphPtr &anf_graph, uint32_t graph_id) {
  MS_LOG(INFO) << "Call GE CompileGraph start, graph id " << graph_id;
  ge::Status ret = ge_session_->CompileGraph(graph_id);
  if (ret != ge::GRAPH_SUCCESS) {
    MS_LOG(ERROR) << "Call GE CompileGraph Failed: " << ge::GEGetErrorMsg();
    return false;
  }
  MS_LOG(INFO) << "Call GE CompileGraph end, graph id " << graph_id;
  if (offline_mode_) {
    MS_LOG(INFO) << "Converter offline mode, skip device resource allocate";
    return true;
  }
  if (!InitMemoryContextManager()) {
    return false;
  }
  if (!InitConstantFeatureDeviceMemory(graph_id)) {
    MS_LOG(ERROR) << "Failed to init constant and feature device memory";
    return false;
  }
  if (!InitRefDataDeviceTensor()) {
    MS_LOG(ERROR) << "Failed to init ref data device data";
    return false;
  }
  // ref data input memories have been allocated
  // for input data memory
  if (!InitInputDeviceTensor(anf_graph)) {
    MS_LOG(ERROR) << "Failed to init input data device data";
    return false;
  }

  // for output memory
  std::vector<AnfWithOutIndex> outputs;
  if (!FuncGraphUtils::GetFuncGraphOutputs(anf_graph, &outputs)) {
    MS_LOG(ERROR) << "Failed to get func graph outputs";
    return false;
  }
  if (outputs.size() != runtime_info_.output_shapes.size()) {
    MS_LOG(ERROR) << "Output count got from graph " << outputs.size() << " != that "
                  << runtime_info_.output_shapes.size() << " got from GE";
    return false;
  }
  outputs_buffer_infos_.resize(outputs.size());
  for (size_t i = 0; i < outputs.size(); i++) {
    auto &output_info = outputs_buffer_infos_[i];
    auto shape = runtime_info_.output_shapes[i];
    auto dtype = static_cast<TypeId>(FuncGraphUtils::GetTensorDataType(outputs[i]));
    if (!InitInOutDeviceBuffer("Output " + std::to_string(i), shape, dtype, &output_info)) {
      return false;
    }
  }
  return true;
}

bool GeGraphExecutor::RunGraphRefMode(uint32_t graph_id, const std::vector<tensor::Tensor> &inputs,
                                      std::vector<tensor::Tensor> *outputs) {
  std::vector<::ge::Tensor> ge_inputs;
  std::vector<::ge::Tensor> ge_outputs;
  InitRealShapeParam(inputs);

  if (!InitInputDataTensor(inputs, &ge_inputs, &ge_outputs)) {
    MS_LOG(ERROR) << "Init input tensor failed in run graph.";
    return false;
  }
  auto stream = context_manager_->GetDefaultStream();
  if (!RunGraphWithStreamAsync(graph_id, stream, ge_inputs, &ge_outputs)) {
    MS_LOG(ERROR) << "Failed in run graph with stream async.";
    return false;
  }
  if (!SyncDeviceOutputsToHost(outputs)) {
    MS_LOG(ERROR) << "Failed in sync device output to host.";
    return false;
  }
  return true;
}

bool GeGraphExecutor::SyncDeviceOutputsToHost(std::vector<tensor::Tensor> *outputs) {
  UpdateOutputShapeInfo();

  if (!outputs->empty()) {
    if (outputs->size() != outputs_buffer_infos_.size()) {
      MS_LOG(ERROR) << "Invalid output size, outputs' size " << outputs->size() << "ge tensor size "
                    << outputs_buffer_infos_.size();
      return false;
    }
    for (size_t i = 0; i < outputs->size(); ++i) {
      auto &output_info = outputs_buffer_infos_[i];
      auto &output = (*outputs)[i];
      if (output.Size() < output_info.max_size) {
        MS_LOG(EXCEPTION) << "Output node " << i << "'s mem size " << output.Size()
                          << " is less than actual output size " << output_info.max_size;
      }
      if ((*outputs)[i].data_c() == nullptr) {
        MS_LOG(ERROR) << "Output data ptr is nullptr.";
        return false;
      }
      auto mem_ret = memory_manager_->MemcpyDevice2Host(reinterpret_cast<uint8_t *>(output.data_c()), output.Size(),
                                                        output_info.device_addr, output_info.max_size);
      if (!mem_ret) {
        MS_LOG(ERROR) << "Failed to copy output data, dst size: " << output.Size()
                      << ", src size: " << output_info.max_size;
        return false;
      }
    }
  } else {
    for (size_t i = 0; i < outputs_buffer_infos_.size(); i++) {
      auto &output_info = outputs_buffer_infos_[i];
      tensor::Tensor ms_tensor(output_info.dtype, output_info.shape);
      auto mem_ret =
        memory_manager_->MemcpyDevice2Host(reinterpret_cast<uint8_t *>(ms_tensor.data_c()), ms_tensor.Size(),
                                           output_info.device_addr, output_info.max_size);
      if (!mem_ret) {
        MS_LOG(ERROR) << "Failed to copy output data, dst size: " << ms_tensor.Size()
                      << ", src size: " << output_info.max_size;
        return false;
      }
      MS_LOG(INFO) << "Output " << i << " shape " << tensor::ShapeToString(ms_tensor.shape_c()) << ", datatype "
                   << ms_tensor.data_type();
      outputs->push_back(ms_tensor);
    }
  }
  return true;
}

bool GeGraphExecutor::RunGraphWithStreamAsync(uint32_t graph_id, void *stream, const std::vector<GeTensor> &inputs,
                                              std::vector<GeTensor> *outputs) {
  MS_EXCEPTION_IF_NULL(outputs);

  MS_LOG(INFO) << "Run the graph in GE with " << inputs.size() << " inputs";
  struct timeval start_time, end_time;
  (void)gettimeofday(&start_time, nullptr);

  ge::Status ret = ge_session_->RunGraphWithStreamAsync(graph_id, stream, inputs, *outputs);
  if (ret != ge::GRAPH_SUCCESS) {
    MS_LOG(ERROR) << "Call GE RunGraphWithStreamAsync Failed, ret is: " << ret;
    return false;
  }
  if (!context_manager_->SyncStream(stream)) {
    MS_LOG(ERROR) << "Sync stream for RunGraphWithStreamAsync failed";
    return false;
  }

  (void)gettimeofday(&end_time, nullptr);
  const uint64_t kUSecondInSecond = 1000000;
  uint64_t cost = kUSecondInSecond * static_cast<uint64_t>(end_time.tv_sec - start_time.tv_sec);
  cost += static_cast<uint64_t>(end_time.tv_usec - start_time.tv_usec);
  MS_LOG(INFO) << "Call GE RunGraphWithStreamAsync Success in " << cost << " us, GE outputs num: " << outputs->size()
               << ", graph id: " << graph_id;

  return true;
}

bool GeGraphExecutor::SetConstMemory(uint32_t graph_id, const void *const memory, size_t size) {
  if (ge_session_ == nullptr) {
    MS_LOG(ERROR) << "The GE session is null, can't run the graph!";
    return false;
  }
  ge::Status ge_ret = ge_session_->SetGraphConstMemoryBase(graph_id, memory, size);
  if (ge_ret != ge::GRAPH_SUCCESS) {
    MS_LOG(ERROR) << "Call GE SetGraphConstMemoryBase Failed, ret is: " << ge_ret;
    return false;
  }
  return true;
}

bool GeGraphExecutor::UpdateFeatureMemory(uint32_t graph_id, const void *const memory, size_t size) {
  if (ge_session_ == nullptr) {
    MS_LOG(ERROR) << "The GE session is null, can't run the graph!";
    return false;
  }
  ge::Status ge_ret = ge_session_->UpdateGraphFeatureMemoryBase(graph_id, memory, size);
  if (ge_ret != ge::GRAPH_SUCCESS) {
    MS_LOG(ERROR) << "Call GE SetGraphConstMemoryBase Failed, ret is: " << ge_ret;
    return false;
  }
  return true;
}

bool GeGraphExecutor::RunGraph(uint32_t graph_id, const std::vector<tensor::Tensor> &inputs,
                               std::vector<tensor::Tensor> *outputs,
                               const std::map<string, string> & /* compile_options */) {
  if (outputs == nullptr) {
    MS_LOG(ERROR) << " Input param is nullptr.";
    return false;
  }
  if (ref_mode_flag_ != transform::RefModeFlag::kRefModeNone) {
    return RunGraphRefMode(graph_id, inputs, outputs);
  }
  MS_LOG(INFO) << "GE run graph " << graph_id << " start.";
  std::vector<::ge::Tensor> ge_inputs;
  for (size_t i = 0; i < inputs.size(); i++) {
    auto &input = inputs[i];
    MS_LOG(INFO) << "Input " << i << " shape " << input.shape_c() << ", datatype " << input.data_type();
    auto ge_tensor =
      transform::TransformUtil::ConvertTensor(std::make_shared<tensor::Tensor>(input), kOpFormat_NCHW, false);
    if (ge_tensor == nullptr) {
      MS_LOG(ERROR) << "Failed to converter input " << i << " ME Tensor to GE Tensor";
      return false;
    }
    ge_inputs.emplace_back(*ge_tensor);
  }
  for (auto &item : ref_data_infos_) {
    ge_inputs.emplace_back(item.ge_tensor);
  }
  std::vector<::ge::Tensor> ge_outputs;
  auto time_start = std::chrono::system_clock::now();
  auto ret = !is_data_flow_graph_ ? RunGeGraphAsync(graph_id, ge_inputs, &ge_outputs)
                                  : RunDataFlowGraphAsync(graph_id, ge_inputs, &ge_outputs);
  if (!ret) {
    MS_LOG(ERROR) << "Exec compute graph failed, graph id " << graph_id;
    return false;
  }
  auto time_cost =
    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - time_start).count();
  MS_LOG(INFO) << "Call GE RunGraph Success in " << time_cost << " us, graph id " << graph_id
               << " the GE outputs num is: " << ge_outputs.size();

  if (!outputs->empty()) {
    if (outputs->size() != ge_outputs.size()) {
      MS_LOG(ERROR) << "Invalid output size, outputs' size " << outputs->size() << "ge tensor size "
                    << ge_outputs.size();
      return false;
    }
    for (size_t i = 0; i < outputs->size(); ++i) {
      const auto &tensor = ge_outputs[i];
      auto &output = (*outputs)[i];
      if (output.Size() < LongToSize(UlongToLong(tensor.GetSize()))) {
        MS_LOG(EXCEPTION) << "Output node " << i << "'s mem size " << output.Size()
                          << " is less than actual output size " << tensor.GetSize();
      }
      if ((*outputs)[i].data_c() == nullptr) {
        MS_LOG(ERROR) << "Output data ptr is nullptr.";
        return false;
      }
      auto mem_ret = common::huge_memcpy(reinterpret_cast<uint8_t *>(output.data_c()), output.Size(), tensor.GetData(),
                                         tensor.GetSize());
      if (mem_ret != EOK) {
        MS_LOG(ERROR) << "Failed to copy output data, dst size: " << output.Size()
                      << ", src size: " << tensor.GetSize();
        return false;
      }
    }
  } else {
    for (size_t i = 0; i < ge_outputs.size(); i++) {
      auto &ge_tensor = ge_outputs[i];
      auto ms_tensor = ConvertGeTensorNoCopy(&ge_tensor, graph_id, i);
      if (ms_tensor == nullptr) {
        MS_LOG(ERROR) << "Failed to converter output " << i << " GE Tensor to ME Tensor";
        return false;
      }
      MS_LOG(INFO) << "Output " << i << " shape " << tensor::ShapeToString(ms_tensor->shape_c()) << ", datatype "
                   << ms_tensor->data_type();
      outputs->push_back(*ms_tensor);
    }
  }
  graph_inputs_[graph_id] = inputs;
  graph_outputs_[graph_id] = *outputs;
  MS_LOG(INFO) << "GE run graph " << graph_id << " end.";
  return true;
}

std::vector<tensor::Tensor> GeGraphExecutor::GetInputInfos(uint32_t graph_id) {
  return graph_inputs_.find(graph_id) != graph_inputs_.end() ? graph_inputs_.at(graph_id)
                                                             : std::vector<tensor::Tensor>();
}

tensor::TensorPtr GeGraphExecutor::ConvertGeTensorNoCopy(::ge::Tensor *ge_tensor_ptr, uint32_t graph_id, size_t idx) {
  auto &ge_tensor = *ge_tensor_ptr;
  auto ge_tensor_desc = ge_tensor.GetTensorDesc();
  auto me_shape = transform::TransformUtil::ConvertGeShape(ge_tensor_desc.GetShape());
  if (original_graph_outputs_.find(graph_id) == original_graph_outputs_.end()) {
    MS_LOG(ERROR) << "Graph original outputs with the given graph id is not found.";
    return nullptr;
  }
  auto original_outputs = original_graph_outputs_[graph_id];
  if (idx >= original_outputs.size()) {
    MS_LOG(ERROR) << "Graph output index is out of range.";
    return nullptr;
  }
  TypeId type_id = static_cast<TypeId>(original_outputs[idx]->data_type_c());
  if (type_id == kTypeUnknown) {
    MS_LOG(ERROR) << "Could not convert Ge Tensor because of unsupported data type: "
                  << static_cast<int>(ge_tensor_desc.GetDataType());
    return nullptr;
  }
  if (ge_tensor_desc.GetPlacement() != ::ge::kPlacementHost) {
    MS_LOG(ERROR) << "It is not supported that graph output data's placement is device now.";
    return nullptr;
  }
  auto &&ge_data_uni = ge_tensor.ResetData();
  auto deleter = ge_data_uni.get_deleter();
  auto ge_data = ge_data_uni.release();
  if (ge_data == nullptr) {
    MS_LOG(ERROR) << "Ge data cannot be nullptr";
    return nullptr;
  }
  constexpr int64_t kTensorAlignBytes = 64;
  if (reinterpret_cast<uintptr_t>(ge_data) % kTensorAlignBytes != 0) {
    MS_LOG(ERROR) << "Skip zero-copy ge tensor " << reinterpret_cast<uintptr_t>(ge_data)
                  << ", bytes not aligned with expected.";
    return nullptr;
  }
  int64_t elem_num = 1;
  for (size_t i = 0; i < me_shape.size(); ++i) {
    elem_num *= me_shape[i];
  }
  if (GetTypeByte(TypeIdToType(type_id)) * elem_num != ge_tensor.GetSize()) {
    MS_LOG(ERROR) << "Output datatype error! Output tensor size from GE RunGraph does not match.";
    return nullptr;
  }
  auto tensor_data = std::make_shared<TensorRefData>(ge_data, elem_num, ge_tensor.GetSize(), me_shape.size(), deleter);
  return std::make_shared<tensor::Tensor>(type_id, me_shape, tensor_data);
}

std::vector<tensor::Tensor> GeGraphExecutor::GetOutputInfos(uint32_t graph_id) {
  return graph_outputs_.find(graph_id) != graph_outputs_.end() ? graph_outputs_.at(graph_id)
                                                               : std::vector<tensor::Tensor>();
}

bool GeGraphExecutor::CreateAsCustomFuncGraph(const FuncGraphPtr &func_graph) {
  Buffer buffer;
  auto files = ReadFileNames(build_cache_dir_);
  for (auto &file : files) {
    if (file.find(".om") != std::string::npos && file.find(graph_name_) != std::string::npos) {
      auto om_path = build_cache_dir_ + "/" + file;
      buffer = ReadFile(om_path);
      break;
    }
  }
  if (buffer.DataSize() == 0 || buffer.Data() == nullptr) {
    MS_LOG(ERROR) << "Failed to read model buffer file, model cache " << build_cache_dir_;
    return false;
  }
  std::map<std::string, ValuePtr> attr_map;
  if (!build_cache_relative_dir_.empty()) {
    attr_map[lite::kNameAttrWeightDir] = MakeValue(build_cache_relative_dir_);
    MS_LOG(INFO) << "Set graph attr " << lite::kNameAttrWeightDir << " to " << build_cache_relative_dir_;
  }
  std::vector<std::string> ref_datas;
  std::transform(ref_data_infos_.begin(), ref_data_infos_.end(), std::back_inserter(ref_datas),
                 [](auto &item) { return item.name; });
  if (!CustomAscendUtils::CreateCustomFuncGraph(func_graph, buffer, graph_name_, attr_map, ref_datas)) {
    MS_LOG(ERROR) << "Create custom func graph failed";
    return false;
  }
  return true;
}

bool GeGraphExecutor::OfflineBuildGraph(const FuncGraphPtr &graph) {
  if (ref_mode_flag_ == transform::RefModeFlag::kRefModeNone) {
    MS_LOG(INFO) << "parameter_as_refdata in ascend_context is none, skip offline build graph";
    return true;
  }
  offline_mode_ = true;
  MS_LOG(INFO) << "Set offline mode";
  uint32_t graph_id = 0;
  if (!CompileGraph(graph, {}, &graph_id)) {
    MS_LOG(ERROR) << "Failed to CompileGraph";
    return false;
  }
  MS_LOG(INFO) << "Call GE CompileGraph start";
  ge::Status ret = ge_session_->CompileGraph(graph_id);
  if (ret != ge::GRAPH_SUCCESS) {
    MS_LOG(ERROR) << "Call GE CompileGraph Failed: " << ge::GEGetErrorMsg();
    return false;
  }
  MS_LOG(INFO) << "Call GE CompileGraph end";
  if (!CreateAsCustomFuncGraph(graph)) {
    MS_LOG(ERROR) << "Failed to CreateAsCustomFuncGraph";
    return false;
  }
  return true;
}

std::map<int64_t, std::shared_ptr<GeSessionContext>> GeSessionManager::ge_session_map_;
std::mutex GeSessionManager::session_mutex_;

std::shared_ptr<ge::Session> GeSessionManager::CreateGeSession(
  int64_t session_id, const std::map<std::string, std::string> &session_options) {
  std::shared_ptr<ge::Session> ge_session = nullptr;
  if (session_id == kUnkonwnSessionId) {
    ge_session = std::make_shared<ge::Session>(session_options);
    if (ge_session == nullptr) {
      MS_LOG(ERROR) << "Failed to create ge session";
      return nullptr;
    }
    MS_LOG(INFO) << "Create ge session successfully, which will not be shared with other graph";
    return ge_session;
  }
  std::lock_guard<std::mutex> lock(session_mutex_);
  auto s_it = ge_session_map_.find(session_id);
  if (s_it != ge_session_map_.end() && s_it->second != nullptr) {
    ge_session = s_it->second->ge_session.lock();
  }
  if (ge_session == nullptr) {
    for (auto &option : session_options) {
      MS_LOG(INFO) << "GE Session (lite session id " << session_id << ") option " << option.first << " = "
                   << option.second;
    }
    ge_session = std::make_shared<ge::Session>(session_options);
    if (ge_session == nullptr) {
      MS_LOG(ERROR) << "Failed to create ge session";
      return nullptr;
    }
    auto session_context = std::make_shared<GeSessionContext>();
    if (session_context == nullptr) {
      MS_LOG(ERROR) << "Failed to create GeSessionContext";
      return nullptr;
    }
    session_context->ge_session = ge_session;
    session_context->session_options = session_options;
    ge_session_map_[session_id] = session_context;
    MS_LOG(INFO) << "Create ge session successfully, lite session id: " << session_id;
  } else {
    auto map_as_string = [](const std::map<std::string, std::string> &options) {
      std::stringstream ss;
      ss << "{";
      for (auto &item : options) {
        ss << "" << item.first << ":" << item.second << ",";
      }
      ss << "}";
      return ss.str();
    };
    auto old_options = s_it->second->session_options;
    if (old_options != session_options) {
      MS_LOG(ERROR)
        << "Session options is not equal in diff config infos when models' weights are shared, last session options: "
        << map_as_string(old_options) << ", current session options: " << map_as_string(session_options);
      return nullptr;
    }
    MS_LOG(INFO) << "Get ge session from session map, lite session id: " << session_id;
  }
  return ge_session;
}

std::set<std::string> GeSessionManager::UpdateSessionVariables(int64_t session_id,
                                                               const std::vector<std::string> &graph_variables) {
  std::set<std::string> new_variables;
  if (session_id == kUnkonwnSessionId) {
    std::transform(graph_variables.begin(), graph_variables.end(), std::inserter(new_variables, new_variables.begin()),
                   [](const auto &item) { return item; });
    return new_variables;
  }
  std::lock_guard<std::mutex> lock(session_mutex_);
  std::shared_ptr<ge::Session> ge_session = nullptr;
  auto s_it = ge_session_map_.find(session_id);
  if (s_it != ge_session_map_.end() && s_it->second != nullptr) {
    ge_session = s_it->second->ge_session.lock();
  }
  if (ge_session == nullptr) {
    std::transform(graph_variables.begin(), graph_variables.end(), std::inserter(new_variables, new_variables.begin()),
                   [](const auto &item) { return item; });
    return new_variables;
  }
  auto &current_session_variables = s_it->second->session_variables;
  for (auto &item : graph_variables) {
    if (current_session_variables.find(item) == current_session_variables.end()) {
      new_variables.insert(item);
      current_session_variables.insert(item);
    }
  }
  return new_variables;
}

void GeSessionManager::TryReleaseGeSessionContext(int64_t session_id) {
  std::lock_guard<std::mutex> lock(session_mutex_);
  auto s_it = ge_session_map_.find(session_id);
  if (s_it != ge_session_map_.end()) {
    if (s_it->second != nullptr) {
      auto ge_session = s_it->second->ge_session.lock();
      if (ge_session == nullptr) {
        ge_session_map_.erase(s_it);
      }
    } else {
      ge_session_map_.erase(s_it);
    }
  }
}

std::shared_ptr<GeSessionContext> GeSessionManager::GetGeSessionContext(int64_t session_id) {
  std::lock_guard<std::mutex> lock(session_mutex_);
  auto s_it = ge_session_map_.find(session_id);
  if (s_it != ge_session_map_.end()) {
    return s_it->second;
  }
  return nullptr;
}

static std::shared_ptr<device::GraphExecutor> GeGraphExecutorCreator(const std::shared_ptr<Context> &ctx,
                                                                     const ConfigInfos &config_infos) {
  auto ge_executor = std::make_shared<GeGraphExecutor>(ctx, config_infos);
  if (ge_executor == nullptr || !ge_executor->Init()) {
    MS_LOG(ERROR) << "Failed to init GeGraphExecutor";
    return nullptr;
  }
  return ge_executor;
}

REG_DELEGATE(kAscend, kProviderGe, GeGraphExecutorCreator)
}  // namespace mindspore
