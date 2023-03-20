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

namespace mindspore {
namespace {
constexpr auto kProviderGe = "ge";
constexpr auto kDump = "dump";
constexpr auto kDumpMode = "dump_mode";
constexpr auto kProfiling = "profiler";

transform::TensorOrderMap GetParams(const FuncGraphPtr &anf_graph) {
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
      res.emplace(para->name(), tensor);
      MS_LOG(INFO) << "Parameter " << para->name() << " has default value.";
    }
  }
  return res;
}
}  // namespace

std::atomic_uint32_t GeGraphExecutor::global_graph_idx_ = 0;
uint32_t GeGraphExecutor::GetNextGraphIdx() { return global_graph_idx_++; }

GeGraphExecutor::~GeGraphExecutor() {
  if (ge_session_) {
    for (auto graph_id : init_graph_id_list_) {
      ge_session_->RemoveGraph(graph_id);
    }
    for (auto graph_id : compute_graph_id_list_) {
      ge_session_->RemoveGraph(graph_id);
    }
    ge_session_ = nullptr;
  }
}

void GeGraphExecutor::GetGeSessionOptions(std::map<std::string, std::string> *ge_options_ptr) {
  MS_EXCEPTION_IF_NULL(ge_options_ptr);
  auto &ge_options = *ge_options_ptr;
  ge_options["ge.trainFlag"] = "0";
  ge_options["ge.enablePrintOpPass"] = "0";
  auto ascend_info = GetAscendDeviceInfo();
  if (ascend_info == nullptr) {
    MS_LOG(ERROR) << "Cannot faid ascend device info";
    return;
  }
  ge_options["ge.exec.device_id"] = std::to_string(ascend_info->GetDeviceID());

  auto config_it = config_infos_.find(lite::kAscendContextSection);
  if (config_it == config_infos_.end()) {
    return;
  }
  auto config = config_it->second;
  if (config.find(lite::kDumpPathKey) != config.end()) {
    auto dump_path = config.at(lite::kDumpPathKey);
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
  if (config.find(lite::kProfilingPathKey) != config.end()) {
    auto profiling_path = config.at(lite::kProfilingPathKey);
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
  if (config.find(lite::kGeVariableMemoryMaxSize) != config.end()) {
    auto variable_memory_max_size = config[lite::kGeVariableMemoryMaxSize];
    ge_options["ge.variableMemoryMaxSize"] = variable_memory_max_size;
  }
  if (config.find(lite::kGeGraphMemoryMaxSize) != config.end()) {
    auto graph_memory_max_size = config[lite::kGeGraphMemoryMaxSize];
    ge_options["ge.graphMemoryMaxSize"] = graph_memory_max_size;
  }
  if (config.find(lite::kGraphCompilerCacheDirKey) != config.end()) {
    auto graph_compiler_cache_dir = config[lite::kGraphCompilerCacheDirKey];
    ge_options["ge.graph_compiler_cache_dir"] = graph_compiler_cache_dir;
  }
}

void GeGraphExecutor::GetGeGraphOptions(const FuncGraphPtr &anf_graph,
                                        std::map<std::string, std::string> *ge_options_ptr) {
  MS_EXCEPTION_IF_NULL(anf_graph);
  MS_EXCEPTION_IF_NULL(ge_options_ptr);
  auto &ge_options = *ge_options_ptr;
  ge_options["ge.graph_key"] = anf_graph->ToString();
  auto device_list = context_->MutableDeviceInfo();
  auto itr =
    std::find_if(device_list.begin(), device_list.end(), [](const std::shared_ptr<DeviceInfoContext> &device_info) {
      return device_info->GetDeviceType() == DeviceType::kAscend;
    });
  if (itr == device_list.end()) {
    MS_LOG(EXCEPTION) << "Can not find ascend device context.";
  }
  auto ascend_device_info = (*itr)->Cast<AscendDeviceInfo>();
  auto precision_mode = ascend_device_info->GetPrecisionMode();
  if (!precision_mode.empty()) {
    ge_options["ge.exec.precision_mode"] = TransforPrecisionToAcl(precision_mode);
  }
  if (config_infos_.find(lite::kAscendContextSection) == config_infos_.end()) {
    return;
  }
  auto config = config_infos_.at(lite::kAscendContextSection);
  ge_options["ge.exec.modify_mixlist"] =
    config.find(lite::kModifyMixList) == config.end() ? "" : config.at(lite::kModifyMixList);
}

bool GeGraphExecutor::CreateSession() {
  if (ge_session_ != nullptr) {
    MS_LOG(INFO) << "Ge session has already been created";
    return true;
  }
  (void)setenv("GE_TRAIN", "0", 1);
  std::map<std::string, std::string> session_options;
  GetGeSessionOptions(&session_options);
  ge_session_ = std::make_shared<ge::Session>(session_options);
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
  auto ge_status = ge_session_->AddGraph(static_cast<uint32_t>(graph_id), *(graph), options);
  if (ge_status != ge::GRAPH_SUCCESS) {
    MS_LOG(ERROR) << "Call GE AddGraph Failed, ret is: " << ge_status;
    return false;
  }
  *graph_id_ret = graph_id;
  return true;
}

bool GeGraphExecutor::CompileGraph(const FuncGraphPtr &anf_graph, const std::map<string, string> &compile_options,
                                   uint32_t *graph_id) {
  if (!CreateSession()) {
    MS_LOG(ERROR) << "Failed to create ge session";
    return false;
  }
  if (anf_graph == nullptr) {
    MS_LOG(ERROR) << "Input param graph is nullptr.";
    return false;
  }
  if (graph_id == nullptr) {
    MS_LOG(ERROR) << "Input param graph_id is nullptr.";
    return false;
  }
  std::map<std::string, std::string> ge_options;
  GetGeGraphOptions(anf_graph, &ge_options);
  auto converter = transform::NewConverter(anf_graph, converter_context);
  transform::BuildGraph(anf_graph->ToString(), converter, GetParams(anf_graph));
  auto err_code = transform::ErrCode(converter);
  if (err_code != 0) {
    transform::ClearGraph();
    MS_LOG(ERROR) << "Convert df graph failed, err:" << err_code;
    return false;
  }
  uint32_t compute_graph_id = 0;
  if (!AddGraph(transform::GetComputeGraph(converter), ge_options, &compute_graph_id)) {
    MS_LOG(ERROR) << "Failed to add compute graph, graph name " << anf_graph->ToString();
    return false;
  }
  compute_graph_id_list_.push_back(compute_graph_id);
  *graph_id = compute_graph_id;

  auto init_graph = transform::GetInitGraph(converter);
  if (init_graph != nullptr) {
    uint32_t init_graph_id = 0;
    if (!AddGraph(init_graph, {}, &init_graph_id)) {
      MS_LOG(ERROR) << "Failed to add init graph, graph name " << anf_graph->ToString();
      return false;
    }
    init_graph_id_list_.push_back(init_graph_id);
    // copy init weight to device
    RunGeInitGraph(init_graph_id);
  } else {
    MS_LOG(INFO) << "There is no init graph for graph " << anf_graph->ToString();
  }
  return true;
}

std::shared_ptr<AscendDeviceInfo> GeGraphExecutor::GetAscendDeviceInfo() {
  if (context_ == nullptr) {
    MS_LOG(ERROR) << "Context cannot be nullptr";
    return nullptr;
  }
  auto device_list = context_->MutableDeviceInfo();
  auto itr =
    std::find_if(device_list.begin(), device_list.end(), [](const std::shared_ptr<DeviceInfoContext> &device_info) {
      return device_info->GetDeviceType() == DeviceType::kAscend;
    });
  if (itr == device_list.end()) {
    MS_LOG(ERROR) << "Can not find ascend device context.";
    return nullptr;
  }
  auto ascend_device_info = (*itr)->Cast<AscendDeviceInfo>();
  return ascend_device_info;
}

bool GeGraphExecutor::RunGeInitGraph(uint32_t init_graph_id) {
  MS_LOG(DEBUG) << "ExecInitGraph start.";
  std::vector<::ge::Tensor> ge_tensors;
  std::vector<::ge::Tensor> ge_outputs;
  auto ge_status = ge_session_->RunGraph(init_graph_id, ge_tensors, ge_outputs);
  if (ge_status != ge::GRAPH_SUCCESS) {
    MS_LOG(ERROR) << "Exec init graph failed, graph id " << init_graph_id;
    return false;
  }
  MS_LOG(INFO) << "Exec init graph success, graph id " << init_graph_id;
  return true;
}

bool GeGraphExecutor::RunGraph(uint32_t graph_id, const std::vector<tensor::Tensor> &inputs,
                               std::vector<tensor::Tensor> *outputs,
                               const std::map<string, string> & /* compile_options */) {
  if (outputs == nullptr) {
    MS_LOG(ERROR) << " Input param is nullptr.";
    return false;
  }
  MS_LOG(INFO) << "GE run graph " << graph_id << " start.";
  std::vector<::ge::Tensor> ge_inputs;
  for (size_t i = 0; i < inputs.size(); i++) {
    auto &input = inputs[i];
    MS_LOG(INFO) << "Input " << i << " shape " << input.shape_c() << ", datatype " << input.data_type();
    auto ge_tensor = transform::TransformUtil::ConvertTensor(std::make_shared<tensor::Tensor>(input), kOpFormat_NCHW);
    if (ge_tensor == nullptr) {
      MS_LOG(ERROR) << "Failed to converter input " << i << " ME Tensor to GE Tensor";
      return false;
    }
    ge_inputs.emplace_back(*ge_tensor);
  }
  std::vector<::ge::Tensor> ge_outputs;

  auto time_start = std::chrono::system_clock::now();
  auto ge_status = ge_session_->RunGraph(graph_id, ge_inputs, ge_outputs);
  auto time_cost =
    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - time_start).count();
  MS_LOG(INFO) << "Call GE RunGraph Success in " << time_cost << " us, graph id " << graph_id
               << " the GE outputs num is: " << ge_outputs.size();

  if (ge_status != ge::GRAPH_SUCCESS) {
    MS_LOG(ERROR) << "Exec compute graph failed, graph id " << graph_id;
    return false;
  }
  MS_LOG(INFO) << "Exec compute graph success, graph id " << graph_id;
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
    MS_LOG(INFO) << "Output is empty.";
    for (size_t i = 0; i < ge_outputs.size(); i++) {
      auto &ge_tensor = ge_outputs[i];
      auto ms_tensor = transform::TransformUtil::ConvertGeTensor(std::make_shared<::ge::Tensor>(ge_tensor));
      if (ms_tensor == nullptr) {
        MS_LOG(ERROR) << "Failed to converter output " << i << " GE Tensor to ME Tensor";
        return false;
      }
      MS_LOG(INFO) << "Output " << i << " shape " << ms_tensor->shape_c() << ", datatype " << ms_tensor->data_type();
      outputs->push_back(*ms_tensor);
    }
  }
  MS_LOG(INFO) << "GE run graph " << graph_id << " end.";
  return true;
}

std::vector<tensor::Tensor> GeGraphExecutor::GetInputInfos(uint32_t graph_id) { return std::vector<tensor::Tensor>(); }

std::vector<tensor::Tensor> GeGraphExecutor::GetOutputInfos(uint32_t graph_id) { return std::vector<tensor::Tensor>(); }

static std::shared_ptr<device::GraphExecutor> GeGraphExecutorCreator(const std::shared_ptr<Context> &ctx,
                                                                     const ConfigInfos &config_infos) {
  return std::make_shared<GeGraphExecutor>(ctx, config_infos);
}

REG_DELEGATE(kAscend, kProviderGe, GeGraphExecutorCreator)
}  // namespace mindspore
