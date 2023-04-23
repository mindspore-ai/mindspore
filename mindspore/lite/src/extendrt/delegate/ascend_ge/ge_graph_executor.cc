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
#include "tools/optimizer/common/gllo_utils.h"
#include "src/extendrt/utils/func_graph_utils.h"
#include "transform/graph_ir/transform_util.h"
#include "flow_graph/data_flow.h"
#ifdef MSLITE_ENABLE_GRAPH_KERNEL
#include "tools/graph_kernel/converter/graph_kernel_optimization.h"
#endif
#include "src/extendrt/utils/tensor_utils.h"
#include "framework/common/ge_inner_error_codes.h"

namespace mindspore {
namespace {
constexpr auto kProviderGe = "ge";
constexpr auto kDump = "dump";
constexpr auto kDumpMode = "dump_mode";
constexpr auto kProfiling = "profiler";
constexpr auto kDataFlowGraphType = "data_flow";
constexpr auto kCustomInputSize = 2;

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
transform::DfGraphPtr GetDataFlowGraph(const FuncGraphPtr &anf_graph,
                                       const std::map<std::string, std::string> &ge_options) {
  MS_EXCEPTION_IF_NULL(anf_graph);
  auto return_node = anf_graph->get_return();
  MS_EXCEPTION_IF_NULL(return_node);
  auto nodes = anf_graph->TopoSort(return_node);
  auto itr = std::find_if(nodes.begin(), nodes.end(), [&](const AnfNodePtr &node) {
    return node->isa<CNode>() && opt::CheckPrimitiveType(node, prim::kPrimCustom);
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
    }
    ge_session_ = nullptr;
  }
}

void GeGraphExecutor::GetGeSessionOptions(std::map<std::string, std::string> *ge_options_ptr) {
  MS_EXCEPTION_IF_NULL(ge_options_ptr);
  auto &ge_options = *ge_options_ptr;
  ge_options["ge.trainFlag"] = "0";
  ge_options["ge.enablePrintOpPass"] = "0";
  auto config_it = config_infos_.find(lite::kGeSessionOptionsSection);
  if (config_it != config_infos_.end()) {
    for (auto &item : config_it->second) {
      ge_options[item.first] = item.second;
      MS_LOG(INFO) << "Set ge session option " << item.first << " to " << item.second;
    }
  }
  auto ascend_info = GetAscendDeviceInfo();
  if (ascend_info == nullptr) {
    MS_LOG(EXCEPTION) << "Failed to get ge session options, can not find ascend device context.";
  }
  ge_options["ge.exec.device_id"] = std::to_string(ascend_info->GetDeviceID());

  config_it = config_infos_.find(lite::kAscendContextSection);
  if (config_it == config_infos_.end()) {
    return;
  }
  auto config = config_it->second;
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
    ge_options["ge.graph_compiler_cache_dir"] = option_id->second;
  }
}

void GeGraphExecutor::GetGeGraphOptions(const FuncGraphPtr &anf_graph,
                                        std::map<std::string, std::string> *ge_options_ptr) {
  MS_EXCEPTION_IF_NULL(anf_graph);
  MS_EXCEPTION_IF_NULL(ge_options_ptr);
  auto &ge_options = *ge_options_ptr;
  ge_options["ge.graph_key"] = anf_graph->ToString();
  auto config_it = config_infos_.find(lite::kGeGraphOptionsSection);
  if (config_it != config_infos_.end()) {
    for (auto &item : config_it->second) {
      ge_options[item.first] = item.second;
      MS_LOG(INFO) << "Set ge graph option " << item.first << " to " << item.second;
    }
  }
  auto ascend_device_info = GetAscendDeviceInfo();
  if (ascend_device_info == nullptr) {
    MS_LOG(EXCEPTION) << "Failed to get graph session options, can not find ascend device context.";
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
#ifdef MSLITE_ENABLE_GRAPH_KERNEL
  if (GraphKernelOptimize(anf_graph, nullptr) != lite::RET_OK) {
    MS_LOG(ERROR) << "Run graphkernel optimization failed.";
    return false;
  }
#endif
  std::map<std::string, std::string> ge_options;
  GetGeGraphOptions(anf_graph, &ge_options);

  transform::DfGraphPtr df_graph = nullptr;
  auto func_type = anf_graph->get_attr(kAttrFuncType);
  is_data_flow_graph_ = func_type != nullptr && GetValue<std::string>(func_type) == kDataFlowGraphType;
  if (!is_data_flow_graph_) {
    auto converter = transform::NewConverter(anf_graph);
    auto params_vals = GetParams(anf_graph);
    transform::BuildGraph(anf_graph->ToString(), converter, params_vals);
    auto err_code = transform::ErrCode(converter);
    if (err_code != 0) {
      transform::ClearGraph();
      MS_LOG(ERROR) << "Convert df graph failed, err:" << err_code;
      return false;
    }
    auto init_graph = transform::GetInitGraph(converter);
    auto init_data_names = converter->GetInitDataNames();
    if (init_graph != nullptr) {
      uint32_t init_graph_id = 0;
      if (!AddGraph(init_graph, {}, &init_graph_id)) {
        MS_LOG(ERROR) << "Failed to add init graph, graph name " << anf_graph->ToString();
        return false;
      }
      std::vector<tensor::TensorPtr> init_data_tensors;
      for (auto &item : init_data_names) {
        auto it = params_vals.find(item);
        if (it == params_vals.end()) {
          MS_LOG(ERROR) << "Cannot find parameter " << item << " in parameter map";
          return false;
        }
        init_data_tensors.push_back(it->second);
      }
      // copy init weight to device
      if (!RunGeInitGraph(init_graph_id, init_data_tensors)) {
        MS_LOG(ERROR) << "Failed to run init graph for " << anf_graph->ToString();
        return false;
      }
    } else {
      MS_LOG(INFO) << "There is no init graph for graph " << anf_graph->ToString();
    }
    df_graph = transform::GetComputeGraph(converter);
  } else {
    df_graph = GetDataFlowGraph(anf_graph, ge_options);
  }
  if (df_graph == nullptr) {
    MS_LOG(ERROR) << "Get df graph failed.";
    return false;
  }
  uint32_t compute_graph_id = 0;
  if (!AddGraph(df_graph, ge_options, &compute_graph_id)) {
    MS_LOG(ERROR) << "Failed to add compute graph, graph name " << anf_graph->ToString();
    return false;
  }
  compute_graph_id_list_.push_back(compute_graph_id);
  *graph_id = compute_graph_id;
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

bool GeGraphExecutor::RunGeInitGraph(uint32_t init_graph_id, const std::vector<tensor::TensorPtr> &init_tensors) {
  MS_LOG(DEBUG) << "ExecInitGraph start.";
  std::vector<::ge::Tensor> ge_inputs;
  for (size_t i = 0; i < init_tensors.size(); i++) {
    auto &input = init_tensors[i];
    auto ge_tensor = transform::TransformUtil::ConvertTensor(input, kOpFormat_NCHW);
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
  std::mutex mutex;
  std::condition_variable condition;
  bool is_finished = false;
  bool end_of_sequence = false;
  std::unique_lock<std::mutex> lock(mutex);
  auto call_back = [=, &is_finished, &end_of_sequence, &condition](ge::Status ge_status,
                                                                   const std::vector<ge::Tensor> &ge_outputs) {
    if (ge_status == ge::GRAPH_SUCCESS) {
      *outputs = ge_outputs;
      is_finished = true;
    } else if (ge_status == ge::END_OF_SEQUENCE) {
      MS_LOG(WARNING) << "RunAsync out of range: End of sequence.";
      end_of_sequence = true;
    } else {
      MS_LOG(ERROR) << "RunAsync failed.";
    }
    condition.notify_all();
    return;
  };
  if (ge_session_ == nullptr) {
    MS_LOG(ERROR) << "The GE session is null, can't run the graph!";
    return false;
  }
  ge::Status ret = ge_session_->RunGraphAsync(graph_id, inputs, call_back);
  if (ret != ge::GRAPH_SUCCESS) {
    MS_LOG(ERROR) << "Call GE RunGraphAsync Failed, ret is: " << ret;
    return false;
  }
  if (!is_finished) {
    condition.wait(lock);
  }
  if (end_of_sequence) {
    throw(std::runtime_error("End of sequence."));
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
      auto ms_tensor = ConvertGeTensorNoCopy(&ge_tensor);
      if (ms_tensor == nullptr) {
        MS_LOG(ERROR) << "Failed to converter output " << i << " GE Tensor to ME Tensor";
        return false;
      }
      MS_LOG(INFO) << "Output " << i << " shape " << ms_tensor->shape_c() << ", datatype " << ms_tensor->data_type();
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

tensor::TensorPtr GeGraphExecutor::ConvertGeTensorNoCopy(::ge::Tensor *ge_tensor_ptr) {
  auto &ge_tensor = *ge_tensor_ptr;
  auto ge_tensor_desc = ge_tensor.GetTensorDesc();
  auto me_shape = transform::TransformUtil::ConvertGeShape(ge_tensor_desc.GetShape());
  TypeId type_id = transform::TransformUtil::ConvertGeDataType(ge_tensor_desc.GetDataType());
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
  auto tensor_data = std::make_shared<TensorRefData>(ge_data, elem_num, ge_tensor.GetSize(), me_shape.size(), deleter);
  return std::make_shared<tensor::Tensor>(type_id, me_shape, tensor_data);
}

std::vector<tensor::Tensor> GeGraphExecutor::GetOutputInfos(uint32_t graph_id) {
  return graph_outputs_.find(graph_id) != graph_outputs_.end() ? graph_outputs_.at(graph_id)
                                                               : std::vector<tensor::Tensor>();
}

static std::shared_ptr<device::GraphExecutor> GeGraphExecutorCreator(const std::shared_ptr<Context> &ctx,
                                                                     const ConfigInfos &config_infos) {
  return std::make_shared<GeGraphExecutor>(ctx, config_infos);
}

REG_DELEGATE(kAscend, kProviderGe, GeGraphExecutorCreator)
}  // namespace mindspore
