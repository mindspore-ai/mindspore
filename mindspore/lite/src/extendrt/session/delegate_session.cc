/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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

#include "extendrt/session/delegate_session.h"
#include <vector>
#include <string>
#include <mutex>
#include <memory>
#include "extendrt/utils/tensor_utils.h"
#include "src/extendrt/utils/kernel_build_utils.h"
#include "extendrt/delegate/factory.h"
#include "extendrt/session/factory.h"
#include "extendrt/utils/tensor_default_impl.h"
#include "src/extendrt/delegate/graph_executor/litert/func_graph_reuse_manager.h"
#include "src/extendrt/delegate/plugin/ascend_ge_executor_plugin.h"
#include "extendrt/utils/func_graph_utils.h"
#include "extendrt/delegate/comm_group_info.h"
#include "backend/common/session/executor.h"
#include "common/common.h"

namespace mindspore {
namespace {
constexpr auto kAscendProviderGe = "ge";
constexpr auto kDataFlowGraphType = "data_flow";
constexpr auto kIsAdapted = "is_adapted";
constexpr auto kHcclPluginFileName = "libhccl.so";

std::mutex kernel_graph_mutex;
std::mutex g_build_graph_mutex;
}  // namespace

typedef enum {
  HCCL_SUCCESS = 0,              /**< success */
  HCCL_E_PARA = 1,               /**< parameter error */
  HCCL_E_PTR = 2,                /**< empty pointer */
  HCCL_E_MEMORY = 3,             /**< memory error */
  HCCL_E_INTERNAL = 4,           /**< internal error */
  HCCL_E_NOT_SUPPORT = 5,        /**< not support feature */
  HCCL_E_NOT_FOUND = 6,          /**< not found specific resource */
  HCCL_E_UNAVAIL = 7,            /**< resource unavailable */
  HCCL_E_SYSCALL = 8,            /**< call system interface error */
  HCCL_E_TIMEOUT = 9,            /**< timeout */
  HCCL_E_OPEN_FILE_FAILURE = 10, /**< open file fail */
  HCCL_E_TCP_CONNECT = 11,       /**< tcp connect fail */
  HCCL_E_ROCE_CONNECT = 12,      /**< roce connect fail */
  HCCL_E_TCP_TRANSFER = 13,      /**< tcp transfer fail */
  HCCL_E_ROCE_TRANSFER = 14,     /**< roce transfer fail */
  HCCL_E_RUNTIME = 15,           /**< call runtime api fail */
  HCCL_E_DRV = 16,               /**< call driver api fail */
  HCCL_E_PROFILING = 17,         /**< call profiling api fail */
  HCCL_E_CCE = 18,               /**< call cce api fail */
  HCCL_E_NETWORK = 19,           /**< call network api fail */
  HCCL_E_AGAIN = 20,             /**< try again */
  HCCL_E_RESERVED                /**< reserved */
} HcclResult;

using GroupInfoMap = std::vector<std::pair<std::string, std::vector<uint32_t>>>;

extern "C" {
HcclResult HcomCreateGroup(const char *, uint32_t, uint32_t *);
HcclResult HcomInitByFile(const char *, const char *);
HcclResult HcomDestroy();
}

constexpr const char *kHcomCreateGroupName = "HcomCreateGroup";
constexpr const char *kHcomInitByFileName = "HcomInitByFile";
constexpr const char *kHcomDestroyName = "HcomDestroy";

using HcomCreateGroupFunObj = std::function<HcclResult(const char *, uint32_t, uint32_t *)>;
using HcomInitByFileFunObj = std::function<HcclResult(const char *, const char *)>;
using HcomDestroyFunObj = std::function<HcclResult()>;
using HcomCreateGroupFunPtr = HcclResult (*)(const char *, uint32_t, uint32_t *);
using HcomInitByFileFunPtr = HcclResult (*)(const char *, const char *);
using HcomDestroyFunPtr = HcclResult (*)();

HcomCreateGroupFunObj HcomCreateGroup_;
HcomInitByFileFunObj HcomInitByFile_;
HcomDestroyFunObj HcomDestroy_;

bool ge_initialize_ = true;
bool init_hccl_exec_ = false;

bool do_hccl_sym_load() {
  void *libhccl = dlopen(kHcclPluginFileName, RTLD_DEEPBIND | RTLD_NOW | RTLD_LOCAL);
  if (libhccl == nullptr) {
    MS_LOG(ERROR) << "Dlopen libhccl" << kHcclPluginFileName << " failed, result = " << GetDlErrorMsg();
    return false;
  }
  HcomCreateGroup_ = DlsymWithCast<HcomCreateGroupFunPtr>(libhccl, kHcomCreateGroupName);
  HcomInitByFile_ = DlsymWithCast<HcomInitByFileFunPtr>(libhccl, kHcomInitByFileName);
  HcomDestroy_ = DlsymWithCast<HcomDestroyFunPtr>(libhccl, kHcomDestroyName);
  if (HcomCreateGroup_ == nullptr || HcomInitByFile_ == nullptr || HcomDestroy_ == nullptr) {
    MS_LOG(ERROR) << "Dlsys libhccl failed, result = " << GetDlErrorMsg();
    return false;
  }
  return true;
}

bool load_hccl_symbols() {
  static std::once_flag g_flag;
  static bool ret = false;
  std::call_once(g_flag, [] { ret = do_hccl_sym_load(); });
  return ret;
}

bool InitHcclExec(const char *rankTablePath, const char *identify) {
  if (ge_initialize_) {
    return true;
  }
  MS_LOG(INFO) << "Start init hccl exec.";
  MS_EXCEPTION_IF_NULL(HcomInitByFile_);
  HcclResult hccl_ret = HcomInitByFile_(rankTablePath, identify);
  if (hccl_ret == HCCL_E_PTR) {
    MS_LOG(WARNING) << "Hccl comm is null, hcom executor initialize is not required";
  } else if (hccl_ret == HCCL_SUCCESS) {
    MS_LOG(INFO) << "Hcom DynamicKernel Initialize success";
  } else {
    MS_LOG(ERROR) << "Hcom DynamicKernel Initialize failed";
    return false;
  }
  init_hccl_exec_ = true;
  MS_LOG(INFO) << "InitHcclExec success";
  return true;
}

bool FinalizeHcclExec() {
  if (!init_hccl_exec_) {
    return true;
  }
  MS_LOG(INFO) << "Start finalize hccl exec.";
  MS_EXCEPTION_IF_NULL(HcomDestroy_);
  HcclResult hccl_ret = HcomDestroy_();
  if (hccl_ret != HCCL_SUCCESS) {
    MS_LOG(ERROR) << "Hcom DynamicKernel Finalize failed";
    return false;
  }
  init_hccl_exec_ = false;
  MS_LOG(INFO) << "HcclExec destroy success";
  return true;
}

GraphSinkSession::~GraphSinkSession() {
  graph_executor_ = nullptr;
  if (is_use_ascend_ge_) {
    lite::AscendGeExecutorPlugin::GetInstance().DestroyGeContext();
    FinalizeHcclExec();
  }
}

Status GraphSinkSession::GeDeviceContextInit(const std::shared_ptr<Context> &context, const ConfigInfos &config_info) {
  return lite::AscendGeExecutorPlugin::GetInstance().InitializeGeContext(context, config_info);
}

Status GraphSinkSession::Init(const std::shared_ptr<Context> &context, const ConfigInfos &config_info) {
  MS_LOG(INFO) << "GraphSinkSession::Init";
  if (graph_executor_ == nullptr) {
    MS_LOG(ERROR) << "GraphSinkSession::Init failed, graph executor is nullptr.";
    return kLiteUninitializedObj;
  }
  auto device_list = context->MutableDeviceInfo();
  for (const auto &device_info : device_list) {
    if (device_info == nullptr) {
      MS_LOG(ERROR) << "GraphSinkSession::Init failed, device info is nullptr.";
      return kLiteUninitializedObj;
    }
    if (device_info->GetDeviceType() == DeviceType::kAscend && device_info->GetProvider() == kAscendProviderGe) {
      MS_LOG(INFO) << "GraphSinkSession::Init ascend helper";
      is_use_ascend_ge_ = true;
      auto ret = GeDeviceContextInit(context, config_info);
      if (!ret) {
        MS_LOG(ERROR) << "GraphSinkSession::Init failed, GE device context init failed.";
        return kLiteError;
      }

      if (!load_hccl_symbols()) {
        return kCoreFailed;
      }
      auto ascend_info = device_info->Cast<mindspore::AscendDeviceInfo>();
      uint32_t device_id = ascend_info->GetDeviceID();
      std::string rank_table_file = "";
      if (config_info.empty() || config_info.find(lite::kAscendContextSection) == config_info.end()) {
        MS_LOG(INFO) << "There is no ascend context info in config file.";
      } else {
        auto config_info_ascend = config_info.at(lite::kAscendContextSection);
        if (config_info_ascend.find(lite::kRankTableFilePathKey) == config_info_ascend.end()) {
          MS_LOG(INFO)
            << "There is no rank table file in Ascend section of config file, distributed inference is not enabled."
            << " If using distributed inference, make sure rank_table_file in the config file,"
            << " device_id and rank_id are set in AscendDeviceInfo.";
        } else {
          rank_table_file = config_info_ascend[lite::kRankTableFilePathKey];
          MS_LOG(INFO) << "Distributed inference is enabled, rank table file: " << rank_table_file;
        }
      }
      auto device_id_s = std::to_string(device_id);
      InitHcclExec(reinterpret_cast<const char *>(rank_table_file.c_str()),
                   reinterpret_cast<const char *>(device_id_s.c_str()));

      auto group_info_file = context->GetGroupInfoFile();
      if (!group_info_file.empty()) {
        MS_LOG(INFO) << "Get env group_info"
                     << " success: " << group_info_file;
        GroupInfoMap group_info_map;
        lite::CommGroupInfo comm_group_info;

        if (!comm_group_info.LoadGroupInfo(group_info_file, &group_info_map)) {
          MS_LOG(ERROR) << "LoadGroupInfo failed.";
          return kMEInvalidInput;
        }
        for (const auto &[group_name, rank_ids] : group_info_map) {
          MS_LOG(INFO) << "group_name" << group_name << "rank_ids" << rank_ids;
          auto rank_size = rank_ids.size();
          auto res = HcomCreateGroup_(reinterpret_cast<const char *>(group_name.c_str()), UlongToUint(rank_size),
                                      std::vector<unsigned int>(rank_ids).data());
          if (res != HCCL_SUCCESS) {
            MS_LOG(ERROR) << "Create group " << group_name << " rank ids " << rank_ids << " failed.";
            return kMEInvalidInput;
          }
        }
        MS_LOG(INFO) << "Create groups by checkpoint file success ";
      }
      break;
    }
  }
  context_ = context;
  return kSuccess;
}

Status GraphSinkSession::CompileGraph(const void *model_data, size_t data_size, uint32_t *graph_id) {
  MS_LOG(INFO) << "GraphSinkSession::CompileGraph";
  // This lock can be removed when LiteRT supports concurrent multithreading compilation.
  std::lock_guard<std::mutex> lock(g_build_graph_mutex);
  auto ret = graph_executor_->CompileGraph(model_data, data_size, options_, graph_id);
  if (!ret) {
    MS_LOG(ERROR) << "GraphSinkSession::CompileGraph compile graph failed";
    return kCoreFailed;
  }
  DelegateGraphInfo graph_info;
  auto status = InitGraphInfo(&graph_info, *graph_id);
  if (!status.IsOk()) {
    MS_LOG(ERROR) << "Failed to get inputs and outputs info from graph";
    return status;
  }
  graph_infos_[*graph_id] = graph_info;
  return kSuccess;
}

Status GraphSinkSession::CompileGraph(FuncGraphPtr graph, const void *data, size_t size, uint32_t *graph_id) {
  MS_LOG(INFO) << "GraphSinkSession::CompileGraph";
  // This lock can be removed when LiteRT supports concurrent multithreading compilation.
  std::lock_guard<std::mutex> lock(g_build_graph_mutex);
  // kernel graph will be removed from GraphSinkSession, and this code will be moved to TensorRT plugin
  auto func_type = graph->get_attr(kAttrFuncType);
  is_data_flow_graph_ = func_type != nullptr && GetValue<std::string>(func_type) == kDataFlowGraphType;
  if (context_ && !context_->MutableDeviceInfo().empty()) {
    auto device_info = context_->MutableDeviceInfo()[0];
    bool is_ge_backend = device_info && device_info->GetDeviceType() == DeviceType::kAscend &&
                         device_info->GetProvider() == kAscendProviderGe;
    bool is_adapted = graph->has_attr(kIsAdapted);  // The funcgraph will only adapted once while running parallel.
    if (is_ge_backend && !is_adapted && !is_data_flow_graph_) {
      lite::AscendGeExecutorPlugin::GetInstance().AdaptGraph(graph);
      graph->set_attr(kIsAdapted, MakeValue(true));
    }
  }
  DelegateGraphInfo graph_info;
  // the funcgraph constructed by flowgraph has no inputs and outputs.
  auto status = !is_data_flow_graph_ ? InitGraphInputsOutputs(graph, &graph_info) : kSuccess;
  if (!status.IsOk()) {
    MS_LOG(ERROR) << "Failed to get inputs and outputs info from graph";
    return status;
  }
  auto ret = graph_executor_->CompileGraph(graph, options_, graph_id);
  if (!ret) {
    MS_LOG(ERROR) << "GraphSinkSession::CompileGraph compile graph failed";
    return kCoreFailed;
  }
  status = !is_data_flow_graph_ ? UpdateGraphInputsOutputs(*graph_id, &graph_info) : kSuccess;
  if (!status.IsOk()) {
    MS_LOG(ERROR) << "Failed to update inputs and outputs info from graph executor";
    return status;
  }
  graph_infos_[*graph_id] = graph_info;
  return kSuccess;
}

Status GraphSinkSession::InitGraphInfo(DelegateGraphInfo *graph_info_ptr, uint32_t graph_id) {
  auto &info = *graph_info_ptr;

  auto new_inputs = graph_executor_->GetInputInfos(graph_id);
  if (new_inputs.empty()) {
    MS_LOG(ERROR) << "Input is empty.";
    return kCoreFailed;
  }
  info.inputs.clear();
  info.input_names.clear();
  for (size_t i = 0; i < new_inputs.size(); i++) {
    auto &input = new_inputs[i];
    info.input_names.push_back(input.name());
    auto data_type = static_cast<enum DataType>(input.data_type());
    auto impl = std::make_shared<TensorDefaultImpl>(info.input_names[i], data_type, input.shape_c());
    info.inputs.push_back(impl);
  }

  auto new_outputs = graph_executor_->GetOutputInfos(graph_id);
  if (new_outputs.empty()) {
    MS_LOG(ERROR) << "Output is empty.";
    return kCoreFailed;
  }

  info.outputs.clear();
  info.output_names.clear();
  for (size_t i = 0; i < new_outputs.size(); i++) {
    auto &output = new_outputs[i];
    info.output_names.push_back(output.name());
    auto data_type = static_cast<enum DataType>(output.data_type());
    auto impl = std::make_shared<TensorDefaultImpl>(info.output_names[i], data_type, output.shape_c());
    info.outputs.push_back(impl);
  }
  return kSuccess;
}

Status GraphSinkSession::InitGraphInputsOutputs(const FuncGraphPtr &graph, DelegateGraphInfo *graph_info_ptr) {
  auto &info = *graph_info_ptr;
  std::vector<tensor::TensorPtr> graph_inputs, graph_outputs;
  {
    std::unique_lock<std::mutex> l(kernel_graph_mutex);
    FuncGraphReuseManager::GetInstance()->GetInOut(config_infos_, &graph_inputs, &graph_outputs, &info.input_names,
                                                   &info.output_names);
    if (graph_inputs.empty() || graph_outputs.empty() || info.input_names.empty() || info.output_names.empty()) {
      FuncGraphUtils::GetFuncGraphInputsInfo(graph, &graph_inputs, &info.input_names);
      FuncGraphUtils::GetFuncGraphOutputsInfo(graph, &graph_outputs, &info.output_names);
      FuncGraphReuseManager::GetInstance()->StoreInOut(config_infos_, graph_inputs, graph_outputs, info.input_names,
                                                       info.output_names);
    } else {
      MS_LOG(INFO) << "the input and output are the same as the last time. We do not need to construct, and we can "
                      "directly use the cached input and output info.";
    }
  }
  if (graph_inputs.size() != info.input_names.size()) {
    MS_LOG(ERROR) << "Graph input size " << graph_inputs.size() << " != input names size " << info.input_names.size();
    return kCoreFailed;
  }
  if (graph_outputs.size() != info.output_names.size()) {
    MS_LOG(ERROR) << "Graph output size " << graph_outputs.size() << " != output names size "
                  << info.output_names.size();
    return kCoreFailed;
  }
  info.inputs.clear();
  for (size_t i = 0; i < info.input_names.size(); i++) {
    auto &input = graph_inputs[i];
    auto data_type = static_cast<enum DataType>(input->data_type());
    auto impl = std::make_shared<TensorDefaultImpl>(info.input_names[i], data_type, input->shape_c());
    info.inputs.push_back(impl);
  }
  info.outputs.clear();
  for (size_t i = 0; i < info.output_names.size(); i++) {
    auto &output = graph_outputs[i];
    auto data_type = static_cast<enum DataType>(output->data_type());
    auto impl = std::make_shared<TensorDefaultImpl>(info.output_names[i], data_type, output->shape_c());
    info.outputs.push_back(impl);
  }
  return kSuccess;
}

Status GraphSinkSession::UpdateGraphInputsOutputs(uint32_t graph_id, DelegateGraphInfo *graph_info_ptr) {
  auto &info = *graph_info_ptr;
  auto new_inputs = graph_executor_->GetInputInfos(graph_id);
  auto generate_tensor_names = [](size_t size, const std::string &prefix) {
    std::vector<std::string> names;
    for (size_t i = 0; i < size; i++) {
      names.push_back(prefix + std::to_string(i));
    }
    return names;
  };
  if (!new_inputs.empty()) {
    info.input_names =
      !info.input_names.empty() ? info.input_names : generate_tensor_names(new_inputs.size(), "input_");
    if (new_inputs.size() != info.input_names.size()) {
      MS_LOG(ERROR) << "Input count " << new_inputs.size() << " get from executor != input names count "
                    << info.input_names.size();
      return kCoreFailed;
    }
    info.inputs.clear();
    for (size_t i = 0; i < new_inputs.size(); i++) {
      auto &input = new_inputs[i];
      auto data_type = static_cast<enum DataType>(input.data_type());
      auto impl = std::make_shared<TensorDefaultImpl>(info.input_names[i], data_type, input.shape_c());
      info.inputs.push_back(impl);
    }
  }
  auto new_outputs = graph_executor_->GetOutputInfos(graph_id);
  if (!new_outputs.empty()) {
    info.output_names =
      !info.output_names.empty() ? info.output_names : generate_tensor_names(new_outputs.size(), "output_");
    if (new_outputs.size() != info.output_names.size()) {
      MS_LOG(ERROR) << "Output count " << new_outputs.size() << " get from executor != output names count "
                    << info.output_names.size();
      return kCoreFailed;
    }
    info.outputs.clear();
    for (size_t i = 0; i < new_outputs.size(); i++) {
      auto &output = new_outputs[i];
      auto data_type = static_cast<enum DataType>(output.data_type());
      auto impl = std::make_shared<TensorDefaultImpl>(info.output_names[i], data_type, output.shape_c());
      info.outputs.push_back(impl);
    }
  }
  return kSuccess;
}

Status GraphSinkSession::RunGraph(uint32_t graph_id, const std::vector<tensor::Tensor> &inputs,
                                  std::vector<tensor::Tensor> *outputs, const MSKernelCallBack &before,
                                  const MSKernelCallBack &after) {
  MS_LOG(INFO) << "GraphSinkSession::RunGraph";
  MS_EXCEPTION_IF_NULL(outputs);
  graph_executor_->SetBefore(before);
  graph_executor_->SetAfter(after);
  bool ret = graph_executor_->RunGraph(graph_id, inputs, outputs, options_);
  if (!ret) {
    MS_LOG(ERROR) << "GraphSinkSession::RunGraph run graph failed";
    return kCoreFailed;
  }
  if (is_data_flow_graph_) {
    DelegateGraphInfo graph_info;
    UpdateGraphInputsOutputs(graph_id, &graph_info);
    graph_infos_[graph_id] = graph_info;
  }
  return kSuccess;
}

Status GraphSinkSession::RunGraph(uint32_t graph_id, const std::vector<tensor::Tensor> &inputs,
                                  std::vector<tensor::Tensor> *outputs) {
  return RunGraph(graph_id, inputs, outputs, nullptr, nullptr);
}

Status GraphSinkSession::Resize(uint32_t graph_id, const std::vector<tensor::Tensor> &inputs,
                                const std::vector<std::vector<int64_t>> &new_shapes) {
  MS_LOG(INFO) << "GraphSinkSession::Resize";
  MS_EXCEPTION_IF_NULL(graph_executor_);
  auto info_it = graph_infos_.find(graph_id);
  if (info_it == graph_infos_.end()) {
    MS_LOG(ERROR) << "Failed to find graph id " << graph_id;
    return kCoreFailed;
  }
  auto &info = info_it->second;
  auto ret = graph_executor_->Resize(graph_id, inputs, new_shapes);
  if (!ret) {
    return kCoreFailed;
  }
  auto new_outputs = graph_executor_->GetOutputInfos(graph_id);
  if (new_outputs.empty()) {
    return kSuccess;
  }
  if (new_outputs.size() != info.outputs.size()) {
    MS_LOG(ERROR) << "Output count " << new_outputs.size() << " get from executor != last output count "
                  << info.outputs.size();
    return kCoreFailed;
  }
  for (size_t i = 0; i < new_shapes.size(); i++) {
    auto &input_shape = new_shapes[i];
    info.inputs[i]->SetShape(input_shape);
    info.inputs[i]->SetData(nullptr, false);  // reset data
  }
  for (size_t i = 0; i < info.outputs.size(); i++) {
    auto &output = new_outputs[i];
    info.outputs[i]->SetShape(output.shape_c());
    info.outputs[i]->SetData(nullptr, false);  // reset data
  }
  return kSuccess;
}
std::vector<MutableTensorImplPtr> GraphSinkSession::GetOutputs(uint32_t graph_id) {
  auto info_it = graph_infos_.find(graph_id);
  if (info_it == graph_infos_.end()) {
    MS_LOG(ERROR) << "Failed to find graph id " << graph_id;
    return {};
  }
  auto &info = info_it->second;
  return info.outputs;
}
std::vector<MutableTensorImplPtr> GraphSinkSession::GetInputs(uint32_t graph_id) {
  auto info_it = graph_infos_.find(graph_id);
  if (info_it == graph_infos_.end()) {
    MS_LOG(ERROR) << "Failed to find graph id " << graph_id;
    return {};
  }
  auto &info = info_it->second;
  return info.inputs;
}
std::vector<std::string> GraphSinkSession::GetOutputNames(uint32_t graph_id) {
  auto info_it = graph_infos_.find(graph_id);
  if (info_it == graph_infos_.end()) {
    MS_LOG(ERROR) << "Failed to find graph id " << graph_id;
    return {};
  }
  auto &info = info_it->second;
  return info.output_names;
}
std::vector<std::string> GraphSinkSession::GetInputNames(uint32_t graph_id) {
  auto info_it = graph_infos_.find(graph_id);
  if (info_it == graph_infos_.end()) {
    MS_LOG(ERROR) << "Failed to find graph id " << graph_id;
    return {};
  }
  auto &info = info_it->second;
  return info.input_names;
}
MutableTensorImplPtr GraphSinkSession::GetOutputByTensorName(uint32_t graph_id, const std::string &tensorName) {
  auto info_it = graph_infos_.find(graph_id);
  if (info_it == graph_infos_.end()) {
    MS_LOG(ERROR) << "Failed to find graph id " << graph_id;
    return {};
  }
  auto &info = info_it->second;
  for (size_t i = 0; i < info.output_names.size(); i++) {
    if (info.output_names[i] == tensorName) {
      return info.outputs[i];
    }
  }
  return nullptr;
}
MutableTensorImplPtr GraphSinkSession::GetInputByTensorName(uint32_t graph_id, const std::string &name) {
  auto info_it = graph_infos_.find(graph_id);
  if (info_it == graph_infos_.end()) {
    MS_LOG(ERROR) << "Failed to find graph id " << graph_id;
    return {};
  }
  auto &info = info_it->second;
  for (size_t i = 0; i < info.input_names.size(); i++) {
    if (info.input_names[i] == name) {
      return info.inputs[i];
    }
  }
  return nullptr;
}
static std::shared_ptr<InferSession> DelegateSessionCreator(const std::shared_ptr<Context> &ctx,
                                                            const ConfigInfos &config_infos) {
  auto &device_contexts = ctx->MutableDeviceInfo();
  if (device_contexts.empty()) {
    return nullptr;
  }
  auto device_type = device_contexts.at(0)->GetDeviceType();
  auto provider = device_contexts.at(0)->GetProvider();

  auto delegate = DelegateRegistry::GetInstance().GetDelegate(device_type, provider, ctx, config_infos);
  if (delegate == nullptr) {
    return nullptr;
  }
  auto session = std::make_shared<GraphSinkSession>(delegate);
  if (provider != kAscendProviderGe) {
    session->Init(ctx);
  }
  session->SetConfigInfo(config_infos);
  return session;
}
REG_SESSION(kDelegateSession, DelegateSessionCreator);
}  // namespace mindspore
