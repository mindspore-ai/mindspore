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

#include <algorithm>
#include <set>
#include <shared_mutex>
#include <cstring>
#include <memory>
#include <unordered_map>
#include "pybind_api/ir/primitive_py.h"
#include "ops/primitive_c.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "src/common/utils.h"
#include "nnacl/op_base.h"
#include "ops/op_utils.h"
#include "extendrt/cxx_api/model/model_impl.h"
#include "extendrt/cxx_api/dlutils.h"
#include "extendrt/cxx_api/file_utils.h"
#include "extendrt/utils/tensor_utils.h"
#include "mindspore/core/utils/ms_context.h"
#include "extendrt/mindir_loader/mindir_model/mindir_model_util.h"
#include "src/extendrt/convert/runtime_convert.h"
#include "src/common/config_file.h"
#include "src/extendrt/utils/serialization.h"
#include "mindapi/ir/func_graph.h"
#include "mindapi/base/base.h"
#include "src/extendrt/delegate/graph_executor/litert/func_graph_reuse_manager.h"
#include "mindspore/core/load_mindir/load_model.h"
#include "src/extendrt/delegate/plugin/tensorrt_executor_plugin.h"
#include "src/extendrt/kernel/ascend/plugin/ascend_kernel_plugin.h"
#include "utils/ms_utils_secure.h"
#include "ops/custom.h"
#include "ops/return.h"
#include "src/extendrt/model_manager.h"
#include "include/api/model_group.h"
#include "src/common/common.h"

namespace mindspore {
namespace {
const char *const kExecutionPlan = "execution_plan";
const char *const kDataFlowGraphType = "data_flow";
const char *const kDataFlowGraphName = "data_flow_graph";
constexpr size_t kMaxSectionNum = 100;
constexpr size_t kMaxConfigNumPerSection = 1000;
std::shared_mutex g_model_converter_lock;
std::mutex g_load_mindir_lock;

FuncGraphPtr CreateFuncGraphFromDataFlow(const void *model_data, size_t data_size) {
  auto func_graph = std::make_shared<FuncGraph>();
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "The func_graph is nullptr.";
    return nullptr;
  }
  func_graph->set_attr(kAttrFuncType, MakeValue(kDataFlowGraphType));

  // Create custom node with the dataFlow graph.
  auto param = func_graph->add_parameter();
  MS_CHECK_TRUE_RET(param != nullptr, nullptr);
  param->set_name(kDataFlowGraphName);
  auto type_ptr = TypeIdToType(kNumberTypeUInt8);
  MS_CHECK_TRUE_RET(type_ptr != nullptr, nullptr);
  ShapeVector shape = {static_cast<int64_t>(data_size)};
  auto param_tensor = std::make_shared<tensor::Tensor>(kNumberTypeUInt8, shape);
  MS_CHECK_TRUE_RET(param_tensor != nullptr, nullptr);
  if (param_tensor->Size() != data_size) {
    MS_LOG(ERROR) << "The data size of param value is not equal to the data size: " << data_size;
    return nullptr;
  }
  auto tensor_data = param_tensor->data_c();
  MS_CHECK_TRUE_RET(tensor_data != nullptr, nullptr);
  if (common::huge_memcpy(reinterpret_cast<uint8_t *>(tensor_data), param_tensor->Size(),
                          reinterpret_cast<uint8_t *>(const_cast<void *>(model_data)), data_size) != EOK) {
    MS_LOG(ERROR) << "Memcpy dataflow graph data failed.";
    return nullptr;
  }
  param->set_default_param(param_tensor);
  auto abstract_tensor = std::make_shared<abstract::AbstractTensor>(type_ptr, shape);
  MS_CHECK_TRUE_RET(abstract_tensor != nullptr, nullptr);
  param->set_abstract(abstract_tensor);

  auto custom_prim = std::make_shared<ops::Custom>();
  MS_CHECK_TRUE_RET(custom_prim != nullptr, nullptr);
  custom_prim->set_type(kDataFlowGraphType);
  auto custom_prim_c = custom_prim->GetPrim();
  MS_CHECK_TRUE_RET(custom_prim_c != nullptr, nullptr);
  CNodePtr custom_cnode = func_graph->NewCNode(custom_prim_c, {param});
  MS_CHECK_TRUE_RET(custom_cnode != nullptr, nullptr);
  custom_cnode->set_fullname_with_scope("Custom_" + std::string(kDataFlowGraphName));
  auto return_prim = std::make_shared<ops::Return>();
  MS_CHECK_TRUE_RET(custom_prim != nullptr, nullptr);
  auto return_prim_c = return_prim->GetPrim();
  MS_CHECK_TRUE_RET(return_prim_c != nullptr, nullptr);
  auto return_cnode = func_graph->NewCNode(return_prim_c, {custom_cnode});
  MS_CHECK_TRUE_RET(return_cnode != nullptr, nullptr);
  return_cnode->set_fullname_with_scope("Return");
  func_graph->set_return(return_cnode);
  return func_graph;
}

std::unordered_map<std::string, mindspore::Format> kStr2FormatMap{{"DEFAULT_FORMAT", mindspore::Format::DEFAULT_FORMAT},
                                                                  {"NCHW", mindspore::Format::NCHW},
                                                                  {"NHWC", mindspore::Format::NHWC},
                                                                  {"NHWC4", mindspore::Format::NHWC4},
                                                                  {"HWKC", mindspore::Format::HWKC},
                                                                  {"HWCK", mindspore::Format::HWCK},
                                                                  {"KCHW", mindspore::Format::KCHW},
                                                                  {"CKHW", mindspore::Format::CKHW},
                                                                  {"KHWC", mindspore::Format::KHWC},
                                                                  {"CHWK", mindspore::Format::CHWK},
                                                                  {"HW", mindspore::Format::HW},
                                                                  {"HW4", mindspore::Format::HW4},
                                                                  {"NC", mindspore::Format::NC},
                                                                  {"NC4", mindspore::Format::NC4},
                                                                  {"NC4HW4", mindspore::Format::NC4HW4},
                                                                  {"NUM_OF_FORMAT", mindspore::Format::NUM_OF_FORMAT},
                                                                  {"NCDHW", mindspore::Format::NCDHW},
                                                                  {"NWC", mindspore::Format::NWC},
                                                                  {"NCW", mindspore::Format::NCW},
                                                                  {"NDHWC", mindspore::Format::NDHWC},
                                                                  {"NC8HW8", mindspore::Format::NC8HW8}};

Status PrimitivePyToC(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  auto node_list = TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    MS_ASSERT(node != nullptr);
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_ASSERT(cnode != nullptr);

    // judge if primitive is PrimitivePy
    auto primpy_ptr = GetValueNode<PrimitivePtr>(cnode->input(0));
    MS_EXCEPTION_IF_NULL(primpy_ptr);
    if (!utils::isa<PrimitivePy>(primpy_ptr)) {
      continue;
    }
    MS_LOG(INFO) << "Transform a primitivePy to primitiveC for node " << cnode->fullname_with_scope();

    auto kernel_name = primpy_ptr->name();
    ops::PrimitiveCPtr primc_ptr = nullptr;
    static auto &primc_fns = ops::OpPrimCRegister::GetInstance().GetPrimCMap();
    auto primc_it = primc_fns.find(kernel_name);
    if (primc_it != primc_fns.end() && primc_it->second) {
      primc_ptr = primc_it->second();
    }
    if (primc_ptr == nullptr) {
      MS_LOG(ERROR) << "OpPrimCRegister can not find " << kernel_name;
      return kLiteError;
    }
    (void)primc_ptr->SetAttrs(primpy_ptr->attrs());

    if (primpy_ptr->HasAttr(ops::kFormat)) {
      MS_LOG(INFO) << "Add attr Original format to " << cnode->fullname_with_scope();
      auto format_str = GetValue<string>(primpy_ptr->GetAttr(ops::kFormat));
      auto format_it = kStr2FormatMap.find(format_str.c_str());
      if (format_it != kStr2FormatMap.end()) {
        MS_LOG(INFO) << "Add attr Original format" << format_it->second << " to " << cnode->fullname_with_scope();
        (void)primc_ptr->AddAttr(mindspore::ops::kOriginalFormat,
                                 std::dynamic_pointer_cast<mindspore::Value>(
                                   api::MakeValue<int64_t>(static_cast<int64_t>(format_it->second))->impl()));
      } else {
        MS_LOG(ERROR) << "Fail to find format " << format_str.c_str() << "in kStr2FormatMap";
        return kLiteError;
      }
    }

    auto new_prim = MakeValue(primc_ptr);
    auto new_value_node = NewValueNode(new_prim);
    new_value_node->set_abstract(new_prim->ToAbstract());
    cnode->set_input(0, new_value_node);
  }
  return kSuccess;
}
}  // namespace

void ModelImpl::SetMsContext() {
  if (MsContext::GetInstance() != nullptr) {
    auto back_policy_env = std::getenv("ASCEND_BACK_POLICY");
    if (back_policy_env != nullptr) {
      (void)MsContext::GetInstance()->set_backend_policy(std::string(back_policy_env));
    }
  }
}

std::mutex ConverterPlugin::mutex_;
ConverterPlugin::ConverterPlugin() = default;

ConverterPlugin::~ConverterPlugin() {
#ifndef _WIN32
  if (handle_ != nullptr) {
    (void)dlclose(handle_);
    handle_ = nullptr;
  }
#endif
}

ConverterPlugin::ConverterFunc ConverterPlugin::GetConverterFunc() {
  std::lock_guard<std::mutex> lock(mutex_);
  static ConverterPlugin instance;
  return instance.GetConverterFuncInner();
}

ConverterPlugin::ConverterFunc ConverterPlugin::GetConverterFuncInner() {
#ifndef _WIN32
  if (converter_func_ == nullptr) {
    std::string plugin_path;
    auto ret = DLSoPath({"libmindspore-lite.so", "_c_lite"}, "libruntime_convert_plugin.so", &plugin_path);
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "Get path of libruntime_convert_plugin.so failed. error: " << ret;
      return nullptr;
    }
    void *function = nullptr;
    ret = DLSoOpen(plugin_path, "RuntimeConvert", &handle_, &function, true);
    if (ret != kSuccess) {
      MS_LOG(WARNING) << "DLSoOpen RuntimeConvert failed, so path: " << plugin_path;
      return nullptr;
    }
    converter_func_ = reinterpret_cast<ConverterPlugin::ConverterFunc>(function);
  }
  return converter_func_;
#else
  MS_LOG(ERROR) << "Not support libruntime_convert_plugin.so in Windows";
  return nullptr;
#endif
}

ModelImpl::ModelImpl() : graph_(nullptr), session_(nullptr), context_(nullptr) {}

FuncGraphPtr ModelImpl::LoadGraphByBufferImpl(const void *model_buff, size_t model_size, ModelType model_type,
                                              const std::shared_ptr<Context> &model_context,
                                              const std::string &model_path) {
  if (model_type != kMindIR) {
    MS_LOG(ERROR) << "Invalid model type";
    return nullptr;
  }
  MS_CHECK_TRUE_MSG(model_context != nullptr, nullptr, "Invalid context pointers.");
  auto status = UpdateSharingWorkspaceConfig(model_buff, model_size, model_path);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "UpdateSharingWorkspaceConfig failed.";
    return nullptr;
  }
  auto mindir_path = GetConfig(lite::kConfigModelFileSection, lite::kConfigMindIRPathKey);
  std::string weight_path = "./";
  std::string base_path = "";
  if (!mindir_path.empty()) {
    base_path = mindir_path;
  } else {
    // user does not set mindir_path, convert from model_path
    base_path = model_path;
  }
  if (base_path.find("/") != std::string::npos) {
    weight_path = base_path.substr(0, base_path.rfind("/"));
  }
  auto dump_path = GetConfig(lite::kAscendContextSection, lite::kDumpPathKey);
  if (!dump_path.empty()) {
    auto dir_pos = model_path.find_last_of('/');
    auto mindir_name = dir_pos != std::string::npos ? model_path.substr(dir_pos + 1) : model_path;
    auto dot_pos = mindir_name.find_last_of('.');
    auto model_name = mindir_name.substr(0, dot_pos);
    (void)UpdateConfig(lite::kAscendContextSection,
                       std::pair<std::string, std::string>(lite::kDumpModelNameKey, model_name));
  }
  FuncGraphPtr func_graph;
  std::string user_info_string;
  {
    std::unique_lock<std::mutex> l(g_load_mindir_lock);
    MindIRLoader mindir_loader(true, nullptr, 0, kDecModeAesGcm, false);
    auto ret = mindir_loader.LoadMindIR(model_buff, model_size, weight_path, &func_graph, &user_info_string);
    if (!ret || func_graph == nullptr) {
      MS_LOG(ERROR) << "Failed to load MindIR model, please check the validity of the model: " << weight_path;
      return nullptr;
    }
    if (!user_info_string.empty()) {
      SetModelInfo(lite::KModelUserInfo, user_info_string);
    }
  }
  if (func_graph->get_attr(lite::kDynamicDimsKey) != nullptr) {
    auto dynamic_dims = GetValue<std::string>(func_graph->get_attr(lite::kDynamicDimsKey));
    SetModelInfo(lite::kDynamicDimsKey, dynamic_dims);
  }
  if (func_graph->get_attr(lite::KModelInputShape) != nullptr) {
    auto input_shape = GetValue<std::string>(func_graph->get_attr(lite::KModelInputShape));
    SetModelInfo(lite::KModelInputShape, input_shape);
  }
  return func_graph;
}

bool ModelImpl::IsEnableModelSharing(const std::string &model_path, ModelGroupFlag *model_group_flag) {
  const std::map<std::string, ModelGroupFlag> &model_path_set = ModelManager::GetInstance().GetModelPath();
  auto it = model_path_set.find(model_path);
  if (it == model_path_set.end()) {
    return false;
  } else {
    *model_group_flag = it->second;
    return true;
  }
}

bool ModelImpl::IsEnableModelSharing(const std::pair<const void *, size_t> &model_buff) {
  const std::set<std::pair<const void *, size_t>> &model_buff_set = ModelManager::GetInstance().GetModelBuff();
  return (model_buff_set.find(model_buff) != model_buff_set.end());
}

Status ModelImpl::UpdateSharingWorkspaceConfig(const void *model_buff, size_t model_size,
                                               const std::string &model_path) {
  bool model_sharing_flag = false;
  ModelGroupFlag model_group_flag = ModelGroupFlag::kUnknown;
  if (!model_path.empty()) {
    model_sharing_flag = IsEnableModelSharing(model_path, &model_group_flag);
  } else {
    model_sharing_flag = IsEnableModelSharing(std::make_pair(model_buff, model_size));
  }
  if (model_sharing_flag) {
    MS_LOG(INFO) << "model_sharing_flag: " << model_sharing_flag;
    auto ret = UpdateConfig(lite::kInnerCommon, std::make_pair(lite::kInnerSharingWorkspace, "true"));
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "UpdateConfig failed.";
      return ret;
    }
    ret = UpdateConfig(lite::kInnerCommon, std::make_pair(lite::kInnerModelPath, model_path));
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "UpdateConfig failed.";
      return ret;
    }
    if (model_group_flag == ModelGroupFlag::kShareWeight) {
      ret = UpdateConfig(lite::kInnerCommon, std::make_pair(lite::kInnerWeightspace, "true"));
      if (ret != kSuccess) {
        MS_LOG(ERROR) << "UpdateConfig " << lite::kInnerCommon << " " << lite::kInnerWeightspace << " failed!";
        return ret;
      }
    } else if (model_group_flag == ModelGroupFlag::kShareWorkspace) {
      ret = UpdateConfig(lite::kInnerCommon, std::make_pair(lite::kInnerWorkspace, "true"));
      if (ret != kSuccess) {
        MS_LOG(ERROR) << "UpdateConfig " << lite::kInnerCommon << " " << lite::kInnerWorkspace << " failed!";
        return ret;
      }
    } else if (model_group_flag == ModelGroupFlag::kShareWeightAndWorkspace) {
      ret = UpdateConfig(lite::kInnerCommon, std::make_pair(lite::kInnerWeightspaceWorkspace, "true"));
      if (ret != kSuccess) {
        MS_LOG(ERROR) << "UpdateConfig " << lite::kInnerCommon << " " << lite::kInnerWeightspaceWorkspace << " failed!";
        return ret;
      }
    }
  }
  return kSuccess;
}

void ModelImpl::UpdateProvider() {
  if (context_ == nullptr) {
    return;
  }
  auto provider = GetConfig(lite::kAscendContextSection, lite::kProvider);
  if (!provider.empty()) {
    for (auto &device_info : context_->MutableDeviceInfo()) {
      if (device_info && device_info->GetDeviceType() == DeviceType::kAscend && device_info->GetProvider().empty()) {
        device_info->SetProvider(provider);
      }
    }
  }
}

Status ModelImpl::BuildByBufferImpl(const void *model_buff, size_t model_size, ModelType model_type,
                                    const std::shared_ptr<Context> &model_context, const std::string &model_path) {
  if (model_buff == nullptr) {
    MS_LOG(ERROR) << "The input model buffer is nullptr.";
    return kLiteError;
  }
  if (model_size == 0) {
    MS_LOG(ERROR) << "The input model buffer size is 0.";
    return kLiteError;
  }
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (session_) {
    MS_LOG(ERROR) << "Model has been called Build";
    return kLiteModelRebuild;
  }
  if (model_context == nullptr) {
    MS_LOG(ERROR) << "Invalid context pointers.";
    return kLiteError;
  }
  SetMsContext();
  auto thread_num = model_context->GetThreadNum();
  if (thread_num < 0) {
    MS_LOG(ERROR) << "Invalid thread num " << thread_num;
    return kLiteError;
  }
  UpdateProvider();
  auto status = UpdateSharingWorkspaceConfig(model_buff, model_size, model_path);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "UpdateSharingWorkspaceConfig failed.";
    return kLiteError;
  }
  auto mindir_path = GetConfig(lite::kConfigModelFileSection, lite::kConfigMindIRPathKey);
  if (mindir_path.empty()) {
    (void)UpdateConfig(lite::kConfigModelFileSection,
                       std::pair<std::string, std::string>(lite::kConfigMindIRPathKey, model_path));
  }
  session_ = InferSession::CreateSession(model_context, config_info_);
  if (session_ == nullptr) {
    MS_LOG(ERROR) << "Create session failed.";
    return kLiteError;
  }
  Status ret;
  if (model_type == kMindIR_Lite) {
    ret = session_->CompileGraph(model_buff, model_size, &graph_id_);
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "compile graph failed.";
      return ret;
    }
    return kSuccess;
  }
  // for model pool
  FuncGraphPtr func_graph = FuncGraphReuseManager::GetInstance()->GetSharedFuncGraph(config_info_);
  if (func_graph != nullptr) {
    MS_LOG(INFO) << "the model buffer is the same as the last time. we can directly use the cached function graph.";
    std::unique_lock<std::shared_mutex> build_lock(g_model_converter_lock);
    return session_->CompileGraph(func_graph, nullptr, 0, &graph_id_);
  }

  if (model_type != ModelType::kDataFlow) {
    func_graph = LoadGraphByBufferImpl(model_buff, model_size, model_type, model_context, model_path);
    if (func_graph == nullptr) {
      MS_LOG(ERROR) << "Failed to load MindIR model, please check the validity of the model: " << model_path;
      return kLiteError;
    }
    // convert and optimize func graph to infer
    ret = ConvertGraphOnline(func_graph, model_context);
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "convert graph failed.";
      return ret;
    }
  } else {
    // new a func graph contains a custom node, which is the data-flow graph.
    func_graph = CreateFuncGraphFromDataFlow(model_buff, model_size);
    if (func_graph == nullptr) {
      MS_LOG(ERROR) << "Create func graph failed from data flow graph.";
      return kLiteError;
    }
  }
  ret = session_->CompileGraph(func_graph, nullptr, 0, &graph_id_);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "compile graph failed.";
    return ret;
  }
  std::shared_lock<std::shared_mutex> build_lock(g_model_converter_lock);
  return FuncGraphReuseManager::GetInstance()->StoreFuncGraph(func_graph, config_info_);
}

Status ModelImpl::Build(const FuncGraphPtr &func_graph, const std::shared_ptr<Context> &model_context) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (session_) {
    MS_LOG(ERROR) << "Model has been called Build";
    return kLiteModelRebuild;
  }
  if (model_context == nullptr) {
    MS_LOG(ERROR) << "Invalid context pointers.";
    return kLiteError;
  }
  SetMsContext();
  auto thread_num = model_context->GetThreadNum();
  if (thread_num < 0) {
    MS_LOG(ERROR) << "Invalid thread num " << thread_num;
    return kLiteError;
  }
  UpdateProvider();
  session_ = InferSession::CreateSession(model_context, config_info_);
  if (session_ == nullptr) {
    MS_LOG(ERROR) << "Create session failed.";
    return kLiteError;
  }
  // get func_graph
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "Input func graph is nullptr";
    return kLiteError;
  }
  // transfer primitivePy to primitiveC
  auto ret = PrimitivePyToC(func_graph);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "transfer primitivePy to primitiveCfailed.";
    return ret;
  }
  // convert and optimize func graph to infer
  ret = ConvertGraphOnline(func_graph, model_context);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "convert graph failed.";
    return ret;
  }
  ret = session_->CompileGraph(func_graph, nullptr, 0, &graph_id_);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "compile graph failed.";
    return ret;
  }
  std::shared_lock<std::shared_mutex> build_lock(g_model_converter_lock);
  return FuncGraphReuseManager::GetInstance()->StoreFuncGraph(func_graph, config_info_);
}

Status ModelImpl::Build(const void *model_data, size_t data_size, ModelType model_type,
                        const std::shared_ptr<Context> &model_context) {
  return BuildByBufferImpl(model_data, data_size, model_type, model_context);
}

Status ModelImpl::Build(const std::string &model_path, ModelType model_type,
                        const std::shared_ptr<Context> &model_context) {
  if (model_path.empty()) {
    MS_LOG(ERROR) << "Model path cannot be empty";
    return kLiteError;
  }
  auto buffer = ReadFile(model_path);
  if (buffer.DataSize() == 0) {
    MS_LOG(ERROR) << "Failed to read buffer from model file: " << model_path;
    return kLiteError;
  }
  return BuildByBufferImpl(buffer.Data(), buffer.DataSize(), model_type, model_context, model_path);
}

Status ModelImpl::ConvertGraphOnline(const FuncGraphPtr &func_graph, const std::shared_ptr<Context> &model_context) {
  MS_ASSERT(func_graph != nullptr);
  auto device_list = model_context->MutableDeviceInfo();
  for (const auto &device_info : device_list) {
    if (device_info == nullptr) {
      continue;
    }
  }
  auto value = func_graph->get_attr(lite::kIsOptimized);
  if (value != nullptr) {
    if (GetValue<bool>(value)) {
      // it does not need to convert, if funcgraph is optimized.
      return kSuccess;
    }
  }

  auto convert = ConverterPlugin::GetConverterFunc();
  if (convert == nullptr) {
    MS_LOG(ERROR) << "get Converter func failed";
    return kLiteError;
  }
  auto api_graph = mindspore::api::MakeShared<mindspore::api::FuncGraph>(func_graph);
  std::unique_lock<std::shared_mutex> build_lock(g_model_converter_lock);
  auto status = convert(api_graph, model_context, config_info_);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "Failed to converter graph";
    return kLiteError;
  }

  return kSuccess;
}  // namespace mindspore

Status ModelImpl::Resize(const std::vector<MSTensor> &inputs, const std::vector<std::vector<int64_t>> &dims) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (session_ == nullptr) {
    MS_LOG(ERROR) << "Model has not been called Build, or Model Build has failed";
    return kLiteError;
  }
  if (inputs.empty()) {
    MS_LOG(ERROR) << "Inputs is null.";
    return kLiteInputParamInvalid;
  }
  if (dims.empty()) {
    MS_LOG(ERROR) << "Dims is null.";
    return kLiteInputParamInvalid;
  }
  for (size_t j = 0; j < dims.size(); j++) {
    auto dims_v = dims[j];
    for (size_t i = 0; i < dims_v.size(); i++) {
      auto dim = dims_v[i];
      if (dim <= 0 || dim > INT_MAX) {
        MS_LOG(ERROR) << "Invalid shape! dim: " << dim;
        return kLiteInputParamInvalid;
      }
    }
  }
  if (inputs.size() != dims.size()) {
    MS_LOG(ERROR) << "The size of inputs does not match the size of dims.";
    return kLiteInputParamInvalid;
  }
  auto model_inputs = session_->GetInputs(graph_id_);
  if (model_inputs.empty()) {
    MS_LOG(ERROR) << "The inputs of model is null.";
    return kLiteParamInvalid;
  }
  if (inputs.size() != model_inputs.size()) {
    MS_LOG(ERROR) << "The size of inputs is incorrect.";
    return kLiteInputParamInvalid;
  }
  std::vector<mindspore::tensor::Tensor> resize_inputs = TensorUtils::MSTensorToTensor(inputs);
  return session_->Resize(graph_id_, resize_inputs, dims);
}

std::vector<MSTensor> ModelImpl::GetInputs() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (session_ == nullptr) {
    MS_LOG(ERROR) << "Model has not been called Build, or Model Build has failed";
    return {};
  }
  auto graph_inputs = session_->GetInputs(graph_id_);
  std::vector<MSTensor> inputs;
  std::transform(graph_inputs.begin(), graph_inputs.end(), std::back_inserter(inputs),
                 [](auto &impl) { return MSTensor(impl); });
  return inputs;
}

std::vector<MSTensor> ModelImpl::GetOutputs() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (session_ == nullptr) {
    MS_LOG(ERROR) << "Model has not been called Build, or Model Build has failed";
    return {};
  }
  auto graph_outputs = session_->GetOutputs(graph_id_);
  std::vector<MSTensor> outputs;
  std::transform(graph_outputs.begin(), graph_outputs.end(), std::back_inserter(outputs),
                 [](auto &impl) { return MSTensor(impl); });
  return outputs;
}

MSTensor ModelImpl::GetInputByTensorName(const std::string &name) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (session_ == nullptr) {
    MS_LOG(ERROR) << "Model has not been called Build, or Model Build has failed";
    return MSTensor(nullptr);
  }
  auto tensor_impl = session_->GetInputByTensorName(graph_id_, name);
  if (tensor_impl == nullptr) {
    MS_LOG(ERROR) << "Model does not contains tensor " << name << " .";
    return MSTensor(nullptr);
  }
  return MSTensor(tensor_impl);
}

std::vector<std::string> ModelImpl::GetOutputTensorNames() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (session_ == nullptr) {
    MS_LOG(ERROR) << "Model has not been called Build, or Model Build has failed";
    return {};
  }
  return session_->GetOutputNames(graph_id_);
}

MSTensor ModelImpl::GetOutputByTensorName(const std::string &name) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (session_ == nullptr) {
    MS_LOG(ERROR) << "Model has not been called Build, or Model Build has failed";
    return MSTensor(nullptr);
  }
  auto tensor_impl = session_->GetOutputByTensorName(graph_id_, name);
  if (tensor_impl == nullptr) {
    MS_LOG(ERROR) << "Model does not contains tensor " << name << " .";
    return MSTensor(nullptr);
  }
  return MSTensor(tensor_impl);
}

Status ModelImpl::UpdateWeights(const std::vector<std::vector<MSTensor>> &weights) {
  std::vector<std::vector<mindspore::tensor::TensorPtr>> new_weights;
  for (auto &weight : weights) {
    std::vector<mindspore::tensor::TensorPtr> new_weight = TensorUtils::MSTensorToTensorPtr(weight);
    new_weights.push_back(new_weight);
  }
  return session_->UpdateWeights(new_weights);
}

Status ModelImpl::Predict(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs,
                          const MSKernelCallBack &before, const MSKernelCallBack &after) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (session_ == nullptr) {
    MS_LOG(ERROR) << "Model has not been called Build, or Model Build has failed";
    return kLiteError;
  }
  MS_EXCEPTION_IF_NULL(outputs);
  std::vector<mindspore::tensor::Tensor> graph_inputs = TensorUtils::MSTensorToTensor(inputs);
  std::vector<mindspore::tensor::Tensor> graph_outputs;
  std::vector<mindspore::tensor::Tensor> org_graph_outputs;
  if (!outputs->empty()) {
    graph_outputs = TensorUtils::MSTensorToTensor(*outputs);
    org_graph_outputs = graph_outputs;
  }
  auto ret = session_->RunGraph(graph_id_, graph_inputs, &graph_outputs, before, after);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "ModelImpl::Predict RunGraph failed with " << ret;
    return ret;
  }
  bool output_remain = false;
  if (!org_graph_outputs.empty() && org_graph_outputs.size() == graph_outputs.size()) {
    output_remain = true;
    for (size_t i = 0; i < org_graph_outputs.size(); i++) {
      if (org_graph_outputs[i].data_ptr() != graph_outputs[i].data_ptr() ||
          org_graph_outputs[i].device_address() != graph_outputs[i].device_address()) {
        output_remain = false;
        break;
      }
    }
  }
  if (!output_remain) {
    auto session_outputs = session_->GetOutputNames(graph_id_);
    if (session_outputs.empty() || session_outputs.size() != graph_outputs.size()) {
      MS_LOG(ERROR) << "output name is wrong.";
      return kLiteError;
    }
    *outputs = TensorUtils::TensorToMSTensor(graph_outputs, session_outputs);
  }
  auto session_outputs = session_->GetOutputs(graph_id_);
  if (graph_outputs.size() != session_outputs.size()) {
    MS_LOG(ERROR) << "Outputs count get from session " << session_outputs.size() << " != outputs count of RunGraph "
                  << graph_outputs.size();
    return kCoreFailed;
  }
  for (size_t i = 0; i < session_outputs.size(); i++) {
    MSTensor session_output(session_outputs[i]);
    auto &execute_output = outputs->at(i);
    session_output.SetShape(execute_output.Shape());
    if (session_output.GetDeviceData() != execute_output.GetDeviceData()) {
      session_output.SetDeviceData(execute_output.GetDeviceData());
    }
    if (execute_output.GetDeviceData() == nullptr && session_output.Data().get() != execute_output.Data().get()) {
      session_output.SetData(execute_output.MutableData(), false);
    }
  }
  return kSuccess;
}

Status ModelImpl::Predict(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs) {
  return Predict(inputs, outputs, nullptr, nullptr);
}

Status ModelImpl::Predict() {
  auto inputs = GetInputs();
  auto outputs = GetOutputs();
  return Predict(inputs, &outputs);
}

bool ModelImpl::HasPreprocess() {
  if (!graph_ || !graph_->graph_data_) {
    MS_LOG(ERROR) << "Model has not been called Build, or Model Build has failed";
    return false;
  }
  return graph_->graph_data_->GetPreprocess().empty() ? false : true;
}

Status ModelImpl::Preprocess(const std::vector<std::vector<MSTensor>> &inputs, std::vector<MSTensor> *outputs) {
#if !defined(_WIN32) && !defined(_WIN64)
  if (session_ == nullptr) {
    MS_LOG(ERROR) << "Model has not been called Build, or Model Build has failed";
    return kLiteError;
  }
  // Config preprocessor, temporary way to let mindspore.so depends on _c_dataengine
  std::string dataengine_so_path;
  Status dlret = DLSoPath({"libmindspore.so"}, "_c_dataengine", &dataengine_so_path);
  CHECK_FAIL_AND_RELEASE(dlret, nullptr, "Parse dataengine_so failed: " + dlret.GetErrDescription());

  // Run preprocess
  if (!HasPreprocess()) {
    MS_LOG(ERROR) << "Attempt to predict with data preprocessor, but no preprocessor is defined in MindIR.";
    return Status(kMEFailed, "Attempt to predict with data preprocessor, but no preprocessor is defined in MindIR.");
  }

  void *handle = nullptr;
  void *function = nullptr;
  dlret = DLSoOpen(dataengine_so_path, "ExecuteRun_C", &handle, &function);
  CHECK_FAIL_AND_RELEASE(dlret, handle, "Parse ExecuteRun_C failed: " + dlret.GetErrDescription());
  auto ExecuteRun =
    (void (*)(const std::vector<std::shared_ptr<dataset::Execute>> &, const std::vector<mindspore::MSTensor> &,
              std::vector<mindspore::MSTensor> *, Status *))(function);

  // perform preprocess on each tensor separately
  std::vector<std::shared_ptr<dataset::Execute>> preprocessor = graph_->graph_data_->GetPreprocess();
  std::vector<std::vector<MSTensor>> output_unbatch;
  std::vector<MSTensor> output_batched;
  for (auto tensor : inputs) {
    std::vector<MSTensor> temp;
    ExecuteRun(preprocessor, tensor, &temp, &dlret);
    CHECK_FAIL_AND_RELEASE(dlret, handle, "Run preprocess failed: " + dlret.GetErrDescription());
    output_unbatch.push_back(temp);
  }

  // Construct a tensor with batch dim
  output_batched.resize(output_unbatch[0].size());
  for (size_t i = 0; i < output_batched.size(); i++) {
    std::vector<int64_t> ori_shape = output_unbatch[0][i].Shape();
    ori_shape.insert(ori_shape.begin(), output_unbatch.size());
    output_batched[i] = mindspore::MSTensor("outputs", output_unbatch[0][i].DataType(), ori_shape, nullptr,
                                            output_unbatch[0][i].DataSize() * output_unbatch.size());
  }

  // Copy unbatch data into tensor
  for (size_t i = 0; i < output_unbatch[0].size(); i++) {
    size_t offset = 0;
    for (size_t j = 0; j < output_unbatch.size(); j++) {
      auto ret =
        memcpy_s(reinterpret_cast<unsigned uint8_t *>(output_batched[i].MutableData()) + offset,
                 output_unbatch[j][i].DataSize(), output_unbatch[j][i].MutableData(), output_unbatch[j][i].DataSize());
      if (ret) {
        MS_LOG(ERROR) << "Memory copy failed to construct High-Dim Tensor.";
        return Status(kMEFailed, "Memory copy failed to construct High-Dim Tensor.");
      }
      offset += output_unbatch[j][i].DataSize();
    }
  }
  *outputs = output_batched;
  DLSoClose(handle);
  return kSuccess;
#else
  MS_LOG(ERROR) << "Data preprocess is not supported on Windows yet.";
  return Status(kMEFailed, "Data preprocess is not supported on Windows yet.");
#endif
}

Status ModelImpl::PredictWithPreprocess(const std::vector<std::vector<MSTensor>> &inputs,
                                        std::vector<MSTensor> *outputs) {
#if !defined(_WIN32) && !defined(_WIN64)
  if (session_ == nullptr) {
    MS_LOG(ERROR) << "Model has not been called Build, or Model Build has failed";
    return kLiteError;
  }
  // Run preprocess
  std::vector<MSTensor> preprocess_outputs;
  Status ret = Preprocess(inputs, &preprocess_outputs);
  if (ret != kSuccess) {
    return ret;
  }

  // Run prediction
  ret = Predict(preprocess_outputs, outputs);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Run predict failed: " << ret.GetErrDescription();
    return ret;
  }
  return kSuccess;
#else
  MS_LOG(ERROR) << "Predict with data preprocess is not supported on Windows yet.";
  return Status(kMEFailed, "Predict with data preprocess is not supported on Windows yet.");
#endif
}

Status ModelImpl::LoadConfig(const std::string &config_path) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (session_) {
    MS_LOG(ERROR) << "Model has been called Build, please call LoadConfig before Build.";
    return kLiteError;
  }
  ConfigInfos all_config_info;
  int ret = lite::GetAllSectionInfoFromConfigFile(config_path, &all_config_info);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "GetAllSectionInfoFromConfigFile fail!ret: " << ret;
    return kLiteFileError;
  }
  for (auto &section : all_config_info) {
    const auto &section_name = section.first;
    auto sec_it = config_info_.find(section_name);
    if (sec_it == config_info_.end()) {
      config_info_.emplace(section.first, section.second);
    } else {
      auto &cur_sec = sec_it->second;
      for (auto &config_item : section.second) {
        cur_sec[config_item.first] = config_item.second;
      }
    }
  }
  return kSuccess;
}

Status ModelImpl::UpdateConfig(const std::string &section, const std::pair<std::string, std::string> &config) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  auto iter = config_info_.find(section);
  if (iter == config_info_.end()) {
    if (config_info_.size() >= kMaxSectionNum) {
      MS_LOG(ERROR) << "config too many sections!";
      return kLiteError;
    }
    config_info_[section][config.first] = config.second;
    return kSuccess;
  }
  if (iter->second.size() >= kMaxConfigNumPerSection) {
    MS_LOG(ERROR) << "config too many items!";
    return kLiteError;
  }
  iter->second[config.first] = config.second;
  return kSuccess;
}

std::string ModelImpl::GetConfig(const std::string &section, const std::string &key) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  auto iter = config_info_.find(section);
  if (iter == config_info_.end()) {
    return "";
  }
  auto elem_iter = iter->second.find(key);
  if (elem_iter == iter->second.end()) {
    return "";
  }
  return elem_iter->second;
}

ModelImpl::~ModelImpl() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  FuncGraphReuseManager::GetInstance()->ReleaseSharedFuncGraph(config_info_);
  session_ = nullptr;
}

bool ModelImpl::CheckModelSupport(DeviceType device_type, ModelType model_type) {
  if (device_type == kCPU) {
    return true;
  }
  if (model_type != kMindIR) {
    return false;
  }

  if (device_type == kGPU) {
    return lite::TensorRTExecutorPlugin::GetInstance().TryRegister().IsOk();
  }
  if (device_type == kAscend) {
    return kernel::AscendKernelPlugin::TryRegister().IsOk();
  }
  return false;
}

Status ModelImpl::Finalize() {
  if (session_ == nullptr) {
    MS_LOG(ERROR) << "session_ is nullptr,please build model first!";
    return kLiteError;
  }
  return session_->Finalize();
}
}  // namespace mindspore
