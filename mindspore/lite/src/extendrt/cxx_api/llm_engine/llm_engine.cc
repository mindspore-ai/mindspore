/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "extendrt/cxx_api/llm_engine/llm_engine.h"
#include <set>
#include "mindspore/lite/src/extendrt/cxx_api/dlutils.h"
#include "mindspore/lite/src/extendrt/cxx_api/file_utils.h"
#include "mindspore/core/load_mindir/load_model.h"
#include "extendrt/cxx_api/llm_engine/llm_engine_plugin.h"
#include "mindspore/lite/src/common/common.h"
#include "mindspore/lite/tools/common/custom_ascend_utils.h"
#include "mindspore/lite/src/extendrt/utils/func_graph_utils.h"

namespace mindspore {
namespace {
constexpr auto kLLMEnginePluginSoName = "libllm_engine_plugin.so";
constexpr auto kLLMEngineCreatePluginFuncName = "CreateLLMEnginePlugin";

Status GetModelInfo(const FuncGraphPtr &func_graph, LLMEngineModelInfo *model_info) {
  if (func_graph == nullptr || model_info == nullptr) {
    return kLiteNullptr;
  }
  std::map<std::string, ValuePtr> attr_map;
  std::vector<std::pair<std::string, tensor::TensorPtr>> ref_datas;
  auto ret =
    CustomAscendUtils::ParseCustomFuncGraph(func_graph, &model_info->om_data, &model_info->name, &attr_map, &ref_datas);
  if (!ret) {
    MS_LOG(ERROR) << "Failed to parse custom func graph";
    return kLiteError;
  }
  for (auto &item : func_graph->get_inputs()) {
    auto shape = FuncGraphUtils::GetTensorShape({item, 0});
    auto type_id = FuncGraphUtils::GetTensorDataType({item, 0});
    model_info->input_shapes.push_back(shape);
    model_info->input_dtypes.push_back(static_cast<TypeId>(type_id));
  }
  for (auto &item : ref_datas) {
    auto &tensor = item.second;
    model_info->ref_input_shapes.push_back(tensor->shape_c());
    model_info->ref_input_dtypes.push_back(static_cast<TypeId>(tensor->data_type()));
  }
  auto get_string_attr = [&attr_map](const std::string &attr_name, std::string *val) {
    auto attr_it = attr_map.find(attr_name);
    if (attr_it == attr_map.end()) {
      MS_LOG(ERROR) << "Failed to attr " << attr_name;
      return false;
    }
    auto &attr_val = attr_it->second;
    if (!attr_val->isa<StringImm>()) {
      MS_LOG(ERROR) << "Failed to attr " << attr_name << ", attr type is " << attr_val->type_name();
      return false;
    }
    *val = GetValue<std::string>(attr_it->second);
    MS_LOG(INFO) << "Get graph attr " << attr_name << " " << *val;
    return true;
  };
  if (!get_string_attr(lite::kNameAttrWeightDir, &model_info->weight_dir)) {
    return kLiteError;
  }
  std::vector<AnfWithOutIndex> outputs;
  if (!FuncGraphUtils::GetFuncGraphOutputs(func_graph, &outputs)) {
    MS_LOG(ERROR) << "Failed to get func graph outputs";
    return kLiteError;
  }
  model_info->output_count = outputs.size();
  return kSuccess;
}

FuncGraphPtr LoadMindIR(const std::string &model_path) {
  if (model_path.empty()) {
    MS_LOG(ERROR) << "Model path cannot be empty";
    return nullptr;
  }
  auto buffer = ReadFile(model_path);
  if (buffer.Data() == nullptr || buffer.DataSize() == 0) {
    MS_LOG(ERROR) << "Failed to read buffer from model file: " << model_path;
    return nullptr;
  }
  std::string weight_path = "./";
  if (model_path.find("/") != std::string::npos) {
    weight_path = model_path.substr(0, model_path.rfind("/"));
  }
  MindIRLoader mindir_loader(true, nullptr, 0, kDecModeAesGcm, false);
  auto func_graph = mindir_loader.LoadMindIR(buffer.Data(), buffer.DataSize(), weight_path);
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "Failed to load MindIR model, please check the validity of the model: " << weight_path;
    return nullptr;
  }
  return func_graph;
}
}  // namespace

typedef LLMEnginePluginBase *(*CreateLLMEnginePluginFunc)();

class LLEnginePluginLoader {
 public:
  static LLEnginePluginLoader &Instance() {
    static LLEnginePluginLoader instance;
    return instance;
  }
  std::shared_ptr<LLMEnginePluginBase> CreatePlugin();

 private:
  void *handle_ = nullptr;
  CreateLLMEnginePluginFunc create_plugin_func_ = nullptr;
  bool Register();
};

bool LLEnginePluginLoader::Register() {
  if (create_plugin_func_ != nullptr) {
    return kSuccess;
  }
  std::string plugin_path;
  auto ret = DLSoPath({"libmindspore-lite.so", "_c_lite"}, kLLMEnginePluginSoName, &plugin_path);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Get real path of " << kLLMEnginePluginSoName << " failed.";
    return false;
  }
  MS_LOG(INFO) << "Find LLMEngine plugin so success, path = " << plugin_path;
  void *function = nullptr;
  ret = DLSoOpen(plugin_path, kLLMEngineCreatePluginFuncName, &handle_, &function);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "DLSoOpen failed, so path: " << plugin_path << ", err: " << ret.ToString();
    return false;
  }
  create_plugin_func_ = reinterpret_cast<CreateLLMEnginePluginFunc>(function);
  if (create_plugin_func_ == nullptr) {
    MS_LOG(ERROR) << "Cast " << kLLMEngineCreatePluginFuncName << " failed.";
    return false;
  }
  MS_LOG(INFO) << "Register LLMEngine plugin success.";
  return true;
}

std::shared_ptr<LLMEnginePluginBase> LLEnginePluginLoader::CreatePlugin() {
  if (!Register()) {
    MS_LOG(ERROR) << "Failed to register " << kLLMEnginePluginSoName;
    return nullptr;
  }
  if (create_plugin_func_ == nullptr) {
    MS_LOG(ERROR) << "Create plugin func is nullptr";
    return nullptr;
  }
  return std::shared_ptr<LLMEnginePluginBase>(create_plugin_func_());
}

LLMEngine::LLMEngine() { plugin_ = LLEnginePluginLoader::Instance().CreatePlugin(); }

Status LLMEngine::Init(const std::vector<std::string> &model_paths, LLMRole role, uint64_t cluster_id,
                       const std::map<std::string, std::string> &options) {
  if (plugin_ == nullptr) {
    MS_LOG(ERROR) << "LLMEngine plugin has not been created";
    return kLiteError;
  }
  std::vector<LLMEngineModelInfo> infos;
  std::set<std::string> names;
  for (auto &model_path : model_paths) {
    auto func_graph = LoadMindIR(model_path);
    if (func_graph == nullptr) {
      MS_LOG(ERROR) << "Failed to load mindir " << model_path;
      return kLiteError;
    }
    LLMEngineModelInfo model_info;
    if (GetModelInfo(func_graph, &model_info) != kSuccess) {
      MS_LOG(ERROR) << "Failed to ge graph info, mindir " << model_path;
      return kLiteError;
    }
    // relative weight path
    if (!model_info.weight_dir.empty() && model_info.weight_dir[0] != '/') {
      if (model_path.find("/") != std::string::npos) {
        model_info.weight_dir = model_path.substr(0, model_path.rfind("/") + 1) + model_info.weight_dir;
        MS_LOG(INFO) << "Update weight dir to " << model_info.weight_dir;
      }
    }
    infos.push_back(model_info);
    names.emplace(model_info.name);
  }
  if (names.size() != infos.size()) {
    for (size_t i = 0; i < infos.size(); i++) {
      infos[i].name += "_U" + std::to_string(i);  // make unique name
    }
  }
  return plugin_->Init(infos, role, cluster_id, options);
}

void LLMEngine::Finalize() {
  if (plugin_ == nullptr) {
    MS_LOG(INFO) << "LLMEngine plugin has not been created";
    return;
  }
  plugin_->Finalize();
}

Status LLMEngine::Predict(const LLMReq &req, const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs) {
  if (plugin_ == nullptr) {
    MS_LOG(ERROR) << "LLMEngine plugin has not been created";
    return kLiteError;
  }
  return plugin_->Predict(req, inputs, outputs);
}

Status LLMEngine::CompleteRequest(const LLMReq &req) {
  if (plugin_ == nullptr) {
    MS_LOG(ERROR) << "LLMEngine plugin has not been created";
    return kLiteError;
  }
  return plugin_->CompleteRequest(req);
}

LLMEngineStatus LLMEngine::FetchStatus() {
  if (plugin_ == nullptr) {
    MS_LOG(ERROR) << "LLMEngine plugin has not been created";
    return LLMEngineStatus();
  }
  return plugin_->FetchStatus();
}
}  // namespace mindspore
