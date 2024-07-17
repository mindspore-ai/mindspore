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

#include "cxx_api/model/acl/model_converter.h"
#include <memory>
#include "include/transform/graph_ir/utils.h"
#include "graph/graph_buffer.h"
#include "graph/graph.h"
#include "cxx_api/model/aoe/auto_tune_process.h"
#include "plugin/device/ascend/optimizer/ge_optimization.h"
#include "transform/symbol/acl_rt_symbol.h"
#include "transform/symbol/acl_symbol.h"
#include "transform/symbol/symbol_utils.h"

namespace mindspore {
namespace {
std::string GetAscendPath() {
  Dl_info info;
  if (dladdr(reinterpret_cast<void *>(aclrtMalloc), &info) == 0) {
    MS_LOG(ERROR) << "Get dladdr failed.";
    return "";
  }
  auto path_tmp = std::string(info.dli_fname);
  const std::string kLatest = "latest";
  auto pos = path_tmp.find(kLatest);
  if (pos == std::string::npos) {
    MS_EXCEPTION(ValueError) << "Get ascend path failed, please check the run package.";
  }
  return path_tmp.substr(0, pos);
}

// todo: acl doesn't support to clear current context
void ClearCurrentRtCtx() {
  aclrtContext tmp_ctx = nullptr;
  auto ret = CALL_ASCEND_API(aclrtCreateContext, &tmp_ctx, 0);
  if (ret != ACL_RT_SUCCESS) {
    MS_LOG(WARNING) << "Call aclrtCreateContext failed, ret = " << ret;
    return;
  }
  ret = CALL_ASCEND_API(aclrtDestroyContext, tmp_ctx);
  if (ret != ACL_RT_SUCCESS) {
    MS_LOG(WARNING) << "Call aclrtDestroyContext failed, ret = " << ret;
    return;
  }
}

transform::TensorOrderMap GetParams(const FuncGraphPtr &anf_graph) {
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

transform::DfGraphPtr ModelConverter::ConvertFuncGraphToAIR(const FuncGraphPtr &anf_graph) const {
  MS_EXCEPTION_IF_NULL(anf_graph);
#ifndef BUILD_LITE
  opt::ReduceOptimization(anf_graph);
#endif
  auto converter = transform::NewConverter(anf_graph, "", transform::RefModeFlag::kRefModeNone, true);
  std::string compute_graph_name = anf_graph->ToString();
  auto option = options_.lock();
  if (option != nullptr && !option->GetDumpModelName().empty()) {
    compute_graph_name = option->GetDumpModelName();
  }
  transform::SetTraining(converter, false);

  transform::BuildGraph(compute_graph_name, converter, GetParams(anf_graph));
  return transform::GetComputeGraph(converter);
}

Buffer ModelConverter::BuildAirModel(const transform::DfGraphPtr &graph,
                                     const std::map<std::string, std::string> &init_options,
                                     const std::map<std::string, std::string> &build_options) const {
  ge::ModelBufferData model;
  auto ret = ge::aclgrphBuildInitialize(init_options);
  if (ret != ge::SUCCESS) {
    MS_LOG(ERROR) << "Call aclgrphBuildInitialize fail: " << CALL_ASCEND_API(aclGetRecentErrMsg);
    return Buffer();
  }

  ret = ge::aclgrphBuildModel(*graph, build_options, model);
  if (ret != ge::SUCCESS) {
    MS_LOG(ERROR) << "Call aclgrphBuildModel fail: " << CALL_ASCEND_API(aclGetRecentErrMsg);
    ge::aclgrphBuildFinalize();
    return Buffer();
  }

  ge::aclgrphBuildFinalize();
  return Buffer(model.data.get(), model.length);
}

Status ModelConverter::SaveModel(const ge::ModelBufferData &model) const {
#ifdef BUILD_LITE
  std::string file_path;
  auto option = options_.lock();
  if (option != nullptr) {
    file_path = option->GetOmFilePath();
  }
  if (file_path.empty()) {
    MS_LOG(INFO) << "File path is empty, there is no need to save model";
    return kSuccess;
  }
  MS_LOG(INFO) << "Om file path: " << file_path;
  auto ret = ge::aclgrphSaveModel(file_path, model);
  if (ret != ge::SUCCESS) {
    MS_LOG(ERROR) << "Call aclgrphSaveModel fail.";
    return kMCFailed;
  }
#endif
  return kSuccess;
}

Buffer ModelConverter::LoadMindIR(const FuncGraphPtr &func_graph) {
  Buffer buffer_ret;
  ClearCurrentRtCtx();
  auto ascend_path = GetAscendPath();
#ifdef MACHINE_LINUX_ARM64
  std::string lib_opsproto_file = ascend_path + "latest/opp/built-in/op_proto/lib/linux/aarch64/libopsproto.so";
#else
  std::string lib_opsproto_file = ascend_path + "latest/opp/built-in/op_proto/lib/linux/x86_64/libopsproto.so";
#endif
  static void *handler = dlopen(lib_opsproto_file.c_str(), RTLD_LAZY);
  if (handler == nullptr) {
    MS_LOG(ERROR) << "dlopen opsproto library failed: " << lib_opsproto_file;
    return buffer_ret;
  }
  auto df_graph = ConvertFuncGraphToAIR(func_graph);
  if (df_graph == nullptr) {
    MS_LOG(ERROR) << "Convert FuncGraph to AscendIR failed.";
    return buffer_ret;
  }
  ge::GraphBuffer model_data;
  auto ge_ret = df_graph->SaveToMem(model_data);
  if (ge_ret != ge::SUCCESS) {
    MS_LOG(ERROR) << "Save ge model to buffer failed.";
    return buffer_ret;
  }
  buffer_ret.SetData(model_data.GetData(), model_data.GetSize());
  Buffer model_result = LoadAscendIRInner(buffer_ret);
  if (model_result.DataSize() == 0) {
    MS_LOG(ERROR) << "Convert model from MindIR to OM failed";
    return {};
  }
  return model_result;
}

Buffer ModelConverter::LoadAscendIRInner(const Buffer &model_data) {
  transform::DfGraphPtr df_graph = std::make_shared<transform::DfGraph>("tmp");
  if (df_graph == nullptr) {
    MS_LOG(ERROR) << "Convert FuncGraph to AscendIR failed.";
    return {};
  }
  auto ret = df_graph->LoadFromMem(static_cast<const uint8_t *>(model_data.Data()), model_data.DataSize());
  if (ret != ge::GRAPH_SUCCESS) {
    MS_LOG(ERROR) << "Convert FuncGraph to AscendIR failed.";
  }

  std::map<std::string, std::string> init_options;
  std::map<std::string, std::string> build_options;
  auto option = options_.lock();
  if (option != nullptr) {
    std::tie(init_options, build_options) = option->GenAclOptions();
  }
#ifdef BUILD_LITE
  if (AutoTuneProcess::AoeOfflineTurningGraph(options_, df_graph) != kSuccess) {
    MS_LOG(ERROR) << "Aoe tune graph failed.";
    return Buffer();
  }
#endif
  return BuildAirModel(df_graph, init_options, build_options);
}
}  // namespace mindspore
