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
#include "extendrt/cxx_api/llm_engine/llm_engine_plugin.h"
#include <algorithm>
#include "mindspore/lite/src/extendrt/cxx_api/dlutils.h"
#include "mindspore/core/load_mindir/load_model.h"
#include "mindspore/ccsrc/transform/graph_ir/transform_util.h"
#include "mindspore/lite/src/extendrt/utils/tensor_utils.h"
#include "mindspore/lite/src/common/common.h"
#include "mindspore/lite/src/common/utils.h"
#include "ge/llm_engine.h"
#include "common/ge_common/ge_inner_error_codes.h"

namespace mindspore {
struct LLMModelInfo {
  std::vector<LLMEngineModelInfo> model_infos;
  std::map<std::string, std::string> options;
  LLMEngineModelInfo postprocess_model;
};

class LLMEnginePlugin : public LLMEnginePluginBase {
 public:
  LLMEnginePlugin(LLMRole role, uint64_t cluster_id, const std::string &batch_mode)
      : LLMEnginePluginBase(role, cluster_id, batch_mode), llm_engine_(std::make_shared<llm::LLMEngine>(cluster_id_)) {}
  ~LLMEnginePlugin();
  Status AddModel(const std::vector<LLMEngineModelInfo> &model_infos, const std::map<std::string, std::string> &options,
                  const LLMEngineModelInfo &postprocess_model, uint64_t *model_id) override;
  Status Init(const std::map<std::string, std::string> &options) override;
  void Finalize() override;
  LLMEngineStatus FetchStatus() override;

  Status Predict(const LLMReq &req, const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs,
                 uint64_t model_id) override;

  Status Predict(const std::vector<LLMReq> &req, const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs,
                 uint64_t model_id) override;
  Status CompleteRequest(const LLMReq &req) override;
  Status PreloadPromptPrefix(const LLMReq &req, const std::vector<MSTensor> &inputs, uint64_t model_id) override;
  Status ReleasePromptPrefix(const LLMReq &req, uint64_t model_id) override;

  Status PullKV(const LLMReq &req, uint64_t model_id) override;
  Status MergeKV(const LLMReq &req, uint32_t batch_index, uint32_t batch_id, uint64_t model_id) override;

  Status LinkClusters(const std::vector<LLMClusterInfo> &, std::vector<Status> *rets, int32_t timeout) override;
  Status UnlinkClusters(const std::vector<LLMClusterInfo> &, std::vector<Status> *rets, int32_t timeout) override;

 private:
  std::shared_ptr<::llm::LLMEngine> llm_engine_ = nullptr;
  bool finalized_ = false;
  bool inited_ = false;
  std::map<uint64_t, LLMModelInfo> model_infos_;

  MSTensor ConvertGeTensorNoCopy(::ge::Tensor *ge_tensor_ptr);
  Status Run(const llm::LLMReq &req, const std::vector<::ge::Tensor> &ge_inputs, std::vector<::ge::Tensor> *ge_outputs,
             uint64_t model_id);
  Status Run(const std::vector<llm::LLMReq> &req, const std::vector<::ge::Tensor> &ge_inputs,
             std::vector<::ge::Tensor> *ge_outputs, uint64_t model_id);
  Status CheckModelInfos(const std::vector<LLMEngineModelInfo> &model_infos);
  void InitInputOptions(const LLMEngineModelInfo &model_info, bool postprocess,
                        std::map<std::string, std::string> *options);
  static void TransLLMReq(const LLMReq &req, llm::LLMReq *llm_req);
  static void TransLLMClusterInfos(const std::vector<LLMClusterInfo> &clusters,
                                   std::vector<llm::ClusterInfo> *llm_clusters);
  Status MSTensorToGeTensor(const std::vector<MSTensor> &inputs, std::vector<::ge::Tensor> *ge_inputs);
  Status OnGeStatus(ge::Status ge_status, const std::string &func_s, const std::string &phase);
};

LLMEnginePluginBase *CreateLLMEnginePlugin(LLMRole role, uint64_t cluster_id, const std::string &batch_mode) {
  return new LLMEnginePlugin(role, cluster_id, batch_mode);
}

LLMEnginePlugin::~LLMEnginePlugin() { LLMEnginePlugin::Finalize(); }

Status LLMEnginePlugin::CheckModelInfos(const std::vector<LLMEngineModelInfo> &model_infos) {
  for (size_t i = 1; i < model_infos.size(); i++) {
    if (model_infos[i].input_shapes != model_infos[0].input_shapes) {
      MS_LOG(ERROR) << "Model " << i << " input shapes " << model_infos[i].input_shapes << " != that "
                    << model_infos[0].input_shapes << " of model 0";
      return kLiteError;
    }
    if (model_infos[i].input_dtypes != model_infos[0].input_dtypes) {
      MS_LOG(ERROR) << "Model " << i << " input dtypes " << model_infos[i].input_dtypes << " != that "
                    << model_infos[0].input_dtypes << " of model 0";
      return kLiteError;
    }
    if (model_infos[i].ref_input_shapes != model_infos[0].ref_input_shapes) {
      MS_LOG(ERROR) << "Model " << i << " ref data input shapes " << model_infos[i].ref_input_shapes << " != that "
                    << model_infos[0].ref_input_shapes << " of model 0";
      return kLiteError;
    }
    if (model_infos[i].ref_input_dtypes != model_infos[0].ref_input_dtypes) {
      MS_LOG(ERROR) << "Model " << i << " ref data input dtypes " << model_infos[i].ref_input_dtypes << " != that "
                    << model_infos[0].ref_input_dtypes << " of model 0";
      return kLiteError;
    }
  }
  return kSuccess;
}

void LLMEnginePlugin::InitInputOptions(const LLMEngineModelInfo &model_info, bool postprocess,
                                       std::map<std::string, std::string> *options_ptr) {
  auto shape_as_string = [](const ShapeVector &shape) {
    std::string str;
    for (size_t i = 0; i < shape.size(); i++) {
      str += std::to_string(shape[i]);
      if (i + 1 < shape.size()) {
        str += ",";
      }
    }
    return str;
  };
  auto dtype_as_string = [](TypeId type_id) {
    auto ge_type = transform::TransformUtil::ConvertDataType(type_id);
    return std::to_string(static_cast<uint64_t>(ge_type));
  };
  std::string input_shapes;
  std::string input_dtypes;
  std::string ref_input_shapes;
  std::string ref_input_dtypes;
  for (auto &item : model_info.input_shapes) {
    input_shapes += shape_as_string(item) + ";";
  }
  for (auto &item : model_info.input_dtypes) {
    input_dtypes += dtype_as_string(item) + ";";
  }
  for (auto &item : model_info.ref_input_shapes) {
    ref_input_shapes += shape_as_string(item) + ";";
  }
  for (auto &item : model_info.ref_input_dtypes) {
    ref_input_dtypes += dtype_as_string(item) + ";";
  }
  auto &options = *options_ptr;
  auto erase_comma = [](const std::string &str) { return str.empty() ? str : str.substr(0, str.size() - 1); };
  if (!postprocess) {
    options["llm.InputShapes"] = erase_comma(input_shapes);
    options["llm.InputDtypes"] = erase_comma(input_dtypes);
    options["llm.RefInputShapes"] = erase_comma(ref_input_shapes);
    options["llm.RefInputDtypes"] = erase_comma(ref_input_dtypes);
    options["llm.OutputNums"] = std::to_string(model_info.output_count);
  } else {
    options["llm.PostProcessInputShapes"] = erase_comma(input_shapes);
    options["llm.PostProcessInputDtypes"] = erase_comma(input_dtypes);
    options["llm.PostProcessOutputNums"] = std::to_string(model_info.output_count);
    options["llm.PostProcessOmCachePath"] = model_info.weight_dir;
  }
}

using ErrorCodeMap = std::unordered_map<ge::Status, std::function<Status(const std::string, const std::string)>>;

static ErrorCodeMap error_map = {
  {ge::GRAPH_SUCCESS, [](const std::string &func_s, const std::string &phase) {
    MS_LOG(INFO) << "End call llm::LLMEngine::" << func_s;
    return kSuccess;}
  },
  {ge::LLM_WAIT_PROC_TIMEOUT, [](const std::string &func_s, const std::string &phase){
    MS_LOG(WARNING) << "Failed to call llm::LLMEngine::" << func_s << " "
                    << ", " << phase << " status: LLM_WAIT_PROC_TIMEOUT";
    return kLiteLLMWaitProcessTimeOut;}
  },
  {ge::LLM_KV_CACHE_NOT_EXIST, [](const std::string &func_s, const std::string &phase){
    MS_LOG(WARNING) << "Failed to call llm::LLMEngine::" << func_s << " " << phase << " status: LLM_KV_CACHE_NOT_EXIST";
    return kLiteLLMKVCacheNotExist;}
  },
  {ge::LLM_REPEAT_REQUEST, [](const std::string &func_s, const std::string &phase){
    MS_LOG(ERROR) << "Failed to call llm::LLMEngine::" << func_s << " " << phase << " status: LLM_REPEAT_REQUEST";
    return kLiteLLMRepeatRequest;}
  },
  {ge::LLM_REQUEST_ALREADY_COMPLETED, [](const std::string &func_s, const std::string &phase){
    MS_LOG(ERROR) << "Failed to call llm::LLMEngine::" << func_s << " " << phase
                  << " receive LLM_REQUEST_ALREADY_COMPLETED";
    return kLiteLLMRequestAlreadyCompleted;}
  },
  {ge::LLM_ENGINE_FINALIZED, [](const std::string &func_s, const std::string &phase){
    MS_LOG(ERROR) << "Failed to call llm::LLMEngine::" << func_s << " " << phase << " status: LLM_ENGINE_FINALIZED";
    return kLiteLLMRequestAlreadyCompleted;}
  },
  {ge::LLM_PARAM_INVALID, [](const std::string &func_s, const std::string &phase){
    MS_LOG(ERROR) << "Failed to call llm::LLMEngine::" << func_s << " " << phase << " status: LLM_PARAM_INVALID";
    return kLiteParamInvalid;}
  },
  {ge::LLM_NOT_YET_LINK, [](const std::string &func_s, const std::string &phase){
    MS_LOG(ERROR) << "Failed to call llm::LLMEngine::" << func_s << " " << phase << " status: LLM_NOT_YET_LINK";
    return kLiteLLMNotYetLink;}
  },
  {ge::LLM_ALREADY_LINK, [](const std::string &func_s, const std::string &phase){
    MS_LOG(ERROR) << "Failed to call llm::LLMEngine::" << func_s << " " << phase << " status: LLM_ALREADY_LINK";
    return kLiteLLMAlreadyLink;}
  },
  {ge::LLM_LINK_FAILED, [](const std::string &func_s, const std::string &phase){
    MS_LOG(ERROR) << "Failed to call llm::LLMEngine::" << func_s << " " << phase << " status: LLM_LINK_FAILED";
    return kLiteLLMLinkFailed;}
  },
  {ge::LLM_UNLINK_FAILED, [](const std::string &func_s, const std::string &phase){
    MS_LOG(ERROR) << "Failed to call llm::LLMEngine::" << func_s << " " << phase << " status: LLM_UNLINK_FAILED";
    return kLiteLLMUnlinkFailed;}
  },
  {ge::LLM_NOTIFY_PROMPT_UNLINK_FAILED, [](const std::string &func_s, const std::string &phase){
    MS_LOG(ERROR) << "Failed to call llm::LLMEngine::" << func_s << " "
                  << phase << " status: LLM_NOTIFY_PROMPT_UNLINK_FAILED";
    return kLiteLLMNofiryPromptUnlinkFailed;}
  },
  {ge::LLM_CLUSTER_NUM_EXCEED_LIMIT, [](const std::string &func_s, const std::string &phase){
    MS_LOG(ERROR) << "Failed to call llm::LLMEngine::" << func_s << " "
                  << phase << " status: LLM_CLUSTER_NUM_EXCEED_LIMIT";
    return kLiteLLMClusterNumExceedLimit;}
  },
  {ge::LLM_PROCESSING_LINK, [](const std::string &func_s, const std::string &phase){
    MS_LOG(ERROR) << "Failed to call llm::LLMEngine::" << func_s << " " << phase << " status: LLM_PROCESSING_LINK";
    return kLiteLLMProcessingLink;}
  },
  {ge::LLM_DEVICE_OUT_OF_MEMORY, [](const std::string &func_s, const std::string &phase){
    MS_LOG(ERROR) << "Failed to call llm::LLMEngine::" << func_s << " "
                  << phase << " status: LLM_DEVICE_OUT_OF_MEMORY";
    return kLiteLLMOutOfMemory;}
  },
  {ge::LLM_PREFIX_ALREADY_EXIST, [](const std::string &func_s, const std::string &phase){
    MS_LOG(ERROR) << "Failed to call llm::LLMEngine::" << func_s << " "
                  << phase << " status: LLM_PREFIX_ALREADY_EXIST";
    return kLiteLLMPrefixAlreadyExist;}
  },
  {ge::LLM_PREFIX_NOT_EXIST, [](const std::string &func_s, const std::string &phase){
    MS_LOG(ERROR) << "Failed to call llm::LLMEngine::" << func_s << " "
                  << phase << " status: LLM_PREFIX_NOT_EXIST";
    return kLiteLLMPrefixNotExist;}
  },
  {ge::LLM_SEQ_LEN_OVER_LIMIT, [](const std::string &func_s, const std::string &phase){
    MS_LOG(ERROR) << "Failed to call llm::LLMEngine::" << func_s << " "
                  << phase << " status: LLM_SEQ_LEN_OVER_LIMIT";
    return kLiteLLMSeqLenOverLimit;}
  },
  {ge::LLM_NO_FREE_BLOCK, [](const std::string &func_s, const std::string &phase){
    MS_LOG(ERROR) << "Failed to call llm::LLMEngine::" << func_s << " "
                  << phase << " status: LLM_NO_FREE_BLOCK";
    return kLiteLLMNoFreeBlock;}
  },
  {ge::LLM_BLOCKS_OUT_OF_MEMORY, [](const std::string &func_s, const std::string &phase){
    MS_LOG(ERROR) << "Failed to call llm::LLMEngine::" << func_s << " "
                  << phase << " status: LLM_BLOCKS_OUT_OF_MEMORY";
    return kLiteLLMBlockOutOfMemory;}
  }
};

Status LLMEnginePlugin::OnGeStatus(ge::Status ge_status, const std::string &func_s, const std::string &phase) {
  Status lite_status;
  if (error_map.count(ge_status) == 1) {
    lite_status = error_map[ge_status](func_s, phase);
  } else {
    MS_LOG(ERROR) << "Failed to call llm::LLMEngine::" << func_s << " " << phase << " status: " << ge_status;
    lite_status = kLiteError;
  }
  return lite_status;
}

Status LLMEnginePlugin::AddModel(const std::vector<LLMEngineModelInfo> &model_infos,
                                 const std::map<std::string, std::string> &options_i,
                                 const LLMEngineModelInfo &postprocess_model, uint64_t *model_id) {
  if (model_infos.empty()) {
    MS_LOG(ERROR) << "Model infos cannot be empty";
    return kLiteError;
  }
  if (model_id == nullptr) {
    MS_LOG(ERROR) << "Input argument model_id is nullptr";
    return kLiteError;
  }
  if (finalized_) {
    MS_LOG(ERROR) << "LLMEngine has been finalized";
    return kLiteLLMEngineFinalized;
  }
  if (inited_) {
    MS_LOG(ERROR) << "LLMEngine has been inited";
    return kLiteError;
  }
  if (llm_engine_ == nullptr) {
    MS_LOG(ERROR) << "LLMEngine object is nullptr";
    return kLiteError;
  }
  MS_LOG(INFO) << "LLMEngine AddLLMModel begin";
  auto options = options_i;
  if (CheckModelInfos(model_infos) != kSuccess) {
    return kLiteError;
  }
  InitInputOptions(model_infos[0], false, &options);
  auto option_it = options.find("llm.OmCachePath");
  if (option_it == options.end()) {
    std::string cache_path;
    for (size_t i = 0; i < model_infos.size(); i++) {
      cache_path += model_infos[i].weight_dir;
      if (i + 1 < model_infos.size()) {
        cache_path += ";";
      }
    }
    MS_LOG(INFO) << "Add option llm.OmCachePath to " << cache_path;
    options["llm.OmCachePath"] = cache_path;
  }
  std::vector<ge::ModelBufferData> model_buffers;
  for (auto &item : model_infos) {
    ge::ModelBufferData buff;
    buff.data = std::shared_ptr<uint8_t>(reinterpret_cast<uint8_t *>(item.om_data->data_c()), [](uint8_t *) {});
    buff.length = item.om_data->Size();
    model_buffers.push_back(buff);
    MS_LOG(INFO) << "Inference model " << item.name << ", model buffer size " << item.om_data->Size();
  }
  std::map<ge::AscendString, std::vector<ge::ModelBufferData>> model_buffers_map;
  model_buffers_map["inference"] = model_buffers;
  if (postprocess_model.om_data != nullptr) {
    InitInputOptions(postprocess_model, true, &options);

    ge::ModelBufferData postprocess_buff;
    postprocess_buff.data =
      std::shared_ptr<uint8_t>(reinterpret_cast<uint8_t *>(postprocess_model.om_data->data_c()), [](uint8_t *) {});
    postprocess_buff.length = postprocess_model.om_data->Size();
    MS_LOG(INFO) << "Postprocess model " << postprocess_model.name << ", model buffer size "
                 << postprocess_model.om_data->Size();
    model_buffers_map["postprocess"] = {postprocess_buff};
  }
  std::map<ge::AscendString, ge::AscendString> model_options;
  for (auto &option : options) {
    model_options[ge::AscendString(option.first.c_str())] = ge::AscendString(option.second.c_str());
    MS_LOG(INFO) << "AddLLMModel option " << option.first << " = " << option.second;
  }
  MS_LOG(INFO) << "Start to call llm::LLMEngine::LLMEngineInitializeV2";
  auto ge_status = llm_engine_->AddLLMModel(model_buffers_map, model_options, *model_id);
  if (ge_status != ge::GRAPH_SUCCESS) {
    return OnGeStatus(ge_status, "AddLLMModel", "return");
  }
  LLMModelInfo info;
  info.model_infos = model_infos;
  info.postprocess_model = postprocess_model;
  info.options = options;
  model_infos_[*model_id] = info;
  MS_LOG(INFO) << "LLMEngine AddLLMModel end";
  return kSuccess;
}

Status LLMEnginePlugin::Init(const std::map<std::string, std::string> &options_i) {
  if (finalized_) {
    MS_LOG(ERROR) << "LLMEngine has been finalized";
    return kLiteLLMEngineFinalized;
  }
  if (inited_) {
    MS_LOG(ERROR) << "LLMEngine has been inited";
    return kLiteError;
  }
  if (llm_engine_ == nullptr) {
    MS_LOG(ERROR) << "LLMEngine object is nullptr";
    return kLiteError;
  }
  MS_LOG(INFO) << "LLMEngine Init begin";
  auto options = options_i;
  options["llm.Role"] = role_ == LLMRole::kLLMRolePrompt ? "Prompt" : "Decoder";
  options["llm.batch_mode"] = batch_mode_;
  std::map<ge::AscendString, ge::AscendString> init_options;
  for (auto &option : options) {
    init_options[ge::AscendString(option.first.c_str())] = ge::AscendString(option.second.c_str());
    MS_LOG(INFO) << "LLMEngineInitializeV2 option " << option.first << " = " << option.second;
  }
  MS_LOG(INFO) << "Start to call llm::LLMEngine::LLMEngineInitialize";
  auto ge_status = llm_engine_->LLMEngineInitializeV2({}, init_options);
  if (ge_status != ge::GRAPH_SUCCESS) {
    return OnGeStatus(ge_status, "LLMEngineInitializeV2", "return");
  }
  model_infos_.clear();
  inited_ = true;
  MS_LOG(INFO) << "LLMEngine Init end";
  return kSuccess;
}

void LLMEnginePlugin::Finalize() {
  if (llm_engine_ != nullptr) {
    MS_LOG(INFO) << "Start to call LLMEngineFinalize";
    auto ge_status = llm_engine_->LLMEngineFinalize();
    llm_engine_ = nullptr;
    finalized_ = true;
    if (ge_status != ge::GRAPH_SUCCESS) {
      MS_LOG(ERROR) << "Failed to call LLMEngineFinalize, status: " << ge_status;
      return;
    }
    MS_LOG(INFO) << "End to call LLMEngineFinalize";
  }
}

Status LLMEnginePlugin::Run(const llm::LLMReq &llm_req, const std::vector<::ge::Tensor> &ge_inputs,
                            std::vector<::ge::Tensor> *outputs, uint64_t model_id) {
  auto time_start = std::chrono::system_clock::now();

  if (role_ == kLLMRolePrompt) {
    MS_LOG(INFO) << "Start to call llm::LLMEngine::RunPrompt";
    auto ge_status = llm_engine_->RunPrompt(llm_req, ge_inputs, *outputs, model_id);
    if (ge_status != ge::GRAPH_SUCCESS) {
      return OnGeStatus(ge_status, "RunPrompt", "return");
    }
  } else {
    if (model_id != 0) {
      MS_LOG(ERROR) << "Decoder only support manual mode when there are more than one LLM Model, current mode "
                    << batch_mode_;
      return kLiteError;
    }
    MS_LOG(INFO) << "Start to call llm::LLMEngine::RunDecoder";
    auto ge_status = llm_engine_->RunDecoder(llm_req, ge_inputs, *outputs);
    if (ge_status != ge::GRAPH_SUCCESS) {
      return OnGeStatus(ge_status, "RunDecoder", "return");
    }
  }
  auto time_cost =
    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - time_start).count();
  MS_LOG(INFO) << "Call LLMEngine RunPrompt or RunDecoder Success in " << time_cost << " us, role "
               << (role_ == LLMRole::kLLMRolePrompt ? "Prompt" : "Decoder") << ", outputs num is: " << outputs->size();
  return kSuccess;
}

Status LLMEnginePlugin::Run(const std::vector<llm::LLMReq> &llm_req, const std::vector<::ge::Tensor> &ge_inputs,
                            std::vector<::ge::Tensor> *outputs, uint64_t model_id) {
  auto time_start = std::chrono::system_clock::now();

  if (role_ == kLLMRolePrompt) {
    MS_LOG(INFO) << "Start to call llm::LLMEngine::RunPrompt";
    auto ge_status = llm_engine_->RunPrompt(llm_req, ge_inputs, *outputs, model_id);
    if (ge_status != ge::GRAPH_SUCCESS) {
      return OnGeStatus(ge_status, "RunPrompt", "return");
    }
  } else {
    MS_LOG(INFO) << "Start to call llm::LLMEngine::RunDecoder";
    auto ge_status = llm_engine_->RunDecoder(llm_req, ge_inputs, *outputs, model_id);
    if (ge_status != ge::GRAPH_SUCCESS) {
      return OnGeStatus(ge_status, "RunDecoder", "return");
    }
  }
  auto time_cost =
    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - time_start).count();
  MS_LOG(INFO) << "Call LLMEngine RunPrompt or RunDecoder Success in " << time_cost << " us, role "
               << (role_ == LLMRole::kLLMRolePrompt ? "Prompt" : "Decoder") << ", outputs num is: " << outputs->size();
  return kSuccess;
}

void LLMEnginePlugin::TransLLMReq(const LLMReq &req, llm::LLMReq *llm_req_ptr) {
  if (llm_req_ptr == nullptr) {
    MS_LOG(ERROR) << "Input argument llm_req_ptr is nullptr";
    return;
  }
  llm::LLMReq &llm_req = *llm_req_ptr;
  llm_req.SetReqId(req.req_id);
  llm_req.SetPromptLength(req.prompt_length);
  llm_req.SetPromptClusterId(req.prompt_cluster_id);
  llm_req.SetDecoderClusterId(req.decoder_cluster_id);
  llm_req.SetPrefixId(req.prefix_id);
}

void LLMEnginePlugin::TransLLMClusterInfos(const std::vector<LLMClusterInfo> &clusters,
                                           std::vector<llm::ClusterInfo> *llm_clusters_ptr) {
  if (llm_clusters_ptr == nullptr) {
    MS_LOG(ERROR) << "Input argument llm_clusters_ptr is nullptr";
    return;
  }
  auto &llm_clusters = *llm_clusters_ptr;
  for (auto &cluster : clusters) {
    llm::ClusterInfo llm_cluster;
    llm_cluster.remote_cluster_id = cluster.remote_cluster_id;
    llm_cluster.remote_role_type = cluster.remote_role_type;
    for (auto &item : cluster.local_ip_infos) {
      llm::IpInfo llm_ip_info;
      llm_ip_info.ip = item.ip;
      llm_ip_info.port = item.port;
      llm_cluster.local_ip_infos.push_back(llm_ip_info);
    }
    for (auto &item : cluster.remote_ip_infos) {
      llm::IpInfo llm_ip_info;
      llm_ip_info.ip = item.ip;
      llm_ip_info.port = item.port;
      llm_cluster.remote_ip_infos.push_back(llm_ip_info);
    }
    llm_clusters.push_back(llm_cluster);
  }
}

Status LLMEnginePlugin::MSTensorToGeTensor(const std::vector<MSTensor> &inputs, std::vector<::ge::Tensor> *ge_inputs) {
  for (size_t i = 0; i < inputs.size(); i++) {
    auto &input = inputs[i];
    MS_LOG(INFO) << "Input " << i << " shape " << input.Shape() << ", datatype " << input.DataType();
    // create ge tensor
    auto desc =
      transform::TransformUtil::GetGeTensorDesc(input.Shape(), static_cast<TypeId>(input.DataType()), kOpFormat_NCHW);
    if (desc == nullptr) {
      MS_LOG(ERROR) << "Failed to get Tensor Desc";
      return kLiteError;
    }
    ge::Tensor tensor(*desc);
    auto data = reinterpret_cast<uint8_t *>(const_cast<void *>(input.Data().get()));
    auto ret = tensor.SetData(data, input.DataSize(), [](uint8_t *) -> void {});
    if (ret != ge::GRAPH_SUCCESS) {
      MS_LOG(ERROR) << "Failed to call ge::Tensor SetData(uint8_t*, size, DeleteFunc), data size " << input.DataSize();
      return kLiteError;
    }
    ge_inputs->emplace_back(tensor);
  }
  return kSuccess;
}

Status LLMEnginePlugin::Predict(const LLMReq &req, const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs,
                                uint64_t model_id) {
  if (outputs == nullptr) {
    MS_LOG(ERROR) << "Input argument outputs is nullptr";
    return kLiteError;
  }
  if (finalized_) {
    MS_LOG(ERROR) << "LLMEngine has been finalized";
    return kLiteLLMEngineFinalized;
  }
  if (!inited_) {
    MS_LOG(ERROR) << "LLMEngine has not been inited or inited failed";
    return kLiteError;
  }
  if (llm_engine_ == nullptr) {
    MS_LOG(ERROR) << "LLMEngine object is nullptr";
    return kLiteError;
  }
  std::vector<::ge::Tensor> ge_inputs;
  auto ret = MSTensorToGeTensor(inputs, &ge_inputs);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Failed to transform MSTensor to Ge Tensor";
    return ret;
  }
  llm::LLMReq llm_req;
  TransLLMReq(req, &llm_req);
  MS_LOG(INFO) << "Start to call predict, req_id " << llm_req.GetReqId() << ", prompt_length "
               << llm_req.GetPromptLength() << ", prompt_cluster_id: " << llm_req.GetPromptClusterId()
               << ", decoder_cluster_id: " << llm_req.GetDecoderClusterId() << ", prefix id " << llm_req.GetPrefixId();
  std::vector<::ge::Tensor> ge_outputs;
  ret = Run(llm_req, ge_inputs, &ge_outputs, model_id);
  if (ret != kSuccess) {
    return ret;
  }
  for (size_t i = 0; i < ge_outputs.size(); i++) {
    auto &ge_tensor = ge_outputs[i];
    auto ms_tensor = ConvertGeTensorNoCopy(&ge_tensor);
    if (ms_tensor == nullptr) {
      MS_LOG(ERROR) << "Failed to converter output " << i << " GE Tensor to ME Tensor";
      return kLiteError;
    }
    MS_LOG(INFO) << "Output " << i << " shape " << ms_tensor.Shape() << ", datatype " << ms_tensor.DataType();
    outputs->push_back(ms_tensor);
  }
  return kSuccess;
}

Status LLMEnginePlugin::Predict(const std::vector<LLMReq> &req, const std::vector<MSTensor> &inputs,
                                std::vector<MSTensor> *outputs, uint64_t model_id) {
  if (outputs == nullptr) {
    MS_LOG(ERROR) << "Input argument outputs is nullptr";
    return kLiteError;
  }
  if (finalized_) {
    MS_LOG(ERROR) << "LLMEngine has been finalized";
    return kLiteLLMEngineFinalized;
  }
  if (!inited_) {
    MS_LOG(ERROR) << "LLMEngine has not been inited or inited failed";
    return kLiteError;
  }
  if (llm_engine_ == nullptr) {
    MS_LOG(ERROR) << "LLMEngine object is nullptr";
    return kLiteError;
  }
  std::vector<::ge::Tensor> ge_inputs;
  auto ret = MSTensorToGeTensor(inputs, &ge_inputs);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Failed to transform MSTensor to Ge Tensor";
    return ret;
  }
  MS_LOG(INFO) << "Start to call predict, requests: ";
  std::vector<llm::LLMReq> llm_reqs;
  (void)std::transform(req.begin(), req.end(), std::back_inserter(llm_reqs), [](const LLMReq &item) {
    llm::LLMReq llm_req;
    TransLLMReq(item, &llm_req);
    MS_LOG(INFO) << "req_id " << llm_req.GetReqId() << ", prompt_length " << llm_req.GetPromptLength()
                 << ", prompt_cluster_id: " << llm_req.GetPromptClusterId()
                 << ", decoder_cluster_id: " << llm_req.GetDecoderClusterId() << ", prefix id "
                 << llm_req.GetPrefixId();
    return llm_req;
  });
  std::vector<::ge::Tensor> ge_outputs;
  ret = Run(llm_reqs, ge_inputs, &ge_outputs, model_id);
  if (ret != kSuccess) {
    return ret;
  }
  for (size_t i = 0; i < ge_outputs.size(); i++) {
    auto &ge_tensor = ge_outputs[i];
    auto ms_tensor = ConvertGeTensorNoCopy(&ge_tensor);
    if (ms_tensor == nullptr) {
      MS_LOG(ERROR) << "Failed to converter output " << i << " GE Tensor to ME Tensor";
      return kLiteError;
    }
    MS_LOG(INFO) << "Output " << i << " shape " << ms_tensor.Shape() << ", datatype " << ms_tensor.DataType();
    outputs->push_back(ms_tensor);
  }
  return kSuccess;
}

Status LLMEnginePlugin::PullKV(const LLMReq &req, uint64_t model_id) {
  if (!inited_) {
    MS_LOG(ERROR) << "LLMEngine has not been inited or inited failed";
    return kLiteError;
  }
  if (llm_engine_ == nullptr) {
    MS_LOG(ERROR) << "LLMEngine object is nullptr";
    return kLiteError;
  }
  llm::LLMReq llm_req;
  TransLLMReq(req, &llm_req);
  MS_LOG(INFO) << "Start to call PullKv, req_id " << llm_req.GetReqId() << ", prompt_length "
               << llm_req.GetPromptLength() << ", prompt_cluster_id: " << llm_req.GetPromptClusterId()
               << ", decoder_cluster_id: " << llm_req.GetDecoderClusterId() << ", prefix_id " << llm_req.GetPrefixId()
               << ", model_id " << model_id;
  auto ge_ret = llm_engine_->PullKv(llm_req, model_id);
  return OnGeStatus(ge_ret, "PullKv", "return");
}

Status LLMEnginePlugin::MergeKV(const LLMReq &req, uint32_t batch_index, uint32_t batch_id, uint64_t model_id) {
  if (!inited_) {
    MS_LOG(ERROR) << "LLMEngine has not been inited or inited failed";
    return kLiteError;
  }
  if (llm_engine_ == nullptr) {
    MS_LOG(ERROR) << "LLMEngine object is nullptr";
    return kLiteError;
  }
  MS_LOG(INFO) << "Start to call MergeKV, req_id " << req.req_id << ", batch_index " << batch_index << ", batch_id "
               << batch_id << ", model_id " << model_id;
  auto ge_ret = llm_engine_->MergeKv(req.req_id, batch_index, batch_id, model_id);
  return OnGeStatus(ge_ret, "MergeKV", "return");
}

Status LLMEnginePlugin::CompleteRequest(const LLMReq &req) {
  if (finalized_) {
    MS_LOG(ERROR) << "LLMEngine has been finalized";
    return kLiteLLMEngineFinalized;
  }
  if (!inited_) {
    MS_LOG(ERROR) << "LLMEngine has not been inited or inited failed";
    return kLiteError;
  }
  if (llm_engine_ == nullptr) {
    MS_LOG(ERROR) << "LLMEngine object is nullptr";
    return kLiteError;
  }
  MS_LOG(INFO) << "Start to call llm::LLMEngine::LLMReqComplete, req_id " << req.req_id << ", prompt_length "
               << req.prompt_length << ", prompt_cluster_id: " << req.prompt_cluster_id;
  llm::LLMReq llm_req;
  TransLLMReq(req, &llm_req);
  auto ge_ret = llm_engine_->LLMReqComplete(llm_req);
  return OnGeStatus(ge_ret, "LLMReqComplete", "return");
}

LLMEngineStatus LLMEnginePlugin::FetchStatus() {
  if (finalized_) {
    MS_LOG(ERROR) << "LLMEngine has been finalized";
    return LLMEngineStatus();
  }
  if (!inited_) {
    MS_LOG(ERROR) << "LLMEngine has not been inited or inited failed";
    return LLMEngineStatus();
  }
  if (llm_engine_ == nullptr) {
    MS_LOG(ERROR) << "LLMEngine object is nullptr";
    return LLMEngineStatus();
  }
  LLMEngineStatus status;
  // When llm_engine_->fetchLLMEngineStatus() is implemented, it will be replaced by return of fetchLLMEngineStatus.
  status.empty_max_prompt_kv = 0;
  return status;
}

Status LLMEnginePlugin::PreloadPromptPrefix(const LLMReq &req, const std::vector<MSTensor> &inputs, uint64_t model_id) {
  if (finalized_) {
    MS_LOG(ERROR) << "LLMEngine has been finalized";
    return kLiteLLMEngineFinalized;
  }
  if (!inited_) {
    MS_LOG(ERROR) << "LLMEngine has not been inited or inited failed";
    return kLiteError;
  }
  if (llm_engine_ == nullptr) {
    MS_LOG(ERROR) << "LLMEngine object is nullptr";
    return kLiteError;
  }
  MS_LOG(INFO) << "Start to call llm::LLMEngine::PreloadPromptPrefix, req_id " << req.req_id << ", prompt_length "
               << req.prompt_length << ", prompt_cluster_id: " << req.prompt_cluster_id << ", prefix_id "
               << req.prefix_id << ", model_id " << model_id;
  std::vector<::ge::Tensor> ge_inputs;
  auto ret = MSTensorToGeTensor(inputs, &ge_inputs);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Failed to transform MSTensor to Ge Tensor";
    return ret;
  }
  llm::LLMReq llm_req;
  TransLLMReq(req, &llm_req);
  auto ge_ret = llm_engine_->PreloadPromptPrefix(llm_req, ge_inputs, model_id);
  return OnGeStatus(ge_ret, "PreloadPromptPrefix", "return");
}

Status LLMEnginePlugin::ReleasePromptPrefix(const LLMReq &req, uint64_t model_id) {
  if (finalized_) {
    MS_LOG(ERROR) << "LLMEngine has been finalized";
    return kLiteLLMEngineFinalized;
  }
  if (!inited_) {
    MS_LOG(ERROR) << "LLMEngine has not been inited or inited failed";
    return kLiteError;
  }
  if (llm_engine_ == nullptr) {
    MS_LOG(ERROR) << "LLMEngine object is nullptr";
    return kLiteError;
  }
  MS_LOG(INFO) << "Start to call llm::LLMEngine::ReleasePromptPrefix, req_id " << req.req_id << ", prompt_length "
               << req.prompt_length << ", prompt_cluster_id: " << req.prompt_cluster_id << ", prefix_id "
               << req.prefix_id << ", model_id " << model_id;
  llm::LLMReq llm_req;
  TransLLMReq(req, &llm_req);
  auto ge_ret = llm_engine_->ReleasePromptPrefix(llm_req, model_id);
  return OnGeStatus(ge_ret, "ReleasePromptPrefix", "return");
}

Status LLMEnginePlugin::LinkClusters(const std::vector<LLMClusterInfo> &clusters, std::vector<Status> *rets,
                                     int32_t timeout) {
  if (finalized_) {
    MS_LOG(ERROR) << "LLMEngine has been finalized";
    return kLiteLLMEngineFinalized;
  }
  if (!inited_) {
    MS_LOG(ERROR) << "LLMEngine has not been inited or inited failed";
    return kLiteError;
  }
  if (llm_engine_ == nullptr) {
    MS_LOG(ERROR) << "LLMEngine object is nullptr";
    return kLiteError;
  }
  if (rets == nullptr) {
    MS_LOG(ERROR) << "Input argument rets is nullptr";
    return kLiteError;
  }
  MS_LOG(INFO) << "Start to call llm::LLMEngine::LinkClusters, cluster size " << clusters.size();
  std::function<std::string(const LLMIpInfo &)> ip_info_as_str = [](const LLMIpInfo &info) {
    return std::to_string(info.ip) + ":" + std::to_string(info.port);
  };
  std::vector<llm::ClusterInfo> llm_clusters;
  for (size_t i = 0; i < clusters.size(); i++) {
    auto &cluster = clusters[i];
    MS_LOG(INFO) << "Cluster " << i << ", remote_cluster_id " << cluster.remote_cluster_id << ", remote_role_type "
                 << cluster.remote_role_type;
    MS_LOG(INFO) << "local ip infos: " << lite::VectorToStr(cluster.local_ip_infos, ip_info_as_str);
    MS_LOG(INFO) << "remote ip infos: " << lite::VectorToStr(cluster.remote_ip_infos, ip_info_as_str);
  }
  TransLLMClusterInfos(clusters, &llm_clusters);
  std::vector<ge::Status> ge_rets;
  auto ret = llm_engine_->LinkClusters(llm_clusters, ge_rets, timeout);
  if (!ge_rets.empty() && llm_clusters.size() != ge_rets.size()) {
    MS_LOG(ERROR) << "Cluster info size " << llm_clusters.size() << "!="
                  << " LinkClusters rets size " << ge_rets.size();
    return kLiteError;
  }
  for (size_t i = 0; i < ge_rets.size(); i++) {
    auto ge_ret = ge_rets[i];
    if (ge_ret != ge::GRAPH_SUCCESS) {
      rets->push_back(OnGeStatus(ge_ret, "LinkClusters", "return"));
      auto &cluster = clusters[i];
      MS_LOG(ERROR) << "Cluster " << i << " error occur, ge error code " << ge_ret << ", remote_cluster_id "
                    << cluster.remote_cluster_id << ", remote_role_type " << cluster.remote_role_type
                    << ", local ip infos: " << lite::VectorToStr(cluster.local_ip_infos, ip_info_as_str)
                    << "remote ip infos: " << lite::VectorToStr(cluster.remote_ip_infos, ip_info_as_str);
    } else {
      rets->push_back(kSuccess);
    }
  }
  return OnGeStatus(ret, "LinkClusters", "return");
}

Status LLMEnginePlugin::UnlinkClusters(const std::vector<LLMClusterInfo> &clusters, std::vector<Status> *rets,
                                       int32_t timeout) {
  if (finalized_) {
    MS_LOG(ERROR) << "LLMEngine has been finalized";
    return kLiteLLMEngineFinalized;
  }
  if (!inited_) {
    MS_LOG(ERROR) << "LLMEngine has not been inited or inited failed";
    return kLiteError;
  }
  if (llm_engine_ == nullptr) {
    MS_LOG(ERROR) << "LLMEngine object is nullptr";
    return kLiteError;
  }
  if (rets == nullptr) {
    MS_LOG(ERROR) << "Input argument rets is nullptr";
    return kLiteError;
  }
  MS_LOG(INFO) << "Start to call llm::LLMEngine::UnlinkClusters, cluster size " << clusters.size();
  std::function<std::string(const LLMIpInfo &)> ip_info_as_str = [](const LLMIpInfo &info) {
    return std::to_string(info.ip) + ":" + std::to_string(info.port);
  };
  std::vector<llm::ClusterInfo> llm_clusters;
  for (size_t i = 0; i < clusters.size(); i++) {
    auto &cluster = clusters[i];
    MS_LOG(INFO) << "Cluster " << i << ", remote_cluster_id " << cluster.remote_cluster_id << ", remote_role_type "
                 << cluster.remote_role_type;
    MS_LOG(INFO) << "local ip infos: " << lite::VectorToStr(cluster.local_ip_infos, ip_info_as_str);
    MS_LOG(INFO) << "remote ip infos: " << lite::VectorToStr(cluster.remote_ip_infos, ip_info_as_str);
  }
  TransLLMClusterInfos(clusters, &llm_clusters);
  std::vector<ge::Status> ge_rets;
  auto ret = llm_engine_->UnlinkClusters(llm_clusters, ge_rets, timeout);
  if (!ge_rets.empty() && llm_clusters.size() != ge_rets.size()) {
    MS_LOG(ERROR) << "Cluster info size " << llm_clusters.size() << "!="
                  << " UnlinkClusters rets size " << ge_rets.size();
    return kLiteError;
  }
  for (size_t i = 0; i < ge_rets.size(); i++) {
    auto ge_ret = ge_rets[i];
    if (ge_ret != ge::GRAPH_SUCCESS) {
      rets->push_back(OnGeStatus(ge_ret, "UnlinkClusters", "return"));
      auto &cluster = clusters[i];
      MS_LOG(ERROR) << "Cluster " << i << " error occur, ge error code " << ge_ret << ", remote_cluster_id "
                    << cluster.remote_cluster_id << ", remote_role_type " << cluster.remote_role_type
                    << ", local ip infos: " << lite::VectorToStr(cluster.local_ip_infos, ip_info_as_str)
                    << "remote ip infos: " << lite::VectorToStr(cluster.remote_ip_infos, ip_info_as_str);
    } else {
      rets->push_back(kSuccess);
    }
  }
  return OnGeStatus(ret, "UnlinkClusters", "return");
}

MSTensor LLMEnginePlugin::ConvertGeTensorNoCopy(::ge::Tensor *ge_tensor_ptr) {
  auto &ge_tensor = *ge_tensor_ptr;
  auto ge_tensor_desc = ge_tensor.GetTensorDesc();
  auto me_shape = transform::TransformUtil::ConvertGeShape(ge_tensor_desc.GetShape());
  if (ge_tensor_desc.GetPlacement() != ::ge::kPlacementHost) {
    MS_LOG(ERROR) << "It is not supported that graph output data's placement is device now.";
    return MSTensor(nullptr);
  }
  auto &&ge_data_uni = ge_tensor.ResetData();
  auto deleter = ge_data_uni.get_deleter();
  auto ge_data = ge_data_uni.release();
  if (ge_data == nullptr) {
    MS_LOG(ERROR) << "Ge data cannot be nullptr";
    return MSTensor(nullptr);
  }
  constexpr int64_t kTensorAlignBytes = 64;
  if (reinterpret_cast<uintptr_t>(ge_data) % kTensorAlignBytes != 0) {
    MS_LOG(ERROR) << "Skip zero-copy ge tensor " << reinterpret_cast<uintptr_t>(ge_data)
                  << ", bytes not aligned with expected.";
    return MSTensor(nullptr);
  }
  int64_t elem_num = 1;
  for (size_t i = 0; i < me_shape.size(); ++i) {
    elem_num *= me_shape[i];
  }
  auto tensor_data = std::make_shared<TensorRefData>(ge_data, elem_num, ge_tensor.GetSize(), me_shape.size(), deleter);
  auto type_id = transform::TransformUtil::ConvertGeDataType(ge_tensor_desc.GetDataType());
  auto tensor = std::make_shared<tensor::Tensor>(type_id, me_shape, tensor_data);
  auto tensor_impl = std::make_shared<TensorTensorImpl>(tensor);
  return MSTensor(tensor_impl);
}
}  // namespace mindspore
