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
#include "mindspore/lite/src/extendrt/cxx_api/llm_engine/llm_engine_mock.h"
#include "common/ge_common/ge_inner_error_codes.h"

#define LLM_RUN_ASYNC

namespace mindspore {
class LLMEnginePlugin : public LLMEnginePluginBase {
 public:
  LLMEnginePlugin();
  ~LLMEnginePlugin();
  Status Init(const std::vector<LLMEngineModelInfo> &model_infos, LLMRole role, uint64_t cluster_id,
              const std::map<std::string, std::string> &options, const std::string &batch_mode,
              const LLMEngineModelInfo &postprocess_model) override;
  void Finalize() override;
  Status Predict(const LLMReq &req, const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs) override;

  Status Predict(const std::vector<LLMReq> &req, const std::vector<MSTensor> &inputs,
                 std::vector<MSTensor> *outputs) override;
  Status CompleteRequest(const LLMReq &req) override;
  LLMEngineStatus FetchStatus() override;
  Status PreloadPromptPrefix(const LLMReq &req, const std::vector<MSTensor> &inputs) override;
  Status ReleasePromptPrefix(const LLMReq &req) override;

  Status PullKV(const LLMReq &req) override;
  Status MergeKV(const LLMReq &req, uint32_t batch_index) override;

 private:
  LLMRole role_ = kLLMRolePrompt;
  uint64_t cluster_id_ = 0;
  std::map<std::string, std::string> options_;
  std::shared_ptr<::llm::LLMEngine> llm_engine_ = nullptr;
  bool finalized_ = false;

  MSTensor ConvertGeTensorNoCopy(::ge::Tensor *ge_tensor_ptr);
  Status Run(llm::LLMReq *req, const std::vector<::ge::Tensor> &ge_inputs, std::vector<::ge::Tensor> *ge_outputs);
  Status CheckModelInfos(const std::vector<LLMEngineModelInfo> &model_infos);
  void InitInputOptions(const LLMEngineModelInfo &model_info, bool postprocess);
  void TransLLMReq(const LLMReq &req, llm::LLMReq *llm_req) const;
  Status MSTensorToGeTensor(const std::vector<MSTensor> &inputs, std::vector<::ge::Tensor> *ge_inputs);
  Status OnGeStatus(ge::Status ge_status, const std::string &func_s, const std::string &phase);
};

LLMEnginePluginBase *CreateLLMEnginePlugin() { return new LLMEnginePlugin(); }

LLMEnginePlugin::LLMEnginePlugin() {}

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

void LLMEnginePlugin::InitInputOptions(const LLMEngineModelInfo &model_info, bool postprocess) {
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
  auto erase_comma = [](const std::string &str) { return str.empty() ? str : str.substr(0, str.size() - 1); };
  if (!postprocess) {
    options_["llm.InputShapes"] = erase_comma(input_shapes);
    options_["llm.InputDtypes"] = erase_comma(input_dtypes);
    options_["llm.RefInputShapes"] = erase_comma(ref_input_shapes);
    options_["llm.RefInputDtypes"] = erase_comma(ref_input_dtypes);
    options_["llm.OutputNums"] = std::to_string(model_info.output_count);
  } else {
    options_["llm.PostProcessInputShapes"] = erase_comma(input_shapes);
    options_["llm.PostProcessInputDtypes"] = erase_comma(input_dtypes);
    options_["llm.PostProcessOutputNums"] = std::to_string(model_info.output_count);
    options_["llm.PostProcessOmCachePath"] = model_info.weight_dir;
  }
}

Status LLMEnginePlugin::OnGeStatus(ge::Status ge_status, const std::string &func_s, const std::string &phase) {
  Status lite_status;
  if (ge_status == ge::GRAPH_SUCCESS) {
    MS_LOG(INFO) << "End call llm::LLMEngine::" << func_s;
    lite_status = kSuccess;
  } else if (ge_status == ge::LLM_WAIT_PROC_TIMEOUT) {
    MS_LOG(WARNING) << "Failed to call llm::LLMEngine::" << func_s << ", " << phase << " status: LLM_WAIT_PROC_TIMEOUT";
    lite_status = kLiteLLMWaitProcessTimeOut;
  } else if (ge_status == ge::LLM_KV_CACHE_NOT_EXIST) {
    MS_LOG(WARNING) << "Failed to call llm::LLMEngine::" << func_s << phase << " status: LLM_KV_CACHE_NOT_EXIST";
    lite_status = kLiteLLMKVCacheNotExist;
  } else if (ge_status == ge::LLM_REPEAT_REQUEST) {
    MS_LOG(ERROR) << "Failed to call llm::LLMEngine::" << func_s << phase << " status: LLM_REPEAT_REQUEST";
    lite_status = kLiteLLMRepeatRequest;
  } else if (ge_status == ge::LLM_REQUEST_ALREADY_COMPLETED) {
    MS_LOG(ERROR) << "Failed to call llm::LLMEngine::" << func_s << phase << " receive LLM_REQUEST_ALREADY_COMPLETED";
    lite_status = kLiteLLMRequestAlreadyCompleted;
  } else if (ge_status == ge::LLM_ENGINE_FINALIZED) {
    MS_LOG(ERROR) << "Failed to call llm::LLMEngine::" << func_s << phase << " status: LLM_ENGINE_FINALIZED";
    lite_status = kLiteLLMEngineFinalized;
  } else if (ge_status == ge::LLM_PARAM_INVALID) {
    MS_LOG(ERROR) << "Failed to call llm::LLMEngine::" << func_s << phase << " status: LLM_PARAM_INVALID";
    lite_status = kLiteParamInvalid;
  } else {
    MS_LOG(ERROR) << "Failed to call llm::LLMEngine::" << func_s << phase << " status: " << ge_status;
    lite_status = kLiteError;
  }
  return lite_status;
}

Status LLMEnginePlugin::Init(const std::vector<LLMEngineModelInfo> &model_infos, LLMRole role, uint64_t cluster_id,
                             const std::map<std::string, std::string> &options, const std::string &batch_mode,
                             const LLMEngineModelInfo &postprocess_model) {
  if (model_infos.empty()) {
    MS_LOG(ERROR) << "Model infos cannot be empty";
    return kLiteError;
  }
  if (finalized_) {
    MS_LOG(ERROR) << "LLMEngine has been finalized";
    return kLiteLLMEngineFinalized;
  }
  if (llm_engine_ != nullptr) {
    MS_LOG(ERROR) << "LLMEngine has been inited";
    return kLiteError;
  }
  MS_LOG(INFO) << "LLMEngine Init begin";
  role_ = role;
  cluster_id_ = cluster_id;
  options_ = options;
  if (CheckModelInfos(model_infos) != kSuccess) {
    return kLiteError;
  }
  InitInputOptions(model_infos[0], false);
  options_["llm.Role"] = role == LLMRole::kLLMRolePrompt ? "Prompt" : "Decoder";
  options_["llm.batch_mode"] = batch_mode;
  auto option_it = options_.find("llm.OmCachePath");
  if (option_it == options_.end()) {
    std::string cache_path;
    for (size_t i = 0; i < model_infos.size(); i++) {
      cache_path += model_infos[i].weight_dir;
      if (i + 1 < model_infos.size()) {
        cache_path += ";";
      }
    }
    MS_LOG(INFO) << "Add option llm.OmCachePath to " << cache_path;
    options_["llm.OmCachePath"] = cache_path;
  }
  std::vector<ge::ModelBufferData> model_buffers;
  for (auto &item : model_infos) {
    ge::ModelBufferData buff;
    buff.data = std::shared_ptr<uint8_t>(reinterpret_cast<uint8_t *>(item.om_data->data_c()), [](uint8_t *) {});
    buff.length = item.om_data->Size();
    model_buffers.push_back(buff);
    MS_LOG(INFO) << "Model " << item.name << ", model buffer size " << item.om_data->Size();
  }
  if (postprocess_model.om_data != nullptr) {
    InitInputOptions(postprocess_model, true);
  }
  std::map<ge::AscendString, ge::AscendString> init_options;
  for (auto &option : options_) {
    init_options[ge::AscendString(option.first.c_str())] = ge::AscendString(option.second.c_str());
    MS_LOG(INFO) << "LLMEngineInitialize option " << option.first << " = " << option.second;
  }
  auto llm_engine = std::make_shared<llm::LLMEngine>(cluster_id);
  if (postprocess_model.om_data == nullptr) {
    MS_LOG(INFO) << "Start to call llm::LLMEngine::LLMEngineInitialize";
    auto ge_status = llm_engine->LLMEngineInitialize(model_buffers, init_options);
    if (ge_status != ge::GRAPH_SUCCESS) {
      MS_LOG(ERROR) << "Failed to call LLMEngineInitialize, status: " << ge_status;
      return kLiteError;
    }
  } else {
    std::map<ge::AscendString, std::vector<ge::ModelBufferData>> model_buffers_map;
    model_buffers_map["inference"] = model_buffers;
    ge::ModelBufferData postprocess_buff;
    postprocess_buff.data =
      std::shared_ptr<uint8_t>(reinterpret_cast<uint8_t *>(postprocess_model.om_data->data_c()), [](uint8_t *) {});
    postprocess_buff.length = postprocess_model.om_data->Size();
    MS_LOG(INFO) << "Model " << postprocess_model.name << ", model buffer size " << postprocess_model.om_data->Size();
    model_buffers_map["postprocess"] = {postprocess_buff};

    MS_LOG(INFO) << "Start to call llm::LLMEngine::LLMEngineInitializeV2";
    auto ge_status = llm_engine->LLMEngineInitializeV2(model_buffers_map, init_options);
    if (ge_status != ge::GRAPH_SUCCESS) {
      MS_LOG(ERROR) << "Failed to call LLMEngineInitializeV2, status: " << ge_status;
      return kLiteError;
    }
  }
  llm_engine_ = llm_engine;
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

#ifndef LLM_RUN_ASYNC
Status LLMEnginePlugin::Run(llm::LLMReq *llm_req, const std::vector<::ge::Tensor> &ge_inputs,
                            std::vector<::ge::Tensor> *outputs) {
  auto time_start = std::chrono::system_clock::now();

  if (role_ == kLLMRolePrompt) {
    MS_LOG(INFO) << "Start to call llm::LLMEngine::RunPrompt";
    auto ret = llm_engine_->RunPrompt(*llm_req, ge_inputs, *outputs);
    if (ret != ge::GRAPH_SUCCESS) {
      MS_LOG(ERROR) << "Failed to call llm::LLMEngine::RunPrompt, status: " << ret;
      return kLiteError;
    }
  } else {
    MS_LOG(INFO) << "Start to call llm::LLMEngine::RunDecoder";
    auto ret = llm_engine_->RunDecoder(*llm_req, ge_inputs, *outputs);
    if (ret != ge::GRAPH_SUCCESS) {
      MS_LOG(ERROR) << "Failed to call llm::LLMEngine::RunDecoder, status: " << ret;
      return kLiteError;
    }
  }
  auto time_cost =
    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - time_start).count();
  MS_LOG(INFO) << "Call LLMEngine RunPrompt or RunDecoder Success in " << time_cost << " us, role "
               << (role_ == LLMRole::kLLMRolePrompt ? "Prompt" : "Decoder") << ", outputs num is: " << outputs->size();
  return kSuccess;
}

#else

Status LLMEnginePlugin::Run(llm::LLMReq *llm_req, const std::vector<::ge::Tensor> &ge_inputs,
                            std::vector<::ge::Tensor> *outputs) {
  auto time_start = std::chrono::system_clock::now();
  std::promise<void> promise;
  Status lite_status = kSuccess;
  auto call_back = [outputs, &promise, &lite_status, this](ge::Status ge_status,
                                                           const std::vector<ge::Tensor> &ge_outputs) {
    if (ge_status == ge::GRAPH_SUCCESS) {
      *outputs = ge_outputs;
    } else {
      auto func_s = role_ == kLLMRolePrompt ? "RunPromptAsync" : "RunDecoderAsync";
      lite_status = OnGeStatus(ge_status, func_s, "callback");
    }
    promise.set_value();
    return;
  };
  if (role_ == kLLMRolePrompt) {
    MS_LOG(INFO) << "Start to call llm::LLMEngine::RunPromptAsync";
    auto ge_status = llm_engine_->RunPromptAsync(*llm_req, ge_inputs, call_back);
    if (ge_status != ge::GRAPH_SUCCESS) {
      return OnGeStatus(ge_status, "RunPromptAsync", "return");
    }
  } else {
    MS_LOG(INFO) << "Start to call llm::LLMEngine::RunDecoderAsync";
    auto ge_status = llm_engine_->RunDecoderAsync(*llm_req, ge_inputs, call_back);
    if (ge_status != ge::GRAPH_SUCCESS) {
      return OnGeStatus(ge_status, "RunPromptAsync", "return");
    }
  }
  auto future = promise.get_future();
  future.wait();
  auto time_cost =
    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - time_start).count();
  auto role = (role_ == LLMRole::kLLMRolePrompt ? "Prompt" : "Decoder");
  if (lite_status != kSuccess) {
    MS_LOG(WARNING) << "Call LLMEngine RunPromptAsync or RunDecoderAsync Failed, time cost " << time_cost
                    << " us, role " << role;
    return lite_status;
  }
  MS_LOG(INFO) << "Call LLMEngine RunPromptAsync or RunDecoderAsync Success in " << time_cost << " us, role " << role
               << ", outputs num is: " << outputs->size();
  return kSuccess;
}
#endif

void LLMEnginePlugin::TransLLMReq(const LLMReq &req, llm::LLMReq *llm_req_ptr) const {
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

Status LLMEnginePlugin::Predict(const LLMReq &req, const std::vector<MSTensor> &inputs,
                                std::vector<MSTensor> *outputs) {
  if (finalized_) {
    MS_LOG(ERROR) << "LLMEngine has been finalized";
    return kLiteLLMEngineFinalized;
  }
  if (llm_engine_ == nullptr) {
    MS_LOG(ERROR) << "LLMEngine has not been inited or inited failed";
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
  ret = Run(&llm_req, ge_inputs, &ge_outputs);
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
                                std::vector<MSTensor> *outputs) {
  if (llm_engine_ == nullptr) {
    MS_LOG(ERROR) << "LLMEngine has not been inited or inited failed";
    return kLiteError;
  }
  std::vector<::ge::Tensor> ge_inputs;
  auto ret = MSTensorToGeTensor(inputs, &ge_inputs);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Failed to transform MSTensor to Ge Tensor";
    return ret;
  }
  std::vector<uint64_t> req_ids;
  (void)std::transform(req.begin(), req.end(), std::back_inserter(req_ids),
                       [](const LLMReq &item) { return item.req_id; });
  MS_LOG(INFO) << "Start to call predict, req_ids " << req_ids;
  std::vector<::ge::Tensor> ge_outputs;
  auto ge_ret = llm_engine_->RunDecoder(req_ids, ge_inputs, ge_outputs);
  if (ge_ret != ge::GRAPH_SUCCESS) {
    return OnGeStatus(ge_ret, "RunDecoder", "return");
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

Status LLMEnginePlugin::PullKV(const LLMReq &req) {
  if (llm_engine_ == nullptr) {
    MS_LOG(ERROR) << "LLMEngine has not been inited or inited failed";
    return kLiteError;
  }
  llm::LLMReq llm_req;
  TransLLMReq(req, &llm_req);
  MS_LOG(INFO) << "Start to call PullKv, req_id " << llm_req.GetReqId() << ", prompt_length "
               << llm_req.GetPromptLength() << ", prompt_cluster_id: " << llm_req.GetPromptClusterId()
               << ", decoder_cluster_id: " << llm_req.GetDecoderClusterId() << ", prefix id " << llm_req.GetPrefixId();
  auto ge_ret = llm_engine_->PullKv(llm_req);
  return OnGeStatus(ge_ret, "PullKv", "return");
}

Status LLMEnginePlugin::MergeKV(const LLMReq &req, uint32_t batch_index) {
  if (llm_engine_ == nullptr) {
    MS_LOG(ERROR) << "LLMEngine has not been inited or inited failed";
    return kLiteError;
  }
  MS_LOG(INFO) << "Start to call MergeKV, req_id " << req.req_id << ", batch_index " << batch_index;
  auto ge_ret = llm_engine_->MergeKv(req.req_id, batch_index);
  return OnGeStatus(ge_ret, "MergeKV", "return");
}

Status LLMEnginePlugin::CompleteRequest(const LLMReq &req) {
  if (finalized_) {
    MS_LOG(ERROR) << "LLMEngine has been finalized";
    return kLiteLLMEngineFinalized;
  }
  if (llm_engine_ == nullptr) {
    MS_LOG(ERROR) << "LLMEngine has not been inited or inited failed";
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
  if (llm_engine_ == nullptr) {
    MS_LOG(ERROR) << "LLMEngine has not been inited or inited failed";
    return LLMEngineStatus();
  }
  LLMEngineStatus status;
  // llm::LLMEngineStatus llm_status = llm_engine_->fetchLLMEngineStatus();
  // status.empty_max_prompt_kv = llm_status.empty_max_prompt_kv;
  return status;
}

Status LLMEnginePlugin::PreloadPromptPrefix(const LLMReq &req, const std::vector<MSTensor> &inputs) {
  if (finalized_) {
    MS_LOG(ERROR) << "LLMEngine has been finalized";
    return kLiteLLMEngineFinalized;
  }
  if (llm_engine_ == nullptr) {
    MS_LOG(ERROR) << "LLMEngine has not been inited or inited failed";
    return kLiteError;
  }
  MS_LOG(INFO) << "Start to call llm::LLMEngine::PreloadPromptPrefix, req_id " << req.req_id << ", prompt_length "
               << req.prompt_length << ", prompt_cluster_id: " << req.prompt_cluster_id << ", prefix_id "
               << req.prefix_id;
  std::vector<::ge::Tensor> ge_inputs;
  auto ret = MSTensorToGeTensor(inputs, &ge_inputs);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Failed to transform MSTensor to Ge Tensor";
    return ret;
  }
  llm::LLMReq llm_req;
  TransLLMReq(req, &llm_req);
  auto ge_ret = llm_engine_->PreloadPromptPrefix(llm_req, ge_inputs);
  return OnGeStatus(ge_ret, "PreloadPromptPrefix", "return");
}

Status LLMEnginePlugin::ReleasePromptPrefix(const LLMReq &req) {
  if (finalized_) {
    MS_LOG(ERROR) << "LLMEngine has been finalized";
    return kLiteLLMEngineFinalized;
  }
  if (llm_engine_ == nullptr) {
    MS_LOG(ERROR) << "LLMEngine has not been inited or inited failed";
    return kLiteError;
  }
  MS_LOG(INFO) << "Start to call llm::LLMEngine::ReleasePromptPrefix, req_id " << req.req_id << ", prompt_length "
               << req.prompt_length << ", prompt_cluster_id: " << req.prompt_cluster_id << ", prefix_id "
               << req.prefix_id;
  llm::LLMReq llm_req;
  TransLLMReq(req, &llm_req);
  auto ge_ret = llm_engine_->ReleasePromptPrefix(llm_req);
  return OnGeStatus(ge_ret, "ReleasePromptPrefix", "return");
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
