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
#include "mindspore/lite/src/extendrt/cxx_api/dlutils.h"
#include "mindspore/core/load_mindir/load_model.h"
#include "mindspore/ccsrc/transform/graph_ir/transform_util.h"
#include "mindspore/lite/src/extendrt/utils/tensor_utils.h"
#include "mindspore/lite/src/common/common.h"
#include "mindspore/lite/src/extendrt/cxx_api/llm_engine/llm_engine_mock.h"

#define LLM_RUN_ASYNC

namespace mindspore {
class LLMEnginePlugin : public LLMEnginePluginBase {
 public:
  LLMEnginePlugin();
  ~LLMEnginePlugin();
  Status Init(const std::vector<LLMEngineModelInfo> &model_infos, LLMRole role, uint64_t cluster_id,
              const std::map<std::string, std::string> &options) override;
  void Finalize() override;
  Status Predict(const LLMReq &req, const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs) override;
  Status CompleteRequest(const LLMReq &req) override;
  LLMEngineStatus FetchStatus() override;

 private:
  LLMRole role_ = kLLMRolePrompt;
  uint64_t cluster_id_ = 0;
  std::map<std::string, std::string> options_;
  std::shared_ptr<::llm::LLMEngine> llm_engine_ = nullptr;

  MSTensor ConvertGeTensorNoCopy(::ge::Tensor *ge_tensor_ptr);
  Status Run(llm::LLMReq *req, const std::vector<::ge::Tensor> &ge_inputs, std::vector<::ge::Tensor> *ge_outputs);
  Status InitInputOptions(const std::vector<LLMEngineModelInfo> &model_infos);
  void TransLLMReq(const LLMReq &req, llm::LLMReq *llm_req) const;
};

LLMEnginePluginBase *CreateLLMEnginePlugin() { return new LLMEnginePlugin(); }

LLMEnginePlugin::LLMEnginePlugin() {}

LLMEnginePlugin::~LLMEnginePlugin() { LLMEnginePlugin::Finalize(); }

Status LLMEnginePlugin::InitInputOptions(const std::vector<LLMEngineModelInfo> &model_infos) {
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
  for (auto &item : model_infos[0].input_shapes) {
    input_shapes += shape_as_string(item) + ";";
  }
  for (auto &item : model_infos[0].input_dtypes) {
    input_dtypes += dtype_as_string(item) + ";";
  }
  for (auto &item : model_infos[0].ref_input_shapes) {
    ref_input_shapes += shape_as_string(item) + ";";
  }
  for (auto &item : model_infos[0].ref_input_dtypes) {
    ref_input_dtypes += dtype_as_string(item) + ";";
  }
  auto erase_comma = [](const std::string &str) { return str.empty() ? str : str.substr(0, str.size() - 1); };
  options_["llm.InputShapes"] = erase_comma(input_shapes);
  options_["llm.InputDtypes"] = erase_comma(input_dtypes);
  options_["llm.RefInputShapes"] = erase_comma(ref_input_shapes);
  options_["llm.RefInputDtypes"] = erase_comma(ref_input_dtypes);
  options_["llm.OutputNums"] = std::to_string(model_infos[0].output_count);
  return kSuccess;
}

Status LLMEnginePlugin::Init(const std::vector<LLMEngineModelInfo> &model_infos, LLMRole role, uint64_t cluster_id,
                             const std::map<std::string, std::string> &options) {
  if (model_infos.empty()) {
    MS_LOG(ERROR) << "Model infos cannot be empty";
    return kLiteError;
  }
  if (llm_engine_ != nullptr) {
    MS_LOG(ERROR) << "LLMEngine has been inited";
    return kLiteError;
  }
  MS_LOG(INFO) << "LLMEngine Init begin";
  role_ = role;
  cluster_id_ = cluster_id;
  options_ = options;
  if (InitInputOptions(model_infos) != kSuccess) {
    return kLiteError;
  }
  options_["llm.Role"] = role == LLMRole::kLLMRolePrompt ? "Prompt" : "Decoder";
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
  std::map<ge::AscendString, ge::AscendString> init_options;
  for (auto &option : options_) {
    init_options[ge::AscendString(option.first.c_str())] = ge::AscendString(option.second.c_str());
    MS_LOG(INFO) << "LLMEngineInitialize option " << option.first << " = " << option.second;
  }
  auto llm_engine = std::make_shared<llm::LLMEngine>(cluster_id);
  MS_LOG(INFO) << "Start to call llm::LLMEngine::LLMEngineInitialize";
  auto ge_status = llm_engine->LLMEngineInitialize(model_buffers, init_options);
  if (ge_status != ge::GRAPH_SUCCESS) {
    MS_LOG(ERROR) << "Failed to call  LLMEngineInitialize";
    return kLiteError;
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
    if (ge_status != ge::GRAPH_SUCCESS) {
      MS_LOG(ERROR) << "Failed to call LLMEngineFinalize";
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
      MS_LOG(ERROR) << "Failed to call llm::LLMEngine::RunPrompt";
      return kLiteError;
    }
  } else {
    MS_LOG(INFO) << "Start to call llm::LLMEngine::RunDecoder";
    auto ret = llm_engine_->RunDecoder(*llm_req, ge_inputs, *outputs);
    if (ret != ge::GRAPH_SUCCESS) {
      MS_LOG(ERROR) << "Failed to call llm::LLMEngine::RunDecoder";
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

  bool is_finished = false;
  std::promise<void> promise;
  auto call_back = [outputs, &is_finished, &promise](ge::Status ge_status, const std::vector<ge::Tensor> &ge_outputs) {
    if (ge_status == ge::GRAPH_SUCCESS) {
      *outputs = ge_outputs;
      is_finished = true;
    } else {
      MS_LOG(ERROR) << "RunPromptAsync or RunDecoderAsync failed, status: " << ge_status;
    }
    promise.set_value();
    return;
  };
  if (role_ == kLLMRolePrompt) {
    MS_LOG(INFO) << "Start to call llm::LLMEngine::RunPromptAsync";
    auto ret = llm_engine_->RunPromptAsync(*llm_req, ge_inputs, call_back);
    if (ret != ge::GRAPH_SUCCESS) {
      MS_LOG(ERROR) << "Failed to call llm::LLMEngine::RunPromptAsync";
      return kLiteError;
    }
  } else {
    MS_LOG(INFO) << "Start to call llm::LLMEngine::RunDecoderAsync";
    auto ret = llm_engine_->RunDecoderAsync(*llm_req, ge_inputs, call_back);
    if (ret != ge::GRAPH_SUCCESS) {
      MS_LOG(ERROR) << "Failed to call llm::LLMEngine::RunDecoderAsync";
      return kLiteError;
    }
  }
  auto future = promise.get_future();
  future.wait();

  auto time_cost =
    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - time_start).count();
  MS_LOG(INFO) << "Call LLMEngine RunPromptAsync or RunDecoderAsync Success in " << time_cost << " us, role "
               << (role_ == LLMRole::kLLMRolePrompt ? "Prompt" : "Decoder") << ", outputs num is: " << outputs->size();
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
}

Status LLMEnginePlugin::Predict(const LLMReq &req, const std::vector<MSTensor> &inputs,
                                std::vector<MSTensor> *outputs) {
  if (llm_engine_ == nullptr) {
    MS_LOG(ERROR) << "LLMEngine has not been inited or inited failed";
    return kLiteError;
  }
  std::vector<::ge::Tensor> ge_inputs;
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
    ge_inputs.emplace_back(tensor);
  }
  llm::LLMReq llm_req;
  TransLLMReq(req, &llm_req);
  MS_LOG(INFO) << "Start to call predict, req_id " << llm_req.GetReqId() << ", prompt_length "
               << llm_req.GetPromptLength() << ", prompt_cluster_id: " << llm_req.GetPromptClusterId()
               << ", decoder_cluster_id: " << llm_req.GetDecoderClusterId();
  std::vector<::ge::Tensor> ge_outputs;
  auto ret = Run(&llm_req, ge_inputs, &ge_outputs);
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

Status LLMEnginePlugin::CompleteRequest(const LLMReq &req) {
  if (llm_engine_ == nullptr) {
    MS_LOG(ERROR) << "LLMEngine has not been inited or inited failed";
    return kLiteError;
  }
  MS_LOG(INFO) << "Start to call llm::LLMEngine::LLMReqComplete, req_id " << req.req_id << ", prompt_length "
               << req.prompt_length << ", prompt_cluster_id: " << req.prompt_cluster_id;
  llm::LLMReq llm_req;
  TransLLMReq(req, &llm_req);
  auto ret = llm_engine_->LLMReqComplete(llm_req);
  if (ret != ge::GRAPH_SUCCESS) {
    MS_LOG(ERROR) << "Failed to call llm::LLMEngine::LLMReqComplete";
    return kLiteError;
  }
  return kSuccess;
}

LLMEngineStatus LLMEnginePlugin::FetchStatus() {
  if (llm_engine_ == nullptr) {
    MS_LOG(ERROR) << "LLMEngine has not been inited or inited failed";
    return LLMEngineStatus();
  }
  LLMEngineStatus status;
  // When llm_engine_->fetchLLMEngineStatus() is implemented, it will be replaced by return of fetchLLMEngineStatus.
  status.empty_max_prompt_kv = 0;
  return status;
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
