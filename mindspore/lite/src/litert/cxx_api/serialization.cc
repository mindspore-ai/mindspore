/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "include/api/serialization.h"
#include <algorithm>
#include <queue>
#include "include/api/graph.h"
#include "include/api/types.h"
#include "include/model.h"
#include "src/litert/cxx_api/expression/net_impl.h"
#include "src/litert/cxx_api/graph/graph_data.h"
#include "src/litert/cxx_api/model/model_impl.h"
#include "src/litert/cxx_api/converters.h"
#include "src/common/log_adapter.h"
#include "src/litert/lite_session.h"

namespace mindspore {
std::function<int(void *)> ExpressionCallback;

Key::Key(const char *dec_key, size_t key_len) {
  len = 0;
  if (key_len >= max_key_len) {
    MS_LOG(ERROR) << "Invalid key len " << key_len << " is more than max key len " << max_key_len;
    return;
  }

  (void)memcpy(key, dec_key, key_len);
  len = key_len;
}

Status Serialization::Load(const void *model_data, size_t data_size, ModelType model_type, Graph *graph,
                           const Key &dec_key, const std::vector<char> &dec_mode) {
  if (dec_key.len != 0 || CharToString(dec_mode) != kDecModeAesGcm) {
    MS_LOG(ERROR) << "Unsupported Feature.";
    return kLiteError;
  }

  if (model_data == nullptr) {
    MS_LOG(ERROR) << "model data is nullptr.";
    return kLiteNullptr;
  }
  if (graph == nullptr) {
    MS_LOG(ERROR) << "graph is nullptr.";
    return kLiteNullptr;
  }
  if (model_type != kMindIR && model_type != kMindIR_Lite) {
    MS_LOG(ERROR) << "Unsupported IR.";
    return kLiteInputParamInvalid;
  }

  size_t lite_buf_size = 0;
  char *lite_buf = nullptr;
  lite::LiteSession session;
  auto buf_model_type = session.LoadModelByBuff(reinterpret_cast<const char *>(model_data), data_size, &lite_buf,
                                                &lite_buf_size, model_type);
  if (buf_model_type == mindspore::ModelType::kUnknownType || lite_buf == nullptr) {
    MS_LOG(ERROR) << "Invalid model_buf";
    return kLiteNullptr;
  }
  auto model = std::shared_ptr<lite::Model>(lite::Model::Import(static_cast<const char *>(lite_buf), data_size));
  if (model == nullptr) {
    MS_LOG(ERROR) << "New model failed.";
    return kLiteNullptr;
  }
  if (buf_model_type == mindspore::ModelType::kMindIR) {
    free(lite_buf);
    lite_buf = nullptr;
  }
  auto graph_data = std::shared_ptr<Graph::GraphData>(new (std::nothrow) Graph::GraphData(model));
  if (graph_data == nullptr) {
    MS_LOG(ERROR) << "New graph data failed.";
    return kLiteMemoryFailed;
  }
  *graph = Graph(graph_data);
  return kSuccess;
}

Status Serialization::Load(const std::vector<char> &file, ModelType model_type, Graph *graph, const Key &dec_key,
                           const std::vector<char> &dec_mode) {
  if (dec_key.len != 0 || CharToString(dec_mode) != kDecModeAesGcm) {
    MS_LOG(ERROR) << "Unsupported Feature.";
    return kLiteError;
  }

  if (graph == nullptr) {
    MS_LOG(ERROR) << "graph is nullptr.";
    return kLiteNullptr;
  }
  if (model_type != kMindIR && model_type != kMindIR_Lite) {
    MS_LOG(ERROR) << "Unsupported IR.";
    return kLiteInputParamInvalid;
  }

  std::string filename(file.data(), file.size());
  if (filename.size() > static_cast<size_t>((std::numeric_limits<int>::max)())) {
    MS_LOG(ERROR) << "file name is too long.";
    return kLiteInputParamInvalid;
  }
  auto pos = filename.find_last_of('.');
  if (pos == std::string::npos || filename.substr(pos + 1) != "ms") {
    filename = filename + ".ms";
  }

  size_t model_size;
  lite::LiteSession session;
  auto model_buf = session.LoadModelByPath(filename, model_type, &model_size, false);
  if (model_buf == nullptr) {
    MS_LOG(ERROR) << "Read model file failed";
    return kLiteNullptr;
  }
  if (graph->IsExecutable()) {
    auto model =
      std::shared_ptr<lite::Model>(lite::ImportFromBuffer(static_cast<const char *>(model_buf), model_size, true));
    if (model == nullptr) {
      MS_LOG(ERROR) << "New model failed.";
      return kLiteNullptr;
    }
    auto graph_data = std::shared_ptr<Graph::GraphData>(new (std::nothrow) Graph::GraphData(model));
    if (graph_data == nullptr) {
      MS_LOG(ERROR) << "New graph data failed.";
      return kLiteMemoryFailed;
    }
    *graph = Graph(graph_data);
    return kSuccess;
  } else {
    auto loader = CreateExpressionLoader();
    if (loader == nullptr) {
      MS_LOG(ERROR) << "Unsupported Feature.";
      delete[] model_buf;
      return kLiteError;
    }
    (void)loader(model_buf, graph);
    delete[] model_buf;
    return kSuccess;
  }
}

Status Serialization::Load(const std::vector<std::vector<char>> &files, ModelType model_type,
                           std::vector<Graph> *graphs, const Key &dec_key, const std::vector<char> &dec_mode) {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return kLiteError;
}

Status Serialization::SetParameters(const std::map<std::vector<char>, Buffer> &parameters, Model *model) {
  MS_LOG(ERROR) << "Unsupported feature.";
  return kMEFailed;
}

Status Serialization::ExportModel(const Model &model, ModelType model_type, Buffer *model_data,
                                  QuantizationType quantization_type, bool export_inference_only,
                                  const std::vector<std::vector<char>> &output_tensor_name) {
  if (model.impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return kLiteUninitializedObj;
  }
  if (!model.impl_->IsTrainModel()) {
    MS_LOG(ERROR) << "Model is not TrainModel.";
    return kLiteError;
  }
  if (model_data == nullptr) {
    MS_LOG(ERROR) << "model_data is nullptr.";
    return kLiteParamInvalid;
  }
  if (model_type != kMindIR && model_type != kMindIR_Lite) {
    MS_LOG(ERROR) << "Unsupported Export Format " << model_type;
    return kLiteParamInvalid;
  }
  if (model.impl_->session_ == nullptr) {
    MS_LOG(ERROR) << "Model session is nullptr.";
    return kLiteError;
  }
  auto ret = model.impl_->session_->Export(model_data, export_inference_only ? lite::MT_INFERENCE : lite::MT_TRAIN,
                                           A2L_ConvertQT(quantization_type), lite::FT_FLATBUFFERS,
                                           VectorCharToString(output_tensor_name));

  return (ret == mindspore::lite::RET_OK) ? kSuccess : kLiteError;
}

Status Serialization::ExportModel(const Model &model, ModelType model_type, const std::vector<char> &model_file,
                                  QuantizationType quantization_type, bool export_inference_only,
                                  const std::vector<std::vector<char>> &output_tensor_name) {
  if (model.impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return kLiteUninitializedObj;
  }
  if (!model.impl_->IsTrainModel()) {
    MS_LOG(ERROR) << "Model is not TrainModel.";
    return kLiteError;
  }
  if (model_type != kMindIR && model_type != kMindIR_Lite) {
    MS_LOG(ERROR) << "Unsupported Export Format " << model_type;
    return kLiteParamInvalid;
  }
  if (model.impl_->session_ == nullptr) {
    MS_LOG(ERROR) << "Model session is nullptr.";
    return kLiteError;
  }
  auto ret = model.impl_->session_->Export(
    CharToString(model_file), export_inference_only ? lite::MT_INFERENCE : lite::MT_TRAIN,
    A2L_ConvertQT(quantization_type), lite::FT_FLATBUFFERS, VectorCharToString(output_tensor_name));

  return (ret == mindspore::lite::RET_OK) ? kSuccess : kLiteError;
}
}  // namespace mindspore
