/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_INCLUDE_API_SERIALIZATION_H
#define MINDSPORE_INCLUDE_API_SERIALIZATION_H

#include <string>
#include <vector>
#include <map>
#include <memory>
#include "include/api/status.h"
#include "include/api/types.h"
#include "include/api/model.h"
#include "include/api/graph.h"
#include "include/api/dual_abi_helper.h"

namespace mindspore {
/// \brief The Serialization class is used to summarize methods for reading and writing model files.
class MS_API Serialization {
 public:
  /// \brief Loads a model file from memory buffer.
  ///
  /// \param[in] model_data A buffer filled by model file.
  /// \param[in] data_size The size of the buffer.
  /// \param[in] model_type The Type of model file, options are ModelType::kMindIR, ModelType::kOM.
  /// \param[out] graph The output parameter, an object saves graph data.
  /// \param[in] dec_key The decryption key, key length is 16, 24, or 32.
  /// \param[in] dec_mode The decryption mode, optional options are AES-GCM, AES-CBC.
  ///
  /// \return Status.
  inline static Status Load(const void *model_data, size_t data_size, ModelType model_type, Graph *graph,
                            const Key &dec_key = {}, const std::string &dec_mode = kDecModeAesGcm);

  /// \brief Loads a model file from path, is not supported on MindSpore Lite.
  ///
  /// \param[in] file The path of model file.
  /// \param[in] model_type The Type of model file, options are ModelType::kMindIR, ModelType::kOM.
  /// \param[out] graph The output parameter, an object saves graph data.
  /// \param[in] dec_key The decryption key, key length is 16, 24, or 32.
  /// \param[in] dec_mode The decryption mode, optional options are AES-GCM, AES-CBC.
  ///
  /// \return Status.
  inline static Status Load(const std::string &file, ModelType model_type, Graph *graph, const Key &dec_key = {},
                            const std::string &dec_mode = kDecModeAesGcm);

  /// \brief Load multiple models from multiple files, MindSpore Lite does not provide this feature.
  ///
  /// \param[in] files The path of model files.
  /// \param[in] model_type The Type of model file, options are ModelType::kMindIR, ModelType::kOM.
  /// \param[out] graph The output parameter, an object saves graph data.
  /// \param[in] dec_key The decryption key, key length is 16, 24, or 32.
  /// \param[in] dec_mode The decryption mode, optional options are AES-GCM, AES-CBC.
  ///
  /// \return Status.
  inline static Status Load(const std::vector<std::string> &files, ModelType model_type, std::vector<Graph> *graphs,
                            const Key &dec_key = {}, const std::string &dec_mode = kDecModeAesGcm);
  static Status SetParameters(const std::map<std::string, Buffer> &parameters, Model *model);
  static Status ExportModel(const Model &model, ModelType model_type, Buffer *model_data);
  static Status ExportModel(const Model &model, ModelType model_type, const std::string &model_file,
                            QuantizationType quantization_type = kNoQuant, bool export_inference_only = true,
                            std::vector<std::string> output_tensor_name = {});

 private:
  static Status Load(const void *model_data, size_t data_size, ModelType model_type, Graph *graph, const Key &dec_key,
                     const std::vector<char> &dec_mode);
  static Status Load(const std::vector<char> &file, ModelType model_type, Graph *graph);
  static Status Load(const std::vector<char> &file, ModelType model_type, Graph *graph, const Key &dec_key,
                     const std::vector<char> &dec_mode);
  static Status Load(const std::vector<std::vector<char>> &files, ModelType model_type, std::vector<Graph> *graphs,
                     const Key &dec_key, const std::vector<char> &dec_mode);
};

Status Serialization::Load(const void *model_data, size_t data_size, ModelType model_type, Graph *graph,
                           const Key &dec_key, const std::string &dec_mode) {
  return Load(model_data, data_size, model_type, graph, dec_key, StringToChar(dec_mode));
}

Status Serialization::Load(const std::string &file, ModelType model_type, Graph *graph, const Key &dec_key,
                           const std::string &dec_mode) {
  return Load(StringToChar(file), model_type, graph, dec_key, StringToChar(dec_mode));
}

Status Serialization::Load(const std::vector<std::string> &files, ModelType model_type, std::vector<Graph> *graphs,
                           const Key &dec_key, const std::string &dec_mode) {
  return Load(VectorStringToChar(files), model_type, graphs, dec_key, StringToChar(dec_mode));
}
}  // namespace mindspore
#endif  // MINDSPORE_INCLUDE_API_SERIALIZATION_H
