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
  /// \param[in] dec_key The decryption key, key length is 16, 24, or 32. Not supported on MindSpore Lite.
  /// \param[in] dec_mode The decryption mode, optional options are AES-GCM, AES-CBC. Not supported on MindSpore Lite.
  ///
  /// \return Status.
  inline static Status Load(const void *model_data, size_t data_size, ModelType model_type, Graph *graph,
                            const Key &dec_key = {}, const std::string &dec_mode = kDecModeAesGcm);

  /// \brief Loads a model file from path.
  ///
  /// \param[in] file The path of model file.
  /// \param[in] model_type The Type of model file, options are ModelType::kMindIR, ModelType::kOM.
  /// \param[out] graph The output parameter, an object saves graph data.
  /// \param[in] dec_key The decryption key, key length is 16, 24, or 32. Not supported on MindSpore Lite.
  /// \param[in] dec_mode The decryption mode, optional options are AES-GCM, AES-CBC. Not supported on MindSpore Lite.
  ///
  /// \return Status.
  inline static Status Load(const std::string &file, ModelType model_type, Graph *graph, const Key &dec_key = {},
                            const std::string &dec_mode = kDecModeAesGcm);

  /// \brief Load multiple models from multiple files, MindSpore Lite does not provide this feature.
  ///
  /// \param[in] files The path of model files.
  /// \param[in] model_type The Type of model file, options are ModelType::kMindIR, ModelType::kOM.
  /// \param[out] graphs The output parameter, an object saves graph data.
  /// \param[in] dec_key The decryption key, key length is 16, 24, or 32.
  /// \param[in] dec_mode The decryption mode, optional options are AES-GCM, AES-CBC.
  ///
  /// \return Status.
  inline static Status Load(const std::vector<std::string> &files, ModelType model_type, std::vector<Graph> *graphs,
                            const Key &dec_key = {}, const std::string &dec_mode = kDecModeAesGcm);

  /// \brief Configure model parameters, MindSpore Lite does not provide this feature.
  ///
  /// \param[in] parameters The parameters.
  /// \param[in] model The model.
  ///
  /// \return Status.
  inline static Status SetParameters(const std::map<std::string, Buffer> &parameters, Model *model);

  /// \brief Export training model from memory buffer, MindSpore Lite does not provide this feature.
  ///
  /// \param[in] model The model data.
  /// \param[in] model_type The model file type.
  /// \param[out] model_data The model buffer.
  /// \param[in] quantization_type The quantification type.
  /// \param[in] export_inference_only Whether to export a reasoning only model.
  /// \param[in] output_tensor_name The set the name of the output tensor of the exported reasoning model, default as
  /// empty, and export the complete reasoning model.
  ///
  /// \return Status.
  inline static Status ExportModel(const Model &model, ModelType model_type, Buffer *model_data,
                                   QuantizationType quantization_type = kNoQuant, bool export_inference_only = true,
                                   const std::vector<std::string> &output_tensor_name = {});

  /// \brief Export training model from file.
  ///
  /// \param[in] model The model data.
  /// \param[in] model_type The model file type.
  /// \param[in] model_file The path of exported model file.
  /// \param[in] quantization_type The quantification type.
  /// \param[in] export_inference_only Whether to export a reasoning only model.
  /// \param[in] output_tensor_name The set the name of the output tensor of the exported reasoning model, default as
  /// empty, and export the complete reasoning model.
  ///
  /// \return Status.
  inline static Status ExportModel(const Model &model, ModelType model_type, const std::string &model_file,
                                   QuantizationType quantization_type = kNoQuant, bool export_inference_only = true,
                                   std::vector<std::string> output_tensor_name = {});

  /// \brief Experimental feature. Export model's weights, which can be used in micro only.
  ///
  /// \param[in] model The model data.
  /// \param[in] model_type The model file type.
  /// \param[in] weight_file The path of exported weight file.
  /// \param[in] is_inference Whether to export weights from a reasoning model. Currently, only support this is `true`.
  /// \param[in] enable_fp16 Float-weight is whether to be saved in float16 format.
  /// \param[in] changeable_weights_name The set the name of these weight tensors, whose shape is changeable.
  ///
  /// \return Status.
  inline static Status ExportWeightsCollaborateWithMicro(const Model &model, ModelType model_type,
                                                         const std::string &weight_file, bool is_inference = true,
                                                         bool enable_fp16 = false,
                                                         const std::vector<std::string> &changeable_weights_name = {});

 private:
  static Status Load(const void *model_data, size_t data_size, ModelType model_type, Graph *graph, const Key &dec_key,
                     const std::vector<char> &dec_mode);
  static Status Load(const std::vector<char> &file, ModelType model_type, Graph *graph);
  static Status Load(const std::vector<char> &file, ModelType model_type, Graph *graph, const Key &dec_key,
                     const std::vector<char> &dec_mode);
  static Status Load(const std::vector<std::vector<char>> &files, ModelType model_type, std::vector<Graph> *graphs,
                     const Key &dec_key, const std::vector<char> &dec_mode);
  static Status SetParameters(const std::map<std::vector<char>, Buffer> &parameters, Model *model);
  static Status ExportModel(const Model &model, ModelType model_type, const std::vector<char> &model_file,
                            QuantizationType quantization_type, bool export_inference_only,
                            const std::vector<std::vector<char>> &output_tensor_name);
  static Status ExportModel(const Model &model, ModelType model_type, Buffer *model_data,
                            QuantizationType quantization_type, bool export_inference_only,
                            const std::vector<std::vector<char>> &output_tensor_name);
  static Status ExportWeightsCollaborateWithMicro(const Model &model, ModelType model_type,
                                                  const std::vector<char> &weight_file, bool is_inference,
                                                  bool enable_fp16,
                                                  const std::vector<std::vector<char>> &changeable_weights_name);
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

Status Serialization::SetParameters(const std::map<std::string, Buffer> &parameters, Model *model) {
  return SetParameters(MapStringToChar<Buffer>(parameters), model);
}

Status Serialization::ExportModel(const Model &model, ModelType model_type, const std::string &model_file,
                                  QuantizationType quantization_type, bool export_inference_only,
                                  std::vector<std::string> output_tensor_name) {
  return ExportModel(model, model_type, StringToChar(model_file), quantization_type, export_inference_only,
                     VectorStringToChar(output_tensor_name));
}

Status Serialization::ExportModel(const Model &model, ModelType model_type, Buffer *model_data,
                                  QuantizationType quantization_type, bool export_inference_only,
                                  const std::vector<std::string> &output_tensor_name) {
  return ExportModel(model, model_type, model_data, quantization_type, export_inference_only,
                     VectorStringToChar(output_tensor_name));
}

Status Serialization::ExportWeightsCollaborateWithMicro(const Model &model, ModelType model_type,
                                                        const std::string &weight_file, bool is_inference,
                                                        bool enable_fp16,
                                                        const std::vector<std::string> &changeable_weights_name) {
  return ExportWeightsCollaborateWithMicro(model, model_type, StringToChar(weight_file), is_inference, enable_fp16,
                                           VectorStringToChar(changeable_weights_name));
}
}  // namespace mindspore
#endif  // MINDSPORE_INCLUDE_API_SERIALIZATION_H
