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
#ifndef MINDSPORE_LITE_EXTENDRT_INFER_SESSION_H
#define MINDSPORE_LITE_EXTENDRT_INFER_SESSION_H
#include <string>
#include <memory>
#include <map>
#include <vector>
#include "include/api/context.h"
#include "include/api/model.h"
#include "include/api/graph.h"
#include "include/api/status.h"
#include "include/common/utils/utils.h"
#include "ir/func_graph.h"
#include "backend/graph_compiler/graph_partition.h"
#include "extendrt/session/type.h"
#include "common/mutable_tensor_impl.h"
#include "src/common/config_infos.h"

namespace mindspore {
class InferSession : public std::enable_shared_from_this<InferSession> {
 public:
  virtual ~InferSession() = default;

  /// \brief Create InferSession object.
  ///
  /// \param[in] context Define model context, which will pass to session.
  /// \param[in] config_info Define config info for model.
  ///
  /// \return The pointer of the model infer session according to model context.
  static std::shared_ptr<InferSession> CreateSession(const std::shared_ptr<Context> &context,
                                                     const ConfigInfos &config_info);

  /// \brief Init InferSession.
  ///
  /// \param[in] context Define model context, which will pass to session.
  ///
  /// \return Status.
  virtual Status Init(const std::shared_ptr<Context> &context, const ConfigInfos &config_info) = 0;

  /// \brief Compile Model Graph.
  ///
  /// \param[in] graph Define FuncGraph pointer reprenst model.
  /// \param[in] data (Deprecated), need delete.
  /// \param[in] size (Deprecated), need delete.
  ///
  /// \return Status.
  virtual Status CompileGraph(FuncGraphPtr graph, const void *data, size_t size, uint32_t *graph_id) = 0;

  /// \brief Compile Model Graph.
  ///
  /// \param[in] model_data Define the buffer read from a model file.
  /// \param[in] data_size Define bytes number of model buffer.
  ///
  /// \return Status.
  virtual Status CompileGraph(const void *model_data, size_t data_size, uint32_t *graph_id) {
    (void)model_data;
    (void)data_size;
    (void)graph_id;
    return kLiteNotSupport;
  }

  /// \brief Run Model Graph to inference.
  ///
  /// \param[in] inputs A vector where model inputs are arranged in sequence.
  /// \param[out] outputs Which is a pointer to a vector. The model outputs are filled in the container in sequence.
  ///
  /// \return Status.
  virtual Status RunGraph(uint32_t graph_id, const std::vector<tensor::Tensor> &inputs,
                          std::vector<tensor::Tensor> *outputs) = 0;

  /// \brief Run Model Graph to inference.
  ///
  /// \param[in] inputs A vector where model inputs are arranged in sequence.
  /// \param[out] outputs Which is a pointer to a vector. The model outputs are filled in the container in sequence.
  /// \param[in] before CallBack before predict.
  /// \param[in] after CallBack after predict.
  ///
  /// \return Status.
  virtual Status RunGraph(uint32_t graph_id, const std::vector<tensor::Tensor> &inputs,
                          std::vector<tensor::Tensor> *outputs, const MSKernelCallBack &before,
                          const MSKernelCallBack &after) = 0;

  /// \brief Resize model inputs shape and memory from specified dims.
  ///
  /// \param[in] inputs Define dst inputs tensors.
  /// \param[in] dims Define dst resize shapes.
  ///
  /// \return Status.
  virtual Status Resize(uint32_t graph_id, const std::vector<tensor::Tensor> &inputs,
                        const std::vector<std::vector<int64_t>> &dims) {
    (void)graph_id;
    (void)inputs;
    (void)dims;
    return kSuccess;
  }

  /// \brief Obtains all output tensors of the model.
  ///
  /// \return The vector that includes all output tensors.
  virtual std::vector<MutableTensorImplPtr> GetOutputs(uint32_t graph_id) = 0;

  /// \brief Obtains all input tensors of the model.
  ///
  /// \return The vector that includes all input tensors.
  virtual std::vector<MutableTensorImplPtr> GetInputs(uint32_t graph_id) = 0;

  /// \brief Obtains all output tensors' name of the model.
  ///
  /// \return The vector that includes all output tensors' name.
  virtual std::vector<std::string> GetOutputNames(uint32_t graph_id) = 0;

  /// \brief Obtains all input tensors' name of the model.
  ///
  /// \return The vector that includes all input tensors' name.
  virtual std::vector<std::string> GetInputNames(uint32_t graph_id) = 0;

  /// \brief Obtains the output tensor of the model by name.
  ///
  /// \return The output tensor with the given name, if the name is not found, an invalid tensor is returned.
  virtual MutableTensorImplPtr GetOutputByTensorName(uint32_t graph_id, const std::string &tensorName) = 0;

  /// \brief Obtains the input tensor of the model by name.
  ///
  /// \return The input tensor with the given name, if the name is not found, an invalid tensor is returned.
  virtual MutableTensorImplPtr GetInputByTensorName(uint32_t graph_id, const std::string &name) = 0;

 protected:
  /// \brief Handle session according to context.
  ///
  /// \param[in] context Define model context, which will pass to session.
  static void HandleContext(const std::shared_ptr<Context> &context, bool use_experimental_rts = false);

  /// \brief Select InferSession type.
  ///
  /// \param[in] context Define model context, which will pass to session.
  ///
  /// \return The Session type, eg kSingleOpSession for Ascend, etc.
  static SessionType SelectSession(const std::shared_ptr<Context> &context, bool use_experimental_rts = false);
};  // namespace mindspore
}  // namespace mindspore
#endif
