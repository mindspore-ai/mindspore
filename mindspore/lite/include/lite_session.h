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

#ifndef MINDSPORE_LITE_INCLUDE_LITE_SESSION_H
#define MINDSPORE_LITE_INCLUDE_LITE_SESSION_H

#ifndef NOT_USE_STL
#include <unordered_map>
#endif  // NOT_USE_STL
#include <vector>
#include <string>
#include "include/ms_tensor.h"
#include "include/model.h"
#include "include/context.h"
#include "include/errorcode.h"
#include "include/lite_types.h"

namespace mindspore {
namespace lite {
class TrainCfg;
}

namespace session {
/// \brief LiteSession defined session in MindSpore Lite for compiling Model and forwarding model.
class MS_API LiteSession {
 public:
  /// \brief Static method to create a LiteSession pointer.
  ///
  /// \param[in] context Define the context of session to be created.
  ///
  /// \return Pointer of MindSpore Lite LiteSession.
  static LiteSession *CreateSession(const lite::Context *context);

  /// \brief Static method to create a LiteSession pointer which has already compiled a model.
  ///
  /// \param[in] model_buf Define the buffer read from a model file.
  /// \param[in] size Define bytes number of model buffer.
  /// \param[in] context Define the context of session to be created.
  ///
  /// \return Pointer of MindSpore Lite LiteSession.
  static LiteSession *CreateSession(const char *model_buf, size_t size, const lite::Context *context);

  /// \brief Destructor of MindSpore Lite LiteSession.
  virtual ~LiteSession() = default;

  /// \brief Attempt to bind or unbind threads in the thread pool to or from the specified cpu core.
  ///
  /// \param[in] if_bind Define whether to bind or unbind threads.
  virtual void BindThread(bool if_bind) = 0;

  /// \brief Compile MindSpore Lite model.
  ///
  /// \note CompileGraph should be called before RunGraph.
  ///
  /// \param[in] model Define the model to be compiled.
  ///
  /// \return STATUS as an error code of compiling graph, STATUS is defined in errorcode.h.
  virtual int CompileGraph(lite::Model *model) = 0;

  /// \brief Get input MindSpore Lite MSTensors of model.
  ///
  /// \return The vector of MindSpore Lite MSTensor.
  virtual Vector<tensor::MSTensor *> GetInputs() const = 0;

  /// \brief Get input MindSpore Lite MSTensors of model by tensor name.
  ///
  /// \param[in] node_name Define tensor name.
  ///
  /// \return The vector of MindSpore Lite MSTensor.
  virtual mindspore::tensor::MSTensor *GetInputsByTensorName(const String &tensor_name) const = 0;

  /// \brief Run session with callback.
  ///
  /// \param[in] before Define a call_back_function to be called before running each node.
  /// \param[in] after Define a call_back_function called after running each node.
  ///
  /// \note RunGraph should be called after CompileGraph.
  ///
  /// \return STATUS as an error code of running graph, STATUS is defined in errorcode.h.
  virtual int RunGraph(const KernelCallBack &before = nullptr, const KernelCallBack &after = nullptr) = 0;

  /// \brief Get output MindSpore Lite MSTensors of model by node name.
  ///
  /// \param[in] node_name Define node name.
  ///
  /// \note Deprecated, replace with GetOutputByTensorName
  ///
  /// \return The vector of MindSpore Lite MSTensor.
  virtual Vector<tensor::MSTensor *> GetOutputsByNodeName(const String &node_name) const = 0;

#ifndef NOT_USE_STL
  /// \brief Get output MindSpore Lite MSTensors of model mapped by tensor name.
  ///
  /// \return The map of output tensor name and MindSpore Lite MSTensor.
  virtual std::unordered_map<String, mindspore::tensor::MSTensor *> GetOutputs() const = 0;
#endif

  /// \brief Get name of output tensors of model compiled by this session.
  ///
  /// \return The vector of string as output tensor names in order.
  virtual Vector<String> GetOutputTensorNames() const = 0;

  /// \brief Get output MindSpore Lite MSTensors of model by tensor name.
  ///
  /// \param[in] tensor_name Define tensor name.
  ///
  /// \return Pointer of MindSpore Lite MSTensor.
  virtual mindspore::tensor::MSTensor *GetOutputByTensorName(const String &tensor_name) const = 0;

  /// \brief Resize inputs shape.
  ///
  /// \param[in] inputs Define the inputs of the model.
  /// \param[in] dims Define the inputs new shape.
  ///
  /// \return STATUS as an error code of resize inputs, STATUS is defined in errorcode.h.
  virtual int Resize(const Vector<tensor::MSTensor *> &inputs, const Vector<Vector<int>> &dims) = 0;

  /// \brief Set model to train mode
  /// \return STATUS as an error code of compiling graph, STATUS is defined in errorcode.h
  virtual int Train() { return mindspore::lite::RET_ERROR; }

  /// \brief Check mode of model
  ///
  /// \return boolean indication if model is in train mode
  virtual bool IsTrain() { return false; }

  /// \brief Set model to eval mode
  /// \return STATUS as an error code of compiling graph, STATUS is defined in errorcode.h
  virtual int Eval() { return mindspore::lite::RET_OK; }

  /// \brief Check mode of model
  ///
  /// \return boolean indication if model is in eval mode
  virtual bool IsEval() { return true; }

  /// \brief Sets the Learning Rate of the training
  ///
  /// \param[in] learning_rate to set
  ///
  /// \return STATUS as an error code of the set operation, STATUS is defined in errorcode.h
  virtual int SetLearningRate(float learning_rate) { return mindspore::lite::RET_ERROR; }

  /// \brief Gets the Learning Rate of the training
  ///
  /// \return learning rate. 0.0 if no optimizer was found
  virtual float GetLearningRate() { return 0.0; }

  /// \brief Setup training with virtual batches
  ///
  /// \param[in] virtual_batch_multiplier - virtual batch multiplier, use any number < 1 to disable
  /// \param[in] lr - learning rate to use for virtual batch, -1 for internal configuration
  /// \param[in] momentum - batch norm momentum to use for virtual batch, -1 for internal configuration

  /// \return STATUS as an error code of the set operation, STATUS is defined in errorcode.h
  virtual int SetupVirtualBatch(int virtual_batch_multiplier, float lr = -1.0f, float momentum = -1.0f) {
    return mindspore::lite::RET_ERROR;
  }

  /// \brief Get output MindSpore Lite MSTensors of Training model prediction
  ///
  /// \return a vector of output tensors (MindSpore Lite MSTensor).
  virtual std::vector<tensor::MSTensor *> GetPredictions() const {
    std::vector<tensor::MSTensor *> outputs;
    return outputs;
  }

  /// \brief Save model
  /// \param[in] file_name pretrained model file name prefix. '.ms' extenension is added if does not exist
  /// \param[in] model_type indication whether to save full model or only the inference part
  /// \param[in] quant_type indication whether to quantize exported model
  /// \param[in] format of exported file (currently only FT_FLATBUFFERS is supported)
  /// \param[in] out_put_tensor_name of exported tensorname
  /// \return STATUS as an error code of the set operation, STATUS is defined in errorcode.h
  virtual int Export(const std::string &file_name, lite::ModelType model_type = lite::MT_TRAIN,
                     lite::QuantizationType quant_type = lite::QT_DEFAULT, lite::FormatType = lite::FT_FLATBUFFERS,
                     std::vector<std::string> out_put_tensor_name = {}) {
    return mindspore::lite::RET_ERROR;
  }

  /// \brief Get model featuremap MindSpore Lite MSTensors of Training model prediction
  ///
  /// \return a vector of output tensors (MindSpore Lite MSTensor).
  virtual std::vector<tensor::MSTensor *> GetFeatureMaps() const {
    std::vector<tensor::MSTensor *> features;
    return features;
  }

  /// \brief update model featuremap save to update_ms_file
  /// \param[in] features new featuremap
  /// \return STATUS as an error code of the set operation, STATUS is defined in errorcode.h
  virtual int UpdateFeatureMaps(const std::vector<tensor::MSTensor *> &features) { return mindspore::lite::RET_ERROR; }

  /// \brief Get model gradient
  ///
  /// \return a vector of gradient tensors (MindSpore Lite MSTensor).
  virtual std::vector<tensor::MSTensor *> GetGradients() const {
    std::vector<tensor::MSTensor *> gradients;
    return gradients;
  }

  /// \brief update model gradient
  ///
  /// \param[in] new gradients
  /// \return STATUS as an error code of the set operation, STATUS is defined in errorcode.h
  virtual int ApplyGradients(const std::vector<tensor::MSTensor *> &gradients) { return mindspore::lite::RET_ERROR; }

  /// \brief Get model optimizer params
  ///
  /// \return a vector of optimizer parameters (MindSpore Lite MSTensor).
  virtual std::vector<tensor::MSTensor *> GetOptimizerParams() const {
    std::vector<tensor::MSTensor *> params;
    return params;
  }

  /// \brief set model optimizer params
  ///
  /// \param[in] new optimizer params
  /// \return STATUS as an error code of the set operation, STATUS is defined in errorcode.h
  virtual int SetOptimizerParams(const std::vector<tensor::MSTensor *> &params) { return mindspore::lite::RET_ERROR; }
};
}  // namespace session
}  // namespace mindspore
#endif  // MINDSPORE_LITE_INCLUDE_LITE_SESSION_H
