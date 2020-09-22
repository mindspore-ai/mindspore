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

#ifndef MINDSPORE_LITE_INTERNAL_INCLUDE_LITE_SESSION_H
#define MINDSPORE_LITE_INTERNAL_INCLUDE_LITE_SESSION_H

#include "internal/include/ms_tensor.h"
#include "internal/include/model.h"
#include "internal/include/context.h"
#include "internal/include/lite_utils.h"

/// \brief LiteSession defined session in MindSpore Lite for compiling Model and forwarding model.
typedef struct LiteSession {
  /// \brief Static method to create a LiteSession pointer.
  ///
  /// \param[in] context Define the context of session to be created.
  ///
  /// \return Pointer of MindSpore Lite LiteSession.
  static LiteSession *CreateSession(Context *context);

  /// \brief Compile MindSpore Lite model.
  ///
  /// \note CompileGraph should be called before RunGraph.
  ///
  /// \param[in] model Define the model to be compiled.
  ///
  /// \return STATUS as an error code of compiling graph, STATUS is defined in errorcode.h.
  int CompileGraph(Model *model);

  /// \brief Get input MindSpore Lite MSTensors of model.
  ///
  /// \return The vector of MindSpore Lite MSTensor.
  TensorPtrVector GetInputs() const;

  /// \brief Get input MindSpore Lite MSTensors of model by node name.
  ///
  /// \param[in] node_name Define node name.
  ///
  /// \return The vector of MindSpore Lite MSTensor.
  TensorPtrVector GetInputsByName(const String &node_name) const;

  /// \brief Get output MindSpore Lite MSTensors of model by node name.
  ///
  /// \param[in] node_name Define node name.
  ///
  /// \return The vector of MindSpore Lite MSTensor.
  TensorPtrVector GetOutputsByNodeName(const String &node_name) const;

  /// \brief Get output MindSpore Lite MSTensors of model mapped by tensor name.
  ///
  /// \return The map of output tensor name and MindSpore Lite MSTensor.
  TensorPtrVector GetOutputs() const;

  /// \brief Get name of output tensors of model compiled by this session.
  ///
  /// \return The vector of string as output tensor names in order.
  StringVector GetOutputTensorNames() const;

  /// \brief Get output MindSpore Lite MSTensors of model by tensor name.
  ///
  /// \param[in] tensor_name Define tensor name.
  ///
  /// \return Pointer of MindSpore Lite MSTensor.
  MSTensor *GetOutputByTensorName(const String &tensor_name) const;

  /// \note RunGraph should be called after CompileGraph.
  int RunGraph();

  /// \brief Resize inputs shape.
  ///
  /// \param[in] inputs Define the new inputs shape.
  /// \param[in] dims Define the inputs new shape.
  ///
  /// \return STATUS as an error code of resize inputs, STATUS is defined in errorcode.h.
  int Resize(const TensorPtrVector &inputs, const Int32VectorVector &dims);
} LiteSession;

#endif  // MINDSPORE_LITE_INCLUDE_LITE_SESSION_H
