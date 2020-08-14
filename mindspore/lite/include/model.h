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

#ifndef MINDSPORE_LITE_INCLUDE_MODEL_H
#define MINDSPORE_LITE_INCLUDE_MODEL_H

#include <string>
#include <vector>
#include <memory>
#include "schema/model_generated.h"

namespace mindspore {
#define MS_API __attribute__((visibility("default")))

/// \brief ModelImpl defined the implement class of Model in MindSpore Lite.
///
/// \note List public class and interface for reference.
class ModelImpl;

namespace lite {
/// \brief Primitive defined as prototype of operator.
///
/// \note List public class and interface for reference.
class Primitive;

/// \brief Model defined model in MindSpore Lite for managing graph.
class MS_API Model {
 public:
  /// \brief Static method to create a Model pointer.
  ///
  /// \param[in] model_buf Define the buffer read from a model file.
  /// \param[in] size Define bytes number of model buffer.
  ///
  /// \return Pointer of MindSpore Lite Model.
  static Model *Import(const char *model_buf, size_t size);

  /// \brief Constructor of MindSpore Lite Model using default value for parameters.
  ///
  /// \return Instance of MindSpore Lite Model.
  Model() = default;

  /// \brief Destructor of MindSpore Lite Model.
  virtual ~Model();

  /// \brief Get MindSpore Lite Primitive by name.
  ///
  /// \param[in] name Define name of primitive to be returned.
  ///
  /// \return the pointer of MindSpore Lite Primitive.
  lite::Primitive *GetOp(const std::string &name) const;

  /// \brief Get graph defined in flatbuffers.
  ///
  /// \return the pointer of graph defined in flatbuffers.
  const schema::MetaGraph *GetMetaGraph() const;

  /// \brief Get MindSpore Lite ModelImpl.
  ///
  /// \return the pointer of MindSpore Lite ModelImpl.
  ModelImpl *model_impl();

  /// \brief Free MetaGraph in MindSpore Lite Model.
  void FreeMetaGraph();

 protected:
  ModelImpl *model_impl_ = nullptr;
};

/// \brief ModelBuilder defined by MindSpore Lite.
class MS_API ModelBuilder {
 public:
  /// \brief OutEdge defined by MindSpore Lite.
  struct OutEdge {
    std::string nodeId;  /**< ID of a node linked by this edge */
    size_t outEdgeIndex; /**< Index of this edge */
  };

  /// \brief Constructor of MindSpore Lite Model using default value for parameters.
  ///
  /// \return Instance of MindSpore Lite ModelBuilder.
  ModelBuilder() = default;

  /// \brief Destructor of MindSpore Lite ModelBuilder.
  virtual ~ModelBuilder() = default;

  /// \brief Add primitive into model builder for model building.
  ///
  /// \param[in] op Define the primitive to be added.
  /// \param[in] inputs Define input edge of primitive to be added.
  ///
  /// \return ID of the added primitive.
  virtual std::string AddOp(const lite::Primitive &op, const std::vector<OutEdge> &inputs) = 0;

  /// \brief Finish constructing the model.
  ///
  /// \return the pointer of MindSpore Lite Model.
  virtual Model *Construct();
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_INCLUDE_MODEL_H
