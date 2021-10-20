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

#ifndef MINDSPORE_LITE_INCLUDE_REGISTRY_MODEL_PARSER_H_
#define MINDSPORE_LITE_INCLUDE_REGISTRY_MODEL_PARSER_H_

#include "api/ir/func_graph.h"
#include "include/registry/converter_context.h"

namespace mindspore {
namespace converter {
/// \brief ModelParser defined a base class to parse model.
class MS_API ModelParser {
 public:
  /// \brief Constructor.
  ModelParser() = default;

  /// \brief Destructor.
  virtual ~ModelParser() = default;

  /// \brief Method to parse model, which must be onnx/caffe/tf/tflite.
  ///
  /// \param[in] flags Define the basic parameters when converting, which defined in parser_context.h.
  ///
  /// \return FuncGraph Pointer, which contains all information about the model.
  virtual api::FuncGraphPtr Parse(const converter::ConverterParameters &flags) { return this->res_graph_; }

 protected:
  api::FuncGraphPtr res_graph_ = nullptr;
};
}  // namespace converter
}  // namespace mindspore

#endif  // MINDSPORE_LITE_INCLUDE_REGISTRY_MODEL_PARSER_H_
