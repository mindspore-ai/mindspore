/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_EXPANDER_INFER_H_
#define MINDSPORE_CORE_EXPANDER_INFER_H_
#include <memory>
#include "common/graph_kernel/bprop/expander/node.h"

namespace mindspore {
namespace expander {
/// \brief ExpanderInfer is the adapter for inferring functions that is called in emitter.
class ExpanderInfer {
 public:
  virtual void Infer(const NodePtr &node) = 0;
};
using ExpanderInferPtr = std::shared_ptr<ExpanderInfer>;

/// \brief CppInfer calls the InferShapeAndType interface of frontend or backend map.
class CppInfer : public ExpanderInfer {
 public:
  void Infer(const NodePtr &node) override;
};
}  // namespace expander
}  // namespace mindspore
#endif  // MINDSPORE_CORE_EXPANDER_INFER_H_
