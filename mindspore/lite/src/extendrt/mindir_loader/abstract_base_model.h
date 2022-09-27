/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_MINDIR_LOADER_ABSTRACT_BASE_MODEL_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_MINDIR_LOADER_ABSTRACT_BASE_MODEL_H_

#include <string>
#include <vector>

#include "include/model.h"
#include "src/tensor.h"
#include "src/litert/kernel_exec.h"

using Model = mindspore::lite::Model;
using LiteGraph = mindspore::lite::LiteGraph;

namespace mindspore::infer {
class AbstractBaseModel : public Model {
 public:
  virtual bool ModelVerify() const = 0;
  // virtual SchemaTensorWrapper *GetSchemaTensor(const size_t &tensor_index) const = 0;
  virtual int ConvertTensors(std::vector<mindspore::lite::Tensor *> *lite_tensors) = 0;
  virtual std::string GetModelPath() const = 0;
  virtual mindspore::kernel::KernelExec *FindBackendKernel(const std::vector<mindspore::lite::Tensor *> &in_tensors,
                                                           const std::vector<mindspore::lite::Tensor *> &out_tensors,
                                                           const LiteGraph::Node *node, lite::InnerContext *context,
                                                           TypeId prefer_data_type) = 0;
};
}  // namespace mindspore::infer

#endif  // MINDSPORE_LITE_SRC_EXTENDRT_MINDIR_LOADER_ABSTRACT_BASE_MODEL_H_
