/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_INFER_MANAGER_H_
#define MINDSPORE_LITE_SRC_RUNTIME_INFER_MANAGER_H_

#include <map>
#include <vector>
#include <set>
#include <string>
#include <memory>
#include "src/common/prim_util.h"
#include "src/tensor.h"
#include "nnacl/tensor_c.h"
#include "nnacl/infer/infer.h"
#include "include/api/kernel.h"
#include "include/api/allocator.h"

namespace mindspore::lite {
int KernelInferShape(const std::vector<lite::Tensor *> &tensors_in, const std::vector<lite::Tensor *> &outputs,
                     OpParameter *parameter, std::shared_ptr<Allocator> allocator = nullptr);
int KernelInferShape(const std::vector<lite::Tensor *> &inputs, const std::vector<lite::Tensor *> &outputs,
                     const void *primitive, std::set<std::string> &&providers, int schema_version,
                     const kernel::Kernel *kernel = nullptr);
typedef bool (*InferChecker)(const std::vector<Tensor *> &, const std::vector<Tensor *> &);
bool InferCheckerAll(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
bool InferCheckerInput(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
bool InferCheckerOutput(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
class InferManager {
 public:
  static InferManager *GetInstance() {
    static InferManager instance;
    return &instance;
  }
  virtual ~InferManager() = default;

  void InsertInferShapeFunc(int prim_type, InferShape func) { infer_shape_funcs_[prim_type] = func; }

  InferShape GetInferShapeFunc(int prim_type) {
    auto iter = infer_shape_funcs_.find(prim_type);
    if (iter == infer_shape_funcs_.end()) {
      return nullptr;
    }
    return iter->second;
  }

 private:
  InferManager() = default;

  std::map<int, InferShape> infer_shape_funcs_;
};

class RegistryInferShape {
 public:
  RegistryInferShape(int prim_type, InferShape func) {
    InferManager::GetInstance()->InsertInferShapeFunc(prim_type, func);
  }
  ~RegistryInferShape() = default;
};

#define REG_INFER_SHAPE(op, prim_type, func) static RegistryInferShape g_##op##InferShape(prim_type, func);
}  // namespace mindspore::lite

#endif  // MINDSPORE_LITE_SRC_RUNTIME_INFER_MANAGER_H_
