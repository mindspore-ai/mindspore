/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef PREDICT_INCLUDE_CONTEXT_H_
#define PREDICT_INCLUDE_CONTEXT_H_

#include <memory>
#include "dlpack/dlpack.h"
#include "include/tensor.h"

#define MSPREDICT_API __attribute__((visibility("default")))

namespace mindspore {
namespace predict {
///\brief Resource management definition of MindSpore predict.
class MSPREDICT_API Context {
 public:
  ///\brief Constructor of MindSpore predict context using default value for parameters.
  ///
  ///\return Instance of MindSpore predict context.
  Context();

  ///\brief Custum constructor of MindSpore predict context using input value for parameters.
  ///
  ///\param[in] threadNum The number of thread during the runtime.
  ///\param[in] allocator The memory management during the runtime
  ///\param[in] deviceCtx The device information during the runtime.
  ///
  ///\return Instance of MindSpore predict context.
  Context(int threadNum, std::shared_ptr<Allocator> allocator, DLContext deviceCtx);

  ///\brief Destructor of MindSpore predict context.
  virtual ~Context();

 public:
  DLContext deviceCtx;
  int threadNum = 1;
  std::shared_ptr<Allocator> allocator;
};
}  // namespace predict
}  // namespace mindspore

#endif  // PREDICT_INCLUDE_CONTEXT_H_
