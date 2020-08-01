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

#ifndef MINDSPORE_LITE_INCLUDE_CONTEXT_H_
#define MINDSPORE_LITE_INCLUDE_CONTEXT_H_

#include <string>
#include <memory>
#include "include/ms_tensor.h"

namespace mindspore::lite {
// brief Allocator defined by MindSpore Lite
//
// note List public class and interface for reference
class Allocator;

enum CpuBindMode {
  MID_CPU = -1,   /**< bind mid cpu first */
  HIGHER_CPU = 1, /**< bind higher cpu first */
  NO_BIND = 0     /**< no bind */
};

typedef enum { DT_CPU, DT_GPU, DT_NPU } DeviceType;

// brief NPUContext defined by MindSpore Lite
typedef struct {
  int freq{3};
  int fmkType{0};
  int modelType{0};
  int deviceType{0};
  std::string modelName = "default";
} NPUContext;

// brief DeviceContext defined by MindSpore Lite
typedef struct {
  DeviceType type;
  NPUContext npuCtx;
} DeviceContext;

// brief Context defined by MindSpore Lite
class MS_API Context {
 public:
  // brief Constructor of MindSpore Lite context using default value for parameters
  //
  // return Instance of MindSpore Lite context.
  Context();

  // brief Constructor of MindSpore Lite context using input value for parameters
  //
  // param[in] threadNum Define the threadNum during the runtime.
  // param[in] allocator Define the allocator for malloc.
  // param[in] deviceCtx Define device information during the runtime.
  Context(int threadNum, std::shared_ptr<Allocator> allocator, DeviceContext deviceCtx);

  // brief Destructor of MindSpore Lite context
  virtual ~Context();

 public:
  DeviceContext deviceCtx;
  int threadNum = 2;
  std::shared_ptr<Allocator> allocator;
  CpuBindMode cpuBindMode = MID_CPU;
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_INCLUDE_CONTEXT_H_
