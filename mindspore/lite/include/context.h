/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed in writing, software
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
/// \brief Allocator defined a memory pool for malloc memory and free memory dynamically.
///
/// \note List public class and interface for reference.
class Allocator;

/// \brief CpuBindMode defined for holding bind cpu strategy argument.
enum CpuBindMode {
  MID_CPU = -1,   /**< bind middle cpu first */
  HIGHER_CPU = 1, /**< bind higher cpu first */
  NO_BIND = 0     /**< no bind */
};

/// \brief DeviceType defined for holding user's preferred backend.
typedef enum {
  DT_CPU, /**< CPU device type */
  DT_GPU, /**< GPU device type */
  DT_NPU  /**< NPU device type */
} DeviceType;

/// \brief DeviceContext defined for holding DeviceType.
typedef struct {
  DeviceType type; /**< device type */
} DeviceContext;

/// \brief Context defined for holding environment variables during runtime.
class MS_API Context {
 public:
  /// \brief Constructor of MindSpore Lite Context using default value for parameters.
  ///
  /// \return Instance of MindSpore Lite Context.
  Context();

  /// \brief Constructor of MindSpore Lite Context using input value for parameters.
  ///
  /// \param[in] thread_num Define the work thread number during the runtime.
  /// \param[in] allocator Define the allocator for malloc.
  /// \param[in] device_ctx Define device information during the runtime.
  Context(int thread_num, std::shared_ptr<Allocator> allocator, DeviceContext device_ctx);

  /// \brief Destructor of MindSpore Lite Context.
  virtual ~Context();

  void InferShapeInterrupt() {
    infer_shape_interrupt_ = true;
  }

 public:
  DeviceContext device_ctx_{DT_CPU};
  int thread_num_ = 2; /**< thread number config for thread pool */
  std::shared_ptr<Allocator> allocator = nullptr;
  CpuBindMode cpu_bind_mode_ = MID_CPU;
  bool infer_shape_interrupt_ = false;
  bool running_ = false;
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_INCLUDE_CONTEXT_H_
