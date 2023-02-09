/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_INCLUDE_LITE_TYPES_H_
#define MINDSPORE_LITE_INCLUDE_LITE_TYPES_H_

#include <memory>

namespace mindspore {
class Allocator;
using AllocatorPtr = std::shared_ptr<Allocator>;

class Delegate;
using DelegatePtr = std::shared_ptr<Delegate>;

namespace lite {
class Tensor;

/// \brief CpuBindMode defined for holding bind cpu strategy argument.
typedef enum {
  NO_BIND,    /**< no bind */
  HIGHER_CPU, /**< bind higher cpu first */
  MID_CPU     /**< bind middle cpu first */
} CpuBindMode;

/// \brief DeviceType defined for holding user's preferred backend.
typedef enum {
  DT_CPU,    /**< CPU device type */
  DT_GPU,    /**< GPU device type */
  DT_NPU,    /**< NPU device type */
  DT_ASCEND, /**< ASCEND device type */
  DT_CUSTOM, /**< EXTEND device type */
  DT_END     /**< NO device type */
} DeviceType;

typedef enum {
  FT_FLATBUFFERS, /**< Flatbuffers format */
  FT_PROTOBUF     /**< Protobuf format */
} FormatType;

typedef enum {
  QT_DEFAULT, /**< the quantization of the original model will apply */
  QT_NONE,    /**< apply no quantization */
  QT_WEIGHT   /**< apply weight quantization */
} QuantizationType;

typedef enum {
  MT_TRAIN,    /**< Both Train and Inference part of the compiled model are serialized */
  MT_INFERENCE /**< Only the Inference part of the compiled model is serialized */
} ModelType;
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_INCLUDE_LITE_TYPES_H_
