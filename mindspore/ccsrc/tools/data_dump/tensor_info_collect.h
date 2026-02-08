/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_TOOLS_DATA_DUMP_TENSOR_INFO_COLLECT_H_
#define MINDSPORE_CCSRC_TOOLS_DATA_DUMP_TENSOR_INFO_COLLECT_H_

#include <map>
#include <string>
#include <vector>

#include "include/runtime/hardware_abstract/kernel_base/kernel.h"

namespace mindspore {
using KernelTensor = kernel::KernelTensor;
using KernelTensorPtr = kernel::KernelTensorPtr;

class TensorInfoForDump {
 public:
  struct KernelTensorMeta {
    explicit KernelTensorMeta(KernelTensorPtr tensor, const TypeId &dtype, const ShapeVector &shape)
        : tensor(tensor), dtype(dtype), shape(shape) {
      MS_EXCEPTION_IF_NULL(tensor);
    }

    KernelTensorPtr tensor;
    TypeId dtype;
    ShapeVector shape;
  };

  TensorInfoForDump(std::string io, uint32_t io_index, std::string format, TypeId host_type,
                    const ShapeVector &host_shape, size_t device_size, KernelTensor *kernel_tensor)
      : io(io),
        io_index(io_index),
        format(format),
        host_type(host_type),
        host_shape(host_shape),
        device_size(device_size),
        kernel_tensor(kernel_tensor) {
    MS_EXCEPTION_IF_NULL(kernel_tensor);
    this->device_ptr = kernel_tensor->device_ptr();
  }

  std::string io;
  uint32_t io_index;
  std::string format;

  TypeId host_type;
  const ShapeVector host_shape;
  size_t device_size;
  KernelTensor *kernel_tensor;
  const void *device_ptr;
  std::map<std::string, std::vector<KernelTensorPtr>> workspace;
  std::map<std::string, KernelTensorMeta> stat_results;
};

class TensorInfoCommForDump {
 public:
  TensorInfoCommForDump(std::string dump_path, std::string op_type, std::string op_name, std::string stat_op_name,
                        uint32_t task_id, uint32_t stream_id)
      : dump_path(dump_path),
        op_type(op_type),
        op_name(op_name),
        stat_op_name(stat_op_name),
        task_id(task_id),
        stream_id(stream_id) {
    this->file_path_prefix =
      dump_path + '/' + op_type + '.' + op_name + '.' + std::to_string(task_id) + '.' + std::to_string(stream_id);
  }

  std::string dump_path;
  std::string op_type;
  std::string op_name;
  std::string stat_op_name;
  uint32_t task_id;
  uint32_t stream_id;
  std::string file_path_prefix;
};

}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_TOOLS_DATA_DUMP_TENSOR_INFO_COLLECT_H_
