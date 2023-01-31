/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
#ifndef AICPU_CONTEXT_CPU_PROTO_TENSOR_IMPL_H
#define AICPU_CONTEXT_CPU_PROTO_TENSOR_IMPL_H
#include <functional>
#include <memory>
#include <string>

#include "cpu_kernel/inc/cpu_tensor_shape.h"
#include "proto/cpu_tensor.pb.h"

namespace aicpu {
class TensorImpl {
  friend class CpuKernelUtils;

 public:
  TensorImpl(
    aicpuops::Tensor *tensor, std::function<void(aicpuops::Tensor *)> delFunc = [](aicpuops::Tensor *p) {})
      : tensor_(tensor, delFunc) {}

  ~TensorImpl() = default;
  TensorImpl(const TensorImpl &) = delete;
  TensorImpl(TensorImpl &&) = delete;
  TensorImpl &operator=(const TensorImpl &) = delete;
  TensorImpl &operator=(TensorImpl &&) = delete;

  /*
   * set tensor shape value to tensor.
   * @param shape: tensor shape value need to set to tensor
   * @return bool: true->success, false->failed
   */
  bool SetTensorShape(const TensorShape *shape);

  /*
   * get tensor shape value of tensor.
   * @return std::shared_ptr<TensorShape>: tensor shape value of tensor
   */
  std::shared_ptr<TensorShape> GetTensorShape() const;

  /*
   * set data type value to tensor.
   * @param type: data type value need to set to tensor
   */
  void SetDataType(DataType type);

  /*
   * get data type value of tensor.
   * @return DataType: data type value of tensor
   */
  DataType GetDataType() const;

  /*
   * set data ptr to tensor.
   * @param addr: tensor data ptr
   */
  void SetData(void *addr);

  /*
   * get data ptr of tensor.
   * @return void *: tensor data ptr
   */
  void *GetData() const;

  /*
   * set data size to tensor.
   * @param size: tensor data size
   */
  void SetDataSize(uint64_t size);

  /*
   * get data size of tensor.
   * @return uint64_t: tensor data size
   */
  uint64_t GetDataSize() const;

  /*
   * get name of tensor.
   * @return std::string: tensor name
   */
  std::string GetName() const;

  /*
   * set name of tensor.
   * @param name: tensor name
   */
  void SetName(const std::string &name);

  /*
   * calculate data size by tensor shape.
   * @return success->not less than 0, failed->less than 0
   */
  int64_t CalcDataSizeByShape() const;

  /*
   * get data elements number.
   * @return success->not less than 0, unknown->less than 0
   */
  int64_t NumElements() const;

  /*
   * get tensor proto.
   */
  aicpuops::Tensor *GetProto() const;

 private:
  std::shared_ptr<aicpuops::Tensor> tensor_{nullptr};
};
}  // namespace aicpu
#endif  // AICPU_CONTEXT_CPU_PROTO_TENSOR_IMPL_H
