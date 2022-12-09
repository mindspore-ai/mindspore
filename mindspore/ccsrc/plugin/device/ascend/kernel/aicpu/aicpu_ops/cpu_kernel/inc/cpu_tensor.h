/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
 * Description: api of tensor
 */

#ifndef CPU_KERNEL_TENSOR_H
#define CPU_KERNEL_TENSOR_H
#include <memory>

#include "cpu_kernel/inc/cpu_tensor_shape.h"

namespace aicpu {
class TensorImpl;
class AICPU_VISIBILITY Tensor {
  friend class CpuKernelUtils;

 public:
  Tensor() = delete;
  ~Tensor() = default;

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
   * calculate data size by tensor shape.
   * @return success->not less than 0, failed->less than 0
   */
  int64_t CalcDataSizeByShape() const;

  /*
   * get data elements number.
   * @return success->not less than 0, unknown->less than 0
   */
  int64_t NumElements() const;

 private:
  explicit Tensor(TensorImpl *impl);

 private:
  std::shared_ptr<TensorImpl> impl_{nullptr};
};
}  // namespace aicpu
#endif  // CPU_KERNEL_TENSOR_H
