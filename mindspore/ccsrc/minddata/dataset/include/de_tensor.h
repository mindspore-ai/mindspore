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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DETENSOR_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DETENSOR_H_
#include <string>
#include <vector>
#include <memory>
#include "include/ms_tensor.h"
#include "minddata/dataset/include/status.h"
#include "minddata/dataset/include/tensor.h"
namespace mindspore {
namespace tensor {
class DETensor : public mindspore::tensor::MSTensor {
 public:
  /// \brief Create a MSTensor pointer.
  /// \param[in] data_type DataTypeId of tensor to be created
  /// \param[in] shape Shape of tensor to be created
  /// \return MSTensor pointer
  static MSTensor *CreateTensor(TypeId data_type, const std::vector<int> &shape);

  /// \brief Create a MSTensor pointer.
  /// \param[in] path Path to file to read
  /// \return MSTensor pointer
  static MSTensor *CreateTensor(const std::string &path);

  /// \brief Create a MSTensor pointer.
  /// \param[in] data_type Data TypeId of tensor to be created
  /// \param[in] shape Shape of tensor to be created
  /// \param[in] data Data pointer
  /// \return MSTensor pointer
  static MSTensor *CreateFromMemory(TypeId data_type, const std::vector<int> &shape, void *data);

  DETensor(TypeId data_type, const std::vector<int> &shape);

  explicit DETensor(std::shared_ptr<dataset::Tensor> tensor_ptr);

  ~DETensor() = default;

  /// \brief Create a duplicate instance, convert the DETensor to the LiteTensor.
  /// \return MSTensor pointer
  MSTensor *ConvertToLiteTensor();

  std::shared_ptr<dataset::Tensor> tensor() const;

  TypeId data_type() const override;

  TypeId set_data_type(const TypeId data_type);

  std::vector<int> shape() const override;

  size_t set_shape(const std::vector<int> &shape);

  int DimensionSize(size_t index) const override;

  int ElementsNum() const override;

  std::size_t hash() const;

  size_t Size() const override;

  void *MutableData() override;

 protected:
  std::shared_ptr<dataset::Tensor> tensor_impl_;
};
}  // namespace tensor
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DETENSOR_H_
