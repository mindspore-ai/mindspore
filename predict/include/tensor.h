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

#ifndef PREDICT_INCLUDE_TENSOR_H_
#define PREDICT_INCLUDE_TENSOR_H_

#include <memory>
#include <vector>
#include "dlpack/dlpack.h"
#include "schema/inner/ms_generated.h"

#define MSPREDICT_API __attribute__((visibility("default")))

namespace mindspore {
namespace predict {
///\brief Allocator definition of MindSpore predict.
class Allocator;

///\brief Tensor definition of MindSpore predict.
class MSPREDICT_API Tensor {
 public:
  ///\brief Constructor of MindSpore predict tensor.
  ///
  ///\param[in] tensor Define the parameters of the tensor.
  ///\param[in] copyData Malloc data for the tensor, and copy origin data from
  /// input tensor.
  ///
  ///\return Instance of MindSpore predict tensor.
  Tensor(const Tensor &tensor, bool copyData = false);

  ///\brief Constructor of MindSpore predict tensor.
  ///
  ///\param[in] dt Data Type of the tensor, see introduction to 'enum DataType'
  /// for supported type.
  ///\param[in] dims Dimension Values such as height and width, which defined
  /// the shape of the tensor.
  ///\param[in] format Tensor format, see introduction to 'enum Format' for
  /// supported format.
  ///\param[in] data Data of the tensor.
  ///
  ///\return Instance of MindSpore predict tensor.
  ///
  ///\note
  /// Length of data should align with dt, format and dims, otherwise the
  /// application might run into unexpected error,
  /// such as segment fault.
  /// For example, dt is DT_FLOAT, format is FORMAT_NCHW, dims is [1,3,300,300],
  /// then minimum length of data should
  /// be 1 * 3 * 300 * 300 * sizeof(float).
  Tensor(DataType dt, const std::vector<int64_t> &dims, Format format, void *data);

  ///\brief Destructor of MindSpore predict tensor.
  ~Tensor();

  ///\brief Get MindSpore predict tensor.
  ///
  ///\param[in] Definition of the tensor.
  ///
  ///\return Address of MindSpore predict tensor.
  static Tensor *CopyFromTensorDef(const TensorDef &tensordef);

  ///\brief Get dtype of MindSpore predict tensor.
  ///
  ///\return Dtype of MindSpore predict tensor.
  DLDataType GetTensorDtype() const;

  ///\brief Get data of MindSpore predict tensor.
  ///
  ///\return Address of MindSpore predict tensor data.
  void *GetData() const;

  ///\brief Set data of MindSpore predict tensor.
  ///
  ///\param[in] data Address for data of the MindSpore predict tensor instance.
  ///
  ///\note
  /// Length of data should align with dt, format and dims, otherwise the
  /// application might run into unexpected error,
  /// such as segment fault.
  /// For example, dt is DT_FLOAT, format is FORMAT_NCHW, dims is [1,3,300,300],
  /// then minimum length of data should
  /// be 1 * 3 * 300 * 300 * sizeof(float).
  void SetData(void *data);

  ///\brief Get data type of MindSpore predict tensor.
  ///
  ///\return Data Type of the tensor.
  DataType GetDataType() const;

  ///\brief Set data type of MindSpore predict tensor.
  ///
  ///\param[in] dt Data Type of the tensor, see introduction to 'enum DataType'
  /// for supported type.
  void SetDataType(DataType dt);

  ///\brief Get number of dimension of MindSpore predict tensor.
  ///
  ///\return Number of dimension of the MindSpore predict tensor.
  int GetNDim() const;

  ///\brief Get dimension of MindSpore predict tensor.
  ///
  ///\return Dimension of the MindSpore predict tensor.
  std::vector<int64_t> GetDims() const;

  ///\brief Set dimension of MindSpore predict tensor.
  ///
  ///\param[in] dims Vector that has values of dimension.
  void SetDims(const std::vector<int64_t> &dims);

  ///\brief Get format of MindSpore predict tensor.
  ///
  ///\return Format of the MindSpore predict tensor.
  Format GetFormat() const { return format; }

  ///\brief Set format of MindSpore predict tensor.
  ///
  ///\param[in] format Format of the tensor.
  void SetFormat(Format format) { this->format = format; }

  ///\brief Get reference count of MindSpore predict tensor.
  ///
  ///\return Reference count of the MindSpore predict tensor.
  int RefCount() { return refCount; }

  ///\brief Increase reference count of MindSpore predict tensor.
  ///
  ///\param[in] ref The increase of the reference count.
  void AddRef(int ref) { refCount += ref; }

  ///\brief Decrease reference count of MindSpore predict tensor.
  ///
  ///\param[in] ref The decrease of the reference count.
  void DefRef(int ref) { refCount -= ref; }

  ///\brief Get element size of MindSpore predict tensor.
  ///
  ///\return Element size of MindSpore predict tensor.
  size_t GetElementSize() const;

  ///\brief Get data size of MindSpore predict tensor.
  ///
  ///\return Data size of MindSpore predict tensor.
  size_t GetDataSize() const;

  ///\brief Get element size of MindSpore predict tensor in NC4HW4 format.
  ///
  ///\param[in] isNhwc Whether the current format is NHWC.
  ///
  ///\return Element size of MindSpore predict tensor in NC4HW4 format.
  size_t GetNC4HW4ElementSize(bool isNhwc);

  ///\brief Get data size of MindSpore predict tensor in NC4HW4 format.
  ///
  ///\param[in] isNhwc Whether the current format is NHWC.
  ///
  ///\return Data size of MindSpore predict tensor in NC4HW4 format.
  size_t GetNC4HW4DataSize(bool isNhwc);

  ///\brief Malloc data for the MindSpore predict tensor.
  ///
  ///\param[in] allocator The malloc source for data.
  ///\param[in] refCount The reference count of the data.
  ///
  ///\return Return RET_OK if the data is successfully allocated, otherwhise return RET_ERROR.
  int MallocData(std::shared_ptr<Allocator> allocator = nullptr, int refCount = 0);

  ///\brief Free the MindSpore predict tensor.
  void FreeTensor();

  ///\brief Free the data of MindSpore predict tensor.
  void ForceFreeData();

  ///\brief Free the data of MindSpore predict tensor.
  void FreeData();

  ///\brief Compare data size of MindSpore predict tensor in NC4HW4 format.
  ///
  ///\param[in] dst The compare tensor.
  ///
  ///\return The result of fuction.
  bool CompareShape(const Tensor &dst);

  ///\brief Compare shape of MindSpore predict tensor with another shape.
  ///
  ///\param[in] other The compare shape information.
  ///
  ///\return The result of function.
  bool CompareShape(const std::vector<int64_t> &other);

  ///\brief Get instance of MindSpore predict tensor.
  ///
  ///\return Instance of MindSpore predict dlTensor.
  DLTensor *GetDLTensor() { return &dlTensor; }

  ///\brief Get height of MindSpore predict tensor.
  ///
  ///\return Height of MindSpore predict tensor.
  int64_t Height() const;

  ///\brief Get width of MindSpore predict tensor.
  ///
  ///\return Width of MindSpore predict tensor.
  int64_t Width() const;

  ///\brief Get channel of MindSpore predict tensor.
  ///
  ///\return Channel of MindSpore predict tensor.
  int64_t Channel() const;

  ///\brief Get batch of MindSpore predict tensor.
  ///
  ///\return Batch of MindSpore predict tensor.
  int64_t Batch() const;

  ///\brief Get stride of MindSpore predict tensor.
  ///
  ///\param[in] index the index of stride.
  ///
  ///\return Stride of MindSpore predict tensor.
  int64_t Stride(int index) const;

  ///\brief Set stride of MindSpore predict tensor by input.
  ///
  ///\param[in] index Index of stride
  ///\param[in] stride The stride to set
  void SetStride(int index, int64_t stride);

  ///\brief Set stride of MindSpore predict tensor by dims.
  void SetStride();
  void SetScale(bool isScale = true);

 private:
  bool isScale = false;
  int refCount = 0;
  int isConst;
  Format format;
  DLTensor dlTensor;
  std::shared_ptr<Allocator> allocator = nullptr;
  std::vector<float> scale;
  std::vector<int> zeroPoint;
};
}  // namespace predict
}  // namespace mindspore

#endif  // PREDICT_INCLUDE_TENSOR_H_
