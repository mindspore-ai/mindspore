/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_INCLUDE_C_API_TENSOE_C_H
#define MINDSPORE_INCLUDE_C_API_TENSOE_C_H

#include <stddef.h>
#include "include/c_api/types_c.h"
#include "include/c_api/data_type_c.h"
#include "include/c_api/format_c.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void *MSTensorHandle;

/// \brief Create a tensor object.
///
/// \param[in] name The name of the tensor.
/// \param[in] type The data type of the tensor.
/// \param[in] shape The shape of the tensor.
/// \param[in] shape_num The num of the shape.
/// \param[in] data The data pointer that points to allocated memory.
/// \param[in] data_len The length of the memory, in bytes.
///
/// \return Tensor object handle.
MS_API MSTensorHandle MSTensorCreate(const char *name, MSDataType type, const int64_t *shape, size_t shape_num,
                                     const void *data, size_t data_len);

/// \brief Destroy the tensor object.
///
/// \param[in] tensor Tensor object handle address.
MS_API void MSTensorDestroy(MSTensorHandle *tensor);

/// \brief Obtain a deep copy of the tensor.
///
/// \param[in] tensor Tensor object handle.
///
/// \return Tensor object handle.
MS_API MSTensorHandle MSTensorClone(MSTensorHandle tensor);

/// \brief Set the name for the tensor.
///
/// \param[in] tensor Tensor object handle.
/// \param[in] name The name of the tensor.
MS_API void MSTensorSetName(MSTensorHandle tensor, const char *name);

/// \brief Obtain the name of the tensor.
///
/// \param[in] tensor Tensor object handle.
///
/// \return The name of the tensor.
MS_API const char *MSTensorGetName(const MSTensorHandle tensor);

/// \brief Set the data type for the tensor.
///
/// \param[in] tensor Tensor object handle.
/// \param[in] type The data type of the tensor.
MS_API void MSTensorSetDataType(MSTensorHandle tensor, MSDataType type);

/// \brief Obtain the data type of the tensor.
///
/// \param[in] tensor Tensor object handle.
///
/// \return The date type of the tensor.
MS_API MSDataType MSTensorGetDataType(const MSTensorHandle tensor);

/// \brief Set the shape for the tensor.
///
/// \param[in] tensor Tensor object handle.
/// \param[in] shape The shape array.
/// \param[in] shape_num Dimension of shape.
MS_API void MSTensorSetShape(MSTensorHandle tensor, const int64_t *shape, size_t shape_num);

/// \brief Obtain the shape of the tensor.
///
/// \param[in] tensor Tensor object handle.
/// \param[out] shape_num Dimension of shape.
///
/// \return The shape array of the tensor.
MS_API const int64_t *MSTensorGetShape(const MSTensorHandle tensor, size_t *shape_num);

/// \brief Set the format for the tensor.
///
/// \param[in] tensor Tensor object handle.
/// \param[in] format The format of the tensor.
MS_API void MSTensorSetFormat(MSTensorHandle tensor, MSFormat format);

/// \brief Obtain the format of the tensor.
///
/// \param[in] tensor Tensor object handle.
///
/// \return The format of the tensor.
MS_API MSFormat MSTensorGetFormat(const MSTensorHandle tensor);

/// \brief Obtain the data for the tensor.
///
/// \param[in] tensor Tensor object handle.
/// \param[in] data A pointer to the data of the tensor.
MS_API void MSTensorSetData(MSTensorHandle tensor, void *data);

/// \brief Obtain the data pointer of the tensor.
///
/// \param[in] tensor Tensor object handle.
///
/// \return The data pointer of the tensor.
MS_API const void *MSTensorGetData(const MSTensorHandle tensor);

/// \brief Obtain the mutable data pointer of the tensor. If the internal data is empty, it will allocate memory.
///
/// \param[in] tensor Tensor object handle.
///
/// \return The data pointer of the tensor.
MS_API void *MSTensorGetMutableData(const MSTensorHandle tensor);

/// \brief Obtain the element number of the tensor.
///
/// \param[in] tensor Tensor object handle.
///
/// \return The element number of the tensor.
MS_API int64_t MSTensorGetElementNum(const MSTensorHandle tensor);

/// \brief Obtain the data size fo the tensor.
///
/// \param[in] tensor Tensor object handle.
///
/// \return The data size of the tensor.
MS_API size_t MSTensorGetDataSize(const MSTensorHandle tensor);

#ifdef __cplusplus
}
#endif
#endif  // MINDSPORE_INCLUDE_C_API_TENSOE_C_H
