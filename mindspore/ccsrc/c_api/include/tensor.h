/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_C_API_IR_FUNC_TENSOR_H_
#define MINDSPORE_CCSRC_C_API_IR_FUNC_TENSOR_H_

#include <stdbool.h>
#include <stdlib.h>
#include "c_api/base/macros.h"
#include "c_api/base/status.h"
#include "c_api/base/types.h"
#include "c_api/base/handle_types.h"
#include "c_api/include/context.h"

#ifdef __cplusplus
extern "C" {
#endif

/// \brief Create a tensor with input data buffer.
///
/// \param[in] res_mgr Resource manager that saves allocated instance resources.
/// \param[in] data The input data to be copied into tensor.
/// \param[in] type [TypeId] Data type of the tensor.
/// \param[in] shape The shape arary of the tensor.
/// \param[in] shape_size The size of shape array, i.e., the rank of the tensor.
/// \param[in] data_len The length of data in bytes.
///
/// \return The pointer of the created tensor instance.
MIND_C_API TensorHandle MSNewTensor(ResMgrHandle res_mgr, void *data, TypeId type, const int64_t shape[],
                                    size_t shape_size, size_t data_len);

/// \brief Create a tensor with path to a space-sperated txt file.
///
/// \param[in] res_mgr Resource manager that saves allocated instance resources.
/// \param[in] type [TypeId] Data type of the tensor.
/// \param[in] shape The shape arary of the tensor.
/// \param[in] shape_size The size of shape array, i.e., the rank of the tensor.
/// \param[in] path path to the file.
///
/// \return The pointer of the created tensor instance.
MIND_C_API TensorHandle MSNewTensorFromFile(ResMgrHandle res_mgr, TypeId type, const int64_t shape[], size_t shape_size,
                                            const char *path);

/// \brief Create a tensor with input data buffer and given source data type.
///
/// \param[in] res_mgr Resource manager that saves allocated instance resources.
/// \param[in] shape The shape arary of the tensor.
/// \param[in] shape_size The size of shape array, i.e., the rank of the tensor.
/// \param[in] data The input data to be copied into tensor.
/// \param[in] tensor_type [TypeId] Data type of the tensor.
/// \param[in] src_type [TypeId] The source data type.
///
/// \return The pointer of the created tensor instance.
MIND_C_API TensorHandle MSNewTensorWithSrcType(ResMgrHandle res_mgr, void *data, const int64_t shape[],
                                               size_t shape_size, TypeId tensor_type, TypeId src_type);

/// \brief Get the raw pointer of tensor data.
///
/// \param[in] res_mgr Resource manager that saves allocated instance resources.
/// \param[in] tensor The pointer of the tensor instance.
///
/// \return The pointer to the tensor data
MIND_C_API void *MSTensorGetData(ResMgrHandle res_mgr, const TensorHandle tensor);

/// \brief Set tensor data type.
///
/// \param[in] res_mgr Resource manager that saves allocated instance resources.
/// \param[in] tensor The pointer of the tensor instance.
/// \param[in] type The data type to be set.
///
/// \return Error code that indicate whether the functions executed successfully.
MIND_C_API STATUS MSTensorSetDataType(ResMgrHandle res_mgr, const TensorHandle tensor, TypeId type);

/// \brief Get tensor data type.
///
/// \param[in] res_mgr Resource manager that saves allocated instance resources.
/// \param[in] tensor The pointer of the tensor instance.
///
/// \return The data type of tensor.
MIND_C_API TypeId MSTensorGetDataType(ResMgrHandle res_mgr, const TensorHandle tensor, STATUS *error);

/// \brief Get the byte size of tensor data.
///
/// \param[in] res_mgr Resource manager that saves allocated instance resources.
/// \param[in] tensor The pointer of the tensor instance.
/// \param[in] error Records error code that indicate whether the functions executed successfully.
///
/// \return The byte size of tensor data.
MIND_C_API size_t MSTensorGetDataSize(ResMgrHandle res_mgr, const TensorHandle tensor, STATUS *error);

/// \brief Get the element number of tensor array.
///
/// \param[in] res_mgr Resource manager that saves allocated instance resources.
/// \param[in] tensor The pointer of the tensor instance.
/// \param[in] error Records error code that indicate whether the functions executed successfully.
///
/// \return The element number of tensor array.
MIND_C_API size_t MSTensorGetElementNum(ResMgrHandle res_mgr, const TensorHandle tensor, STATUS *error);

/// \brief Get the dimension of tensor.
///
/// \param[in] res_mgr Resource manager that saves allocated instance resources.
/// \param[in] tensor The pointer of the tensor instance.
/// \param[in] error Records error code that indicate whether the functions executed successfully.
///
/// \return The dimension of tensor.
MIND_C_API size_t MSTensorGetDimension(ResMgrHandle res_mgr, const TensorHandle tensor, STATUS *error);

/// \brief Set the shape of tensor array.
///
/// \param[in] res_mgr Resource manager that saves allocated instance resources.
/// \param[in] tensor The pointer of the tensor instance.
/// \param[in] shape The shape array.
/// \param[in] dim The the dimension of tensor, i.e., size of shape array.
///
/// \return Error code indicates whether the function executed successfully.
MIND_C_API STATUS MSTensorSetShape(ResMgrHandle res_mgr, const TensorHandle tensor, int64_t shape[], size_t dim);

/// \brief Get the shape of tensor array.
///
/// \param[in] res_mgr Resource manager that saves allocated instance resources.
/// \param[in] tensor The pointer of the tensor instance.
/// \param[in] shape The shape array.
/// \param[in] dim The the dimension of tensor, i.e., size of shape array.
///
/// \return Error code indicates whether the function executed successfully.
MIND_C_API STATUS MSTensorGetShape(ResMgrHandle res_mgr, const TensorHandle tensor, int64_t shape[], size_t dim);

#ifdef __cplusplus
}
#endif
#endif  // MINDSPORE_CCSRC_C_API_IR_FUNC_TENSOR_H_
