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

#ifndef MINDSPORE_CCSRC_C_API_INCLUDE_VALUE_H_
#define MINDSPORE_CCSRC_C_API_INCLUDE_VALUE_H_

#include <stdbool.h>
#include <stdlib.h>
#include "include/c_api/ms/base/macros.h"
#include "include/c_api/ms/base/handle_types.h"
#include "include/c_api/ms/base/types.h"
#include "include/c_api/ms/context.h"

#ifdef __cplusplus
extern "C" {
#endif

/// \brief Create new Int64 scalar value.
///
/// \param[in] res_mgr Resource Handle that manages the nodes of the funcGraph.
/// \param[in] v Given value.
///
/// \return Value handle.
MIND_C_API ValueHandle MSNewValueInt64(ResMgrHandle res_mgr, const int64_t v);

/// \brief Create new flaot32 scalar value.
///
/// \param[in] res_mgr Resource Handle that manages the nodes of the funcGraph.
/// \param[in] v Given value.
///
/// \return Value handle.
MIND_C_API ValueHandle MSNewValueFloat32(ResMgrHandle res_mgr, const float v);

/// \brief Create new Bool scalar value.
///
/// \param[in] res_mgr Resource Handle that manages the nodes of the funcGraph.
/// \param[in] v Given value.
///
/// \return Value handle.
MIND_C_API ValueHandle MSNewValueBool(ResMgrHandle res_mgr, const bool v);

/// \brief Create new value of DataType.
///
/// \param[in] res_mgr Resource Handle that manages the nodes of the funcGraph.
/// \param[in] type Given data type.
///
/// \return Value handle.
MIND_C_API ValueHandle MSNewValueType(ResMgrHandle res_mgr, DataTypeC type);

/// \brief Create new vector of Strings Value.
///
/// \param[in] res_mgr Resource Handle that manages the nodes of the funcGraph.
/// \param[in] strs Given value.
/// \param[in] vec_len Length of the string vector.
///
/// \return Value handle.
MIND_C_API ValueHandle MSNewValueStrings(ResMgrHandle res_mgr, const char *strs[], size_t vec_len);

/// \brief Create new Value with array.
///
/// \param[in] res_mgr Resource Handle that manages the nodes of the funcGraph.
/// \param[in] value Given array.
/// \param[in] vec_size Given array size.
/// \param[in] data_type Datatype of the array.
///
/// \return Value handle
MIND_C_API ValueHandle MSNewValueArray(ResMgrHandle res_mgr, void *value, size_t vec_size, DataTypeC data_type);
#ifdef __cplusplus
}
#endif
#endif  // MINDSPORE_CCSRC_C_API_INCLUDE_VALUE_H_
