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

#ifndef MINDSPORE_CCSRC_C_API_BASE_HANDLE_TYPES_H_
#define MINDSPORE_CCSRC_C_API_BASE_HANDLE_TYPES_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef void *Handle;
typedef void *TensorHandle;
typedef void *NodeHandle;
typedef void *AttrHandle;
typedef void *GraphHandle;
typedef void *FuncGraphMgrHandle;
typedef void *ResMgrHandle;

typedef const void *ConstHandle;
typedef const void *ConstTensorHandle;
typedef const void *ConstNodeHandle;
typedef const void *ConstAttrHandle;
typedef const void *ConstGraphHandle;
typedef const void *ConstFuncGraphMgrHandle;
typedef const void *ConstResMgrHandle;

#ifdef __cplusplus
}
#endif
#endif  // MINDSPORE_CCSRC_C_API_BASE_HANDLE_TYPES_H_
