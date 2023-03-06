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

#ifndef MINDSPORE_CCSRC_C_API_INCLUDE_CONTEXT_H_
#define MINDSPORE_CCSRC_C_API_INCLUDE_CONTEXT_H_

#include <stdint.h>
#include <stdlib.h>
#include "c_api/base/macros.h"
#include "c_api/base/status.h"
#include "c_api/base/handle_types.h"

#ifdef __cplusplus
extern "C" {
#endif
/// \brief Create a resource manager. Tracking allocated objects(i.e. FuncGraph, Nodes...).
MIND_C_API ResMgrHandle MSResourceManagerCreate();

/// \brief Release a resource manager and the enclosed objects.
///
/// \param[in] res_mgr Resource Handle that manages the nodes of the funcGraph.
MIND_C_API void MSResourceManagerDestroy(ResMgrHandle res_mgr);

/// \brief Set context to Graph or pynative mood.
///
/// \param[in] eager_mode True: Pynative, False: Graph.
MIND_C_API void MSSetEagerMode(bool eager_mode);

/// \brief Select MindRT, VM or GE as Backend Policy.
///
/// \param[in] policy select one from {"ge", "vm", "ms", "ge_only", "vm_prior"}.
///
/// \return Error code indicates whether the function executed successfully.
MIND_C_API STATUS MSSetBackendPolicy(const char *policy);

/// \brief set device type context.
///
/// \param[in] device type name of the device i.e. CPU, GPU, Ascend.
MIND_C_API void MSSetDeviceTarget(const char *device);

/// \brief Get device type context.
///
/// \param[in] str_buf The char array to contain the device target string.
/// \param[in] str_len The size of the char array.
///
/// \return name of the device under current context
MIND_C_API STATUS MSGetDeviceTarget(char str_buf[], size_t str_len);

/// \brief Set Device ID context.
///
/// \param[in] deviceId Device ID.
MIND_C_API void MSSetDeviceId(uint32_t deviceId);

/// \brief Set flag for saving the graph.
///
/// \param[in] save_mode The flag to control the amount of saved graph files. There are 3 levels:
/// 1: Basic level. Only save frontend IR graph and some common backend IR graphs.
/// 2: Advanced level. Save all graphs except for verbose IR graphs and dot files.
/// 3: Full level. Save all IR graphs.
MIND_C_API void MSSetGraphsSaveMode(int save_mode);

MIND_C_API void MSSetGraphsSavePath(const char *save_path);

/// \brief Set flag for auto shape and type infer
///
/// \param res_mgr Resource Handle that manages the nodes of the funcGraph.
/// \param infer True: Use inner auto infer, False: Not use auto infer.
MIND_C_API void MSSetInfer(ResMgrHandle res_mgr, bool infer);

/// \brief Get flag for auto shape and type infer
///
/// \param res_mgr Resource Handle that manages the nodes of the funcGraph.
MIND_C_API bool MSGetInfer(ResMgrHandle res_mgr);

#ifdef __cplusplus
}
#endif
#endif  // MINDSPORE_CCSRC_C_API_INCLUDE_CONTEXT_H_
