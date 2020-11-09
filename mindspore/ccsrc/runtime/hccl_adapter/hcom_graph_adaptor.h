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

#ifndef MINDSPORE_RUNTIME_HCCL_ADAPTER_HCOM_GRAPH_ADAPTOR_H
#define MINDSPORE_RUNTIME_HCCL_ADAPTER_HCOM_GRAPH_ADAPTOR_H

#include <string>
#include <map>
#include <memory>
#include "mindspore/core/ir/anf.h"
#include "common/opskernel/ops_kernel_info_store.h"

extern "C" {
ge::Status Initialize(const std::map<std::string, std::string> &);
ge::Status Finalize();
void GetOpsKernelInfoStores(std::map<std::string, std::shared_ptr<ge::OpsKernelInfoStore>> &);
}

#endif  // MINDSPORE_RUNTIME_HCCL_ADAPTER_HCOM_GRAPH_ADAPTOR_H
