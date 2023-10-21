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

#ifndef MINDSPORE_LITE_TOOLS_GRAPH_KERNEL_COMMON_UTILS_H_
#define MINDSPORE_LITE_TOOLS_GRAPH_KERNEL_COMMON_UTILS_H_
#include <vector>
#include <string>
#include <memory>
#include "nnacl/tensor_c.h"
#include "mindspore/ccsrc/kernel/kernel_build_info.h"
#include "include/backend/kernel_info.h"

constexpr auto kAkgKernelSo = "akgkernels.so";
namespace mindspore::graphkernel {
std::vector<std::string> SplitString(const std::string &raw_str, char delimiter);

int GetCustomShape(const std::string &attr, std::vector<std::vector<int>> *shapes);

int CalculateDynamicBatchSize(const TensorC *const *inputs, size_t inputs_size,
                              const std::vector<std::vector<int>> &shapes, const std::vector<size_t> &index,
                              int *batch);
void GetCustomIndex(const std::string &dynamic_input_index, std::vector<size_t> *index);
int GetCustomShape(const std::string &attr, std::vector<std::vector<int>> *shapes);
void SetKernelInfoWithFormatToAnfNode(const AnfNodePtr &node, const std::vector<std::string> &format);
kernel::KernelBuildInfoPtr GetKernelInfo(const AnfNodePtr &node);
void SetAnfKernelInfoFormatFromAToB(const AnfNodePtr &node_a, const CNodePtr &node_b,
                                    const std::vector<std::string> &formats);
std::string GetOutputFormatFromAnfNode(const AnfNodePtr &node, size_t output_idx);
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_LITE_TOOLS_GRAPH_KERNEL_COMMON_UTILS_H_
