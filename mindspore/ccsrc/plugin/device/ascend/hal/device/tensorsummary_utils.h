/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_DEVICE_TENSORSUMMARY_UTILS_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_DEVICE_TENSORSUMMARY_UTILS_H_

#include <vector>
#include <string>
#include <functional>
#include "acl/acl_tdt.h"
#include "ir/tensor.h"

namespace mindspore::device::ascend {
const std::vector<string> summary_channel_names{"ms_tensor_summary", "ms_image_summary", "ms_scalar_summary",
                                                "ms_histogram_summary"};
void SummaryReceiveData(acltdtDataset *acl_dataset, const string &channel_name);
};  // namespace mindspore::device::ascend

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_DEVICE_TENSORSUMMARY_UTILS_H_
