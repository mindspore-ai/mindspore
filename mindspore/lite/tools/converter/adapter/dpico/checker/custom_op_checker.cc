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

#include "checker/custom_op_checker.h"
#include <vector>
#include <string>

namespace mindspore {
namespace dpico {
bool CustomOpChecker::Check(api::CNodePtr, int32_t output_num, mindspore::Format) { return true; }
OpCheckerRegistrar g_DecBboxChecker("DecBBox", new CustomOpChecker());
OpCheckerRegistrar g_DetectionOutputChecker("DetectionOutput", new CustomOpChecker());
OpCheckerRegistrar g_ExtractChecker("Extract", new CustomOpChecker());
OpCheckerRegistrar g_PassThroughChecker("PassThrough", new CustomOpChecker());
OpCheckerRegistrar g_PSROIPoolingChecker("PsRoiPool", new CustomOpChecker());
OpCheckerRegistrar g_ROIPoolingChecker("ROIPooling", new CustomOpChecker());
OpCheckerRegistrar g_ShuffleChannelChecker("ShuffleChannel", new CustomOpChecker());
OpCheckerRegistrar g_ThresholdChecker("Threshold", new CustomOpChecker());
OpCheckerRegistrar g_RNNChecker("Rnn", new CustomOpChecker());
OpCheckerRegistrar g_BiLstmChecker("BiLstm", new CustomOpChecker());
OpCheckerRegistrar g_GruChecker("Gru", new CustomOpChecker());
OpCheckerRegistrar g_NopChecker("Nop", new CustomOpChecker());
}  // namespace dpico
}  // namespace mindspore
