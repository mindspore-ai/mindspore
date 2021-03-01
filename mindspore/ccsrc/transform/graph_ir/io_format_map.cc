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

#include "transform/graph_ir/io_format_map.h"

namespace mindspore {
namespace transform {
std::unordered_map<std::string, std::string> IOFormatMap::io_format_map_ = {{"BasicLSTMCell", "ND"},
                                                                            {"BasicLSTMCellInputGrad", "ND"},
                                                                            {"BasicLSTMCellCStateGrad", "ND"},
                                                                            {"Dequant", "ND"},
                                                                            {"DynamicGRUV2", "ND"},
                                                                            {"DynamicGRUV2Grad", "ND"},
                                                                            {"DynamicRNN", "ND"},
                                                                            {"DynamicRNNGrad", "ND"},
                                                                            {"MatMul", "ND"},
                                                                            {"Quant", "ND"},
                                                                            {"BasicLSTMCellWeightGrad", "HWCN"},
                                                                            {"ExtractImagePatches", "NCHW"},
                                                                            {"Conv3D", "format"},
                                                                            {"Conv3DBackpropFilter", "format"},
                                                                            {"Conv3DBackpropInput", "format"},
                                                                            {"Conv3DTranspose", "format"}};
std::unordered_map<std::string, std::string> &IOFormatMap::get() { return io_format_map_; }
}  // namespace transform
}  // namespace mindspore
