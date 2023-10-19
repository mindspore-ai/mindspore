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
mindspore::HashMap<std::string, std::string> IOFormatMap::io_format_map_ = {{"BNTrainingReduce", "NCHW"},
                                                                            {"BNTrainingUpdate", "NCHW"},
                                                                            {"BNTrainingUpdateGrad", "NCHW"},
                                                                            {"BNTrainingReduceGrad", "NCHW"},
                                                                            {"BNInfer", "NCHW"},
                                                                            {"BNInferGrad", "NCHW"},
                                                                            {"Conv2D", "NCHW"},
                                                                            {"DepthwiseConv2D", "NCHW"},
                                                                            {"DepthwiseConv2dNative", "NCHW"},
                                                                            {"Conv2DBackpropInput", "NCHW"},
                                                                            {"Conv2DBackpropFilter", "NCHW"},
                                                                            {"BasicLSTMCellWeightGrad", "HWCN"},
                                                                            {"ExtractImagePatches", "NCHW"},
                                                                            {"ApplyMomentum", "NCHW"},
                                                                            {"FullConnection", "NCHW"},
                                                                            {"PReLU", "NCHW"},
                                                                            {"Scale", "NCHW"},
                                                                            {"GridSampler2D", "NCHW"},
                                                                            {"ResizeBilinearV2", "NCHW"},
                                                                            {"Conv3D", "format"},
                                                                            {"MaxPool3D", "NCDHW"},
                                                                            {"MaxPool3DGrad", "NCDHW"},
                                                                            {"AvgPool3D", "NCDHW"},
                                                                            {"AvgPool3DGrad", "NCDHW"},
                                                                            {"Conv3DBackpropFilter", "format"},
                                                                            {"Conv3DBackpropInput", "format"},
                                                                            {"Conv3DTranspose", "format"}};
mindspore::HashMap<std::string, std::string> &IOFormatMap::get() { return io_format_map_; }
}  // namespace transform
}  // namespace mindspore
