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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_SHAPE_INFO_CONTAINER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_SHAPE_INFO_CONTAINER_H_

#include <vector>
#include <string>
#include <map>
#include <memory>
#include "tools/converter/micro/coder/config.h"
#include "include/model.h"
#include "src/tensor.h"

namespace mindspore::lite::micro {
class OperatorCoder;
class ShapeInfoContainer {
 public:
  ShapeInfoContainer() = default;
  ~ShapeInfoContainer() = default;

  int Init(const std::vector<std::unique_ptr<OperatorCoder>> &nodes_coder,
           const std::map<Tensor *, std::vector<std::vector<int>>> &graph_inputs);

  const std::map<const Tensor *, std::vector<std::vector<int>>> &GetVarTensorInfos() const {
    return var_tensor_shapes_;
  }

  const std::map<int, std::vector<int>> GetShapesWholeScenes() const { return shapes_whole_scenes_; }

  const std::map<const Tensor *, std::vector<std::string>> &GetWholeTemplateShape() const { return shape_templates_; }

  std::vector<std::string> GetTemplateShape(const Tensor *tensor) const;

  std::vector<int> GetRealNums(const std::string &shape_var) const;

 private:
  int DoInferShape(const std::vector<Tensor *> &in_tensors, std::vector<Tensor *> *out_tensors, OpParameter *op_param,
                   const void *primitive);
  int DetermineShapeVarInfos();
  std::map<const Tensor *, std::vector<std::vector<int>>> var_tensor_shapes_;
  std::map<const Tensor *, std::vector<std::string>> shape_templates_;
  std::map<std::string, std::vector<int>> shape_to_nums_;
  std::map<int, std::vector<int>> shapes_whole_scenes_;
};
}  // namespace mindspore::lite::micro
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_SHAPE_INFO_CONTAINER_H_
