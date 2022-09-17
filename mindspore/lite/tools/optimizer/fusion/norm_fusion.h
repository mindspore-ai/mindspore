/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_NORM_FUSION_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_NORM_FUSION_H_

#include <vector>
#include <memory>
#include <string>
#include <map>
#include "schema/inner/model_generated.h"
#include "tools/optimizer/common/pattern_process_pass_extends.h"
#include "include/common/utils/utils.h"
#include "tools/optimizer/common/gllo_utils.h"

namespace mindspore {
namespace opt {

/// fuse layer_norm or instance_norm into one operator
class NormFusion : public LitePatternProcessPass {
 public:
  explicit NormFusion(const std::string &name = "NormFusion", bool multigraph = true)
      : LitePatternProcessPass(name, multigraph) {
    InitShapeSizeInferFuncMap();
  }

  ~NormFusion() override = default;

  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 protected:
  bool Init() const;

 private:
  void InitShapeSizeInferFuncMap();
  bool GetNormTypeAndAxis(const FuncGraphPtr &func_graph, const CNodePtr &input_cnode,
                          const std::vector<int> &mean_axes, const std::vector<int> &params_shape,
                          schema::PrimitiveType *type, int *begin_norm_axis, int *begin_params_axis) const;
  bool CheckPattern(const FuncGraphPtr &func_graph, const EquivPtr &equiv, schema::PrimitiveType *type, float *epsilon,
                    int *begin_norm_axis, int *begin_params_axis) const;
  CNodePtr CreateNormNode(const FuncGraphPtr &func_graph, const EquivPtr &equiv, const schema::PrimitiveType type,
                          float epsilon, int begin_norm_axis, int begin_params_axis) const;
  std::map<string, int> ShapeSizeInfer(const FuncGraphPtr &func_graph) const;

 protected:
  mutable VarPtr input_ = nullptr;
  mutable VarPtr mean1_ = nullptr;
  mutable VarPtr mean1_axes_ = nullptr;
  mutable VarPtr mean2_ = nullptr;
  mutable VarPtr mean2_axes_ = nullptr;
  mutable VarPtr gamma_ = nullptr;
  mutable VarPtr beta_ = nullptr;
  mutable VarPtr epsilon_ = nullptr;
  std::map<schema::PrimitiveType, std::function<int(std::vector<int>, const schema::PrimitiveT &)>>
    shape_size_infer_registry_;
};

/// fuse tf layer_norm or instance_norm into one operator
class TfNormFusion : public NormFusion {
 public:
  explicit TfNormFusion(const std::string &name = "TfNormFusion", bool multigraph = true)
      : NormFusion(name, multigraph) {}

  ~TfNormFusion() override = default;

 private:
  const BaseRef DefinePattern() const override;
};

/// fuse onnx layer_norm into one operator
class OnnxLayerNormFusion : public NormFusion {
 public:
  explicit OnnxLayerNormFusion(const std::string &name = "OnnxLayerNormFusion", bool multigraph = true)
      : NormFusion(name, multigraph) {}

  ~OnnxLayerNormFusion() override = default;

 private:
  const BaseRef DefinePattern() const override;
};

/// fuse onnx layer_norm into one operator with little variance
class OnnxLayerNormFusion2 : public NormFusion {
 public:
  explicit OnnxLayerNormFusion2(const std::string &name = "OnnxLayerNormFusion2", bool multigraph = true)
      : NormFusion(name, multigraph) {}

  ~OnnxLayerNormFusion2() override = default;

 private:
  const BaseRef DefinePattern() const override;
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_FUSION_NORM_FUSION_H_
