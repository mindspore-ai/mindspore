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

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>

#include "backend/common/graph_kernel/expanders/op_desc_registry.h"
#include "tools/graph_kernel/converter/expanders/activation.h"
#include "mindapi/base/types.h"
#include "ir/dtype.h"

namespace mindspore::graphkernel::expanders {
NodePtr GetPadResult(const inner::GraphBuilder &gb, const NodePtr &input_node, inner::DAttrs attrs,
                     const inner::DShape &input_shape, bool do_transform) {
  NodePtr input_pad;
  int64_t pad_mode = GetValue<int64_t>(attrs["pad_mode"]);
  if (pad_mode == PadMode::PAD) {
    std::vector<int64_t> pads = GetValue<std::vector<int64_t>>(attrs["pad"]);
    auto pad_n = pads[0];
    auto pad_h = pads[1];
    auto pad_w = pads[2];
    auto pad_c = pads[3];
    if (pad_n == 0 && pad_h == 0 && pad_w == 0 && pad_c == 0) {
      input_pad = input_node;
    } else {
      ShapeVector head_pad, tail_pad;
      if (do_transform) {
        head_pad = {0, 0, pad_n, pad_w, 0};
        tail_pad = {0, 0, pad_h, pad_c, 0};
      } else {
        head_pad = {0, pad_n, pad_w, 0};
        tail_pad = {0, pad_h, pad_c, 0};
      }
      input_pad =
        gb.Emit("PadAkg", {input_node},
                {{"head", MakeValue(head_pad)}, {"tail", MakeValue(tail_pad)}, {"pad_val", MakeValue((int64_t)0)}});
    }
  } else if (pad_mode == PadMode::SAME) {
    auto input_h = input_shape[1];
    auto input_w = input_shape[2];
    auto stride_h = GetValue<std::vector<int64_t>>(attrs["strides"])[0];
    auto stride_w = GetValue<std::vector<int64_t>>(attrs["strides"])[1];
    auto kernel_h = GetValue<std::vector<int64_t>>(attrs["kernel_size"])[0];
    auto kernel_w = GetValue<std::vector<int64_t>>(attrs["kernel_size"])[1];
    int64_t pad_h, pad_w;
    if (input_h % stride_h == 0) {
      pad_h = std::max(kernel_h - stride_h, int64_t(0));
    } else {
      pad_h = std::max(kernel_h - (input_h % stride_h), int64_t(0));
    }
    if (input_w % stride_w == 0) {
      pad_w = std::max(kernel_w - stride_w, int64_t(0));
    } else {
      pad_w = std::max(kernel_w - (input_w % stride_w), int64_t(0));
    }
    if (pad_h == 0 && pad_w == 0) {
      input_pad = input_node;
    } else {
      ShapeVector head_pad, tail_pad;
      auto pad_top = pad_h / 2;
      auto pad_bottom = pad_h - pad_top;
      auto pad_left = pad_w / 2;
      auto pad_right = pad_w - pad_left;
      if (do_transform) {
        head_pad = {0, 0, pad_top, pad_left, 0};
        tail_pad = {0, 0, pad_bottom, pad_right, 0};
      } else {
        head_pad = {0, pad_top, pad_left, 0};
        tail_pad = {0, pad_bottom, pad_right, 0};
      }
      input_pad =
        gb.Emit("PadAkg", {input_node},
                {{"head", MakeValue(head_pad)}, {"tail", MakeValue(tail_pad)}, {"pad_val", MakeValue((int64_t)0)}});
    }
  } else {
    input_pad = input_node;
  }
  return input_pad;
}

class PoolFusion : public OpDesc {
 public:
  explicit PoolFusion(const std::string &pool_type) : pool_type_(pool_type) {}
  ~PoolFusion() = default;

 protected:
  NodePtrList Expand(const NodePtrList &inputs) override {
    const auto &input = inputs[0];
    auto input_shape = input->shape;
    auto input_format = input->format;
    bool do_transform = attrs_.count("layout_axis") != 0;
    NodePtr input_tp;
    std::string layout_format;
    if (do_transform) {
      auto layout_axis = GetValue<int64_t>(attrs_["layout_axis"]);
      layout_format = "NCHW" + std::to_string(layout_axis) + "c";
      input_tp = gb.Emit("LayoutTransform", {input},
                         {{"src_format", MakeValue(input_format)}, {"dst_format", MakeValue(layout_format)}});
    } else {
      input_tp = input;
      layout_format = "NHWC";
    }

    NodePtr input_pad = GetPadResult(gb, input_tp, attrs_, input_shape, do_transform);

    inner::DAttrs attr_list = {
      {"pool_type", MakeValue(pool_type_)}, {"data_layout", MakeValue(layout_format)}, {"strides", attrs_["strides"]}};
    if (attrs_["kernel_size"] != nullptr) {
      attr_list["kernel_size"] = attrs_["kernel_size"];
    }
    if (attrs_.count("global") != 0) {
      attr_list["global"] = attrs_["global"];
    } else {
      attr_list["global"] = MakeValue(false);
    }
    if (attrs_.count("round_mode") != 0) {
      attr_list["round_mode"] = attrs_["round_mode"];
    } else {
      attr_list["round_mode"] = MakeValue(0);
    }
    auto pool_res = gb.Emit("Pool2D", {input_pad}, attr_list);
    NodePtr result;
    if (do_transform) {
      result = gb.Emit("LayoutTransform", {pool_res},
                       {{"src_format", MakeValue(layout_format)}, {"dst_format", MakeValue(input_format)}});
    } else {
      result = pool_res;
    }
    return {result};
  }

 private:
  std::string pool_type_;
};

class MaxPoolFusion : public PoolFusion {
 public:
  MaxPoolFusion() : PoolFusion("max") {
    (void)validators_.emplace_back(std::make_unique<CheckActivationType>(ActivationType::NO_ACTIVATION));
  }
  ~MaxPoolFusion() = default;
};
EXPANDER_OP_DESC_REGISTER("MaxPoolFusion", MaxPoolFusion);

class AvgPoolFusion : public PoolFusion {
 public:
  AvgPoolFusion() : PoolFusion("avg") {
    (void)validators_.emplace_back(std::make_unique<CheckActivationType>(ActivationType::NO_ACTIVATION));
  }
  ~AvgPoolFusion() = default;
};
EXPANDER_OP_DESC_REGISTER("AvgPoolFusion", AvgPoolFusion);
}  // namespace mindspore::graphkernel::expanders
