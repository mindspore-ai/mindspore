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

#include <memory>

#include "backend/common/graph_kernel/expanders/op_desc_registry.h"
#include "tools/graph_kernel/converter/conv_tuning_expander.h"
#include "tools/graph_kernel/converter/expanders/activation.h"
#include "mindapi/base/types.h"
#include "ir/dtype.h"

namespace mindspore::graphkernel::expanders {
class CheckValidAttr : public Validator {
 public:
  bool Check(const OpDesc &e) override {
    const auto kernel_size = GetValue<std::vector<int64_t>>(e.Attrs().find("kernel_size")->second);
    const auto stride = GetValue<std::vector<int64_t>>(e.Attrs().find("stride")->second);
    const auto dilation = GetValue<std::vector<int64_t>>(e.Attrs().find("dilation")->second);
    if (InvalidConvAttr(kernel_size, stride, dilation)) {
      return false;
    }
    return true;
  }
};

class CheckDepthWise : public Validator {
 public:
  bool Check(const OpDesc &e) override {
    if (e.Attrs().count("is_depth_wise") != 0) {
      if (e.Attrs().count("group") == 0) {
        return false;
      }
      const auto group = GetValue<int64_t>(e.Attrs().find("group")->second);
      const auto c_in = e.InputsInfo()[0].shape[3];
      if (group != c_in) {
        return false;
      }
    }
    return true;
  }
};

class Conv2DFusion : public OpDesc {
 public:
  Conv2DFusion() {
    std::initializer_list<std::string> attrs{"kernel_size", "out_channel", "stride",    "dilation",
                                             "in_channel",  "pad_list",    "pad_mode",  "weight_coo",
                                             "weight_coi",  "weight_cio",  "weight_cii"};
    (void)validators_.emplace_back(std::make_unique<CheckAttr>(attrs));
    (void)validators_.emplace_back(std::make_unique<CheckDepthWise>());
    (void)validators_.emplace_back(std::make_unique<CheckValidAttr>());
  }
  ~Conv2DFusion() = default;

 protected:
  NodePtrList Expand(const NodePtrList &inputs) override {
    const auto &data = inputs[0];
    const auto &weight = inputs[1];
    auto data_shape = data->shape;
    auto data_format = data->format;

    // pad_top, pad_bottom, pad_left, pad_right
    std::vector<int64_t> pads = GetValue<std::vector<int64_t>>(attrs_["pad_list"]);
    auto n = data_shape[0];
    auto h = data_shape[1];
    auto w = data_shape[2];
    auto c_i_i = GetValue<int64_t>(attrs_["weight_cii"]);
    auto c_i_o = GetValue<int64_t>(attrs_["weight_cio"]);
    auto c_o_i = GetValue<int64_t>(attrs_["weight_coi"]);

    ShapeVector data_rs_shape{n, h, w, c_i_o, c_i_i};
    std::string conv_format = "NCHW" + std::to_string(c_i_i) + "c";
    auto data_tp = gb.Emit("LayoutTransform", {data},
                           {{"src_format", MakeValue(data_format)}, {"dst_format", MakeValue(conv_format)}});

    // PAD: NCHWc->NCHWc
    auto pad_n = pads[0];
    auto pad_h = pads[1];
    auto pad_w = pads[2];
    auto pad_c = pads[3];
    ShapeVector head_pad{0, 0, pad_n, pad_w, 0};
    ShapeVector tail_pad{0, 0, pad_h, pad_c, 0};

    inner::NodePtr data_pad;
    if (pad_n == 0 && pad_h == 0 && pad_w == 0 && pad_c == 0) {
      data_pad = data_tp;
    } else {
      data_pad =
        gb.Emit("PadAkg", {data_tp},
                {{"head", MakeValue(head_pad)}, {"tail", MakeValue(tail_pad)}, {"pad_val", MakeValue((int64_t)0)}});
    }

    // update attrs after pad
    auto updated_attrs = attrs_;
    updated_attrs["pad_mode"] = MakeValue("VALID");
    auto pad_val = MakeValue((int64_t)0);
    updated_attrs["pad_list"] = MakeValue({pad_val, pad_val, pad_val, pad_val});
    updated_attrs["data_format"] = MakeValue(kOpFormat_NC1HWC0);
    std::string conv_out_format = "NCHW" + std::to_string(c_o_i) + "c";
    updated_attrs["conv_out_format"] = MakeValue(conv_out_format);
    auto result_nchwc = gb.Emit("Conv2D", {data_pad, weight}, updated_attrs);

    inner::NodePtr result_nchwc_bias;
    constexpr size_t has_bias_inputs_size = 3;
    if (inputs.size() == has_bias_inputs_size) {
      const auto &bias = inputs[2];
      auto bias_dim = bias->shape[0];
      auto conv_c = result_nchwc->shape[4];
      ShapeVector bias_shape{1, bias_dim / conv_c, 1, 1, conv_c};
      auto bias_nchwc = gb.Reshape(bias, bias_shape);
      result_nchwc_bias = gb.Add(result_nchwc, bias_nchwc);
    } else {
      result_nchwc_bias = result_nchwc;
    }

    inner::NodePtr result_nchwc_act;
    if (attrs_.find("activation_type") != attrs_.end()) {
      auto act_type = GetValue<int64_t>(attrs_["activation_type"]);
      result_nchwc_act = GetActivationExpander(gb, {result_nchwc_bias}, act_type);
    } else {
      result_nchwc_act = result_nchwc_bias;
    }

    auto result_rs = gb.Emit("LayoutTransform", {result_nchwc_act},
                             {{"src_format", MakeValue(conv_out_format)}, {"dst_format", MakeValue(data_format)}});

    return {result_rs};
  }
};
EXPANDER_OP_DESC_REGISTER("Conv2DFusion", Conv2DFusion);
}  // namespace mindspore::graphkernel::expanders
