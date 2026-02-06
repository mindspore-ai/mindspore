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

#include <algorithm>
#include <memory>
#include <vector>
#include <string>
#include "backend/ms_backend/graph_fusion/expanders/op_desc_registry.h"
#include "backend/ms_backend/graph_fusion/model/op_node.h"

namespace mindspore::graphkernel::expanders {
constexpr int M_ALIGN = 32;
constexpr int N_ALIGN = 32;
constexpr int K_ALIGN = 16;
constexpr int K_LIMIT = 800;
constexpr int64_t MNK_LIMIT = 30000000000;
constexpr int N0_CHANNEL_ALIGN = 32;
constexpr int N1_CHANNEL_ALIGN = 32;
constexpr int C_CHANNEL_ALIGN = 16;
constexpr int OUT_NHW_ALIGN = 128;
constexpr size_t kIdxN = 0;
constexpr size_t kIdxH = 1;
constexpr size_t kIdxW = 2;
constexpr size_t kIdxC = 3;
constexpr size_t kTop = 0;
constexpr size_t kBottom = 1;
constexpr size_t kLeft = 2;
constexpr size_t kRight = 3;

using inner::DFormat;
// Conv2D expander
// Currently, only Conv2D that meets several conditions can be expanded, other cases will be skipped.
// Conditions to expand:
//   inputs are NHWC format and float16.
//   attr groups and group are 1.
//   attr dilation are all 1.
//   N channel of inputs > 16.
//   C channel of inputs > 8.
//   output N*H*W are multiplies of 128.
class Conv2D : public OpDesc {
 public:
  Conv2D() {
    auto support_format = std::make_unique<SupportFormat>();
    support_format->AddFormat({kOpFormat_DEFAULT, kOpFormat_DEFAULT});
    support_format->AddFormat({kOpFormat_NHWC, kOpFormat_NHWC});
    (void)validators_.emplace_back(std::move(support_format));
    std::initializer_list<std::string> attrs{"format", "pad_list",    "pad_mode", "groups",
                                             "group",  "kernel_size", "stride",   "dilation"};
    (void)validators_.emplace_back(std::make_unique<CheckAttr>(attrs));
  }
  ~Conv2D() = default;

 private:
  bool OptimizeToMatmul() {
    auto h = shape_1_pad_[kIdxH];
    auto w = shape_1_pad_[kIdxW];
    const auto all_one = ShapeVector{1, 1, 1, 1};
    if (h == 1 && w == 1 && stride_ == all_one && dilation_ == all_one && m_ % M_ALIGN == 0 && n_ % N_ALIGN == 0 &&
        k_ % K_ALIGN == 0) {
      return true;
    }
    return false;
  }

  // common check for inputs and attrs
  bool CommonCheck() {
    auto type_0 = inputs_info_[0].type;
    auto type_1 = inputs_info_[1].type;
    if (type_0 != TypeId::kNumberTypeFloat16 || type_1 != TypeId::kNumberTypeFloat16) {
      MS_LOG(INFO) << "For 'Conv2D', inputs data type should be both float16, but got " << type_0 << " and " << type_1;
      return false;
    }

    if (inputs_info_[0].format != kOpFormat_NHWC && inputs_info_[1].format != kOpFormat_NHWC &&
        GetValue<std::string>(attrs_["format"]) != kOpFormat_NHWC) {
      MS_LOG(INFO) << "For now, Conv2D would expand only when inputs are NHWC format";
      return false;
    }

    auto groups = GetValue<int64_t>(attrs_["groups"]);
    auto group = GetValue<int64_t>(attrs_["group"]);
    if (groups != 1 || group != 1) {
      MS_LOG(INFO) << "For 'Conv2D', value of attr 'groups' and 'group' should be both 1, but got " << groups << " and "
                   << group;
      return false;
    }

    if (dilation_ != ShapeVector{1, 1, 1, 1}) {
      MS_LOG(INFO) << "For 'Conv2D', value of attr 'dilation' should be [1, 1, 1, 1], but got " << dilation_;
      return false;
    }
    return true;
  }

 protected:
  void Init() override {
    dst_type_ = outputs_info_[0].type;
    dst_format_ = outputs_info_[0].format;
    shape_0_pad_ = inputs_info_[0].shape;
    shape_1_pad_ = inputs_info_[1].shape;
    stride_ = GetAxisList(attrs_["stride"]);
    dilation_ = GetAxisList(attrs_["dilation"]);
  }
  bool CheckInputs() override {
    if (!CommonCheck()) {
      return false;
    }

    auto pad_list = GetAxisList(attrs_["pad_list"]);
    auto shape_0 = inputs_info_[0].shape;
    auto shape_1 = inputs_info_[1].shape;
    constexpr size_t i4 = 4;
    if ((pad_list.size() != i4) || (shape_0.size() != i4) || (shape_1.size() != i4) || (stride_.size() != i4)) {
      MS_LOG(INFO)
        << "For 'Conv2D', pad_list, shape of input0, shape of input1 and stride all should be of size 4, but got "
        << pad_list.size() << ", " << shape_0.size() << ", " << shape_1.size() << ", " << stride_.size();
      return false;
    }

    auto n0 = shape_0[kIdxN];
    auto h0 = shape_0[kIdxH];
    auto w0 = shape_0[kIdxW];
    auto c0 = shape_0[kIdxC];

    auto n1 = shape_1[kIdxN];
    auto h1 = shape_1[kIdxH];
    auto w1 = shape_1[kIdxW];
    auto c1 = shape_1[kIdxC];

    if (n0 % N0_CHANNEL_ALIGN != 0) {
      MS_LOG(INFO) << "For 'Conv2D', N channel of first input should be multiples of " << N0_CHANNEL_ALIGN
                   << ", but got " << n0;
      return false;
    }
    if (n1 % N1_CHANNEL_ALIGN != 0) {
      MS_LOG(INFO) << "For 'Conv2D', N channel of second input should be multiples of " << N1_CHANNEL_ALIGN
                   << ", but got " << n1;
      return false;
    }
    if (c0 != c1 || (c0 % C_CHANNEL_ALIGN) != 0) {
      MS_LOG(INFO) << "For 'Conv2D', C channel of inputs should be multiples of " << C_CHANNEL_ALIGN << ", but got "
                   << c0 << " and " << c1;
      return false;
    }

    // n0 pad
    n0 = ((n0 + N0_CHANNEL_ALIGN - 1) / N0_CHANNEL_ALIGN) * N0_CHANNEL_ALIGN;
    // h0, w0 pad
    has_pad_ = inner::Conv2dOp::HadPad(pad_list, GetValue<std::string>(attrs_["pad_mode"]));
    if (has_pad_) {
      h0 = h0 + pad_list[kTop] + pad_list[kBottom];
      w0 = w0 + pad_list[kLeft] + pad_list[kRight];
    }

    // c0, c1 pad
    c0 = (c0 + C_CHANNEL_ALIGN - 1) / C_CHANNEL_ALIGN * C_CHANNEL_ALIGN;
    c1 = c0;

    // n1 pad
    n1 = (n1 + N1_CHANNEL_ALIGN - 1) / N1_CHANNEL_ALIGN * N1_CHANNEL_ALIGN;

    // check if can optimize to matmul
    m_ = n0 * h0 * w0;
    n_ = n1;
    k_ = c1;
    can_optimize_to_matmul_ = OptimizeToMatmul();
    // expand requirement
    if (can_optimize_to_matmul_) {
      if (k_ > K_LIMIT) {
        MS_LOG(INFO) << "For 'Conv2D', if transformed to 'MatMul', C0 should not be larger than " << K_LIMIT
                     << ", but got " << k_;
        return false;
      }
      if (m_ * n_ * k_ >= MNK_LIMIT) {
        MS_LOG(INFO) << "For 'Conv2D', if transformed to 'MatMul', The total size should not be larger than "
                     << MNK_LIMIT << ", but got " << m_ * n_ * k_;
        return false;
      }
    } else {
      constexpr size_t idx_h = 2;
      constexpr size_t idx_w = 3;
      auto out_h = (h0 - h1) / stride_[idx_h] + 1;
      auto out_w = (w0 - w1) / stride_[idx_w] + 1;
      if ((n0 * out_h * out_w) % OUT_NHW_ALIGN != 0) {
        MS_LOG(INFO) << "For 'Conv2D', N(" << n0 << ") * H(" << out_h << ") * W(" << out_w
                     << ") of output should be multiplies of " << OUT_NHW_ALIGN;
        return false;
      }
      if (stride_ != ShapeVector{1, 1, 2, 2}) {
        MS_LOG(INFO) << "For 'Conv2D', value of attr 'stride' should be [1, 1, 2, 2], but got " << stride_;
        return false;
      }
    }

    shape_0_pad_ = ShapeVector{n0, h0, w0, c0};
    shape_1_pad_ = ShapeVector{n1, h1, w1, c1};

    return true;
  }

  NodePtrList Expand(const NodePtrList &inputs) override {
    auto input_0 = inputs[0];
    auto input_1 = inputs[1];
    auto shape_0 = input_0->shape;
    auto shape_1 = input_1->shape;
    auto pad_value = 0;

    // input_0 pad
    ShapeVector input_0_pad_before = {0, 0, 0, 0};
    ShapeVector input_0_pad_after = {0, 0, 0, 0};
    if (has_pad_) {
      auto pad_list = GetAxisList(attrs_["pad_list"]);
      input_0_pad_before = {0, pad_list[kTop], pad_list[kLeft], 0};
      input_0_pad_after = {0, pad_list[kBottom], pad_list[kRight], 0};
    }
    input_0_pad_after[kIdxN] = shape_0_pad_[kIdxN] - shape_0[kIdxN];
    input_0_pad_after[kIdxC] = shape_0_pad_[kIdxC] - shape_0[kIdxC];

    if (input_0_pad_before != ShapeVector{0, 0, 0, 0} || input_0_pad_after != ShapeVector{0, 0, 0, 0}) {
      input_0 = gb.Emit("PadAkg", {input_0},
                        {{"head", MakeValue(input_0_pad_before)},
                         {"tail", MakeValue(input_0_pad_after)},
                         {"pad_val", MakeValue(pad_value)}});
    }

    // input_1 pad
    ShapeVector input_1_pad_before = {0, 0, 0, 0};
    ShapeVector input_1_pad_after = {shape_1_pad_[kIdxN] - shape_1[kIdxN], 0, 0, shape_1_pad_[kIdxC] - shape_1[kIdxC]};
    if (input_1_pad_after != ShapeVector{0, 0, 0, 0}) {
      input_1 = gb.Emit("PadAkg", {input_1},
                        {{"head", MakeValue(input_1_pad_before)},
                         {"tail", MakeValue(input_1_pad_after)},
                         {"pad_val", MakeValue(pad_value)}});
    }
    NodePtr result;
    if (can_optimize_to_matmul_) {
      auto a = gb.Reshape(input_0, ShapeVector{m_, k_});
      auto b = gb.Reshape(input_1, ShapeVector{n_, k_});
      auto c = gb.MatMul(a, b, dst_type_, false, true);
      result = gb.Emit(
        "Reshape",
        {c, gb.Tensor(ShapeVector{shape_0_pad_[kIdxN], shape_0_pad_[kIdxH], shape_0_pad_[kIdxW], shape_1_pad_[kIdxN]})},
        {{"format", MakeValue(dst_format_)}});
    } else {
      auto attrs = attrs_;
      attrs["pad_list"] = MakeValue(ShapeVector{0, 0, 0, 0});
      attrs["dst_type"] = TypeIdToType(dst_type_);
      result = gb.Emit("Conv2D", {input_0, input_1}, attrs);
    }
    // unpad
    ShapeVector unpad_after = {input_0_pad_after[kIdxN], 0, 0, input_1_pad_after[kIdxN]};
    if (unpad_after != ShapeVector{0, 0, 0, 0}) {
      result = gb.Emit("UnPadAkg", {result}, {{"tail", MakeValue(unpad_after)}});
    }
    return {result};
  }

  TypeId dst_type_ = TypeId::kTypeUnknown;
  DFormat dst_format_{};
  bool has_pad_ = false;
  bool can_optimize_to_matmul_ = false;
  ShapeVector shape_0_pad_{};
  ShapeVector shape_1_pad_{};
  int64_t m_ = 0;
  int64_t n_ = 0;
  int64_t k_ = 0;
  ShapeVector stride_{};
  ShapeVector dilation_{};
};
EXPANDER_OP_DESC_REGISTER("Conv2D", Conv2D);
}  // namespace mindspore::graphkernel::expanders
