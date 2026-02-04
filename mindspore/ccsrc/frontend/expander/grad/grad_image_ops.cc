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
#include <set>
#include "frontend/expander/bprop/bprop_irbuilder.h"
#include "frontend/expander/grad/grad_utils.h"
#include "utils/ms_context.h"

namespace mindspore::expander::bprop {
REG_BPROP_BUILDERS_BEGIN(GradImageOps)
REG_BPROP_BUILDER("ResizeBicubic").SetUnusedInputs({i1, i4}).SetBody(BODYFUNC(ib) {
  auto images = ib->GetInput(i0);
  const auto &size = ib->GetInput(i1);
  const auto &align_corners = ib->GetInput(i2);
  const auto &half_pixel_centers = ib->GetInput(i3);
  auto dout = ib->GetInput(i5);
  // ResizeBicubicGrad do not support fp16 on ascend platform
  auto type_id = ib->GetDtypeId(dout);
  if (type_id == TypeId::kNumberTypeFloat16) {
    dout = ib->Cast(dout, kFloat32);
    images = ib->Cast(images, kFloat32);
  } else if (type_id == TypeId::kNumberTypeFloat64) {
    dout = ib->Cast(dout, kFloat32);
  }
  auto dx = ib->Emit("ResizeBicubicGrad", {dout, images, align_corners, half_pixel_centers});
  if (type_id == TypeId::kNumberTypeFloat16) {
    dx = ib->Cast(dx, type_id);
  }
  return {dx, ib->OutZeros(size), ib->OutZeros(align_corners), ib->OutZeros(half_pixel_centers)};
});

REG_BPROP_BUILDER("CropAndResize").SetUnusedInputs({i3, i4}).SetBody(BODYFUNC(ib) {
  std::set<TypeId> allowed_types = {kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeFloat64};
  auto method = GetValue<std::string>(ib->GetAttr("method"));
  auto target = ib->GetTargetFromContext();
  auto is_ascend_cpu = (target == kAscendDevice || target == kCPUDevice);
  auto x = ib->GetInput(i0);
  const auto &boxes = ib->GetInput(i1);
  const auto &box_index = ib->GetInput(i2);
  const auto &crop_size = ib->GetInput(i3);
  const auto &dout = ib->GetInput(i5);
  if (method != "bilinear") {
    if (!is_ascend_cpu) {
      return {ib->OutZeros(x), ib->OutZeros(boxes), ib->OutZeros(box_index), ib->OutZeros(crop_size)};
    }
  }
  auto image_type = ib->GetDtype(x);
  if (allowed_types.count(image_type->type_id()) != 0) {
    x = ib->Cast(x, kFloat32);
  }
  auto x_shape = ib->GetShape(x);
  NodePtr image_size{nullptr};
  if (IsDynamic(x_shape)) {
    image_size = ib->Emit("TupleToTensor", {ib->Shape(x), ib->Value<int64_t>(kInt32->type_id())});
  } else {
    image_size = ib->Tensor(x_shape, kInt32);
  }
  const auto max_byte = static_cast<int64_t>(2e9);  // max bytes of image gradient
  NodePtr dimage = x->need_compute_grad_out()
                     ? ib->Emit("CropAndResizeGradImage", {dout, boxes, box_index, image_size},
                                {{"method", MakeValue(method)}, {"T", image_type}, {"max_Byte", MakeValue(max_byte)}})
                     : ib->OutZeros(x);
  NodePtr dbox = boxes->need_compute_grad_out() ? ib->Emit("CropAndResizeGradBoxes", {dout, x, boxes, box_index},
                                                           {{"method", MakeValue("bilinear")}})
                                                : ib->OutZeros(boxes);
  return {dimage, dbox, ib->OutZeros(box_index), ib->OutZeros(crop_size)};
});

REG_BPROP_BUILDER("ScaleAndTranslate").SetUnusedInputs({i1, i4}).SetBody(BODYFUNC(ib) {
  const auto &images = ib->GetInput(i0);
  const auto &size = ib->GetInput(i1);
  const auto &scale = ib->GetInput(i2);
  const auto &translation = ib->GetInput(i3);
  const auto &dout = ib->GetInput(i5);
  auto images_dtype = ib->GetDtype(images);
  auto images_fp32 = (images_dtype->type_id() != kNumberTypeFloat32) ? ib->Cast(images, kFloat32) : images;
  auto grad0_fp32 = ib->Emit("ScaleAndTranslateGrad", {dout, images_fp32, scale, translation},
                             {{"kernel_type", ib->GetAttr("kernel_type")}, {"antialias", ib->GetAttr("antialias")}});
  auto grad0 = (images_dtype->type_id() != kNumberTypeFloat32) ? ib->Cast(grad0_fp32, images_dtype) : grad0_fp32;
  return {grad0, ib->OutZeros(size), ib->OutZeros(scale), ib->OutZeros(translation)};
});

REG_BPROP_BUILDER("RGBToHSV").SetBody(BODYFUNC(ib) {
  auto images = ib->GetInput(i0);
  auto out = ib->GetInput(i1);
  auto dout = ib->GetInput(i2);
  auto images_dtype = ib->GetDtype(images);
  if (images_dtype->type_id() != kNumberTypeFloat32) {
    images = ib->Cast(images, kFloat32);
    out = ib->Cast(out, kFloat32);
    dout = ib->Cast(dout, kFloat32);
  }
  auto tensor_0 = ib->Tensor(0, kFloat32);
  auto tensor_1 = ib->Tensor(1, kFloat32);
  auto negative_one = ib->ValueByType(-1, kFloat32);
  auto value_sixty = ib->ValueByType(60, kFloat32);
  auto value_360 = ib->ValueByType(360, kFloat32);
  auto crcp = [&tensor_1, &ib](const NodePtr &x) { return ib->DivNoNan(tensor_1, x); };

  auto reds = ib->StridedSlice(images, {{-1, {0}}});
  auto greens = ib->StridedSlice(images, {{-1, {1}}});
  auto blues = ib->StridedSlice(images, {{-1, {2}}});
  auto dsr1 = ib->Greater(reds, tensor_0, kFloat32);
  auto saturation = ib->StridedSlice(out, {{-1, {1}}});
  auto value = ib->StridedSlice(out, {{-1, {2}}});
  auto red_biggest = ib->Cast(ib->LogicalAnd(ib->GreaterEqual(reds, blues), ib->GreaterEqual(reds, greens)), kFloat32);
  auto green_biggest = ib->Cast(ib->LogicalAnd(ib->Greater(greens, reds), ib->GreaterEqual(greens, blues)), kFloat32);
  auto blue_biggest = ib->Cast(ib->LogicalAnd(ib->Greater(blues, reds), ib->Greater(blues, greens)), kFloat32);
  auto red_smallest = ib->Cast(ib->LogicalAnd(ib->Less(reds, blues), ib->Less(reds, greens)), kFloat32);
  auto green_smallest = ib->Cast(ib->LogicalAnd(ib->LessEqual(greens, reds), ib->Less(greens, blues)), kFloat32);
  auto blue_smallest = ib->Cast(ib->LogicalAnd(ib->LessEqual(blues, reds), ib->LessEqual(blues, greens)), kFloat32);
  auto dv_dr = red_biggest;
  auto dv_dg = green_biggest;
  auto dv_db = blue_biggest;

  auto dsr2 = red_biggest * (green_smallest * greens + blue_smallest * blues) * crcp(ib->Square(reds));
  auto dsr3 = ib->Muls(red_smallest, negative_one) * crcp((green_biggest * greens) + (blue_biggest * blues));
  auto ds_dr = dsr1 * (dsr2 + dsr3);
  auto dsg1 = ib->Greater(greens, tensor_0, kFloat32);
  auto dsg2 = green_biggest * (red_smallest * reds + blue_smallest * blues) * crcp(ib->Square(greens));
  auto dsg3 = ib->Muls(green_smallest, negative_one) * crcp((red_biggest * reds) + (blue_biggest * blues));
  auto ds_dg = dsg1 * (dsg2 + dsg3);

  auto dsb1 = ib->Greater(blues, tensor_0, kFloat32);
  auto dsb2 = blue_biggest * (green_smallest * greens + red_smallest * reds) * crcp(ib->Square(blues));
  auto dsb3 = ib->Muls(blue_smallest, negative_one) * crcp((green_biggest * greens) + (red_biggest * reds));
  auto ds_db = dsb1 * (dsb2 + dsb3);

  auto dhr1 = (greens - blues) * crcp(ib->Square(saturation)) * crcp(ib->Square(value));
  auto dh_dr_1 =
    ib->Muls((ib->Muls(ib->Greater(reds, tensor_0, kFloat32) * red_biggest, negative_one) * dhr1), value_sixty);
  auto dhr2 = red_smallest * (blues - greens) * crcp(ib->Square(reds - greens));
  auto dh_dr_2 = ib->Muls((ib->Greater(greens, tensor_0, kFloat32) * green_biggest * dhr2), value_sixty);
  auto dhr3 = ib->Muls(blue_smallest, negative_one) * crcp(greens - blues);
  auto dh_dr_3 = ib->Muls((ib->Greater(greens, tensor_0, kFloat32) * green_biggest * dhr3), value_sixty);
  auto dhr4 = red_smallest * (blues - greens) * crcp(ib->Square(blues - reds));
  auto dh_dr_4 = ib->Muls((ib->Greater(blues, tensor_0, kFloat32) * blue_biggest * dhr4), value_sixty);
  auto dhr5 = green_smallest * crcp(blues - greens);
  auto dh_dr_5 = ib->Muls((ib->Greater(blues, tensor_0, kFloat32) * blue_biggest * dhr5), value_sixty);
  auto dh_dr = ib->Divs((dh_dr_1 + dh_dr_2 + dh_dr_3 + dh_dr_4 + dh_dr_5), value_360);

  auto dhg1 = (blues - reds) * crcp(ib->Square(saturation)) * crcp(ib->Square(value));
  auto dh_dg_1 =
    ib->Muls((ib->Muls(ib->Greater(greens, tensor_0, kFloat32) * green_biggest, negative_one) * dhg1), value_sixty);
  auto dhg2 = green_smallest * (reds - blues) * crcp(ib->Square(reds - greens));
  auto dh_dg_2 = ib->Muls((ib->Greater(reds, tensor_0, kFloat32) * red_biggest * dhg2), value_sixty);
  auto dhg3 = blue_smallest * crcp(reds - blues);
  auto dh_dg_3 = ib->Muls((ib->Greater(reds, tensor_0, kFloat32) * red_biggest * dhg3), value_sixty);
  auto dhg4 = green_smallest * (reds - blues) * crcp(ib->Square(blues - greens));
  auto dh_dg_4 = ib->Muls((ib->Greater(blues, tensor_0, kFloat32) * blue_biggest * dhg4), value_sixty);
  auto dhg5 = ib->Muls(red_smallest, negative_one) * crcp(blues - reds);
  auto dh_dg_5 = ib->Muls((ib->Greater(blues, tensor_0, kFloat32) * blue_biggest * dhg5), value_sixty);
  auto dh_dg = ib->Divs((dh_dg_1 + dh_dg_2 + dh_dg_3 + dh_dg_4 + dh_dg_5), value_360);

  auto dhb1 = (reds - greens) * crcp(ib->Square(saturation)) * crcp(ib->Square(value));
  auto dh_db_1 =
    ib->Muls((ib->Muls(ib->Greater(blues, tensor_0, kFloat32) * blue_biggest, negative_one) * dhb1), value_sixty);
  auto dhb2 = blue_smallest * (greens - reds) * crcp(ib->Square(reds - blues));
  auto dh_db_2 = ib->Muls((ib->Greater(reds, tensor_0, kFloat32) * red_biggest * dhb2), value_sixty);
  auto dhb3 = ib->Muls(green_smallest, negative_one) * crcp(reds - greens);
  auto dh_db_3 = ib->Muls((ib->Greater(reds, tensor_0, kFloat32) * red_biggest * dhb3), value_sixty);
  auto dhb4 = blue_smallest * (greens - reds) * crcp(ib->Square(greens - blues));
  auto dh_db_4 = ib->Muls((ib->Greater(greens, tensor_0, kFloat32) * green_biggest * dhb4), value_sixty);
  auto dhb5 = red_smallest * crcp(greens - reds);
  auto dh_db_5 = ib->Muls((ib->Greater(greens, tensor_0, kFloat32) * green_biggest * dhb5), value_sixty);
  auto dh_db = ib->Divs((dh_db_1 + dh_db_2 + dh_db_3 + dh_db_4 + dh_db_5), value_360);

  auto dout_r = ib->StridedSlice(dout, {{-1, {0}}});
  auto dout_g = ib->StridedSlice(dout, {{-1, {1}}});
  auto dout_b = ib->StridedSlice(dout, {{-1, {2}}});
  int64_t axis = -1;
  auto dv_drgb = ib->Stack({dout_b * dv_dr, dout_b * dv_dg, dout_b * dv_db}, axis);
  auto ds_drgb = ib->Stack({dout_g * ds_dr, dout_g * ds_dg, dout_g * ds_db}, axis);
  auto dh_drgb = ib->Stack({dout_r * dh_dr, dout_r * dh_dg, dout_r * dh_db}, axis);
  auto doutient_input = dv_drgb + ds_drgb + dh_drgb;
  return {doutient_input};
});

REG_BPROP_BUILDER("ResizeV2").SetUnusedInputs({i3, i4}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &roi = ib->GetInput(i1);
  const auto &scales = ib->GetInput(i2);
  const auto &sizes = ib->GetInput(i3);
  const auto &dout = ib->GetInput(i5);
  auto input_size = ib->Shape(x, true);
  auto dx = ib->Emit(
    "ResizeV2Grad", {dout, roi, scales, input_size},
    {{"coordinate_transformation_mode", ib->GetAttr("coordinate_transformation_mode")}, {"mode", ib->GetAttr("mode")}});
  return {dx, ib->OutZeros(roi), ib->OutZeros(scales), ib->OutZeros(sizes)};
});
REG_BPROP_BUILDERS_END
}  // namespace mindspore::expander::bprop
