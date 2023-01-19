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
#include "pipeline/pynative/grad/bprop_expander/bprop_irbuilder.h"
#include "pipeline/pynative/grad/bprop_expander/grad_ops/common_utils.h"
#include "include/common/utils/utils.h"
#include "utils/ms_context.h"

namespace mindspore::expander::bprop {
REG_BPROP_BUILDERS_BEGIN(GradImageOps)
REG_BPROP_BUILDER("ResizeBicubic").SetUnusedInputs({i1, i2}).SetBody(BODYFUNC(ib) {
  auto images = ib->GetInput(kIndex0);
  auto size = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto dx = ib->Emit(
    "ResizeBicubicGrad", {dout, images},
    {{"align_corners", ib->GetAttr("align_corners")}, {"half_pixel_centers", ib->GetAttr("half_pixel_centers")}});
  return {dx, ib->ZerosLike(size)};
});

REG_BPROP_BUILDER("CropAndResize").SetUnusedInputs({i3, i4}).SetBody(BODYFUNC(ib) {
  std::set<TypeId> allowed_types = {kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeFloat64};
  auto method = GetValue<std::string>(ib->GetAttr("method"));
  auto target = ib->GetTargetFromContext();
  auto is_ascend_cpu = (target == kAscendDevice || target == kCPUDevice);
  auto x = ib->GetInput(kIndex0);
  auto boxes = ib->GetInput(kIndex1);
  auto box_index = ib->GetInput(kIndex2);
  auto crop_size = ib->GetInput(kIndex3);
  auto dout = ib->GetInput(kIndex5);
  if (method != "bilinear") {
    if (!is_ascend_cpu) {
      return {ib->ZerosLike(x), ib->ZerosLike(boxes), ib->ZerosLike(box_index), ib->ZerosLike(crop_size)};
    }
  }
  auto image_type = ib->GetDtype(x);
  if (allowed_types.count(image_type->type_id()) != 0) {
    x = ib->Cast(x, kFloat32);
  }
  auto image_size = ib->Tensor(ib->GetShape(x), kInt32);
  const int64_t max_byte = 2e9;  // max bytes of image gradient
  auto dimage = ib->Emit("CropAndResizeGradImage", {dout, boxes, box_index, image_size},
                         {{"method", MakeValue(method)}, {"T", image_type}, {"max_Byte", MakeValue(max_byte)}});
  auto dbox = ib->Emit("CropAndResizeGradBoxes", {dout, x, boxes, box_index}, {{"method", MakeValue("bilinear")}});
  return {dimage, dbox, ib->ZerosLike(box_index), ib->ZerosLike(crop_size)};
});

REG_BPROP_BUILDER("ScaleAndTranslate").SetUnusedInputs({i1, i4}).SetBody(BODYFUNC(ib) {
  auto images = ib->GetInput(kIndex0);
  auto size = ib->GetInput(kIndex1);
  auto scale = ib->GetInput(kIndex2);
  auto translation = ib->GetInput(kIndex3);
  auto dout = ib->GetInput(kIndex5);
  auto images_dtype = ib->GetDtype(images);
  auto images_fp32 = (images_dtype->type_id() != kNumberTypeFloat32) ? ib->Cast(images, kFloat32) : images;
  auto grad0_fp32 = ib->Emit("ScaleAndTranslateGrad", {dout, images_fp32, scale, translation},
                             {{"kernel_type", ib->GetAttr("kernel_type")}, {"antialias", ib->GetAttr("antialias")}});
  auto grad0 = (images_dtype->type_id() != kNumberTypeFloat32) ? ib->Cast(grad0_fp32, images_dtype) : grad0_fp32;
  return {grad0, ib->ZerosLike(size), ib->ZerosLike(scale), ib->ZerosLike(translation)};
});

REG_BPROP_BUILDER("RGBToHSV").SetBody(BODYFUNC(ib) {
  auto images = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto images_dtype = ib->GetDtype(images);
  if (images_dtype->type_id() != kNumberTypeFloat32) {
    images = ib->Cast(images, kFloat32);
    out = ib->Cast(out, kFloat32);
    dout = ib->Cast(dout, kFloat32);
  }
  auto tensor_0 = ib->Tensor(0, kFloat32);
  auto tensor_1 = ib->Tensor(1, kFloat32);
  auto tensor_n1 = ib->Tensor(-1, kFloat32);
  auto tensor_60 = ib->Tensor(60, kFloat32);
  auto tensor_360 = ib->Tensor(360, kFloat32);
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
  auto dsr3 = red_smallest * tensor_n1 * crcp((green_biggest * greens) + (blue_biggest * blues));
  auto ds_dr = dsr1 * (dsr2 + dsr3);
  auto dsg1 = ib->Greater(greens, tensor_0, kFloat32);
  auto dsg2 = green_biggest * (red_smallest * reds + blue_smallest * blues) * crcp(ib->Square(greens));
  auto dsg3 = green_smallest * tensor_n1 * crcp((red_biggest * reds) + (blue_biggest * blues));
  auto ds_dg = dsg1 * (dsg2 + dsg3);

  auto dsb1 = ib->Greater(blues, tensor_0, kFloat32);
  auto dsb2 = blue_biggest * (green_smallest * greens + red_smallest * reds) * crcp(ib->Square(blues));
  auto dsb3 = blue_smallest * tensor_n1 * crcp((green_biggest * greens) + (red_biggest * reds));
  auto ds_db = dsb1 * (dsb2 + dsb3);

  auto dhr1 = (greens - blues) * crcp(ib->Square(saturation)) * crcp(ib->Square(value));
  auto dh_dr_1 = tensor_60 * (ib->Greater(reds, tensor_0, kFloat32) * red_biggest * tensor_n1 * dhr1);
  auto dhr2 = red_smallest * (blues - greens) * crcp(ib->Square(reds - greens));
  auto dh_dr_2 = tensor_60 * (ib->Greater(greens, tensor_0, kFloat32) * green_biggest * dhr2);
  auto dhr3 = (blue_smallest * tensor_n1) * crcp(greens - blues);
  auto dh_dr_3 = tensor_60 * (ib->Greater(greens, tensor_0, kFloat32) * green_biggest * dhr3);
  auto dhr4 = red_smallest * (blues - greens) * crcp(ib->Square(blues - reds));
  auto dh_dr_4 = tensor_60 * (ib->Greater(blues, tensor_0, kFloat32) * blue_biggest * dhr4);
  auto dhr5 = green_smallest * crcp(blues - greens);
  auto dh_dr_5 = tensor_60 * (ib->Greater(blues, tensor_0, kFloat32) * blue_biggest * dhr5);
  auto dh_dr = (dh_dr_1 + dh_dr_2 + dh_dr_3 + dh_dr_4 + dh_dr_5) / tensor_360;

  auto dhg1 = (blues - reds) * crcp(ib->Square(saturation)) * crcp(ib->Square(value));
  auto dh_dg_1 = tensor_60 * (ib->Greater(greens, tensor_0, kFloat32) * green_biggest * tensor_n1 * dhg1);
  auto dhg2 = green_smallest * (reds - blues) * crcp(ib->Square(reds - greens));
  auto dh_dg_2 = tensor_60 * (ib->Greater(reds, tensor_0, kFloat32) * red_biggest * dhg2);
  auto dhg3 = blue_smallest * crcp(reds - blues);
  auto dh_dg_3 = tensor_60 * (ib->Greater(reds, tensor_0, kFloat32) * red_biggest * dhg3);
  auto dhg4 = green_smallest * (reds - blues) * crcp(ib->Square(blues - greens));
  auto dh_dg_4 = tensor_60 * (ib->Greater(blues, tensor_0, kFloat32) * blue_biggest * dhg4);
  auto dhg5 = red_smallest * tensor_n1 * crcp(blues - reds);
  auto dh_dg_5 = tensor_60 * (ib->Greater(blues, tensor_0, kFloat32) * blue_biggest * dhg5);
  auto dh_dg = (dh_dg_1 + dh_dg_2 + dh_dg_3 + dh_dg_4 + dh_dg_5) / tensor_360;

  auto dhb1 = (reds - greens) * crcp(ib->Square(saturation)) * crcp(ib->Square(value));
  auto dh_db_1 = tensor_60 * (ib->Greater(blues, tensor_0, kFloat32) * blue_biggest * tensor_n1 * dhb1);
  auto dhb2 = blue_smallest * (greens - reds) * crcp(ib->Square(reds - blues));
  auto dh_db_2 = tensor_60 * (ib->Greater(reds, tensor_0, kFloat32) * red_biggest * dhb2);
  auto dhb3 = green_smallest * tensor_n1 * crcp(reds - greens);
  auto dh_db_3 = tensor_60 * (ib->Greater(reds, tensor_0, kFloat32) * red_biggest * dhb3);
  auto dhb4 = blue_smallest * (greens - reds) * crcp(ib->Square(greens - blues));
  auto dh_db_4 = tensor_60 * (ib->Greater(greens, tensor_0, kFloat32) * green_biggest * dhb4);
  auto dhb5 = red_smallest * crcp(greens - reds);
  auto dh_db_5 = tensor_60 * (ib->Greater(greens, tensor_0, kFloat32) * green_biggest * dhb5);
  auto dh_db = (dh_db_1 + dh_db_2 + dh_db_3 + dh_db_4 + dh_db_5) / tensor_360;

  auto dout_r = ib->StridedSlice(dout, {{-1, {0}}});
  auto dout_g = ib->StridedSlice(dout, {{-1, {1}}});
  auto dout_b = ib->StridedSlice(dout, {{-1, {2}}});
  auto axis = MakeValue<int64_t>(-1);
  auto dv_drgb = ib->Emit("Stack", {ib->MakeTuple({dout_b * dv_dr, dout_b * dv_dg, dout_b * dv_db})}, {{"axis", axis}});
  auto ds_drgb = ib->Emit("Stack", {ib->MakeTuple({dout_g * ds_dr, dout_g * ds_dg, dout_g * ds_db})}, {{"axis", axis}});
  auto dh_drgb = ib->Emit("Stack", {ib->MakeTuple({dout_r * dh_dr, dout_r * dh_dg, dout_r * dh_db})}, {{"axis", axis}});
  auto doutient_input = dv_drgb + ds_drgb + dh_drgb;
  return {doutient_input};
});
REG_BPROP_BUILDERS_END
}  // namespace mindspore::expander::bprop
