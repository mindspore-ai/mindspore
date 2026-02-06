/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "ops/ops_func_impl/simple_infer.h"
#include "infer/ops_func_impl/upsample_linear1d.h"
#include "infer/ops_func_impl/upsample_linear1d_grad.h"
#include "infer/ops_func_impl/upsample_nearest1d.h"
#include "infer/ops_func_impl/upsample_nearest1d_grad.h"
#include "infer/ops_func_impl/upsample_nearest2d.h"
#include "infer/ops_func_impl/upsample_nearest2d_grad.h"
#include "infer/ops_func_impl/upsample_nearest3d.h"
#include "infer/ops_func_impl/upsample_nearest3d_grad.h"
#include "infer/ops_func_impl/upsample_bicubic2d.h"
#include "infer/ops_func_impl/upsample_bicubic2d_grad.h"
#include "infer/ops_func_impl/upsample_bilinear2d.h"
#include "infer/ops_func_impl/upsample_bilinear2d_grad.h"
#include "infer/ops_func_impl/upsample_trilinear3d.h"
#include "infer/ops_func_impl/upsample_trilinear3d_grad.h"
#include "primitive/auto_generate/gen_ops_primitive_u.h"

namespace mindspore::ops {
REGISTER_SIMPLE_INFER(kNameUpsampleLinear1D, UpsampleLinear1DFuncImpl)
REGISTER_SIMPLE_INFER(kNameUpsampleNearest1D, UpsampleNearest1DFuncImpl)
REGISTER_SIMPLE_INFER(kNameUpsampleNearest2D, UpsampleNearest2DFuncImpl)
REGISTER_SIMPLE_INFER(kNameUpsampleNearest3D, UpsampleNearest3DFuncImpl)
REGISTER_SIMPLE_INFER(kNameUpsampleBilinear2D, UpsampleBilinear2DFuncImpl)
REGISTER_SIMPLE_INFER(kNameUpsampleBicubic2D, UpsampleBicubic2DFuncImpl)
REGISTER_SIMPLE_INFER(kNameUpsampleTrilinear3D, UpsampleTrilinear3DFuncImpl)
REGISTER_SIMPLE_INFER(kNameUpsampleLinear1DGrad, UpsampleLinear1DGradFuncImpl)
REGISTER_SIMPLE_INFER(kNameUpsampleNearest1DGrad, UpsampleNearest1DGradFuncImpl)
REGISTER_SIMPLE_INFER(kNameUpsampleNearest2DGrad, UpsampleNearest2DGradFuncImpl)
REGISTER_SIMPLE_INFER(kNameUpsampleNearest3DGrad, UpsampleNearest3DGradFuncImpl)
REGISTER_SIMPLE_INFER(kNameUpsampleBilinear2DGrad, UpsampleBilinear2DGradFuncImpl)
REGISTER_SIMPLE_INFER(kNameUpsampleBicubic2DGrad, UpsampleBicubic2DGradFuncImpl)
REGISTER_SIMPLE_INFER(kNameUpsampleTrilinear3DGrad, UpsampleTrilinear3DGradFuncImpl)
}  // namespace mindspore::ops
