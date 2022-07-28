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

#include "frontend/parallel/ops_info/elementary_function_info.h"

#include "frontend/parallel/dynamic_creator.h"

namespace mindspore {
namespace parallel {
REGISTER(ExpInfo);
REGISTER(LogInfo);
REGISTER(CosInfo);
REGISTER(ACosInfo);
REGISTER(LogicalNotInfo);
REGISTER(AbsInfo);
REGISTER(SignInfo);
REGISTER(FloorInfo);
REGISTER(RoundInfo);
REGISTER(ReciprocalInfo);
REGISTER(InvInfo);
REGISTER(RsqrtInfo);
REGISTER(TanInfo);
REGISTER(SinInfo);
REGISTER(SinhInfo);
REGISTER(Log1pInfo);
REGISTER(Expm1Info);
REGISTER(CoshInfo);
REGISTER(CeilInfo);
REGISTER(AtanhInfo);
REGISTER(AtanInfo);
REGISTER(AsinInfo);
REGISTER(AsinhInfo);
REGISTER(AcoshInfo);
REGISTER(ErfInfo);
REGISTER(ErfcInfo);
REGISTER(ZerosLikeInfo);
REGISTER(OnesLikeInfo);
REGISTER(BesselI0eInfo);
REGISTER(BesselI1eInfo);
}  // namespace parallel
}  // namespace mindspore
