/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_ELEMENTARY_FUNCTION_INFO_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_ELEMENTARY_FUNCTION_INFO_H_

#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include "ir/value.h"
#include "frontend/parallel/auto_parallel/operator_costmodel.h"
#include "frontend/parallel/ops_info/activation_info.h"
#include "frontend/parallel/strategy.h"

namespace mindspore {
namespace parallel {
class ExpInfo : public ActivationOther {
 public:
  ExpInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape, const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<ExpCost>()) {}
  ~ExpInfo() override = default;
};

class LogInfo : public ActivationOther {
 public:
  LogInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape, const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<LogCost>()) {}
  ~LogInfo() override = default;
};

class CosInfo : public ActivationOther {
 public:
  CosInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape, const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<CosCost>()) {}
  ~CosInfo() override = default;
};

class ACosInfo : public ActivationOther {
 public:
  ACosInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
           const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<ACosCost>()) {}
  ~ACosInfo() override = default;
};

class LogicalNotInfo : public ActivationOther {
 public:
  LogicalNotInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                 const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<LogicalNotCost>()) {}
  ~LogicalNotInfo() override = default;
};

class AbsInfo : public ActivationOther {
 public:
  AbsInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape, const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<AbsCost>()) {}
  ~AbsInfo() override = default;
};

class SignInfo : public ActivationOther {
 public:
  SignInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
           const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<SignCost>()) {}
  ~SignInfo() override = default;
};

class FloorInfo : public ActivationOther {
 public:
  FloorInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
            const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<FloorCost>()) {}
  ~FloorInfo() override = default;
};

class RoundInfo : public ActivationOther {
 public:
  RoundInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
            const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<RoundCost>()) {}
  ~RoundInfo() override = default;
};

class ReciprocalInfo : public ActivationOther {
 public:
  ReciprocalInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                 const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<ReciprocalCost>()) {}
  ~ReciprocalInfo() override = default;
};

class InvInfo : public ActivationOther {
 public:
  InvInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape, const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<InvCost>()) {}
  ~InvInfo() override = default;
};

class RsqrtInfo : public ActivationOther {
 public:
  RsqrtInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
            const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<RsqrtCost>()) {}
  ~RsqrtInfo() override = default;
};

class TanInfo : public ActivationOther {
 public:
  TanInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape, const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<TanCost>()) {}
  ~TanInfo() override = default;
};

class SinInfo : public ActivationOther {
 public:
  SinInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape, const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<SinCost>()) {}
  ~SinInfo() override = default;
};

class SinhInfo : public ActivationOther {
 public:
  SinhInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
           const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<SinhCost>()) {}
  ~SinhInfo() override = default;
};

class Log1pInfo : public ActivationOther {
 public:
  Log1pInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
            const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<Log1pCost>()) {}
  ~Log1pInfo() override = default;
};

class Expm1Info : public ActivationOther {
 public:
  Expm1Info(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
            const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<Expm1Cost>()) {}
  ~Expm1Info() override = default;
};

class CoshInfo : public ActivationOther {
 public:
  CoshInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
           const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<CoshCost>()) {}
  ~CoshInfo() override = default;
};

class CeilInfo : public ActivationOther {
 public:
  CeilInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
           const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<CeilCost>()) {}
  ~CeilInfo() override = default;
};

class AtanhInfo : public ActivationOther {
 public:
  AtanhInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
            const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<AtanhCost>()) {}
  ~AtanhInfo() override = default;
};

class AtanInfo : public ActivationOther {
 public:
  AtanInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
           const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<AtanCost>()) {}
  ~AtanInfo() override = default;
};

class AsinInfo : public ActivationOther {
 public:
  AsinInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
           const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<AsinCost>()) {}
  ~AsinInfo() override = default;
};

class AsinhInfo : public ActivationOther {
 public:
  AsinhInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
            const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<AsinhCost>()) {}
  ~AsinhInfo() override = default;
};

class AcoshInfo : public ActivationOther {
 public:
  AcoshInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
            const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<AcoshCost>()) {}
  ~AcoshInfo() override = default;
};

class ErfInfo : public ActivationOther {
 public:
  ErfInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape, const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<ErfCost>()) {}
  ~ErfInfo() override = default;
};

class ErfcInfo : public ActivationOther {
 public:
  ErfcInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
           const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<ErfcCost>()) {}
  ~ErfcInfo() override = default;
};

class ZerosLikeInfo : public ActivationOther {
 public:
  ZerosLikeInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<ZerosLikeCost>()) {}
  ~ZerosLikeInfo() override = default;
};

class OnesLikeInfo : public ActivationOther {
 public:
  OnesLikeInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
               const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<OnesLikeCost>()) {}
  ~OnesLikeInfo() override = default;
};

class BesselI0eInfo : public ActivationOther {
 public:
  BesselI0eInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<BesselI0eCost>()) {}
  ~BesselI0eInfo() override = default;
};

class BesselI1eInfo : public ActivationOther {
 public:
  BesselI1eInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                const PrimitiveAttrs &attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs, std::make_shared<BesselI1eCost>()) {}
  ~BesselI1eInfo() override = default;
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_ELEMENTARY_FUNCTION_INFO_H_
