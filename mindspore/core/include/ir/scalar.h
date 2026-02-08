/**
 * Copyright 2019-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_IR_SCALAR_H_
#define MINDSPORE_CORE_IR_SCALAR_H_

#include <type_traits>
#include <algorithm>
#include <cmath>
#include <vector>
#include <string>
#include <memory>
#include <sstream>
#include <utility>
#include <cfloat>
#include <functional>
#include <complex>

#include "base/base.h"
#include "ir/dtype/number.h"
#include "utils/hashing.h"
#include "base/bfloat16.h"
#include "base/float16.h"

using std::fabs;

namespace mindspore {
// Half floating error threshold : 2^(-10)
constexpr float HALF_EPSILON = 0.0009765625f;

template <typename T>
inline std::string scalar_float_to_string(T v) {
  std::stringstream ss;
  ss << v;
  return ss.str();
}

template <typename T>
inline std::string scalar_to_string(T v) {
  return std::to_string(v);
}

/// \brief Scalar defines interface for scalar data.
class MS_CORE_API Scalar : public Value {
 public:
  /// \brief The default constructor for Scalar.
  Scalar() = default;
  /// \brief The constructor for Scalar.
  ///
  /// \param[in] t The type of scalar.
  explicit Scalar(const TypePtr t) : Value(t) {}
  /// \brief The destructor of Scalar.
  ~Scalar() override = default;
  MS_DECLARE_PARENT(Scalar, Value)
  /// \brief Check whether the value of scalar is zero.
  ///
  /// \return Return true if the value of scalar is zero ,else return false.
  virtual bool IsZero() = 0;
  /// \brief Check whether the value of scalar is zero.
  ///
  /// \return Return true if the value of scalar is zero ,else return false.
  virtual bool IsOne() = 0;
  abstract::AbstractBasePtr ToAbstract() override;

 protected:
  std::size_t hash_ = 0;
};
using ScalarPtr = std::shared_ptr<Scalar>;

/// \brief BoolImm defines interface for bool data.
class MS_CORE_API BoolImm final : public Scalar {
 public:
  /// \brief The constructor of BoolImm.
  ///
  /// \param[in] b The value of bool data.
  explicit BoolImm(bool b) : Scalar(kBool), v_(b) { hash_ = hash_combine({tid(), std::hash<bool>{}(v_)}); }
  /// \brief The destructor of BoolImm.
  ~BoolImm() override = default;
  MS_DECLARE_PARENT(BoolImm, Scalar)
  std::size_t hash() const override { return hash_; }
  /// \brief Get the value of BoolImm.
  ///
  /// \return Return the value of BoolImm.
  bool value() const { return v_; }
  bool IsZero() override { return v_ == false; }
  bool IsOne() override { return v_ == true; }
  bool operator==(const Value &other) const override;
  /// \brief Compare two BoolImm objects is equal.
  ///
  /// \param[in] other The other BoolImm to be compared with.
  /// \return Return true if other's value and the value of current object are the same,else return false.
  bool operator==(const BoolImm &other) const;
  std::string ToString() const override {
    if (v_) {
      return "true";
    } else {
      return "false";
    }
  }

  std::string DumpText() const override {
    std::ostringstream oss;
    oss << "Bool(" << v_ << ")";
    return oss.str();
  }

 private:
  bool v_;
};
using BoolImmPtr = std::shared_ptr<BoolImm>;
IMM_TRAITS(BoolImmPtr, bool)

/// \brief IntegerImm defines interface for integer data.
class MS_CORE_API IntegerImm : public Scalar {
 public:
  /// \brief The default constructor for IntegerImm.
  IntegerImm() = default;
  /// \brief The constructor for IntegerImm.
  ///
  /// \param[in] t The type of IntegerImm.
  explicit IntegerImm(const TypePtr &t) : Scalar(t) {}
  /// \brief The destructor of Scalar.
  ~IntegerImm() override = default;
  MS_DECLARE_PARENT(IntegerImm, Scalar)
};

/// \brief Int8Imm defines interface for int8 data.
class MS_CORE_API Int8Imm final : public IntegerImm {
 public:
  /// \brief The default constructor for Int8Imm.
  Int8Imm() : IntegerImm(kInt8), v_(0) {}
  /// \brief The constructor for Int8Imm.
  ///
  /// \param[in] v The value of Int8Imm.
  explicit Int8Imm(int8_t v) : IntegerImm(kInt8), v_(v) { hash_ = hash_combine({tid(), std::hash<int>{}(v_)}); }
  /// \brief The destructor of Int8Imm.
  ~Int8Imm() override = default;
  MS_DECLARE_PARENT(Int8Imm, IntegerImm)
  std::size_t hash() const override { return hash_; }
  bool IsZero() override { return v_ == 0; }
  bool IsOne() override { return v_ == 1; }
  /// \brief Get the value of Int8Imm.
  ///
  /// \return Return the value of Int8Imm.
  int8_t value() const { return v_; }
  bool operator==(const Value &other) const override;
  /// \brief Compare two Int8Imm objects is equal.
  ///
  /// \param[in] other The other Int8Imm to be compared with.
  /// \return Return true if other's value and the value of current object are the same,else return false.
  bool operator==(const Int8Imm &other) const;
  std::string ToString() const override { return scalar_to_string(v_); }

  std::string DumpText() const override {
    std::ostringstream oss;
    oss << "I8(" << int(v_) << ")";
    return oss.str();
  }

 private:
  int8_t v_;
};
using Int8ImmPtr = std::shared_ptr<Int8Imm>;
IMM_TRAITS(Int8ImmPtr, int8_t)
/// \brief Int16Imm defines interface for int16 data.
class MS_CORE_API Int16Imm final : public IntegerImm {
 public:
  /// \brief The default constructor for Int16Imm.
  Int16Imm() : IntegerImm(kInt16), v_(0) {}
  /// \brief The constructor for Int16Imm.
  ///
  /// \param[in] v The value of Int16Imm.
  explicit Int16Imm(int16_t v) : IntegerImm(kInt16), v_(v) { hash_ = hash_combine({tid(), std::hash<int>{}(v_)}); }
  /// \brief The destructor of Int16Imm.
  ~Int16Imm() override = default;
  MS_DECLARE_PARENT(Int16Imm, IntegerImm)
  std::size_t hash() const override { return hash_; }
  bool IsZero() override { return v_ == 0; }
  bool IsOne() override { return v_ == 1; }
  /// \brief Get the value of Int16Imm.
  ///
  /// \return Return the value of Int16Imm.
  int16_t value() const { return v_; }
  bool operator==(const Value &other) const override;
  /// \brief Compare two Int16Imm objects is equal.
  ///
  /// \param[in] other The other Int16Imm to be compared with.
  /// \return Return true if other's value and the value of current object are the same,else return false.
  bool operator==(const Int16Imm &other) const;
  std::string ToString() const override { return scalar_to_string(v_); }

  std::string DumpText() const override {
    std::ostringstream oss;
    oss << "I16(" << int(v_) << ")";
    return oss.str();
  }

 private:
  int16_t v_;
};
using Int16ImmPtr = std::shared_ptr<Int16Imm>;
IMM_TRAITS(Int16ImmPtr, int16_t)

/// \brief Int32Imm defines interface for int32 data.
class MS_CORE_API Int32Imm final : public IntegerImm {
 public:
  /// \brief The default constructor for Int32Imm.
  Int32Imm() : IntegerImm(kInt32), v_(0) {}
  /// \brief The constructor for Int32Imm.
  ///
  /// \param[in] v The value of Int32Imm.
  explicit Int32Imm(int v) : IntegerImm(kInt32), v_(v) { hash_ = hash_combine({tid(), std::hash<int>{}(v_)}); }
  /// \brief The destructor of Int32Imm.
  ~Int32Imm() override = default;
  MS_DECLARE_PARENT(Int32Imm, IntegerImm)
  std::size_t hash() const override { return hash_; }
  bool IsZero() override { return v_ == 0; }
  bool IsOne() override { return v_ == 1; }
  /// \brief Get the value of Int32Imm.
  ///
  /// \return Return the value of Int32Imm.
  int32_t value() const { return v_; }
  bool operator==(const Value &other) const override;
  /// \brief Compare two Int32Imm objects is equal.
  ///
  /// \param[in] other The other Int32Imm to be compared with.
  /// \return Return true if other's value and the value of current object are the same,else return false.
  bool operator==(const Int32Imm &other) const;
  std::string ToString() const override { return scalar_to_string(v_); }

  std::string DumpText() const override {
    std::ostringstream oss;
    oss << "I32(" << int(v_) << ")";
    return oss.str();
  }

 private:
  int32_t v_;
};
using Int32ImmPtr = std::shared_ptr<Int32Imm>;
IMM_TRAITS(Int32ImmPtr, int32_t)

/// \brief Int64Imm defines interface for int64 data.
class MS_CORE_API Int64Imm final : public IntegerImm {
 public:
  /// \brief The default constructor for Int64Imm.
  Int64Imm() : IntegerImm(kInt64), v_(0) {}
  /// \brief The constructor for Int64Imm.
  ///
  /// \param[in] v The value of Int64Imm.
  explicit Int64Imm(int64_t v) : IntegerImm(kInt64), v_(v) { hash_ = hash_combine({tid(), std::hash<int64_t>{}(v_)}); }
  /// \brief The destructor of Int64Imm.
  ~Int64Imm() override = default;
  MS_DECLARE_PARENT(Int64Imm, IntegerImm)
  std::size_t hash() const override { return hash_; }
  bool IsZero() override { return v_ == 0; }
  bool IsOne() override { return v_ == 1; }
  /// \brief Get the value of Int64Imm.
  ///
  /// \return Return the value of Int64Imm.
  int64_t value() const { return v_; }
  bool operator==(const Value &other) const override;
  /// \brief Compare two Int64Imm objects is equal.
  ///
  /// \param[in] other The other Int64Imm to be compared with.
  /// \return Return true if other's value and the value of current object are the same,else return false.
  bool operator==(const Int64Imm &other) const;
  std::string ToString() const override { return scalar_to_string(v_); }

  std::string DumpText() const override {
    std::ostringstream oss;
    oss << "I64(" << v_ << ")";
    return oss.str();
  }

 private:
  int64_t v_;
};
using Int64ImmPtr = std::shared_ptr<Int64Imm>;
IMM_TRAITS(Int64ImmPtr, int64_t)
/// \brief UInt8Imm defines interface for uint8 data.
class MS_CORE_API UInt8Imm final : public IntegerImm {
 public:
  /// \brief The default constructor for UInt8Imm.
  UInt8Imm() : IntegerImm(kUInt8), v_(0) {}
  /// \brief The constructor for UInt8Imm.
  ///
  /// \param[in] v The value of UInt8Imm.
  explicit UInt8Imm(uint8_t v) : IntegerImm(kUInt8), v_(v) {
    hash_ = hash_combine({tid(), std::hash<unsigned int>{}(v_)});
  }
  /// \brief The destructor of UInt8Imm.
  ~UInt8Imm() override = default;
  MS_DECLARE_PARENT(UInt8Imm, IntegerImm)
  std::size_t hash() const override { return hash_; }
  bool IsZero() override { return v_ == 0; }
  bool IsOne() override { return v_ == 1; }
  /// \brief Get the value of UInt8Imm.
  ///
  /// \return Return the value of UInt8Imm.
  uint8_t value() const { return v_; }
  bool operator==(const Value &other) const override;
  /// \brief Compare two UInt8Imm objects is equal.
  ///
  /// \param[in] other The other UInt8Imm to be compared with.
  /// \return Return true if other's value and the value of current object are the same,else return false.
  bool operator==(const UInt8Imm &other) const;
  std::string ToString() const override { return scalar_to_string(v_); }

  std::string DumpText() const override {
    std::ostringstream oss;
    oss << "U8(" << unsigned(v_) << ")";
    return oss.str();
  }

 private:
  uint8_t v_;
};
using UInt8ImmPtr = std::shared_ptr<UInt8Imm>;
IMM_TRAITS(UInt8ImmPtr, uint8_t);

/// \brief UInt16Imm defines interface for uint16 data.
class MS_CORE_API UInt16Imm final : public IntegerImm {
 public:
  /// \brief The default constructor for UInt16Imm.
  UInt16Imm() : IntegerImm(kUInt16), v_(0) {}
  /// \brief The constructor for UInt16Imm.
  ///
  /// \param[in] v The value of UInt16Imm.
  explicit UInt16Imm(uint16_t v) : IntegerImm(kUInt16), v_(v) {
    hash_ = hash_combine({tid(), std::hash<unsigned int>{}(v_)});
  }
  /// \brief The destructor of UInt16Imm.
  ~UInt16Imm() override = default;
  MS_DECLARE_PARENT(UInt16Imm, IntegerImm)
  std::size_t hash() const override { return hash_; }
  bool IsZero() override { return v_ == 0; }
  bool IsOne() override { return v_ == 1; }
  /// \brief Get the value of UInt16Imm.
  ///
  /// \return Return the value of UInt16Imm.
  uint16_t value() const { return v_; }
  bool operator==(const Value &other) const override;
  /// \brief Compare two UInt16Imm objects is equal.
  ///
  /// \param[in] other The other UInt16Imm to be compared with.
  /// \return Return true if other's value and the value of current object are the same,else return false.
  bool operator==(const UInt16Imm &other) const;
  std::string ToString() const override { return scalar_to_string(v_); }

  std::string DumpText() const override {
    std::ostringstream oss;
    oss << "U16(" << unsigned(v_) << ")";
    return oss.str();
  }

 private:
  uint16_t v_;
};
using UInt16ImmPtr = std::shared_ptr<UInt16Imm>;
IMM_TRAITS(UInt16ImmPtr, uint16_t);

/// \brief UInt32Imm defines interface for uint32 data.
class MS_CORE_API UInt32Imm final : public IntegerImm {
 public:
  /// \brief The default constructor for UInt32Imm.
  UInt32Imm() : IntegerImm(kUInt32), v_(0) {}
  /// \brief The constructor for UInt32Imm.
  ///
  /// \param[in] v The value of UInt32Imm.
  explicit UInt32Imm(uint32_t v) : IntegerImm(kUInt32), v_(v) {
    hash_ = hash_combine({tid(), std::hash<unsigned int>{}(v_)});
  }
  /// \brief The destructor of UInt32Imm.
  ~UInt32Imm() override = default;
  MS_DECLARE_PARENT(UInt32Imm, IntegerImm)
  std::size_t hash() const override { return hash_; }
  bool IsZero() override { return v_ == 0; }
  bool IsOne() override { return v_ == 1; }
  /// \brief Get the value of UInt32Imm.
  ///
  /// \return Return the value of UInt32Imm.
  uint32_t value() const { return v_; }
  bool operator==(const Value &other) const override;
  /// \brief Compare two UInt32Imm objects is equal.
  ///
  /// \param[in] other The other UInt32Imm to be compared with.
  /// \return Return true if other's value and the value of current object are the same,else return false.
  bool operator==(const UInt32Imm &other) const;
  std::string ToString() const override { return scalar_to_string(v_); }

  std::string DumpText() const override {
    std::ostringstream oss;
    oss << "U32(" << unsigned(v_) << ")";
    return oss.str();
  }

 private:
  uint32_t v_;
};
using UInt32ImmPtr = std::shared_ptr<UInt32Imm>;
IMM_TRAITS(UInt32ImmPtr, uint32_t);
/// \brief UInt64Imm defines interface for uint64 data.
class MS_CORE_API UInt64Imm final : public IntegerImm {
 public:
  /// \brief The default constructor for UInt64Imm.
  UInt64Imm() : IntegerImm(kUInt64), v_(0) {}
  /// \brief The constructor for UInt64Imm.
  ///
  /// \param[in] v The value of UInt64Imm.
  explicit UInt64Imm(uint64_t v) : IntegerImm(kUInt64), v_(v) {
    hash_ = hash_combine({tid(), std::hash<uint64_t>{}(v)});
  }
  /// \brief The destructor of UInt64Imm.
  ~UInt64Imm() override = default;
  MS_DECLARE_PARENT(UInt64Imm, IntegerImm)
  std::size_t hash() const override { return hash_; }
  bool IsZero() override { return v_ == 0; }
  bool IsOne() override { return v_ == 1; }
  /// \brief Get the value of UInt64Imm.
  ///
  /// \return Return the value of UInt64Imm.
  uint64_t value() const { return v_; }
  bool operator==(const Value &other) const override;
  /// \brief Compare two UInt64Imm objects is equal.
  ///
  /// \param[in] other The other UInt64Imm to be compared with.
  /// \return Return true if other's value and the value of current object are the same,else return false.
  bool operator==(const UInt64Imm &other) const;
  std::string ToString() const override { return scalar_to_string(v_); }

  std::string DumpText() const override {
    std::ostringstream oss;
    oss << "U64(" << v_ << ")";
    return oss.str();
  }

 private:
  uint64_t v_;
};
using UInt64ImmPtr = std::shared_ptr<UInt64Imm>;
IMM_TRAITS(UInt64ImmPtr, uint64_t);

#if defined(__APPLE__)
using SizetImmPtr = std::shared_ptr<UInt64Imm>;
IMM_TRAITS(SizetImmPtr, size_t);
#endif

/// \brief FloatImm defines interface for float data.
class MS_CORE_API FloatImm : public Scalar {
 public:
  /// \brief The default constructor for FloatImm.
  FloatImm() = default;
  /// \brief The constructor for FloatImm.
  ///
  /// \param[in] t The value of FloatImm.
  explicit FloatImm(const TypePtr &t) : Scalar(t) {}
  /// \brief The destructor of FloatImm.
  ~FloatImm() override = default;
  MS_DECLARE_PARENT(FloatImm, Scalar)
};
using FloatImmPtr = std::shared_ptr<FloatImm>;

/// \brief FP16Imm defines interface for float16 data.
class MS_CORE_API FP16Imm final : public FloatImm {
 public:
  /// \brief The default constructor for FP16Imm.
  FP16Imm() : FloatImm(kFloat16), v_(0.0) {}
  /// \brief The constructor for FP16Imm.
  ///
  /// \param[in] v The value of FP16Imm.
  explicit FP16Imm(float16 v) : FloatImm(kFloat16), v_(v) {
    hash_ = hash_combine(tid(), [&]() {
      uint16_t value;
      auto ret = memcpy_s(&value, sizeof(value), &v_, sizeof(uint16_t));
      if (ret != EOK) {
        MS_LOG(EXCEPTION) << "Failed to copy data, memcpy_s errorno: " << ret;
      }
      return std::hash<uint16_t>{}(value);
    }());
  }
  /// \brief The destructor of FP16Imm.
  ~FP16Imm() override = default;
  MS_DECLARE_PARENT(FP16Imm, FloatImm)
  std::size_t hash() const override { return hash_; }
  bool IsZero() override { return fabs(half_to_float(v_)) <= HALF_EPSILON; }
  bool IsOne() override { return fabs(half_to_float(v_) - 1.0) <= HALF_EPSILON; }
  /// \brief Get the value of FP16Imm.
  ///
  /// \return Return the value of FP16Imm.
  float16 value() const { return v_; }
  bool operator==(const Value &other) const override;
  /// \brief Compare two FP16Imm objects is equal.
  ///
  /// \param[in] other The other FP16Imm to be compared with.
  /// \return Return true if other's value and the value of current object are the same,else return false.
  bool operator==(const FP16Imm &other) const;
  std::string ToString() const override { return scalar_float_to_string(v_); }

  std::string DumpText() const override {
    std::ostringstream oss;
    oss << "F16(" << v_ << ")";
    return oss.str();
  }

 private:
  float16 v_;
};
using FP16ImmPtr = std::shared_ptr<FP16Imm>;
IMM_TRAITS(FP16ImmPtr, float16)

/// \brief FP32Imm defines interface for float32 data.
class MS_CORE_API FP32Imm final : public FloatImm {
 public:
  /// \brief The default constructor for FP32Imm.
  FP32Imm() : FloatImm(kFloat32), v_(0.0) {}
  /// \brief The constructor for FP32Imm.
  ///
  /// \param[in] v The value of FP32Imm.
  explicit FP32Imm(float v) : FloatImm(kFloat32), v_(v) { hash_ = hash_combine({tid(), std::hash<float>{}(v_)}); }
  /// \brief The destructor of FP32Imm.
  ~FP32Imm() override = default;
  MS_DECLARE_PARENT(FP32Imm, FloatImm)
  std::size_t hash() const override { return hash_; }
  bool IsZero() override { return fabs(v_) <= FLT_EPSILON; }
  bool IsOne() override { return fabs(v_ - 1.0) <= FLT_EPSILON; }
  /// \brief Get the value of FP32Imm.
  ///
  /// \return Return the value of FP32Imm.
  float value() const { return v_; }
  /// \brief Get the double type value of FP32Imm.
  ///
  /// \return Return the double type value of FP32Imm.
  double prim_value() const { return prim_v_; }
  /// \brief Set the double type value of FP32Imm.
  ///
  /// \param[prim_v] double type value for FP32IMM.
  void set_prim_value(double prim_v) { prim_v_ = prim_v; }
  bool operator==(const Value &other) const override;
  /// \brief Compare two FP32Imm objects is equal.
  ///
  /// \param[in] other The other FP32Imm to be compared with.
  /// \return Return true if other's value and the value of current object are the same,else return false.
  bool operator==(const FP32Imm &other) const;

  std::string ToString() const override { return scalar_float_to_string(v_); }

  std::string DumpText() const override {
    std::ostringstream oss;
    oss << "F32(" << v_ << ")";
    return oss.str();
  }

 private:
  float v_;
  double prim_v_{0.0};
};
using FP32ImmPtr = std::shared_ptr<FP32Imm>;
IMM_TRAITS(FP32ImmPtr, float)

/// \brief FP64Imm defines interface for float64 data.
class MS_CORE_API FP64Imm final : public FloatImm {
 public:
  /// \brief The default constructor for FP64Imm.
  FP64Imm() : FloatImm(kFloat64), v_(0.0) {}
  /// \brief The constructor for FP64Imm.
  ///
  /// \param[in] v The value of FP64Imm.
  explicit FP64Imm(double v) : FloatImm(kFloat64), v_(v) { hash_ = hash_combine({tid(), std::hash<double>{}(v_)}); }
  /// \brief The destructor of FP64Imm.
  ~FP64Imm() override = default;
  MS_DECLARE_PARENT(FP64Imm, FloatImm)
  std::size_t hash() const override { return hash_; }
  bool IsZero() override { return fabs(v_) <= DBL_EPSILON; }
  bool IsOne() override { return fabs(v_ - 1.0) <= DBL_EPSILON; }
  /// \brief Get the value of FP64Imm.
  ///
  /// \return Return the value of FP64Imm.
  double value() const { return v_; }
  bool operator==(const Value &other) const override;
  /// \brief Compare two FP64Imm objects is equal.
  ///
  /// \param[in] other The other FP64Imm to be compared with.
  /// \return Return true if other's value and the value of current object are the same,else return false.
  bool operator==(const FP64Imm &other) const;
  std::string ToString() const override { return scalar_float_to_string(v_); }

  std::string DumpText() const override {
    std::ostringstream oss;
    oss << "F64(" << v_ << ")";
    return oss.str();
  }

 private:
  double v_;
};
using FP64ImmPtr = std::shared_ptr<FP64Imm>;
IMM_TRAITS(FP64ImmPtr, double)

/// \brief BF16Imm defines interface for bfloat16 data.
class MS_CORE_API BF16Imm final : public FloatImm {
 public:
  /// \brief The default constructor for BF16Imm.
  BF16Imm() : FloatImm(kBFloat16), v_(0.0) {}
  /// \brief The constructor for BF16Imm.
  ///
  /// \param[in] v The value of BF16Imm.
  explicit BF16Imm(bfloat16 v) : FloatImm(kBFloat16), v_(v) {
    hash_ = hash_combine({tid(), std::hash<bfloat16>{}(v_)});
  }
  /// \brief The destructor of BF16Imm.
  ~BF16Imm() override = default;
  MS_DECLARE_PARENT(BF16Imm, FloatImm)
  std::size_t hash() const override { return hash_; }
  bool IsZero() override { return v_ == BFloat16(0.0); }
  bool IsOne() override { return v_ == BFloat16(1.0); }
  /// \brief Get the value of BF16Imm.
  ///
  /// \return Return the value of BF16Imm.
  bfloat16 value() const { return v_; }
  bool operator==(const Value &other) const override;
  /// \brief Compare two BF16Imm objects is equal.
  ///
  /// \param[in] other The other BF16Imm to be compared with.
  /// \return Return true if other's value and the value of current object are the same,else return false.
  bool operator==(const BF16Imm &other) const;
  std::string ToString() const override { return scalar_float_to_string(v_); }

  std::string DumpText() const override {
    std::ostringstream oss;
    oss << "BF16(" << v_ << ")";
    return oss.str();
  }

 private:
  bfloat16 v_;
};
using BF16ImmPtr = std::shared_ptr<BF16Imm>;
IMM_TRAITS(BF16ImmPtr, bfloat16)

/// \brief ComplexImm defines interface for complex data.
class MS_CORE_API ComplexImm : public Scalar {
 public:
  /// \brief The default constructor for ComplexImm.
  ComplexImm() = default;
  /// \brief The constructor for ComplexImm.
  ///
  /// \param[in] t The value of ComplexImm.
  explicit ComplexImm(const TypePtr &t) : Scalar(t) {}
  /// \brief The destructor of ComplexImm.
  ~ComplexImm() override = default;
  MS_DECLARE_PARENT(ComplexImm, Scalar)
};
using ComplexImmPtr = std::shared_ptr<ComplexImm>;

/// \brief Complex64Imm defines interface for complex64 data.
class MS_CORE_API Complex64Imm final : public ComplexImm {
 public:
  /// \brief The default constructor for Complex64Imm.
  Complex64Imm() : ComplexImm(kComplex64), v_(0.0) {}
  /// \brief The constructor for Complex64Imm.
  ///
  /// \param[in] v The value of Complex64Imm.
  explicit Complex64Imm(std::complex<float> v) : ComplexImm(kComplex64), v_(v) {
    hash_ = hash_combine({tid(), hash_combine(std::hash<float>{}(v_.real()), std::hash<float>{}(v_.imag()))});
  }
  /// \brief The destructor of Complex64Imm.
  ~Complex64Imm() override = default;
  MS_DECLARE_PARENT(Complex64Imm, ComplexImm)
  std::size_t hash() const override { return hash_; }
  bool IsZero() override { return v_ == std::complex<float>(0.0); }
  bool IsOne() override { return v_ == std::complex<float>(1.0); }
  /// \brief Get the value of Complex64Imm.
  ///
  /// \return Return the value of Complex64Imm.
  std::complex<float> value() const { return v_; }
  bool operator==(const Value &other) const override;
  /// \brief Compare two Complex64Imm objects is equal.
  ///
  /// \param[in] other The other Complex64Imm to be compared with.
  /// \return Return true if other's value and the value of current object are the same,else return false.
  bool operator==(const Complex64Imm &other) const;
  std::string ToString() const override { return scalar_float_to_string(v_); }

  std::string DumpText() const override {
    std::ostringstream oss;
    oss << "Complex64(" << v_ << ")";
    return oss.str();
  }

 private:
  std::complex<float> v_;
};
using Complex64ImmPtr = std::shared_ptr<Complex64Imm>;
IMM_TRAITS(Complex64ImmPtr, std::complex<float>)

/// \brief Complex128Imm defines interface for complex128 data.
class MS_CORE_API Complex128Imm final : public ComplexImm {
 public:
  /// \brief The default constructor for Complex128Imm.
  Complex128Imm() : ComplexImm(kComplex128), v_(0.0) {}
  /// \brief The constructor for Complex128Imm.
  ///
  /// \param[in] v The value of Complex128Imm.
  explicit Complex128Imm(std::complex<double> v) : ComplexImm(kComplex128), v_(v) {
    hash_ = hash_combine({tid(), hash_combine(std::hash<double>{}(v_.real()), std::hash<double>{}(v_.imag()))});
  }
  /// \brief The destructor of Complex128Imm.
  ~Complex128Imm() override = default;
  MS_DECLARE_PARENT(Complex128Imm, ComplexImm)
  std::size_t hash() const override { return hash_; }
  bool IsZero() override { return v_ == std::complex<double>(0.0); }
  bool IsOne() override { return v_ == std::complex<double>(1.0); }
  /// \brief Get the value of Complex128Imm.
  ///
  /// \return Return the value of Complex128Imm.
  std::complex<double> value() const { return v_; }
  bool operator==(const Value &other) const override;
  /// \brief Compare two Complex128Imm objects is equal.
  ///
  /// \param[in] other The other Complex128Imm to be compared with.
  /// \return Return true if other's value and the value of current object are the same,else return false.
  bool operator==(const Complex128Imm &other) const;
  std::string ToString() const override { return scalar_float_to_string(v_); }

  std::string DumpText() const override {
    std::ostringstream oss;
    oss << "Complex128(" << v_ << ")";
    return oss.str();
  }

 private:
  std::complex<double> v_;
};
using Complex128ImmPtr = std::shared_ptr<Complex128Imm>;
IMM_TRAITS(Complex128ImmPtr, std::complex<double>)
}  // namespace mindspore

#endif  // MINDSPORE_CORE_IR_SCALAR_H_
