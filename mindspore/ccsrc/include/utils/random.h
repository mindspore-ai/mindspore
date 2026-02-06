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

#ifndef MINDSPORE_CCSRC_INCLUDE_COMMON_RANDOM_H_
#define MINDSPORE_CCSRC_INCLUDE_COMMON_RANDOM_H_

#include <cstdint>
#include <cmath>
#include <array>
#include <limits>
#include <random>
#include <vector>
#include <optional>
#include <algorithm>
#include <utility>
#include "include/utils/thread_pool.h"
#include "include/utils/utils.h"
#include "utils/log_adapter.h"

namespace mindspore::random {
//
// Generate random numbers into a buffer.
//
template <typename T, typename Generator, typename Distribution, typename... Args>
void GenerateRandoms(std::uint64_t seed, size_t skip, T *buf, size_t size, Args... args) {
  MS_EXCEPTION_IF_NULL(buf);
  Generator gen{seed};
  gen.discard(skip);
  Distribution dis{args...};
  for (size_t i = 0; i < size; ++i) {
    buf[i] = T(dis(gen));
  }
}

// Compute number of task and batch size of each task.
static inline std::pair<size_t, size_t> ComputeTaskNumSize(size_t total_size, size_t thread_num) {
  constexpr size_t min_parallel_size = 1024;
  if (thread_num == 0 || total_size <= min_parallel_size) {
    return {1, total_size};
  }
  constexpr size_t block_size = 4;
  const size_t block_count = (total_size + block_size - 1) / block_size;
  if (block_count <= thread_num) {
    return {block_count, block_size};
  }
  const size_t blocks_per_thread = (block_count + thread_num - 1) / thread_num;
  const size_t task_num = (block_count + blocks_per_thread - 1) / blocks_per_thread;
  const size_t batch_size = blocks_per_thread * block_size;
  return {task_num, batch_size};
}

//
// Parallel generate random numbers into a buffer.
//
template <typename T, typename Generator, typename Distribution, typename... Args>
void GenerateRandomsParallel(std::uint64_t input_seed, T *buf, size_t buf_size, Args... args) {
  MS_EXCEPTION_IF_NULL(buf);

  // Calculate number of tasks and batch size.
  auto &thread_pool = common::ThreadPool::GetInstance();
  auto [task_num, batch_size] = ComputeTaskNumSize(buf_size, thread_pool.GetSyncRunThreadNum());

  // Generate random seed if required.
  std::uint64_t seed = input_seed;
  if (seed == 0) {
    MS_LOG(INFO) << "The input seed for GenerateRandomsParallel is 0.";
  }

  if (task_num == 1) {
    // Use single thread for small data size.
    GenerateRandoms<T, Generator, Distribution>(seed, 0, buf, buf_size, args...);
    return;
  }

  // Prepare parallel tasks.
  std::vector<common::Task> tasks;
  tasks.reserve(task_num);
  T *task_buf = buf;
  size_t skip = 0;
  for (size_t i = 0; i < task_num; ++i) {
    const auto task_size = ((i == task_num - 1) ? (buf_size - (task_num - 1) * batch_size) : batch_size);
    (void)tasks.emplace_back([seed, skip, task_buf, task_size, args...]() {
      GenerateRandoms<T, Generator, Distribution>(seed, skip, task_buf, task_size, args...);
      return common::SUCCESS;
    });
    skip += task_size;
    task_buf += task_size;
  }
  // Parallel execute tasks by thread pool.
  (void)thread_pool.SyncRun(tasks);
}

//
// Philox is a random number generator that is suitable for parallel random number generating.
//
class Philox {
 public:
  explicit Philox(uint64_t seed)
      : key_({static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> kShift32)}),
        counter_({0, 0, static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> kShift32)}),
        results_({}) {}

  Philox(uint64_t seed, uint64_t seed2)
      : key_({static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> kShift32)}),
        counter_({0, 0, static_cast<uint32_t>(seed2), static_cast<uint32_t>(seed2 >> kShift32)}),
        results_({}) {}

  ~Philox() = default;

  uint32_t operator()() {
    if (index_ == kCounterNum) {
      results_ = next();
      index_ = 0;
    }
    return results_[index_++];
  }

  void discard(uint64_t step) {
    if (index_ == kCounterNum) {
      const auto count = (step / kCounterNum);
      skip(count);
      const auto remain = (step % kCounterNum);
      if (remain > 0) {
        results_ = next();
        index_ = remain;
      }
    } else {
      const auto pos = index_ + step;
      if (pos <= kCounterNum) {
        index_ = pos;
      } else {
        const auto count = (pos - kCounterNum) / kCounterNum;
        skip(count);
        const auto remain = (pos % kCounterNum);
        if (remain > 0) {
          results_ = next();
          index_ = remain;
        } else {
          index_ = kCounterNum;
        }
      }
    }
  }

  static constexpr uint32_t min() { return 0; }
  static constexpr uint32_t max() { return std::numeric_limits<uint32_t>::max(); }

 private:
  static constexpr int kShift32 = 32;
  static constexpr size_t kCounterNum = 4;
  static constexpr size_t kKeyNum = 2;
  static constexpr size_t kIndex0 = 0;
  static constexpr size_t kIndex1 = 1;
  static constexpr size_t kIndex2 = 2;
  static constexpr size_t kIndex3 = 3;
  static constexpr uint32_t kMagic0 = 0xD2511F53;
  static constexpr uint32_t kMagic1 = 0xCD9E8D57;
  static constexpr uint32_t kKeyStep0 = 0x9E3779B9;
  static constexpr uint32_t kKeyStep1 = 0xBB67AE85;

  using Counter = std::array<uint32_t, kCounterNum>;
  using Key = std::array<uint32_t, kKeyNum>;

  Key key_;
  Counter counter_;
  Counter results_;
  size_t index_ = kCounterNum;

  static void compute(uint32_t *counter, const uint32_t *key) {
    const uint64_t t0 = static_cast<uint64_t>(kMagic0) * counter[kIndex0];
    const uint32_t l0 = static_cast<uint32_t>(t0);
    const uint32_t h0 = static_cast<uint32_t>(t0 >> kShift32);
    const uint64_t t1 = static_cast<uint64_t>(kMagic1) * counter[kIndex2];
    const uint32_t l1 = static_cast<uint32_t>(t1);
    const uint32_t h1 = static_cast<uint32_t>(t1 >> kShift32);
    counter[kIndex0] = (h1 ^ counter[kIndex1] ^ key[kIndex0]);
    counter[kIndex1] = l1;
    counter[kIndex2] = (h0 ^ counter[kIndex3] ^ key[kIndex1]);
    counter[kIndex3] = l0;
  }

  static void raise_key(uint32_t *key) {
    key[kIndex0] += kKeyStep0;
    key[kIndex1] += kKeyStep1;
  }

  // Generate next 4 random numbers and advance counter.
  Counter next() {
    Counter result = counter_;
    Key key = key_;
    // For performance reason, we do not use loop here,
    // but manually call compute() 10 times.
    compute(result.data(), key.data());
    raise_key(key.data());
    compute(result.data(), key.data());
    raise_key(key.data());
    compute(result.data(), key.data());
    raise_key(key.data());
    compute(result.data(), key.data());
    raise_key(key.data());
    compute(result.data(), key.data());
    raise_key(key.data());
    compute(result.data(), key.data());
    raise_key(key.data());
    compute(result.data(), key.data());
    raise_key(key.data());
    compute(result.data(), key.data());
    raise_key(key.data());
    compute(result.data(), key.data());
    raise_key(key.data());
    compute(result.data(), key.data());
    skip_one();
    return result;
  }

  // Advance counter for one step.
  void skip_one() {
    if (++counter_[kIndex0] == 0) {
      if (++counter_[kIndex1] == 0) {
        if (++counter_[kIndex2] == 0) {
          ++counter_[kIndex3];
        }
      }
    }
  }

  // Skip the given number of samples of 4 uint32.
  void skip(uint64_t count) {
    const uint32_t lo = static_cast<uint32_t>(count);
    uint32_t hi = static_cast<uint32_t>(count >> kShift32);
    counter_[kIndex0] += lo;
    if (counter_[kIndex0] < lo) {
      ++hi;
    }
    counter_[kIndex1] += hi;
    if (counter_[kIndex1] < hi) {
      if (++counter_[kIndex2] == 0) {
        ++counter_[kIndex3];
      }
    }
  }
};

//
// Uniform distribution.
//
template <typename T>
class UniformDistribution {
 public:
  UniformDistribution(T a, T b) : a_(a), b_(b) {}
  ~UniformDistribution() = default;

  template <typename Generator>
  T operator()(Generator &&g) const {
    const auto min_num = g.min();
    const auto max_num = g.max();
    const long double range = static_cast<long double>(max_num) - static_cast<long double>(min_num) + 1.0L;
    T s = static_cast<T>(T(g() - min_num) / range);
    if (s >= T(1)) {
      s = std::nextafter(T(1), T(0));
    }
    return (b_ - a_) * s + a_;
  }

 private:
  T a_;
  T b_;
};  // namespace mindspore::random

//
// Normal distribution.
//
template <typename T>
class NormalDistribution {
 public:
  NormalDistribution(T mean, T sigma) : mean_(mean), sigma_(sigma) {}
  ~NormalDistribution() = default;

  template <typename Generator>
  T operator()(Generator &&g) const {
    if (has_next_) {
      has_next_ = false;
      return next_;
    }
    // Box-Muller transform algorithm:
    // z1 = sqrt(-2 * ln(u1)) * cos(2 * pi * u2)
    // z2 = sqrt(-2 * ln(u1)) * sin(2 * pi * u2)
    constexpr T pi = 3.1415926f;
    constexpr T threshold = 1.0e-7f;
    const T u1 = std::max(to_float(g()), threshold);
    const T u2 = std::max(to_float(g()), threshold);
    const T x = std::sqrt(-2.0f * std::log(u1)) * sigma_;
    const T y = 2.0f * pi * u2;
    next_ = mean_ + (x * std::sin(y));
    has_next_ = true;
    return mean_ + (x * std::cos(y));
  }

 private:
  T mean_;
  T sigma_;
  mutable T next_ = 0;
  mutable bool has_next_ = false;

  static T to_float(uint32_t input) {
    constexpr uint32_t mask = 0x7fffffu;
    constexpr uint32_t exp = (127 << 23);
    union {
      uint32_t int_val;
      float float_val;
    } val;
    val.int_val = (input & mask) | exp;
    return T(val.float_val - 1.0f);
  }
};

//
// Truncated normal distribution.
//
template <typename T>
class TruncatedNormal {
 public:
  TruncatedNormal(T a, T b, T mean, T sigma) : lower_(a), upper_(b), mean_(mean), sigma_(sigma) {
    if (sigma <= 0) {
      MS_LOG(EXCEPTION) << "TruncatedNormal: invalid sigma " << sigma << ".";
    } else {
      alpha_ = (a - mean) / sigma;
      beta_ = (b - mean) / sigma;
    }
  }

  ~TruncatedNormal() = default;

  template <typename Generator>
  T operator()(Generator &&g) const {
    // Inverse CDF (Cumulative Distribution Function) method.
    const T u = std_uniform_(g);
    const T cdf_a = cdf(alpha_);
    const T cdf_b = cdf(beta_);
    const T p = cdf_a + u * (cdf_b - cdf_a);
    const T x = quantile(p);
    return mean_ + x * sigma_;
  }

 private:
  UniformDistribution<T> std_uniform_{0.0f, 1.0f};
  T lower_;
  T upper_;
  T mean_;
  T sigma_;
  T alpha_;
  T beta_;

  static constexpr T kRootTwo = 1.4142135f;

  static T cdf(T x) {
    const T diff = x / kRootTwo;
    return std::erfc(-diff) / 2.0f;
  }

  static T quantile(T p) {
    auto z = 2.0f * p;
    const T x = erfc_inv(z);
    return -x * kRootTwo;
  }

  static T erfc_inv(T z) {
    // Keep z in range (0, 2).
    if (z <= 0) {
      z = std::nextafterf(0.0f, 2.0f);
    } else if (z >= 2.0f) {
      z = std::nextafterf(2.0f, 0.0f);
    }
    T p, q, s;
    if (z > 1.0f) {
      q = 2.0f - z;
      p = 1.0f - q;
      s = -1;
    } else {
      p = 1.0f - z;
      q = z;
      s = 1;
    }
    return s * erf_inv_imp(p, q);
  }

  // The algorithm and polynomia constants are borrow from boost.
  static T erf_inv_imp(T p, T q) {
    if (p <= 0.5f) {
      constexpr float Y = 0.0891314744949340820313f;
      constexpr T P[] = {T(-0.000508781949658280665617), T(-0.00836874819741736770379), T(0.0334806625409744615033),
                         T(-0.0126926147662974029034),   T(-0.0365637971411762664006),  T(0.0219878681111168899165),
                         T(0.00822687874676915743155),   T(-0.00538772965071242932965)};
      constexpr T Q[] = {T(1.0),
                         T(-0.970005043303290640362),
                         T(-1.56574558234175846809),
                         T(1.56221558398423026363),
                         T(0.662328840472002992063),
                         T(-0.71228902341542847553),
                         T(-0.0527396382340099713954),
                         T(0.0795283687341571680018),
                         T(-0.00233393759374190016776),
                         T(0.000886216390456424707504)};
      T g = p * (p + 10.0f);
      T r = eval_polynomial(P, p) / eval_polynomial(Q, p);
      return g * Y + g * r;
    }
    if (q >= 0.25f) {
      constexpr float Y = 2.249481201171875f;
      constexpr T P[] = {T(-0.202433508355938759655), T(0.105264680699391713268), T(8.37050328343119927838),
                         T(17.6447298408374015486),   T(-18.8510648058714251895), T(-44.6382324441786960818),
                         T(17.445385985570866523),    T(21.1294655448340526258),  T(-3.67192254707729348546)};
      constexpr T Q[] = {T(1.0),
                         T(6.24264124854247537712),
                         T(3.9713437953343869095),
                         T(-28.6608180499800029974),
                         T(-20.1432634680485188801),
                         T(48.5609213108739935468),
                         T(10.8268667355460159008),
                         T(-22.6436933413139721736),
                         T(1.72114765761200282724)};
      T g = std::sqrt(-2.0f * std::log(q));
      T xs = q - 0.25f;
      T r = eval_polynomial(P, xs) / eval_polynomial(Q, xs);
      return g / (Y + r);
    }
    // Avoid static check warning for 'function body too long'.
    return erf_inv_imp2(q);
  }

  static T erf_inv_imp2(T q) {
    T x = std::sqrt(-std::log(q));
    if (x < 3.0f) {
      constexpr float Y = 0.807220458984375f;
      constexpr T P[] = {T(-0.131102781679951906451),   T(-0.163794047193317060787),   T(0.117030156341995252019),
                         T(0.387079738972604337464),    T(0.337785538912035898924),    T(0.142869534408157156766),
                         T(0.0290157910005329060432),   T(0.00214558995388805277169),  T(-0.679465575181126350155e-6),
                         T(0.285225331782217055858e-7), T(-0.681149956853776992068e-9)};
      constexpr T Q[] = {T(1.0),
                         T(3.46625407242567245975),
                         T(5.38168345707006855425),
                         T(4.77846592945843778382),
                         T(2.59301921623620271374),
                         T(0.848854343457902036425),
                         T(0.152264338295331783612),
                         T(0.01105924229346489121)};
      T xs = x - 1.125f;
      T R = eval_polynomial(P, xs) / eval_polynomial(Q, xs);
      return Y * x + R * x;
    }
    if (x < 6.0f) {
      constexpr float Y = 0.93995571136474609375f;
      constexpr T P[] = {T(-0.0350353787183177984712),  T(-0.00222426529213447927281),  T(0.0185573306514231072324),
                         T(0.00950804701325919603619),  T(0.00187123492819559223345),   T(0.000157544617424960554631),
                         T(0.460469890584317994083e-5), T(-0.230404776911882601748e-9), T(0.266339227425782031962e-11)};
      constexpr T Q[] = {T(1.0),
                         T(1.3653349817554063097),
                         T(0.762059164553623404043),
                         T(0.220091105764131249824),
                         T(0.0341589143670947727934),
                         T(0.00263861676657015992959),
                         T(0.764675292302794483503e-4)};
      T xs = x - 3.0f;
      T R = eval_polynomial(P, xs) / eval_polynomial(Q, xs);
      return Y * x + R * x;
    }
    if (x < 18.0f) {
      constexpr float Y = 0.98362827301025390625f;
      constexpr T P[] = {T(-0.0167431005076633737133),  T(-0.00112951438745580278863),   T(0.00105628862152492910091),
                         T(0.000209386317487588078668), T(0.149624783758342370182e-4),   T(0.449696789927706453732e-6),
                         T(0.462596163522878599135e-8), T(-0.281128735628831791805e-13), T(0.99055709973310326855e-16)};
      constexpr T Q[] = {T(1.0),
                         T(0.591429344886417493481),
                         T(0.138151865749083321638),
                         T(0.0160746087093676504695),
                         T(0.000964011807005165528527),
                         T(0.275335474764726041141e-4),
                         T(0.282243172016108031869e-6)};
      T xs = x - 6.0f;
      T R = eval_polynomial(P, xs) / eval_polynomial(Q, xs);
      return Y * x + R * x;
    }
    if (x < 44.0f) {
      constexpr float Y = 0.99714565277099609375f;
      constexpr T P[] = {T(-0.0024978212791898131227),   T(-0.779190719229053954292e-5), T(0.254723037413027451751e-4),
                         T(0.162397777342510920873e-5),  T(0.396341011304801168516e-7),  T(0.411632831190944208473e-9),
                         T(0.145596286718675035587e-11), T(-0.116765012397184275695e-17)};
      constexpr T Q[] = {T(1.0),
                         T(0.207123112214422517181),
                         T(0.0169410838120975906478),
                         T(0.000690538265622684595676),
                         T(0.145007359818232637924e-4),
                         T(0.144437756628144157666e-6),
                         T(0.509761276599778486139e-9)};
      T xs = x - 18.0f;
      T R = eval_polynomial(P, xs) / eval_polynomial(Q, xs);
      return Y * x + R * x;
    }
    constexpr float Y = 0.99941349029541015625f;
    constexpr T P[] = {T(-0.000539042911019078575891), T(-0.28398759004727721098e-6),  T(0.899465114892291446442e-6),
                       T(0.229345859265920864296e-7),  T(0.225561444863500149219e-9),  T(0.947846627503022684216e-12),
                       T(0.135880130108924861008e-14), T(-0.348890393399948882918e-21)};
    constexpr T Q[] = {T(1.0),
                       T(0.0845746234001899436914),
                       T(0.00282092984726264681981),
                       T(0.468292921940894236786e-4),
                       T(0.399968812193862100054e-6),
                       T(0.161809290887904476097e-8),
                       T(0.231558608310259605225e-11)};
    T xs = x - 44.0f;
    T R = eval_polynomial(P, xs) / eval_polynomial(Q, xs);
    return Y * x + R * x;
  }

  // We use template function to unrolling polynomial evaluations
  // at compile time to improve performance.
  template <size_t N>
  static T eval_polynomial(const T (&arr)[N], T x) {
    T sum = arr[N - 1];
    if constexpr (N > 1) {
      eval_polynomial_loop<N - kIndex2>(arr, x, &sum);
    }
    return sum;
  }

  template <size_t Index>
  static void eval_polynomial_loop(const T *arr, T x, T *sum) {
    *sum *= x;
    *sum += arr[Index];
    if constexpr (Index > 0) {
      eval_polynomial_loop<Index - 1>(arr, x, sum);
    }
  }
};

//
// Constant distribution.
//
template <typename T>
class ConstantDistribution {
 public:
  explicit ConstantDistribution(T value) : value_(value) {}
  ~ConstantDistribution() = default;

  template <typename Generator>
  T operator()(Generator &&) const {
    return value_;
  }

 private:
  T value_;
};
}  // namespace mindspore::random

#endif  // MINDSPORE_CCSRC_INCLUDE_COMMON_RANDOM_H_
