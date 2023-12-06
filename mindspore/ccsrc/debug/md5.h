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

#ifndef MINDSPORE_MD5_H
#define MINDSPORE_MD5_H
#include <string>
#include <vector>
#include <iostream>
#include <bitset>
using std::cout;
using std::endl;
using std::string;
using std::vector;

namespace mindspore {
// 这些参数是处理幻数的函数使用的循环左移的位数，64个字节分4轮处理，每轮调用16次函数，一轮中使用的一个参数4次
// 第一轮
#define s11 7
#define s12 12
#define s13 17
#define s14 22
// 第二轮
#define s21 5
#define s22 9
#define s23 14
#define s24 20
// 第三轮
#define s31 4
#define s32 11
#define s33 16
#define s34 23
// 第四轮
#define s41 6
#define s42 10
#define s43 15
#define s44 21

// 循环左移函数
#define SHIFT_LEFT(num, n) (((num) << (n)) | ((num) >> (32 - n)))

// 这4个宏函数用于处理 FF GG HH II的对应的函数中的另外3个不处理的幻数
#define F(x, y, z) ((x & y) | ((~x) & z))
#define G(x, y, z) ((x & z) | (y & (~z)))
#define H(x, y, z) (x ^ y ^ z)
#define I(x, y, z) (y ^ (x | (~z)))

// 用于更新幻数
#define FF(a, b, c, d, M, SL, AC)       \
  {                                     \
    (a) += F((b), (c), (d)) + (M) + AC; \
    (a) = SHIFT_LEFT((a), (SL));        \
    (a) += (b);                         \
  }
#define GG(a, b, c, d, M, SL, AC)       \
  {                                     \
    (a) += G((b), (c), (d)) + (M) + AC; \
    (a) = SHIFT_LEFT((a), (SL));        \
    (a) += (b);                         \
  }
#define HH(a, b, c, d, M, SL, AC)       \
  {                                     \
    (a) += H((b), (c), (d)) + (M) + AC; \
    (a) = SHIFT_LEFT((a), (SL));        \
    (a) += (b);                         \
  }
#define II(a, b, c, d, M, SL, AC)       \
  {                                     \
    (a) += I((b), (c), (d)) + (M) + AC; \
    (a) = SHIFT_LEFT((a), (SL));        \
    (a) += (b);                         \
  }

// 对于MD5编码，输入的是字符串数据，将字符转化成比特位，但是最后进行幻数处理都是使用4个字节的数据进行处
// 最后保留的数据都是 M*64 个字节
class MD5 {
  typedef unsigned char byte;
  typedef unsigned int bit32;
  typedef uint64_t bit64;

 public:
  /*
   * 功能：
   * 将字符串数据读入对象中
   * 获取MD5的编码
   */
  explicit MD5(const string &message);
  MD5();
  const string &Get_MD5_Code(void);

  // 用于测试md4Code的长度是否符合要求
  void test_md5code(void);
  // 用于测试md5Code的元素是否补充正确
  void test_md5Code_value(void);

 private:
  // 把message转换成二进制/16进制码
  void str_2_HEX(void);
  // 准备扩增数据扩增到M*64个字节
  void Data_Amplification(void);

  // 幻数处理_md5Code,更新幻数
  void Magic_Number_Update();
  // 记录最后的4个幻数
  void Read_Magic_Num(void);
  string bit32_to_string(const bit32 &input);

 private:
  string _str;  // 保存输入信息的数据

  // 四个幻数-小端存储模式:     A    B    C    D
  vector<bit32> _magicNum = {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476};

  vector<bit32> _magicNumUpdate = {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476};

  // 用于保存转化后的M*64个字节的数据
  vector<std::vector<bit32>> _md5Code;

  // 用于保存转换后的MD5编码，16个byte
  string _code;
};
}  // namespace mindspore
#endif  // MINDSPORE_MD5_H
