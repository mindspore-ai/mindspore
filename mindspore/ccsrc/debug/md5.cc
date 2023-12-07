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

#include "debug/md5.h"

namespace mindspore {
MD5::MD5(const string &message) : _str(message) {
  if (_str.size() != 0) {
    str_2_HEX();
    Data_Amplification();
    Magic_Number_Update();
    Read_Magic_Num();
  } else {
    // cout << "输入字符串为空" << endl;
  }
}

void MD5::str_2_HEX(void) {
  // 测试用例数字，用字符串表示的数字
  // 测试用例所有符号，根据ASSIC码8个字节可以表示一个字符
  // 遍历_str，然后依次用8bit进行表示字符，然后依次位移，用4个字节表示，然后放进_md5Code
  int i = 0;
  bit32 temp_vc = 0;
  vector<bit32> vc;
  for (const auto &e : _str) {
    byte assic = e - 0;
    temp_vc |= assic;
    i++;
    if (i == 4) {
      i = 0;
      vc.push_back(temp_vc);
      temp_vc = 0;
      // 64个字节大小的数组是为了后面好处理
      if (vc.size() == 16) {
        _md5Code.push_back(vc);  // 满足16个元素就压入
        vc.resize(0);
      }
    }

    temp_vc <<= 8;
  }
  // 用于处理多出来的数据
  if (temp_vc) {
    vc.push_back(temp_vc);
  }
  if (!vc.empty()) {
    _md5Code.push_back(vc);
  }
  // 这样_md5Code就是一个最后一个一维vector可能不满64,剩下准备补位
}

/*
 * 将字符串数据转换成4个字节为元素类型的二维数组中，有可能会出现最后一个元素其实并没有满4个字节，所以要靠
 * 统计字符串长度后，计算后重新 |
 */
void MD5::Data_Amplification(void) {
  int last_num_value = 4 - _str.size() % 4;  // 找出最后一个 元素 还需要几个字节满4个字节
  int last_count = 0;
  int last_index = (_md5Code.size() - 1);
  int last_row_index = _md5Code[last_index].size() - 1;
  // 先对二维数组的最后一个元素进行处理
  if (last_num_value != 4) {
    last_count = last_num_value;
    // 对最后一个元素补位
    _md5Code[last_index][last_row_index] |= (1 << 7);
    last_num_value -= 1;
    while (last_num_value) {
      _md5Code[last_index][last_row_index] <<= 8;
      _md5Code[last_index][last_row_index] |= 0;
      last_num_value -= 1;
    }
  }

  // 还需要计算需要多少个字节才能满足（M*64+56）字节
  // 判断二维数组的最后一行的元素个数才能满足16个元素或者14个元素
  int last_num = _md5Code[last_index].size();
  // 如果上面的补充最后一个元素，那么剩下的都是0，如果没补充就需要补充一个1
  if (last_count == 0) {
    // 需要补充一个1
    // 还要防止本行已满16
    if (last_num >= 16) {
      // 重新开辟下一行
      _md5Code.push_back(vector<bit32>());
      last_index += 1;
      last_num = 0;  // 最后一行的元素数量置零
    }
    const bit32 one = 1;
    bit32 i = one << 31;
    last_num += 1;
    _md5Code[last_index].push_back(i);
  }
  int aim_count = 0;
  // 直接考虑还需要多少满足14，继续补位
  // 补0补满14即可
  if (last_num < 14) {
    aim_count = 14 - last_num;
    // 再补16个0
  } else if (last_num == 14) {
    aim_count = 16;
    // 补满当前行，然后再补14个元素
  } else if (last_num > 14 && last_num <= 16) {
    aim_count = 16 - last_num + 14;
  }

  while (aim_count) {
    if (_md5Code[last_index].size() >= 16) {  // 要先判断是否要添加一行，再进行元素插入。
      _md5Code.push_back(vector<bit32>());
      last_index += 1;
    }
    _md5Code[last_index].push_back(0);
    aim_count -= 1;
  }
  // 补位之后，数组的最后一行就有14个元素，还需要把最后两个元素用于存储输入字符串的长度
  // 使用2个32bit大小的元素表示字符串长度，但是最多就只有64个比特位，其所能表示的最大长度是0xFFFFFFFFFFFFFFFF，大约是2^64
  // 所以这8个字节就于字符串长度位或，保留后64位比特位
  bit64 len = _str.size();
  bit64 x = 0x0000000000000000;
  x |= len;
  bit32 xl = (x >> 8 * 4) & (0xFFFFFFFF);  // 二进制位要移动32个比特位，（8个16进制位 * 每4个二进制为表示一个16进制位）
  bit32 xr = (x & 0xFFFFFFFF);
  _md5Code[last_index].push_back(xl);
  _md5Code[last_index].push_back(xr);
}

void MD5::Magic_Number_Update(void) {
  // 4个标准幻数都是4个字节-32个比特
  // md5Code的每一行都是由16个元素，每个元素4个字节
  // 有4个幻数处理的函数，每个元素都要把这4个函数执行一次
  // 区更新这四个幻数，然后把根性下来的幻数与初始进入计算的幻数进行相加，然后完成一次更新
  // 对于处理的顺序，我没理解，我们好像所有的实现的顺序都是这样的，可能是算法的一部分吧。
  int len = _md5Code.size();
  for (int i = 0; i < len; i++) {
    /* Round 1 */
    vector<bit32> &x = _md5Code[i];
    bit32 &a = _magicNumUpdate[0];
    bit32 &b = _magicNumUpdate[1];
    bit32 &c = _magicNumUpdate[2];
    bit32 &d = _magicNumUpdate[3];
    FF(a, b, c, d, x[0], s11, 0xd76aa478);
    FF(d, a, b, c, x[1], s12, 0xe8c7b756);
    FF(c, d, a, b, x[2], s13, 0x242070db);
    FF(b, c, d, a, x[3], s14, 0xc1bdceee);
    FF(a, b, c, d, x[4], s11, 0xf57c0faf);
    FF(d, a, b, c, x[5], s12, 0x4787c62a);
    FF(c, d, a, b, x[6], s13, 0xa8304613);
    FF(b, c, d, a, x[7], s14, 0xfd469501);
    FF(a, b, c, d, x[8], s11, 0x698098d8);
    FF(d, a, b, c, x[9], s12, 0x8b44f7af);
    FF(c, d, a, b, x[10], s13, 0xffff5bb1);
    FF(b, c, d, a, x[11], s14, 0x895cd7be);
    FF(a, b, c, d, x[12], s11, 0x6b901122);
    FF(d, a, b, c, x[13], s12, 0xfd987193);
    FF(c, d, a, b, x[14], s13, 0xa679438e);
    FF(b, c, d, a, x[15], s14, 0x49b40821);

    /* Round 2 */
    GG(a, b, c, d, x[1], s21, 0xf61e2562);
    GG(d, a, b, c, x[6], s22, 0xc040b340);
    GG(c, d, a, b, x[11], s23, 0x265e5a51);
    GG(b, c, d, a, x[0], s24, 0xe9b6c7aa);
    GG(a, b, c, d, x[5], s21, 0xd62f105d);
    GG(d, a, b, c, x[10], s22, 0x2441453);
    GG(c, d, a, b, x[15], s23, 0xd8a1e681);
    GG(b, c, d, a, x[4], s24, 0xe7d3fbc8);
    GG(a, b, c, d, x[9], s21, 0x21e1cde6);
    GG(d, a, b, c, x[14], s22, 0xc33707d6);
    GG(c, d, a, b, x[3], s23, 0xf4d50d87);
    GG(b, c, d, a, x[8], s24, 0x455a14ed);
    GG(a, b, c, d, x[13], s21, 0xa9e3e905);
    GG(d, a, b, c, x[2], s22, 0xfcefa3f8);
    GG(c, d, a, b, x[7], s23, 0x676f02d9);
    GG(b, c, d, a, x[12], s24, 0x8d2a4c8a);

    /* Round 3 */
    HH(a, b, c, d, x[5], s31, 0xfffa3942);
    HH(d, a, b, c, x[8], s32, 0x8771f681);
    HH(c, d, a, b, x[11], s33, 0x6d9d6122);
    HH(b, c, d, a, x[14], s34, 0xfde5380c);
    HH(a, b, c, d, x[1], s31, 0xa4beea44);
    HH(d, a, b, c, x[4], s32, 0x4bdecfa9);
    HH(c, d, a, b, x[7], s33, 0xf6bb4b60);
    HH(b, c, d, a, x[10], s34, 0xbebfbc70);
    HH(a, b, c, d, x[13], s31, 0x289b7ec6);
    HH(d, a, b, c, x[0], s32, 0xeaa127fa);
    HH(c, d, a, b, x[3], s33, 0xd4ef3085);
    HH(b, c, d, a, x[6], s34, 0x4881d05);
    HH(a, b, c, d, x[9], s31, 0xd9d4d039);
    HH(d, a, b, c, x[12], s32, 0xe6db99e5);
    HH(c, d, a, b, x[15], s33, 0x1fa27cf8);
    HH(b, c, d, a, x[2], s34, 0xc4ac5665);

    /* Round 4 */
    II(a, b, c, d, x[0], s41, 0xf4292244);
    II(d, a, b, c, x[7], s42, 0x432aff97);
    II(c, d, a, b, x[14], s43, 0xab9423a7);
    II(b, c, d, a, x[5], s44, 0xfc93a039);
    II(a, b, c, d, x[12], s41, 0x655b59c3);
    II(d, a, b, c, x[3], s42, 0x8f0ccc92);
    II(c, d, a, b, x[10], s43, 0xffeff47d);
    II(b, c, d, a, x[1], s44, 0x85845dd1);
    II(a, b, c, d, x[8], s41, 0x6fa87e4f);
    II(d, a, b, c, x[15], s42, 0xfe2ce6e0);
    II(c, d, a, b, x[6], s43, 0xa3014314);
    II(b, c, d, a, x[13], s44, 0x4e0811a1);
    II(a, b, c, d, x[4], s41, 0xf7537e82);
    II(d, a, b, c, x[11], s42, 0xbd3af235);
    II(c, d, a, b, x[2], s43, 0x2ad7d2bb);
    II(b, c, d, a, x[9], s44, 0xeb86d391);

    // 将新的x加上初始进入改行计算的函数
    _magicNum[0] += _magicNumUpdate[0];
    _magicNum[1] += _magicNumUpdate[1];
    _magicNum[2] += _magicNumUpdate[2];
    _magicNum[3] += _magicNumUpdate[3];

    _magicNumUpdate[0] = _magicNum[0];
    _magicNumUpdate[1] = _magicNum[1];
    _magicNumUpdate[2] = _magicNum[2];
    _magicNumUpdate[3] = _magicNum[3];
  }
}

void MD5::Read_Magic_Num(void) {
  for (int i = 0; i < 4; i++) {
    _code += bit32_to_string(_magicNum[i]);
  }
}

string MD5::bit32_to_string(const bit32 &input) {
  string ret;
  // 目的就是把其中的4个字节转为16进制的字符表示
  // 一个bit32中有8个4个比特位
  // 4个比特位表示一个16进制字符
  for (int i = 0; i < 8; i++) {
    byte tmp = 0xF;
    tmp &= (input >> i * 4);
    if (tmp <= 9 && tmp >= 0) {
      ret += tmp + '0';
    } else if (tmp <= 15 && tmp >= 10) {
      ret += (tmp - 10) + 'A';
    }
  }
  // reverse(ret.begin(), ret.end());
  return ret;
}

const string &MD5::Get_MD5_Code(void) { return _code; }

void MD5::test_md5code(void) {
  int i = 0;
  for (const auto &e : _md5Code) {
    cout << "第" << i << "行长度：" << e.size() << endl;
    i++;
  }
}

void MD5::test_md5Code_value(void) {
  int row = _md5Code.size();
  for (int i = 0; i < row; i++) {
    int col = _md5Code[i].size();
    for (int j = 0; j < col; j++) {
      printf("%08x ", _md5Code[i][j]);
    }
    cout << endl;
  }
}
}  // namespace mindspore
