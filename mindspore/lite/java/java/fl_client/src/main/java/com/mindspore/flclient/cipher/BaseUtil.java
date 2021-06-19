/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 * <p>
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * <p>
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p>
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.mindspore.flclient.cipher;

import java.io.UnsupportedEncodingException;
import java.math.BigInteger;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.List;

public class BaseUtil {
    private static final char[] HEX_DIGITS = new char[]{'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F'};

    public BaseUtil() {
    }

    public static String byte2HexString(byte[] bytes) {
        if (null == bytes) {
            return null;
        } else if (bytes.length == 0) {
            return "";
        } else {
            char[] chars = new char[bytes.length * 2];

            for (int i = 0; i < bytes.length; ++i) {
                int b = bytes[i];
                chars[i * 2] = HEX_DIGITS[(b & 240) >> 4];
                chars[i * 2 + 1] = HEX_DIGITS[b & 15];
            }
            return new String(chars);
        }
    }

    public static byte[] hexString2ByteArray(String str) {
        int length = str.length() / 2;
        byte[] bytes = new byte[length];
        byte[] source = str.getBytes(Charset.forName("UTF-8"));

        for (int i = 0; i < bytes.length; ++i) {
            byte bh = Byte.decode("0x" + new String(new byte[]{source[i * 2]}, Charset.forName("UTF-8")));
            bh = (byte) (bh << 4);
            byte bl = Byte.decode("0x" + new String(new byte[]{source[i * 2 + 1]}, Charset.forName("UTF-8")));
            bytes[i] = (byte) (bh ^ bl);
        }
        return bytes;
    }

    public static BigInteger byteArray2BigInteger(byte[] bytes) {

        BigInteger bigInteger = BigInteger.ZERO;
        for (int i = 0; i < bytes.length; ++i) {
            int intI = bytes[i];
            if (intI < 0) {
                intI = intI + 256;
            }
            BigInteger bi = new BigInteger(String.valueOf(intI));
            bigInteger = bigInteger.multiply(BigInteger.valueOf(256)).add(bi);
        }
        return bigInteger;
    }

    public static BigInteger string2BigInteger(String str) throws UnsupportedEncodingException {
        StringBuilder res = new StringBuilder();
        byte[] bytes = String.valueOf(str).getBytes("UTF-8");
        BigInteger bigInteger = BigInteger.ZERO;
        for (int i = 0; i < str.length(); ++i) {
            BigInteger bi = new BigInteger(String.valueOf(bytes[i]));
            bigInteger = bigInteger.multiply(BigInteger.valueOf(256)).add(bi);
        }
        return bigInteger;
    }

    public static String bigInteger2String(BigInteger bigInteger) throws UnsupportedEncodingException {
        StringBuilder res = new StringBuilder();
        List<Integer> lists = new ArrayList<>();
        BigInteger bi = bigInteger;
        BigInteger DIV = BigInteger.valueOf(256);
        while (bi.compareTo(BigInteger.ZERO) > 0) {
            lists.add(bi.mod(DIV).intValue());
            bi = bi.divide(DIV);
        }
        for (int i = lists.size() - 1; i >= 0; --i) {
            res.append((char) (int) (lists.get(i)));
        }
        return res.toString();
    }

    public static byte[] bigInteger2byteArray(BigInteger bigInteger) throws UnsupportedEncodingException {
        List<Integer> lists = new ArrayList<>();
        BigInteger bi = bigInteger;
        BigInteger DIV = BigInteger.valueOf(256);
        while (bi.compareTo(BigInteger.ZERO) > 0) {
            lists.add(bi.mod(DIV).intValue());
            bi = bi.divide(DIV);
        }
        byte[] res = new byte[lists.size()];
        for (int i = lists.size() - 1; i >= 0; --i) {
            res[lists.size() - i - 1] = ((byte) (int) (lists.get(i)));
        }
        return res;
    }

    public static byte[] integer2byteArray(Integer num) {
        List<Integer> lists = new ArrayList<>();
        Integer bi = num;
        Integer DIV = 256;
        while (bi > 0) {
            lists.add(bi % DIV);
            bi = bi / DIV;
        }
        byte[] res = new byte[lists.size()];
        for (int i = lists.size() - 1; i >= 0; --i) {
            res[lists.size() - i - 1] = ((byte) (int) (lists.get(i)));
        }
        return res;
    }

    public static Integer byteArray2Integer(byte[] bytes) {

        Integer num = 0;
        for (int i = 0; i < bytes.length; ++i) {
            int intI = bytes[i];
            if (intI < 0) {
                intI = intI + 256;
            }
            num = num * 256 + intI;
        }
        return num;
    }
}