/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
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

package com.mindspore.flclient.cipher;

import java.io.UnsupportedEncodingException;
import java.math.BigInteger;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.List;

/**
 * Define conversion methods between basic data types.
 *
 * @since 2021-06-30
 */
public class BaseUtil {
    private static final char[] HEX_DIGITS = new char[]{'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B',
            'C', 'D', 'E', 'F'};

    /**
     * Convert byte[] to String in hexadecimal format.
     *
     * @param bytes the byte[] object.
     * @return the String object converted from byte[].
     */
    public static String byte2HexString(byte[] bytes) {
        if (bytes == null) {
            return null;
        } else if (bytes.length == 0) {
            return "";
        } else {
            char[] chars = new char[bytes.length * 2];

            for (int i = 0; i < bytes.length; ++i) {
                int byteNum = bytes[i];
                chars[i * 2] = HEX_DIGITS[(byteNum & 240) >> 4];
                chars[i * 2 + 1] = HEX_DIGITS[byteNum & 15];
            }
            return new String(chars);
        }
    }

    /**
     * Convert String in hexadecimal format to byte[].
     *
     * @param str the String object.
     * @return the byte[] converted from String object.
     */
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

    /**
     * Convert byte[] to BigInteger.
     *
     * @param bytes the byte[] object.
     * @return the BigInteger object converted from byte[].
     */
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

    /**
     * Convert String to BigInteger.
     *
     * @param str the String object.
     * @return the BigInteger object converted from String object.
     * @throws UnsupportedEncodingException if the encoding is not supported.
     */
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

    /**
     * Convert BigInteger to String.
     *
     * @param bigInteger the BigInteger object.
     * @return the String object converted from BigInteger.
     */
    public static String bigInteger2String(BigInteger bigInteger) {
        StringBuilder res = new StringBuilder();
        List<Integer> lists = new ArrayList<>();
        BigInteger bi = bigInteger;
        BigInteger div = BigInteger.valueOf(256);
        while (bi.compareTo(BigInteger.ZERO) > 0) {
            lists.add(bi.mod(div).intValue());
            bi = bi.divide(div);
        }
        for (int i = lists.size() - 1; i >= 0; --i) {
            res.append((char) (int) (lists.get(i)));
        }
        return res.toString();
    }

    /**
     * Convert BigInteger to byte[].
     *
     * @param bigInteger the BigInteger object.
     * @return the byte[] object converted from BigInteger.
     */
    public static byte[] bigInteger2byteArray(BigInteger bigInteger) {
        List<Integer> lists = new ArrayList<>();
        BigInteger bi = bigInteger;
        BigInteger div = BigInteger.valueOf(256);
        while (bi.compareTo(BigInteger.ZERO) > 0) {
            lists.add(bi.mod(div).intValue());
            bi = bi.divide(div);
        }
        byte[] res = new byte[lists.size()];
        for (int i = lists.size() - 1; i >= 0; --i) {
            res[lists.size() - i - 1] = ((byte) (int) (lists.get(i)));
        }
        return res;
    }

    /**
     * Convert Integer to byte[].
     *
     * @param num the Integer object.
     * @return the byte[] object converted from Integer.
     */
    public static byte[] integer2byteArray(Integer num) {
        List<Integer> lists = new ArrayList<>();
        Integer bi = num;
        Integer div = 256;
        while (bi > 0) {
            lists.add(bi % div);
            bi = bi / div;
        }
        byte[] res = new byte[lists.size()];
        for (int i = lists.size() - 1; i >= 0; --i) {
            res[lists.size() - i - 1] = ((byte) (int) (lists.get(i)));
        }
        return res;
    }

    /**
     * Convert byte[] to Integer.
     *
     * @param bytes the byte[] object.
     * @return the Integer object converted from byte[].
     */
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