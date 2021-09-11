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

import com.mindspore.flclient.Common;

import java.math.BigInteger;
import java.util.Random;
import java.util.logging.Logger;

/**
 * Define functions that for splitting secret and combining secret shards.
 *
 * @since 2021-06-30
 */
public class ShareSecrets {
    private static final Logger LOGGER = Logger.getLogger(ShareSecrets.class.toString());

    private BigInteger prime;
    private final int minNum;
    private final int totalNum;
    private final Random random;

    /**
     * Defines the constructor of the class ShareSecrets.
     *
     * @param minNum   minimum number of fragments required to reconstruct a secret.
     * @param totalNum total clients number.
     */
    public ShareSecrets(final int minNum, final int totalNum) {
        if (minNum <= 0) {
            LOGGER.severe(Common.addTag("the argument <k> is not valid: <= 0, it should be > 0"));
            throw new IllegalArgumentException();
        }
        if (totalNum <= 0) {
            LOGGER.severe(Common.addTag("the argument <n> is not valid: <= 0, it should be > 0"));
            throw new IllegalArgumentException();
        }
        if (minNum > totalNum) {
            LOGGER.severe(Common.addTag("the argument <k, n> is not valid: k > n, it should k <= n"));
            throw new IllegalArgumentException();
        }
        this.minNum = minNum;
        this.totalNum = totalNum;
        random = Common.getSecureRandom();
    }

    /**
     * Splits a secret into a specified number of secret fragments.
     *
     * @param bytes     the secret need to be split.
     * @param primeByte teh big prime number used to combine secret fragments.
     * @return the secret fragments.
     */
    public SecretShares[] split(final byte[] bytes, byte[] primeByte) {
        if (bytes == null || bytes.length == 0) {
            LOGGER.severe(Common.addTag("the input argument <bytes> is null"));
            return new SecretShares[0];
        }
        if (primeByte == null || primeByte.length == 0) {
            LOGGER.severe(Common.addTag("the input argument <primeByte> is null"));
            return new SecretShares[0];
        }
        BigInteger secret = BaseUtil.byteArray2BigInteger(bytes);
        final int modLength = secret.bitLength() + 1;
        prime = BaseUtil.byteArray2BigInteger(primeByte);
        final BigInteger[] coefficient = new BigInteger[minNum - 1];

        for (int i = 0; i < minNum - 1; i++) {
            coefficient[i] = randomZp(prime);
        }

        final SecretShares[] shares = new SecretShares[totalNum];
        for (int i = 1; i <= totalNum; i++) {
            BigInteger accumulate = secret;

            for (int j = 1; j < minNum; j++) {
                final BigInteger b1 = BigInteger.valueOf(i).modPow(BigInteger.valueOf(j), prime);
                final BigInteger b2 = coefficient[j - 1].multiply(b1).mod(prime);

                accumulate = accumulate.add(b2).mod(prime);
            }
            shares[i - 1] = new SecretShares(i, accumulate);
        }
        return shares;
    }

    /**
     * Combine secret fragments.
     *
     * @param shares    the secret fragments.
     * @param primeByte teh big prime number used to combine secret fragments.
     * @return the secrets combined by secret fragments.
     */
    public BigInteger combine(final SecretShares[] shares, final byte[] primeByte) {
        if (shares == null || shares.length == 0) {
            LOGGER.severe(Common.addTag("the input argument <shares> is null"));
            return BigInteger.ZERO;
        }
        if (primeByte == null || primeByte.length == 0) {
            LOGGER.severe(Common.addTag("the input argument <primeByte> is null"));
            return BigInteger.ZERO;
        }
        BigInteger primeNum = BaseUtil.byteArray2BigInteger(primeByte);
        BigInteger accumulate = BigInteger.ZERO;
        for (int j = 0; j < minNum; j++) {
            BigInteger num = BigInteger.ONE;
            BigInteger den = BigInteger.ONE;

            BigInteger tmp;

            for (int m = 0; m < minNum; m++) {
                if (j != m) {
                    num = num.multiply(BigInteger.valueOf(shares[m].getNumber())).mod(primeNum);
                    tmp = BigInteger.valueOf(shares[j].getNumber()).multiply(BigInteger.valueOf(-1));
                    tmp = BigInteger.valueOf(shares[m].getNumber()).add(tmp).mod(primeNum);
                    den = den.multiply(tmp).mod(primeNum);
                }
            }
            final BigInteger value = shares[j].getShares();

            tmp = den.modInverse(primeNum);
            tmp = tmp.multiply(num).mod(primeNum);
            tmp = tmp.multiply(value).mod(primeNum);
            accumulate = accumulate.add(tmp).mod(primeNum);
        }
        return accumulate;
    }

    private BigInteger randomZp(final BigInteger num) {
        while (true) {
            final BigInteger rand = new BigInteger(num.bitLength(), random);
            if (rand.compareTo(BigInteger.ZERO) > 0 && rand.compareTo(num) < 0) {
                return rand;
            }
        }
    }

    /**
     * Define the structure for store secret fragments.
     */
    public final class SecretShares {
        private final int number;
        private final BigInteger share;

        public SecretShares(final int number, final BigInteger share) {
            this.number = number;
            this.share = share;
        }

        public int getNumber() {
            return number;
        }

        public BigInteger getShares() {
            return share;
        }

        @Override
        public String toString() {
            return "SecretShares [number=" + number + ", share=" + share + "]";
        }
    }
}

