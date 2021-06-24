/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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


public class ShareSecrets {
    private static final Logger LOGGER = Logger.getLogger(ShareSecrets.class.toString());

    public final class SecretShare {
        public SecretShare(final int num, final BigInteger share) {
            this.num = num;
            this.share = share;
        }

        public int getNum() {
            return num;
        }

        public BigInteger getShare() {
            return share;
        }

        @Override
        public String toString() {
            return "SecretShare [num=" + num + ", share=" + share + "]";
        }

        private final int num;
        private final BigInteger share;
    }

    public ShareSecrets(final int k, final int n) {
        this.k = k;
        this.n = n;

        random = new Random();
    }

    public SecretShare[] split(final byte[] bytes, byte[] primeByte) {
        BigInteger secret = BaseUtil.byteArray2BigInteger(bytes);
        final int modLength = secret.bitLength() + 1;
        prime = BaseUtil.byteArray2BigInteger(primeByte);
        final BigInteger[] coeff = new BigInteger[k - 1];

        LOGGER.info(Common.addTag("Prime Number: " + prime));

        for (int i = 0; i < k - 1; i++) {
            coeff[i] = randomZp(prime);
            LOGGER.info(Common.addTag("a" + (i + 1) + ": " + coeff[i]));
        }

        final SecretShare[] shares = new SecretShare[n];
        for (int i = 1; i <= n; i++) {
            BigInteger accum = secret;

            for (int j = 1; j < k; j++) {
                final BigInteger t1 = BigInteger.valueOf(i).modPow(BigInteger.valueOf(j), prime);
                final BigInteger t2 = coeff[j - 1].multiply(t1).mod(prime);

                accum = accum.add(t2).mod(prime);
            }
            shares[i - 1] = new SecretShare(i, accum);
            LOGGER.info(Common.addTag("Share " + shares[i - 1]));
        }
        return shares;
    }

    public BigInteger getPrime() {
        return prime;
    }

    public BigInteger combine(final SecretShare[] shares, final byte[] primeByte) {
        BigInteger primeNum = BaseUtil.byteArray2BigInteger(primeByte);
        BigInteger accum = BigInteger.ZERO;
        for (int j = 0; j < k; j++) {
            BigInteger num = BigInteger.ONE;
            BigInteger den = BigInteger.ONE;

            BigInteger tmp;

            for (int m = 0; m < k; m++) {
                if (j != m) {
                    num = num.multiply(BigInteger.valueOf(shares[m].getNum())).mod(primeNum);
                    tmp = BigInteger.valueOf(shares[j].getNum()).multiply(BigInteger.valueOf(-1));
                    tmp = BigInteger.valueOf(shares[m].getNum()).add(tmp).mod(primeNum);
                    den = den.multiply(tmp).mod(primeNum);
                }
            }
            final BigInteger value = shares[j].getShare();

            tmp = den.modInverse(primeNum);
            tmp = tmp.multiply(num).mod(primeNum);
            tmp = tmp.multiply(value).mod(primeNum);
            accum = accum.add(tmp).mod(primeNum);
            LOGGER.info(Common.addTag("value: " + value + ", tmp: " + tmp + ", accum: " + accum));
        }
        LOGGER.info(Common.addTag("The secret is: " + accum));
        return accum;
    }

    private BigInteger randomZp(final BigInteger p) {
        while (true) {
            final BigInteger r = new BigInteger(p.bitLength(), random);
            if (r.compareTo(BigInteger.ZERO) > 0 && r.compareTo(p) < 0) {
                return r;
            }
        }
    }

    private BigInteger prime;
    private final int k;
    private final int n;
    private final Random random;
    private final int SECRET_MAX_LEN = 32;
}

