#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <Windows.h>
#include <ctime>
#define BLOCKS				2048
//#define BLOCKS              8192
//#define BLOCKS				64
#define THREADS				1024

__int64 trial = 1, keys = 10;   double PCFreq = 0.0;    __int64 CounterStart = 0;
#define bit8 unsigned char
#define bit16 unsigned short int
#define bit32 unsigned int
#define bit64 unsigned __int64 
bit8 S7[128] = {   54, 50, 62, 56, 22, 34, 94, 96, 38,  6, 63, 93,  2, 18,123, 33,   55,113, 39,114, 21, 67, 65, 12, 47, 73, 46, 27, 25,111,124, 81,   53,  9,121, 79, 52, 60, 58, 48,101,127, 40,120,104, 70, 71, 43,   20,122, 72, 61, 23,109, 13,100, 77,  1, 16,  7, 82, 10,105, 98,  117,116, 76, 11, 89,106,  0,125,118, 99, 86, 69, 30, 57,126, 87,  112, 51, 17,  5, 95, 14, 90, 84, 91,  8, 35,103, 32, 97, 28, 66,  102, 31, 26, 45, 75,  4, 85, 92, 37, 74, 80, 49, 68, 29,115, 44,   64,107,108, 24,110, 83, 36, 78, 42, 19, 15, 41, 88,119, 59,  3};
bit16 S9[512] = {  167,239,161,379,391,334,  9,338, 38,226, 48,358,452,385, 90,397,  183,253,147,331,415,340, 51,362,306,500,262, 82,216,159,356,177,  175,241,489, 37,206, 17,  0,333, 44,254,378, 58,143,220, 81,400,   95,  3,315,245, 54,235,218,405,472,264,172,494,371,290,399, 76,  165,197,395,121,257,480,423,212,240, 28,462,176,406,507,288,223,  501,407,249,265, 89,186,221,428,164, 74,440,196,458,421,350,163,  232,158,134,354, 13,250,491,142,191, 69,193,425,152,227,366,135,  344,300,276,242,437,320,113,278, 11,243, 87,317, 36, 93,496, 27,  487,446,482, 41, 68,156,457,131,326,403,339, 20, 39,115,442,124,  475,384,508, 53,112,170,479,151,126,169, 73,268,279,321,168,364,  363,292, 46,499,393,327,324, 24,456,267,157,460,488,426,309,229,  439,506,208,271,349,401,434,236, 16,209,359, 52, 56,120,199,277,  465,416,252,287,246,  6, 83,305,420,345,153,502, 65, 61,244,282,  173,222,418, 67,386,368,261,101,476,291,195,430, 49, 79,166,330,  280,383,373,128,382,408,155,495,367,388,274,107,459,417, 62,454,  132,225,203,316,234, 14,301, 91,503,286,424,211,347,307,140,374,   35,103,125,427, 19,214,453,146,498,314,444,230,256,329,198,285,   50,116, 78,410, 10,205,510,171,231, 45,139,467, 29, 86,505, 32,   72, 26,342,150,313,490,431,238,411,325,149,473, 40,119,174,355, 185,233,389, 71,448,273,372, 55,110,178,322, 12,469,392,369,190,   1,109,375,137,181, 88, 75,308,260,484, 98,272,370,275,412,111, 336,318,  4,504,492,259,304, 77,337,435, 21,357,303,332,483, 18,  47, 85, 25,497,474,289,100,269,296,478,270,106, 31,104,433, 84,  414,486,394, 96, 99,154,511,148,413,361,409,255,162,215,302,201,  266,351,343,144,441,365,108,298,251, 34,182,509,138,210,335,133,  311,352,328,141,396,346,123,319,450,281,429,228,443,481, 92,404,  485,422,248,297, 23,213,130,466, 22,217,283, 70,294,360,419,127,  312,377,  7,468,194,  2,117,295,463,258,224,447,247,187, 80,398,  284,353,105,390,299,471,470,184, 57,200,348, 63,204,188, 33,451,   97, 30,310,219, 94,160,129,493, 64,179,263,102,189,207,114,402,  438,477,387,122,192, 42,381,  5,145,118,180,449,293,323,136,380,   43, 66, 60,455,341,445,202,432,  8,237, 15,376,436,464, 59,461};
__device__ bit32 arithmeticRightShift(bit32 x, bit32 n) { return (x >> n) | (x << (-n & 31)); }
bit16 LeftShift(bit16 x, bit16 n) { return (x << n) | (x >> (-n & 15)); }
__device__ bit16 LeftShiftd(bit16 x, bit16 n) { return (x << n) | (x >> (-n & 15)); }
__shared__ bit8 S7S[128];
__shared__ bit16 S9S[512];
__shared__ bit16 constants[8];
bit16 constant[8] ={ 0x0123, 0x4567, 0x89AB,0xCDEF, 0xFEDC, 0xBA98, 0x7654, 0x3210};
//__shared__ bit16 S9S2[256][32][2];

bit16 FI(bit16 input, bit16 roundkey) {
    bit16 left, right, round_key_1, round_key_2, tmp_l, tmp_r;
    left = input >> 7;
    right = input & 0b1111111;
    round_key_1 = roundkey >> 9;
    round_key_2 = roundkey & 0b111111111;
    tmp_l = right;
    tmp_r = S9[left] ^ right;
    left = tmp_r ^ round_key_2;
    right = S7[tmp_l] ^ (tmp_r & 0b1111111) ^ round_key_1;
    tmp_l = right;
    tmp_r = S9[left] ^ right;
    left = S7[tmp_l] ^ (tmp_r & 0b1111111);
    right = tmp_r;
    return (left << 9) | right;
}
__device__ bit16 FId(bit16 input, bit16 roundkey) {
    bit16 left, right, round_key_1, round_key_2, tmp_l, tmp_r;
    left = input >> 7;
    right = input & 0b1111111;
    round_key_1 = roundkey >> 9;
    round_key_2 = roundkey & 0b111111111;
    tmp_l = right;
    tmp_r = S9S[left] ^ right;
    left = tmp_r ^ round_key_2;
    right = S7S[tmp_l] ^ (tmp_r & 0b1111111) ^ round_key_1;
    tmp_l = right;
    tmp_r = S9S[left] ^ right;
    left = S7S[tmp_l] ^ (tmp_r & 0b1111111);
/*    right = tmp_r;
    return (left << 9) | right;*/
    return (left << 9) | tmp_r;
}
/*__device__ bit16 FId(bit16 input, bit16 roundkey) {
    bit16 left, right, round_key_1, round_key_2, tmp_l, tmp_r;
    int warpThreadIndex = threadIdx.x & 31;
    left = input >> 7;
    right = input & 0b1111111;
    round_key_1 = roundkey >> 9;
    round_key_2 = roundkey & 0b111111111;
    tmp_l = right;
    tmp_r = S9S2[left/2][warpThreadIndex][left%2] ^ right;
    left = tmp_r ^ round_key_2;
    right = S7S[tmp_l] ^ (tmp_r & 0b1111111) ^ round_key_1;
    tmp_l = right;
    tmp_r = S9S2[left/2][warpThreadIndex][left%2] ^ right;
    left = S7S[tmp_l] ^ (tmp_r & 0b1111111);
    right = tmp_r;
    return (left << 9) | right;
}*/
bit32 FO(bit32 input, bit16 KO1, bit16 KO2, bit16 KO3, bit16 KI1, bit16 KI2, bit16 KI3) {
    bit16 in_left, in_right, out_left, out_right;
    in_left = input >> 16;
    in_right = input & 0xFFFF;
    out_left = in_right;
    out_right = FI(in_left ^ KO1, KI1) ^ in_right;
    in_left = out_right;
    in_right = FI(out_left ^ KO2, KI2) ^ out_right;
    out_left = in_right;
    out_right = FI(in_left ^ KO3, KI3) ^ in_right;
    return (out_left << 16) | out_right;
}
__device__ bit32 FOd(bit32 input, bit16 KO1, bit16 KO2, bit16 KO3, bit16 KI1, bit16 KI2, bit16 KI3) {
    bit16 in_left, in_right, out_left, out_right;
    in_left = input >> 16;
    in_right = input & 0xFFFF;
    out_left = in_right;
    out_right = FId(in_left ^ KO1, KI1) ^ in_right;
    in_left = out_right;
    in_right = FId(out_left ^ KO2, KI2) ^ out_right;
    out_left = in_right;
    out_right = FId(in_left ^ KO3, KI3) ^ in_right;
    return (out_left << 16) | out_right;
}
bit32 FL(bit32 input, bit16 KL1, bit16 KL2) {
    bit32 in_left, in_right, out_right, out_left;
    in_left = input >> 16;
    in_right = input & 0xFFFF;
    out_right = in_right ^ LeftShift(in_left & KL1, 1);
    out_left = in_left ^ LeftShift(out_right | KL2, 1);
    return (out_left << 16) | out_right;
}
__device__ bit32 FLd(bit32 input, bit16 KL1, bit16 KL2) {
    bit16 in_left, in_right, out_right, out_left;
    in_left = input >> 16;
    in_right = input & 0xFFFF;
    out_right = in_right ^ LeftShiftd(in_left & KL1, 1);
    out_left = in_left ^ LeftShiftd(out_right | KL2, 1);
    return (out_left << 16) | out_right;
}
void encryption(bit32 left, bit32 right, bit32 cipher_left, bit32 cipher_right) {
    bit32 in_left = left, in_right = right, temp;
    bit16 k1= 0, k2=0, k3=0, k4=0, k5=0, k6=0, k7=0, k8=0;
    bit16 KL1, KL2, KO1, KO2, KO3, KI1, KI2, KI3;
    //    C1 0x0123
    //    C2 0x4567
    //    C3 0x89AB
    //    C4 0xCDEF
    //    C5 0xFEDC
    //    C6 0xBA98
    //    C7 0x7654
    //    C8 0x3210
    // Round 1
    KL1 = LeftShift(k1, 1);    KL2 = k3 ^ 0x89AB;
    KO1 = LeftShift(k2, 5);    KO2 = LeftShift(k6, 8);    KO3 = LeftShift(k7, 13);
    KI1 = k5 ^ 0xFEDC;    KI2 = k4 ^ 0xCDEF;    KI3 = k8 ^ 0x3210;
    temp = FL(in_left, KL1, KL2);
    temp = FO(temp, KO1, KO2, KO3, KI1, KI2, KI3);
    temp ^= in_right;    in_right = in_left;    in_left = temp;
    // Round 2
    KL1 = LeftShift(k2, 1);    KL2 = k4 ^ 0xCDEF;
    KO1 = LeftShift(k3, 5);    KO2 = LeftShift(k7, 8);    KO3 = LeftShift(k8, 13);
    KI1 = k6 ^ 0xBA98;    KI2 = k5 ^ 0xFEDC;    KI3 = k1 ^ 0x0123;
    temp = FO(in_left, KO1, KO2, KO3, KI1, KI2, KI3);
    temp = FL(temp, KL1, KL2);    
    temp ^= in_right;    in_right = in_left;    in_left = temp;
    // Round 3
    KL1 = LeftShift(k3, 1);    KL2 = k5 ^ 0xFEDC;
    KO1 = LeftShift(k4, 5);    KO2 = LeftShift(k8, 8);    KO3 = LeftShift(k1, 13);
    KI1 = k7 ^ 0x7654;    KI2 = k6 ^ 0xBA98;    KI3 = k2 ^ 0x4567;
    temp = FL(in_left, KL1, KL2);
    temp = FO(temp, KO1, KO2, KO3, KI1, KI2, KI3);
    temp ^= in_right;    in_right = in_left;    in_left = temp;
    // Round 4
    KL1 = LeftShift(k4, 1);    KL2 = k6 ^ 0xBA98;
    KO1 = LeftShift(k5, 5);    KO2 = LeftShift(k1, 8);    KO3 = LeftShift(k2, 13);
    KI1 = k8 ^ 0x3210;    KI2 = k7 ^ 0x7654;    KI3 = k3 ^ 0x89AB;
    temp = FO(in_left, KO1, KO2, KO3, KI1, KI2, KI3);
    temp = FL(temp, KL1, KL2);
    temp ^= in_right;    in_right = in_left;    in_left = temp;
    // Round 5
    KL1 = LeftShift(k5, 1);    KL2 = k7 ^ 0x7654;
    KO1 = LeftShift(k6, 5);    KO2 = LeftShift(k2, 8);    KO3 = LeftShift(k3, 13);
    KI1 = k1 ^ 0x0123;    KI2 = k8 ^ 0x3210;    KI3 = k4 ^ 0xCDEF;
    temp = FL(in_left, KL1, KL2);
    temp = FO(temp, KO1, KO2, KO3, KI1, KI2, KI3);
    temp ^= in_right;    in_right = in_left;    in_left = temp;
    // Round 6
    KL1 = LeftShift(k6, 1);    KL2 = k8 ^ 0x3210;
    KO1 = LeftShift(k7, 5);    KO2 = LeftShift(k3, 8);    KO3 = LeftShift(k4, 13);
    KI1 = k2 ^ 0x4567;    KI2 = k1 ^ 0x0123;    KI3 = k5 ^ 0xFEDC;
    temp = FO(in_left, KO1, KO2, KO3, KI1, KI2, KI3);
    temp = FL(temp, KL1, KL2);
    temp ^= in_right;    in_right = in_left;    in_left = temp;
    // Round 7
    KL1 = LeftShift(k7, 1);    KL2 = k1 ^ 0x0123;
    KO1 = LeftShift(k8, 5);    KO2 = LeftShift(k4, 8);    KO3 = LeftShift(k5, 13);
    KI1 = k3 ^ 0x89AB;    KI2 = k2 ^ 0x4567;    KI3 = k6 ^ 0xBA98;
    temp = FL(in_left, KL1, KL2);
    temp = FO(temp, KO1, KO2, KO3, KI1, KI2, KI3);
    temp ^= in_right;    in_right = in_left;    in_left = temp;
    // Round 8
    KL1 = LeftShift(k8, 1);    KL2 = k2 ^ 0x4567;
    KO1 = LeftShift(k1, 5);    KO2 = LeftShift(k5, 8);    KO3 = LeftShift(k6, 13);
    KI1 = k4 ^ 0xCDEF;    KI2 = k3 ^ 0x89AB;    KI3 = k7 ^ 0x7654;
    temp = FO(in_left, KO1, KO2, KO3, KI1, KI2, KI3);
    temp = FL(temp, KL1, KL2);
    temp ^= in_right;    in_right = in_left;    in_left = temp;
    printf("%08x%08x\n", in_left, in_right);


    if (in_left == cipher_left)
        if (in_right == cipher_right)
            printf("Can you see me?\n"); 
}
__global__ void KASUMI64Exhaustive(bit32 left, bit32 right, bit32 cipher_left, bit32 cipher_right, bit8* S7G, bit16* S9G) {
    bit32 threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
//    int warpThreadIndex = threadIdx.x & 31;
    if (threadIdx.x < 512) {
        if (threadIdx.x < 128) S7S[threadIdx.x] = S7G[threadIdx.x];
        S9S[threadIdx.x] = S9G[threadIdx.x];
    }
    __syncthreads();
    bit32 in_left, in_right, temp;
    bit16 k1 = threadIndex/65536, k2 = threadIndex % 65536, k3 = 0, k4 = 0, k5 = k1, k6 = k2, k7 = 0, k8 = 0;
    bit16 KL1, KL2, KO1, KO2, KO3, KI1, KI2, KI3;
//#pragma unroll
    for (int j = 0; j < 1; j++) {
        for (int i = 0; i < 65536; i++) {
            in_left = left; in_right = right;
            // Round 1
            KL1 = LeftShiftd(k1, 1);    KL2 = k3 ^ 0x89AB;
            KO1 = LeftShiftd(k2, 5);    KO2 = LeftShiftd(k6, 8);    KO3 = LeftShiftd(k7, 13);
            KI1 = k5 ^ 0xFEDC;    KI2 = k4 ^ 0xCDEF;    KI3 = k8 ^ 0x3210;
            temp = FLd(in_left, KL1, KL2);
            temp = FOd(temp, KO1, KO2, KO3, KI1, KI2, KI3);
            temp ^= in_right;    in_right = in_left;    in_left = temp;
            // Round 2
            KL1 = LeftShiftd(k2, 1);    KL2 = k4 ^ 0xCDEF;
            KO1 = LeftShiftd(k3, 5);    KO2 = LeftShiftd(k7, 8);    KO3 = LeftShiftd(k8, 13);
            KI1 = k6 ^ 0xBA98;    KI2 = k5 ^ 0xFEDC;    KI3 = k1 ^ 0x0123;
            temp = FOd(in_left, KO1, KO2, KO3, KI1, KI2, KI3);
            temp = FLd(temp, KL1, KL2);
            temp ^= in_right;    in_right = in_left;    in_left = temp;
            // Round 3
            KL1 = LeftShiftd(k3, 1);    KL2 = k5 ^ 0xFEDC;
            KO1 = LeftShiftd(k4, 5);    KO2 = LeftShiftd(k8, 8);    KO3 = LeftShiftd(k1, 13);
            KI1 = k7 ^ 0x7654;    KI2 = k6 ^ 0xBA98;    KI3 = k2 ^ 0x4567;
            temp = FLd(in_left, KL1, KL2);
            temp = FOd(temp, KO1, KO2, KO3, KI1, KI2, KI3);
            temp ^= in_right;    in_right = in_left;    in_left = temp;
            // Round 4
            KL1 = LeftShiftd(k4, 1);    KL2 = k6 ^ 0xBA98;
            KO1 = LeftShiftd(k5, 5);    KO2 = LeftShiftd(k1, 8);    KO3 = LeftShiftd(k2, 13);
            KI1 = k8 ^ 0x3210;    KI2 = k7 ^ 0x7654;    KI3 = k3 ^ 0x89AB;
            temp = FOd(in_left, KO1, KO2, KO3, KI1, KI2, KI3);
            temp = FLd(temp, KL1, KL2);
            temp ^= in_right;    in_right = in_left;    in_left = temp;
            // Round 5
            KL1 = LeftShiftd(k5, 1);    KL2 = k7 ^ 0x7654;
            KO1 = LeftShiftd(k6, 5);    KO2 = LeftShiftd(k2, 8);    KO3 = LeftShiftd(k3, 13);
            KI1 = k1 ^ 0x0123;    KI2 = k8 ^ 0x3210;    KI3 = k4 ^ 0xCDEF;
            temp = FLd(in_left, KL1, KL2);
            temp = FOd(temp, KO1, KO2, KO3, KI1, KI2, KI3);
            temp ^= in_right;    in_right = in_left;    in_left = temp;
            // Round 6
            KL1 = LeftShiftd(k6, 1);    KL2 = k8 ^ 0x3210;
            KO1 = LeftShiftd(k7, 5);    KO2 = LeftShiftd(k3, 8);    KO3 = LeftShiftd(k4, 13);
            KI1 = k2 ^ 0x4567;    KI2 = k1 ^ 0x0123;    KI3 = k5 ^ 0xFEDC;
            temp = FOd(in_left, KO1, KO2, KO3, KI1, KI2, KI3);
            temp = FLd(temp, KL1, KL2);
            temp ^= in_right;    in_right = in_left;    in_left = temp;
            // Round 7
            KL1 = LeftShiftd(k7, 1);    KL2 = k1 ^ 0x0123;
            KO1 = LeftShiftd(k8, 5);    KO2 = LeftShiftd(k4, 8);    KO3 = LeftShiftd(k5, 13);
            KI1 = k3 ^ 0x89AB;    KI2 = k2 ^ 0x4567;    KI3 = k6 ^ 0xBA98;
            temp = FLd(in_left, KL1, KL2);
            temp = FOd(temp, KO1, KO2, KO3, KI1, KI2, KI3);
            temp ^= in_right;
            if (temp == cipher_right) {
                in_right = in_left;    in_left = temp;
                // Round 8
                KL1 = LeftShiftd(k8, 1);    KL2 = k2 ^ 0x4567;
                KO1 = LeftShiftd(k1, 5);    KO2 = LeftShiftd(k5, 8);    KO3 = LeftShiftd(k6, 13);
                KI1 = k4 ^ 0xCDEF;    KI2 = k3 ^ 0x89AB;    KI3 = k7 ^ 0x7654;
                temp = FOd(in_left, KO1, KO2, KO3, KI1, KI2, KI3);
                temp = FLd(temp, KL1, KL2);
                temp ^= in_right; //   in_right = in_left;    in_left = temp;
                if (temp == cipher_left) printf("The secret key is %08x%08x\n", threadIndex, i);
            }
            k8++; k4 = k8;
        }
        k7++; k3 = k7;
        /*       KL1 = LeftShiftd(k8, 1);    KL2 = k2 ^ 0x4567;
        KO1 = LeftShiftd(k1, 5);    KO2 = LeftShiftd(k5, 8);    KO3 = LeftShiftd(k6, 13);
        KI1 = k4 ^ 0xCDEF;    KI2 = k3 ^ 0x89AB;    KI3 = k7 ^ 0x7654;
        temp = FOd(in_left, KO1, KO2, KO3, KI1, KI2, KI3);
        temp = FLd(temp, KL1, KL2);
        temp ^= in_right; //   in_right = in_left;    in_left = temp;
        //printf("%08x%08x\n", in_left, in_right);

        if (temp == cipher_left)
            if (in_left == cipher_right)
                printf("The secret key is %08x%08x\n", threadIndex, i);
        k8++; k4 = k8;*/
    }
}
__global__ void KASUMI64EncryptionTMTO(bit32 left, bit32 right, bit32 cipher_left, bit32 cipher_right, bit8* S7G, bit16* S9G) {
    bit32 threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    //    int warpThreadIndex = threadIdx.x & 31;
    if (threadIdx.x < 512) {
        if (threadIdx.x < 128) S7S[threadIdx.x] = S7G[threadIdx.x];
        S9S[threadIdx.x] = S9G[threadIdx.x];
    }
    __syncthreads();
    bit32 in_left, in_right, temp;
    bit16 k1 = threadIndex / 65536, k2 = threadIndex % 65536, k3 = 0, k4 = 0, k5 = k1, k6 = k2, k7 = 0, k8 = 0;
    bit16 KL1, KL2, KO1, KO2, KO3, KI1, KI2, KI3;
    //#pragma unroll
    for (int j = 0; j < 1; j++) {
        for (int i = 0; i < 65536; i++) {
            in_left = left; in_right = right;
            // Round 1
            KL1 = LeftShiftd(k1, 1);    KL2 = k3 ^ 0x89AB;
            KO1 = LeftShiftd(k2, 5);    KO2 = LeftShiftd(k6, 8);    KO3 = LeftShiftd(k7, 13);
            KI1 = k5 ^ 0xFEDC;    KI2 = k4 ^ 0xCDEF;    KI3 = k8 ^ 0x3210;
            temp = FLd(in_left, KL1, KL2);
            temp = FOd(temp, KO1, KO2, KO3, KI1, KI2, KI3);
            temp ^= in_right;    in_right = in_left;    in_left = temp;
            // Round 2
            KL1 = LeftShiftd(k2, 1);    KL2 = k4 ^ 0xCDEF;
            KO1 = LeftShiftd(k3, 5);    KO2 = LeftShiftd(k7, 8);    KO3 = LeftShiftd(k8, 13);
            KI1 = k6 ^ 0xBA98;    KI2 = k5 ^ 0xFEDC;    KI3 = k1 ^ 0x0123;
            temp = FOd(in_left, KO1, KO2, KO3, KI1, KI2, KI3);
            temp = FLd(temp, KL1, KL2);
            temp ^= in_right;    in_right = in_left;    in_left = temp;
            // Round 3
            KL1 = LeftShiftd(k3, 1);    KL2 = k5 ^ 0xFEDC;
            KO1 = LeftShiftd(k4, 5);    KO2 = LeftShiftd(k8, 8);    KO3 = LeftShiftd(k1, 13);
            KI1 = k7 ^ 0x7654;    KI2 = k6 ^ 0xBA98;    KI3 = k2 ^ 0x4567;
            temp = FLd(in_left, KL1, KL2);
            temp = FOd(temp, KO1, KO2, KO3, KI1, KI2, KI3);
            temp ^= in_right;    in_right = in_left;    in_left = temp;
            // Round 4
            KL1 = LeftShiftd(k4, 1);    KL2 = k6 ^ 0xBA98;
            KO1 = LeftShiftd(k5, 5);    KO2 = LeftShiftd(k1, 8);    KO3 = LeftShiftd(k2, 13);
            KI1 = k8 ^ 0x3210;    KI2 = k7 ^ 0x7654;    KI3 = k3 ^ 0x89AB;
            temp = FOd(in_left, KO1, KO2, KO3, KI1, KI2, KI3);
            temp = FLd(temp, KL1, KL2);
            temp ^= in_right;    in_right = in_left;    in_left = temp;
            // Round 5
            KL1 = LeftShiftd(k5, 1);    KL2 = k7 ^ 0x7654;
            KO1 = LeftShiftd(k6, 5);    KO2 = LeftShiftd(k2, 8);    KO3 = LeftShiftd(k3, 13);
            KI1 = k1 ^ 0x0123;    KI2 = k8 ^ 0x3210;    KI3 = k4 ^ 0xCDEF;
            temp = FLd(in_left, KL1, KL2);
            temp = FOd(temp, KO1, KO2, KO3, KI1, KI2, KI3);
            temp ^= in_right;    in_right = in_left;    in_left = temp;
            // Round 6
            KL1 = LeftShiftd(k6, 1);    KL2 = k8 ^ 0x3210;
            KO1 = LeftShiftd(k7, 5);    KO2 = LeftShiftd(k3, 8);    KO3 = LeftShiftd(k4, 13);
            KI1 = k2 ^ 0x4567;    KI2 = k1 ^ 0x0123;    KI3 = k5 ^ 0xFEDC;
            temp = FOd(in_left, KO1, KO2, KO3, KI1, KI2, KI3);
            temp = FLd(temp, KL1, KL2);
            temp ^= in_right;    in_right = in_left;    in_left = temp;
            // Round 7
            KL1 = LeftShiftd(k7, 1);    KL2 = k1 ^ 0x0123;
            KO1 = LeftShiftd(k8, 5);    KO2 = LeftShiftd(k4, 8);    KO3 = LeftShiftd(k5, 13);
            KI1 = k3 ^ 0x89AB;    KI2 = k2 ^ 0x4567;    KI3 = k6 ^ 0xBA98;
            temp = FLd(in_left, KL1, KL2);
            temp = FOd(temp, KO1, KO2, KO3, KI1, KI2, KI3);
            temp ^= in_right;
 
                in_right = in_left;    in_left = temp;
                // Round 8
                KL1 = LeftShiftd(k8, 1);    KL2 = k2 ^ 0x4567;
                KO1 = LeftShiftd(k1, 5);    KO2 = LeftShiftd(k5, 8);    KO3 = LeftShiftd(k6, 13);
                KI1 = k4 ^ 0xCDEF;    KI2 = k3 ^ 0x89AB;    KI3 = k7 ^ 0x7654;
                temp = FOd(in_left, KO1, KO2, KO3, KI1, KI2, KI3);
                temp = FLd(temp, KL1, KL2);
                temp ^= in_right; //   in_right = in_left;    in_left = temp;
                if (temp == cipher_left && in_left ==cipher_right) printf("The secret key is %08x%08x\n", threadIndex, i);
            
            k8++; k4 = k8;
        }
        k7++; k3 = k7;
        /*       KL1 = LeftShiftd(k8, 1);    KL2 = k2 ^ 0x4567;
        KO1 = LeftShiftd(k1, 5);    KO2 = LeftShiftd(k5, 8);    KO3 = LeftShiftd(k6, 13);
        KI1 = k4 ^ 0xCDEF;    KI2 = k3 ^ 0x89AB;    KI3 = k7 ^ 0x7654;
        temp = FOd(in_left, KO1, KO2, KO3, KI1, KI2, KI3);
        temp = FLd(temp, KL1, KL2);
        temp ^= in_right; //   in_right = in_left;    in_left = temp;
        //printf("%08x%08x\n", in_left, in_right);

        if (temp == cipher_left)
            if (in_left == cipher_right)
                printf("The secret key is %08x%08x\n", threadIndex, i);
        k8++; k4 = k8;*/
    }
}
__global__ void KASUMI64ExhaustiveConstantsRegister(bit32 left, bit32 right, bit32 cipher_left, bit32 cipher_right, bit8* S7G, bit16* S9G) {
    bit32 threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    //    int warpThreadIndex = threadIdx.x & 31;
    if (threadIdx.x < 512) {
        if (threadIdx.x < 128) S7S[threadIdx.x] = S7G[threadIdx.x];
        S9S[threadIdx.x] = S9G[threadIdx.x];
    }
    __syncthreads();
    bit32 in_left, in_right, temp;
    bit16 k1 = threadIndex / 65536, k2 = threadIndex % 65536, k3 = 0, k4 = 0, k5 = k1, k6 = k2, k7 = 0, k8 = 0;
    bit16 KL1, KL2, KO1, KO2, KO3, KI1, KI2, KI3;
    bit16 c1 = 0x0123, c2 = 0x4567, c3 = 0x89AB, c4 = 0xCDEF, c5 = 0xFEDC, c6 = 0xBA98, c7 = 0x7654, c8 = 0x3210;

#pragma unroll
    for (int j = 0; j < 16; j++) {
    for (int i = 0; i < 65536; i++) {
        in_left = left; in_right = right;
        // Round 1
        KL1 = LeftShiftd(k1, 1);    KL2 = k3 ^ c3;
        KO1 = LeftShiftd(k2, 5);    KO2 = LeftShiftd(k6, 8);    KO3 = LeftShiftd(k7, 13);
        KI1 = k5 ^ c5;    KI2 = k4 ^ c4;    KI3 = k8 ^ c8;
        temp = FLd(in_left, KL1, KL2);
        temp = FOd(temp, KO1, KO2, KO3, KI1, KI2, KI3);
        temp ^= in_right;    in_right = in_left;    in_left = temp;
        // Round 2
        KL1 = LeftShiftd(k2, 1);    KL2 = k4 ^ c4;
        KO1 = LeftShiftd(k3, 5);    KO2 = LeftShiftd(k7, 8);    KO3 = LeftShiftd(k8, 13);
        KI1 = k6 ^ c6;    KI2 = k5 ^ c5;    KI3 = k1 ^ c1;
        temp = FOd(in_left, KO1, KO2, KO3, KI1, KI2, KI3);
        temp = FLd(temp, KL1, KL2);
        temp ^= in_right;    in_right = in_left;    in_left = temp;
        // Round 3
        KL1 = LeftShiftd(k3, 1);    KL2 = k5 ^ c5;
        KO1 = LeftShiftd(k4, 5);    KO2 = LeftShiftd(k8, 8);    KO3 = LeftShiftd(k1, 13);
        KI1 = k7 ^ c7;    KI2 = k6 ^ c6;    KI3 = k2 ^ c2;
        temp = FLd(in_left, KL1, KL2);
        temp = FOd(temp, KO1, KO2, KO3, KI1, KI2, KI3);
        temp ^= in_right;    in_right = in_left;    in_left = temp;
        // Round 4
        KL1 = LeftShiftd(k4, 1);    KL2 = k6 ^ c6;
        KO1 = LeftShiftd(k5, 5);    KO2 = LeftShiftd(k1, 8);    KO3 = LeftShiftd(k2, 13);
        KI1 = k8 ^ c8;    KI2 = k7 ^ c7;    KI3 = k3 ^ c3;
        temp = FOd(in_left, KO1, KO2, KO3, KI1, KI2, KI3);
        temp = FLd(temp, KL1, KL2);
        temp ^= in_right;    in_right = in_left;    in_left = temp;
        // Round 5
        KL1 = LeftShiftd(k5, 1);    KL2 = k7 ^ c7;
        KO1 = LeftShiftd(k6, 5);    KO2 = LeftShiftd(k2, 8);    KO3 = LeftShiftd(k3, 13);
        KI1 = k1 ^ c1;    KI2 = k8 ^ c8;    KI3 = k4 ^ c4;
        temp = FLd(in_left, KL1, KL2);
        temp = FOd(temp, KO1, KO2, KO3, KI1, KI2, KI3);
        temp ^= in_right;    in_right = in_left;    in_left = temp;
        // Round 6
        KL1 = LeftShiftd(k6, 1);    KL2 = k8 ^ c8;
        KO1 = LeftShiftd(k7, 5);    KO2 = LeftShiftd(k3, 8);    KO3 = LeftShiftd(k4, 13);
        KI1 = k2 ^ c2;    KI2 = k1 ^ c1;    KI3 = k5 ^ c5;
        temp = FOd(in_left, KO1, KO2, KO3, KI1, KI2, KI3);
        temp = FLd(temp, KL1, KL2);
        temp ^= in_right;    in_right = in_left;    in_left = temp;
        // Round 7
        KL1 = LeftShiftd(k7, 1);    KL2 = k1 ^ c1;
        KO1 = LeftShiftd(k8, 5);    KO2 = LeftShiftd(k4, 8);    KO3 = LeftShiftd(k5, 13);
        KI1 = k3 ^ c3;    KI2 = k2 ^ c2;    KI3 = k6 ^ c6;
        temp = FLd(in_left, KL1, KL2);
        temp = FOd(temp, KO1, KO2, KO3, KI1, KI2, KI3);
        temp ^= in_right;
        if (temp == cipher_right) {
            in_right = in_left;    in_left = temp;
            // Round 8
            KL1 = LeftShiftd(k8, 1);    KL2 = k2 ^ c2;
            KO1 = LeftShiftd(k1, 5);    KO2 = LeftShiftd(k5, 8);    KO3 = LeftShiftd(k6, 13);
            KI1 = k4 ^ c4;    KI2 = k3 ^ c3;    KI3 = k7 ^ c7;
            temp = FOd(in_left, KO1, KO2, KO3, KI1, KI2, KI3);
            temp = FLd(temp, KL1, KL2);
            temp ^= in_right; //   in_right = in_left;    in_left = temp;
            if (temp == cipher_left) printf("The secret key is %08x%08x\n", threadIndex, i);
        }
        k8++; k4 = k8;
    }
    k7++; k3 = k7;
        /*       KL1 = LeftShiftd(k8, 1);    KL2 = k2 ^ 0x4567;
        KO1 = LeftShiftd(k1, 5);    KO2 = LeftShiftd(k5, 8);    KO3 = LeftShiftd(k6, 13);
        KI1 = k4 ^ 0xCDEF;    KI2 = k3 ^ 0x89AB;    KI3 = k7 ^ 0x7654;
        temp = FOd(in_left, KO1, KO2, KO3, KI1, KI2, KI3);
        temp = FLd(temp, KL1, KL2);
        temp ^= in_right; //   in_right = in_left;    in_left = temp;
        //printf("%08x%08x\n", in_left, in_right);

        if (temp == cipher_left)
            if (in_left == cipher_right)
                printf("The secret key is %08x%08x\n", threadIndex, i);
        k8++; k4 = k8;*/
    }
}
__global__ void KASUMI64ExhaustiveConstantsRegisterTMTO(bit32 left, bit32 right, bit32 cipher_left, bit32 cipher_right, bit8* S7G, bit16* S9G, bit32 cipherl[], bit32 cipherr[]) {
    bit32 threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    //    int warpThreadIndex = threadIdx.x & 31;
    if (threadIdx.x < 512) {
        if (threadIdx.x < 128) S7S[threadIdx.x] = S7G[threadIdx.x];
        S9S[threadIdx.x] = S9G[threadIdx.x];
    }
    __syncthreads();
    bit32 in_left, in_right, temp;
    bit16 k1 = threadIndex / 65536, k2 = threadIndex % 65536, k3 = 0, k4 = 0, k5 = k1, k6 = k2, k7 = 0, k8 = 0;
    bit16 KL1, KL2, KO1, KO2, KO3, KI1, KI2, KI3;
    bit16 c1 = 0x0123, c2 = 0x4567, c3 = 0x89AB, c4 = 0xCDEF, c5 = 0xFEDC, c6 = 0xBA98, c7 = 0x7654, c8 = 0x3210;

#pragma unroll
    for (int j = 0; j < 16; j++) {
        for (int i = 0; i < 65536; i++) {
            in_left = left; in_right = right;
            // Round 1
            KL1 = LeftShiftd(k1, 1);    KL2 = k3 ^ c3;
            KO1 = LeftShiftd(k2, 5);    KO2 = LeftShiftd(k6, 8);    KO3 = LeftShiftd(k7, 13);
            KI1 = k5 ^ c5;    KI2 = k4 ^ c4;    KI3 = k8 ^ c8;
            temp = FLd(in_left, KL1, KL2);
            temp = FOd(temp, KO1, KO2, KO3, KI1, KI2, KI3);
            temp ^= in_right;    in_right = in_left;    in_left = temp;
            // Round 2
            KL1 = LeftShiftd(k2, 1);    KL2 = k4 ^ c4;
            KO1 = LeftShiftd(k3, 5);    KO2 = LeftShiftd(k7, 8);    KO3 = LeftShiftd(k8, 13);
            KI1 = k6 ^ c6;    KI2 = k5 ^ c5;    KI3 = k1 ^ c1;
            temp = FOd(in_left, KO1, KO2, KO3, KI1, KI2, KI3);
            temp = FLd(temp, KL1, KL2);
            temp ^= in_right;    in_right = in_left;    in_left = temp;
            // Round 3
            KL1 = LeftShiftd(k3, 1);    KL2 = k5 ^ c5;
            KO1 = LeftShiftd(k4, 5);    KO2 = LeftShiftd(k8, 8);    KO3 = LeftShiftd(k1, 13);
            KI1 = k7 ^ c7;    KI2 = k6 ^ c6;    KI3 = k2 ^ c2;
            temp = FLd(in_left, KL1, KL2);
            temp = FOd(temp, KO1, KO2, KO3, KI1, KI2, KI3);
            temp ^= in_right;    in_right = in_left;    in_left = temp;
            // Round 4
            KL1 = LeftShiftd(k4, 1);    KL2 = k6 ^ c6;
            KO1 = LeftShiftd(k5, 5);    KO2 = LeftShiftd(k1, 8);    KO3 = LeftShiftd(k2, 13);
            KI1 = k8 ^ c8;    KI2 = k7 ^ c7;    KI3 = k3 ^ c3;
            temp = FOd(in_left, KO1, KO2, KO3, KI1, KI2, KI3);
            temp = FLd(temp, KL1, KL2);
            temp ^= in_right;    in_right = in_left;    in_left = temp;
            // Round 5
            KL1 = LeftShiftd(k5, 1);    KL2 = k7 ^ c7;
            KO1 = LeftShiftd(k6, 5);    KO2 = LeftShiftd(k2, 8);    KO3 = LeftShiftd(k3, 13);
            KI1 = k1 ^ c1;    KI2 = k8 ^ c8;    KI3 = k4 ^ c4;
            temp = FLd(in_left, KL1, KL2);
            temp = FOd(temp, KO1, KO2, KO3, KI1, KI2, KI3);
            temp ^= in_right;    in_right = in_left;    in_left = temp;
            // Round 6
            KL1 = LeftShiftd(k6, 1);    KL2 = k8 ^ c8;
            KO1 = LeftShiftd(k7, 5);    KO2 = LeftShiftd(k3, 8);    KO3 = LeftShiftd(k4, 13);
            KI1 = k2 ^ c2;    KI2 = k1 ^ c1;    KI3 = k5 ^ c5;
            temp = FOd(in_left, KO1, KO2, KO3, KI1, KI2, KI3);
            temp = FLd(temp, KL1, KL2);
            temp ^= in_right;    in_right = in_left;    in_left = temp;
            // Round 7
            KL1 = LeftShiftd(k7, 1);    KL2 = k1 ^ c1;
            KO1 = LeftShiftd(k8, 5);    KO2 = LeftShiftd(k4, 8);    KO3 = LeftShiftd(k5, 13);
            KI1 = k3 ^ c3;    KI2 = k2 ^ c2;    KI3 = k6 ^ c6;
            temp = FLd(in_left, KL1, KL2);
            temp = FOd(temp, KO1, KO2, KO3, KI1, KI2, KI3);
            temp ^= in_right;
            
                in_right = in_left;    in_left = temp;
                // Round 8
                KL1 = LeftShiftd(k8, 1);    KL2 = k2 ^ c2;
                KO1 = LeftShiftd(k1, 5);    KO2 = LeftShiftd(k5, 8);    KO3 = LeftShiftd(k6, 13);
                KI1 = k4 ^ c4;    KI2 = k3 ^ c3;    KI3 = k7 ^ c7;
                temp = FOd(in_left, KO1, KO2, KO3, KI1, KI2, KI3);
                temp = FLd(temp, KL1, KL2);
                temp ^= in_right; //   in_right = in_left;    in_left = temp;
 //               if (temp == cipher_left && in_left == cipher_right) printf("The secret key is %08x%08x\n", threadIndex, i);          
                cipherl[threadIndex] = temp; cipherr[threadIndex] = in_left; // In TMTO we will be writing the results to the memory, so we have this extra step
            k8++; k4 = k8;
        }
        k7++; k3 = k7;
        /*       KL1 = LeftShiftd(k8, 1);    KL2 = k2 ^ 0x4567;
        KO1 = LeftShiftd(k1, 5);    KO2 = LeftShiftd(k5, 8);    KO3 = LeftShiftd(k6, 13);
        KI1 = k4 ^ 0xCDEF;    KI2 = k3 ^ 0x89AB;    KI3 = k7 ^ 0x7654;
        temp = FOd(in_left, KO1, KO2, KO3, KI1, KI2, KI3);
        temp = FLd(temp, KL1, KL2);
        temp ^= in_right; //   in_right = in_left;    in_left = temp;
        //printf("%08x%08x\n", in_left, in_right);

        if (temp == cipher_left)
            if (in_left == cipher_right)
                printf("The secret key is %08x%08x\n", threadIndex, i);
        k8++; k4 = k8;*/
    }
}
__global__ void KASUMI64ExhaustiveConstants(bit32 left, bit32 right, bit32 cipher_left, bit32 cipher_right, bit8* S7G, bit16* S9G, bit16* constantG) {
    bit32 threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    //    int warpThreadIndex = threadIdx.x & 31;
    if (threadIdx.x < 512) {
        if (threadIdx.x < 128) S7S[threadIdx.x] = S7G[threadIdx.x];
        if (threadIdx.x < 8) constants[threadIdx.x] = constantG[threadIdx.x];
        S9S[threadIdx.x] = S9G[threadIdx.x];
    }
    __syncthreads();
    bit32 in_left = left, in_right = right, temp;
    bit16 k1 = threadIndex / 65536, k2 = threadIndex % 65536, k3 = 0, k4 = 0, k5 = threadIndex / 65536, k6 = threadIndex % 65536, k7 = 0, k8 = 0;
    bit16 KL1, KL2, KO1, KO2, KO3, KI1, KI2, KI3;

#pragma unroll
    for (int j = 0; j < 16; j++) {
    for (int i = 0; i < 65536; i++) {
        in_left = left; in_right = right;
        // Round 1
        KL1 = LeftShiftd(k1, 1);    KL2 = k3 ^ constants[2];
        KO1 = LeftShiftd(k2, 5);    KO2 = LeftShiftd(k6, 8);    KO3 = LeftShiftd(k7, 13);
        KI1 = k5 ^ constants[4];    KI2 = k4 ^ constants[3];    KI3 = k8 ^ constants[7];
        temp = FLd(in_left, KL1, KL2);
        temp = FOd(temp, KO1, KO2, KO3, KI1, KI2, KI3);
        temp ^= in_right;    in_right = in_left;    in_left = temp;
        // Round 2
        KL1 = LeftShiftd(k2, 1);    KL2 = k4 ^ constants[3];
        KO1 = LeftShiftd(k3, 5);    KO2 = LeftShiftd(k7, 8);    KO3 = LeftShiftd(k8, 13);
        KI1 = k6 ^ constants[5];    KI2 = k5 ^ constants[4];    KI3 = k1 ^ constants[0];
        temp = FOd(in_left, KO1, KO2, KO3, KI1, KI2, KI3);
        temp = FLd(temp, KL1, KL2);
        temp ^= in_right;    in_right = in_left;    in_left = temp;
        // Round 3
        KL1 = LeftShiftd(k3, 1);    KL2 = k5 ^ constants[4];
        KO1 = LeftShiftd(k4, 5);    KO2 = LeftShiftd(k8, 8);    KO3 = LeftShiftd(k1, 13);
        KI1 = k7 ^ constants[6];    KI2 = k6 ^ constants[5];    KI3 = k2 ^ constants[1];
        temp = FLd(in_left, KL1, KL2);
        temp = FOd(temp, KO1, KO2, KO3, KI1, KI2, KI3);
        temp ^= in_right;    in_right = in_left;    in_left = temp;
        // Round 4
        KL1 = LeftShiftd(k4, 1);    KL2 = k6 ^ constants[5];
        KO1 = LeftShiftd(k5, 5);    KO2 = LeftShiftd(k1, 8);    KO3 = LeftShiftd(k2, 13);
        KI1 = k8 ^ constants[7];    KI2 = k7 ^ constants[6];    KI3 = k3 ^ constants[2];
        temp = FOd(in_left, KO1, KO2, KO3, KI1, KI2, KI3);
        temp = FLd(temp, KL1, KL2);
        temp ^= in_right;    in_right = in_left;    in_left = temp;
        // Round 5
        KL1 = LeftShiftd(k5, 1);    KL2 = k7 ^ constants[6];
        KO1 = LeftShiftd(k6, 5);    KO2 = LeftShiftd(k2, 8);    KO3 = LeftShiftd(k3, 13);
        KI1 = k1 ^ constants[0];    KI2 = k8 ^ constants[7];    KI3 = k4 ^ constants[3];
        temp = FLd(in_left, KL1, KL2);
        temp = FOd(temp, KO1, KO2, KO3, KI1, KI2, KI3);
        temp ^= in_right;    in_right = in_left;    in_left = temp;
        // Round 6
        KL1 = LeftShiftd(k6, 1);    KL2 = k8 ^ constants[7];
        KO1 = LeftShiftd(k7, 5);    KO2 = LeftShiftd(k3, 8);    KO3 = LeftShiftd(k4, 13);
        KI1 = k2 ^ constants[1];    KI2 = k1 ^ constants[0];    KI3 = k5 ^ constants[4];
        temp = FOd(in_left, KO1, KO2, KO3, KI1, KI2, KI3);
        temp = FLd(temp, KL1, KL2);
        temp ^= in_right;    in_right = in_left;    in_left = temp;
        // Round 7
        KL1 = LeftShiftd(k7, 1);    KL2 = k1 ^ constants[0];
        KO1 = LeftShiftd(k8, 5);    KO2 = LeftShiftd(k4, 8);    KO3 = LeftShiftd(k5, 13);
        KI1 = k3 ^ constants[2];    KI2 = k2 ^ constants[1];    KI3 = k6 ^ constants[5];
        temp = FLd(in_left, KL1, KL2);
        temp = FOd(temp, KO1, KO2, KO3, KI1, KI2, KI3);
        temp ^= in_right;//    in_right = in_left;    in_left = temp;
        if (temp == cipher_right) {
            in_right = in_left;    in_left = temp;
            // Round 8
            KL1 = LeftShiftd(k8, 1);    KL2 = k2 ^ constants[1];
            KO1 = LeftShiftd(k1, 5);    KO2 = LeftShiftd(k5, 8);    KO3 = LeftShiftd(k6, 13);
            KI1 = k4 ^ constants[3];    KI2 = k3 ^ constants[2];    KI3 = k7 ^ constants[6];
            temp = FOd(in_left, KO1, KO2, KO3, KI1, KI2, KI3);
            temp = FLd(temp, KL1, KL2);
            temp ^= in_right; //   in_right = in_left;    in_left = temp;
            if (temp == cipher_left) printf("The secret key is %08x%08x\n", threadIndex, i);
        }



 /*       // Round 8
        KL1 = LeftShiftd(k8, 1);    KL2 = k2 ^ constants[1];
        KO1 = LeftShiftd(k1, 5);    KO2 = LeftShiftd(k5, 8);    KO3 = LeftShiftd(k6, 13);
        KI1 = k4 ^ constants[3];    KI2 = k3 ^ constants[2];    KI3 = k7 ^ constants[6];
        temp = FOd(in_left, KO1, KO2, KO3, KI1, KI2, KI3);
        temp = FLd(temp, KL1, KL2);
        temp ^= in_right;    in_right = in_left;    in_left = temp;
        //printf("%08x%08x\n", in_left, in_right);

        if (in_left == cipher_left)
            if (in_right == cipher_right)
                printf("The secret key is %08x%08x\n", threadIndex, i);*/
        k8++; k4 = k8;
    }
    k7++; k3 = k7;
    }
}

/*__global__ void KASUMI64Exhaustive32Tables(bit32 left, bit32 right, bit32 cipher_left, bit32 cipher_right, bit8* S7G, bit16* S9G) {
    bit32 threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    //    int warpThreadIndex = threadIdx.x & 31;
    if (threadIdx.x < 512) {
        if (threadIdx.x < 128) S7S[threadIdx.x] = S7G[threadIdx.x];
 //       for (int i=0;i<32; i++)
 //           S9S2[threadIdx.x / 4][i][threadIdx.x % 4] = S9G[threadIdx.x];
    }
    __syncthreads();

    bit32 in_left = left, in_right = right, temp;
    bit16 k1 = threadIndex / 65536, k2 = threadIndex % 65536, k3 = 0, k4 = 0, k5 = 0, k6 = 0, k7 = 0, k8 = 0;
    bit16 KL1, KL2, KO1, KO2, KO3, KI1, KI2, KI3;

    for (int i = 0; i < 65536; i++) {
        in_left = left; in_right = right;
        // Round 1
        KL1 = LeftShiftd(k1, 1);    KL2 = k3 ^ 0x89AB;
        KO1 = LeftShiftd(k2, 5);    KO2 = LeftShiftd(k6, 8);    KO3 = LeftShiftd(k7, 13);
        KI1 = k5 ^ 0xFEDC;    KI2 = k4 ^ 0xCDEF;    KI3 = k8 ^ 0x3210;
        temp = FLd(in_left, KL1, KL2);
        temp = FOd(temp, KO1, KO2, KO3, KI1, KI2, KI3);
        temp ^= in_right;    in_right = in_left;    in_left = temp;
        // Round 2
        KL1 = LeftShiftd(k2, 1);    KL2 = k4 ^ 0xCDEF;
        KO1 = LeftShiftd(k3, 5);    KO2 = LeftShiftd(k7, 8);    KO3 = LeftShiftd(k8, 13);
        KI1 = k6 ^ 0xBA98;    KI2 = k5 ^ 0xFEDC;    KI3 = k1 ^ 0x0123;
        temp = FOd(in_left, KO1, KO2, KO3, KI1, KI2, KI3);
        temp = FLd(temp, KL1, KL2);
        temp ^= in_right;    in_right = in_left;    in_left = temp;
        // Round 3
        KL1 = LeftShiftd(k3, 1);    KL2 = k5 ^ 0xFEDC;
        KO1 = LeftShiftd(k4, 5);    KO2 = LeftShiftd(k8, 8);    KO3 = LeftShiftd(k1, 13);
        KI1 = k7 ^ 0x7654;    KI2 = k6 ^ 0xBA98;    KI3 = k2 ^ 0x4567;
        temp = FLd(in_left, KL1, KL2);
        temp = FOd(temp, KO1, KO2, KO3, KI1, KI2, KI3);
        temp ^= in_right;    in_right = in_left;    in_left = temp;
        // Round 4
        KL1 = LeftShiftd(k4, 1);    KL2 = k6 ^ 0xBA98;
        KO1 = LeftShiftd(k5, 5);    KO2 = LeftShiftd(k1, 8);    KO3 = LeftShiftd(k2, 13);
        KI1 = k8 ^ 0x3210;    KI2 = k7 ^ 0x7654;    KI3 = k3 ^ 0x89AB;
        temp = FOd(in_left, KO1, KO2, KO3, KI1, KI2, KI3);
        temp = FLd(temp, KL1, KL2);
        temp ^= in_right;    in_right = in_left;    in_left = temp;
        // Round 5
        KL1 = LeftShiftd(k5, 1);    KL2 = k7 ^ 0x7654;
        KO1 = LeftShiftd(k6, 5);    KO2 = LeftShiftd(k2, 8);    KO3 = LeftShiftd(k3, 13);
        KI1 = k1 ^ 0x0123;    KI2 = k8 ^ 0x3210;    KI3 = k4 ^ 0xCDEF;
        temp = FLd(in_left, KL1, KL2);
        temp = FOd(temp, KO1, KO2, KO3, KI1, KI2, KI3);
        temp ^= in_right;    in_right = in_left;    in_left = temp;
        // Round 6
        KL1 = LeftShiftd(k6, 1);    KL2 = k8 ^ 0x3210;
        KO1 = LeftShiftd(k7, 5);    KO2 = LeftShiftd(k3, 8);    KO3 = LeftShiftd(k4, 13);
        KI1 = k2 ^ 0x4567;    KI2 = k1 ^ 0x0123;    KI3 = k5 ^ 0xFEDC;
        temp = FOd(in_left, KO1, KO2, KO3, KI1, KI2, KI3);
        temp = FLd(temp, KL1, KL2);
        temp ^= in_right;    in_right = in_left;    in_left = temp;
        // Round 7
        KL1 = LeftShiftd(k7, 1);    KL2 = k1 ^ 0x0123;
        KO1 = LeftShiftd(k8, 5);    KO2 = LeftShiftd(k4, 8);    KO3 = LeftShiftd(k5, 13);
        KI1 = k3 ^ 0x89AB;    KI2 = k2 ^ 0x4567;    KI3 = k6 ^ 0xBA98;
        temp = FLd(in_left, KL1, KL2);
        temp = FOd(temp, KO1, KO2, KO3, KI1, KI2, KI3);
        temp ^= in_right;    in_right = in_left;    in_left = temp;
        // Round 8
        KL1 = LeftShiftd(k8, 1);    KL2 = k2 ^ 0x4567;
        KO1 = LeftShiftd(k1, 5);    KO2 = LeftShiftd(k5, 8);    KO3 = LeftShiftd(k6, 13);
        KI1 = k4 ^ 0xCDEF;    KI2 = k3 ^ 0x89AB;    KI3 = k7 ^ 0x7654;
        temp = FOd(in_left, KO1, KO2, KO3, KI1, KI2, KI3);
        temp = FLd(temp, KL1, KL2);
        temp ^= in_right;    in_right = in_left;    in_left = temp;
        //printf("%08x%08x\n", in_left, in_right);

        if (in_left == cipher_left)
            if (in_right == cipher_right)
                printf("The secret key is %08x%08x\n", threadIndex, i);
        k8++;
    }
}*/

int main(void) {
	cudaSetDevice(0);
    bit32 plaintextl = 0, plaintextr=0, ciphertextl= 0xf54cfbf7, ciphertextr= 0x5f3b5699;
    // Allocate tables
    bit8 *S7d;
    bit16 *S9d;
    bit16* constantd;
    float milliseconds = 0;

    bit32* ciphertextl_d, *ciphertextr_d;
    bit32* cipherl, * cipherr;

    cipherl = (bit32*)calloc(BLOCKS * THREADS, sizeof(bit32));
    cipherr = (bit32*)calloc(BLOCKS * THREADS, sizeof(bit32));

    cudaMalloc((void**)&ciphertextl_d, BLOCKS * THREADS * sizeof(bit32));
    cudaMalloc((void**)&ciphertextr_d, BLOCKS * THREADS * sizeof(bit32));


    cudaMallocManaged(&S7d, 128 * sizeof(bit8));
    cudaMallocManaged(&constantd, 8 * sizeof(bit16));
    cudaMallocManaged(&S9d, 512 * sizeof(bit16));
    for (int i = 0; i < 128; i++) S7d[i] = S7[i];
    for (int i = 0; i < 512; i++) S9d[i] = S9[i];
    for (int i = 0; i < 8; i++) constantd[i] = constant[i];

    
    cudaDeviceSynchronize(); clock_t beginTime = clock();
    cudaEvent_t start, stop;	cudaEventCreate(&start);	cudaEventCreate(&stop);	cudaEventRecord(start);
//    KASUMI64Exhaustive << <BLOCKS, THREADS >> > (plaintextl, plaintextr, ciphertextl, ciphertextr, S7d, S9d);
//    KASUMI64EncryptionTMTO << <BLOCKS, THREADS >> > (plaintextl, plaintextr, ciphertextl, ciphertextr, S7d, S9d);
//    KASUMI64ExhaustiveConstants << <BLOCKS, THREADS >> > (plaintextl, plaintextr, ciphertextl, ciphertextr, S7d, S9d, constantd);
   KASUMI64ExhaustiveConstantsRegister << <BLOCKS, THREADS >> > (plaintextl, plaintextr, ciphertextl, ciphertextr, S7d, S9d); //best
//    KASUMI64ExhaustiveConstantsRegisterTMTO << <BLOCKS, THREADS >> > (plaintextl, plaintextr, ciphertextl, ciphertextr, S7d, S9d, ciphertextl_d,ciphertextr_d); //best
 //   KASUMI64Exhaustive32Tables << <BLOCKS, THREADS >> > (plaintextl, plaintextr, ciphertextl, ciphertextr, S7d, S9d);
    cudaEventRecord(stop);	cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);	printf("Time elapsed: %f milliseconds ", milliseconds);

    cudaMemcpy(cipherl, ciphertextl_d, BLOCKS * THREADS * sizeof(bit32), cudaMemcpyDeviceToHost);
    cudaMemcpy(cipherr, ciphertextr_d, BLOCKS * THREADS * sizeof(bit32), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    printf("Time elapsed: %f sec\n", float(clock() - beginTime) / CLOCKS_PER_SEC);
    printf("-------------------------------\n");
//    encryption(0,0,0xf54cfbf7,0x5f3b5699);
	printf("%s\n", cudaGetErrorString(cudaGetLastError()));
}

