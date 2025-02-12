# CUDA_KASUMI

These CUDA Optimizations are used in the ToSC publication **GPU Assisted Brute Force Cryptanalysis of GPRS, GSM, RFID, and TETRA - Brute Force Cryptanalysis of KASUMI, SPECK, and TEA3** by Cihangir Tezcan and Gregor Leander.

It measures how many seconds it takes for your GPU to perform 2^{41} key trials and this number can be modified inside the code.

Use **−maxrregcount = 64** command while compiling to limit the register count to 64. Otherwise you would get **too many resources requested for launch** error at the run time.

Since our optimizations allow **2^{35.72}** keys per second on an RTX 4090, it takes 10.35 years for a single RTX 4090 to break KASUMI-64. Or to break KASUMI-64 in a year, 11 RTX 4090 GPUs are enough.
