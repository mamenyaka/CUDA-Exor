CUDA Exor decrypter
============

This program can decript an exor encrypted text with a key of size 8.


How to use
============

Use on of the demos, or create one of your own

>$ javac Encrypter.java

>$ java Encrypter 12345678 &lt;in &gt;out

Compile the decrypter (you need a NVidia GPU with CUDA architecture >= 2.1)

>$ nvcc exor.cu -o exor -arch=sm_21 CUDA.cu

Then run it

>$ ./exor &lt;secret1 &gt;out
