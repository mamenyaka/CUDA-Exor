CUDA Exor decrypter
============

This program can decript an exor encrypted text with a key of size 8.


How to use
============

Use on of the demos, or create one of your own

$ javac Encrypter.java
$ java Encrypter 12345678 <in >out

Compile the decrypter (you need to install the CUDA Toolkit first)

$ nvcc exor.cu -o exor -arch=sm_21 CUDA.cu

Then run it

$ ./exor /<secret0 />out
