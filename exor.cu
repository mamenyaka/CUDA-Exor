#include <stdio.h>

#define MAX_SECRET 8000000
#define KEY_SIZE 8
#define BUFFER 512

__global__
void exor(const int size, const char *secret, char *key)
{
    char temp[KEY_SIZE];

    temp[0] = blockIdx.x/10 + 48;
    temp[1] = blockIdx.x%10 + 48;
    temp[2] = blockIdx.y/10 + 48;
    temp[3] = blockIdx.y%10 + 48;
    temp[4] = blockIdx.z + 48;
    temp[5] = threadIdx.x + 48;
    temp[6] = threadIdx.y + 48;
    temp[7] = threadIdx.z + 48;

    for(int i = 0; i < size; i++)
    {
        switch(secret[i] ^ temp[i % KEY_SIZE])
        {
        case '|':
        case '~':
        case '^':
        case '*':
        case '+':
        case '_':
        case '{':
        case '}':
        case '\\':
        case '#':
            return;
        }
    }

    for(int i = 0; i < KEY_SIZE; i++)
        key[i] = temp[i];
}

int
main()
{
    char secret[MAX_SECRET], key[KEY_SIZE+1];
    char *p = secret;

    while (int n = fread((void *) p, 1, (p - secret + BUFFER < MAX_SECRET) ? BUFFER : secret + MAX_SECRET - p, stdin))
        p += n;

    int size = p - secret;

    char *d_secret, *d_key;
    cudaMalloc((void **) &d_secret, size);
    cudaMalloc((void **) &d_key, KEY_SIZE);

    cudaMemcpy(d_secret, secret, size, cudaMemcpyHostToDevice);

    dim3 blocksPerGrid(100, 100, 10);
    dim3 threadsPerBlock(10, 10, 10);

    exor<<<blocksPerGrid, threadsPerBlock>>>(size, d_secret, d_key);

    cudaMemcpy(key, d_key, KEY_SIZE, cudaMemcpyDeviceToHost);

    cudaFree(d_secret);
    cudaFree(d_key);
    cudaDeviceReset();

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
	fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
	return -1;
    }

    for(int i = 0; i < size; i++)
        secret[i] ^= key[i % KEY_SIZE];

    secret[size] = '\0';
    key[KEY_SIZE] = '\0';
    printf("%s\nKey: %s\n", secret, key);

    fprintf(stderr, "Done\n");
    return 0;
}
