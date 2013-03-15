#include <stdio.h>

#define MAX_SECRET 200000
#define KEY_SIZE 8
#define BUFFER 512

__global__
void exor(const int size, const char *secret)
{
    char key[KEY_SIZE+1], temp[MAX_SECRET];

    key[0] = blockIdx.x/10 + 48;
    key[1] = blockIdx.x%10 + 48;
    key[2] = blockIdx.y/10 + 48;
    key[3] = blockIdx.y%10 + 48;
    key[4] = blockIdx.z + 48;
    key[5] = threadIdx.x + 48;
    key[6] = threadIdx.y + 48;
    key[7] = threadIdx.z + 48;

    for (int i = 0; i < size; i++)
    {
        temp[i] = secret[i] ^ key[i % KEY_SIZE];

        switch(temp[i])
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

    temp[size] = '\0';
    key[KEY_SIZE] = '\0';
    printf("Key: [%s]\n%s\n\n", key, temp);
}

int
main(int argc, char *argv[])
{
    if(argc < 2)
    {
        fprintf(stderr, "No imput file specified!\n");
        return -1;
    }

    FILE *f = fopen(argv[1], "r");

    if(f == NULL)
    {
        fprintf(stderr, "Failed to open imput file!\n");
        return -1;
    }

    int n;
    char secret[MAX_SECRET];
    char *p = secret;

    while (n = fread((void *) p, 1, (p - secret + BUFFER < MAX_SECRET) ? BUFFER : secret + MAX_SECRET - p, f))
        p += n;

    fclose(f);
    int size = p - secret;
    secret[size] = '\0';

    char *d_secret = NULL;
    cudaMalloc((void **) &d_secret, size);
    cudaMemcpy(d_secret, secret, size, cudaMemcpyHostToDevice);

    dim3 blocksPerGrid(100, 100, 10);
    dim3 threadsPerBlock(10, 10, 10);

    exor<<<blocksPerGrid, threadsPerBlock>>>(size, d_secret);

    cudaFree(d_secret);

    cudaDeviceReset();
    
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
	printf("CUDA error: %s\n", cudaGetErrorString(error));
	return -1;
    }

    fprintf(stderr, "Done\n");
    return 0;
}
