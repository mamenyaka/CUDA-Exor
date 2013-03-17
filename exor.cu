#include <stdio.h>

#define MAX_SECRET 200000
#define KEY_SIZE 8
#define BUFFER 512

__global__
void exor(char *secret)
{
    char key[KEY_SIZE];
    
    __syncthreads();

    key[0] = blockIdx.x/10 + 48;
    key[1] = blockIdx.x%10 + 48;
    key[2] = blockIdx.y/10 + 48;
    key[3] = blockIdx.y%10 + 48;
    key[4] = blockIdx.z + 48;
    key[5] = threadIdx.x + 48;
    key[6] = threadIdx.y + 48;
    key[7] = threadIdx.z + 48;

    int i = 0;
    while(secret[i] != '\0')
    {
        switch(secret[i] ^ key[i % KEY_SIZE])
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
        
        i++;
    }

    i = 0;
    while(secret[i] != '\0')
    {
        secret[i] ^= key[i % KEY_SIZE];
        i++;
    }
    
    secret[i++] = '\n';
    
    for(int j = 0; j < 8; j++)
        secret[i++] = key[j];
        
    secret[i++] = '\0';
}

int
main(int argc, char *argv[])
{
    if(argc < 2)
    {
        fprintf(stderr, "No imput file specified!\n");
        return -1;
    }

    FILE *in = fopen(argv[1], "r");

    if(in == NULL)
    {
        fprintf(stderr, "Failed to open imput file!\n");
        return -1;
    }

    int n;
    char secret[MAX_SECRET];
    char *p = secret;

    while (n = fread((void *) p, 1, (p - secret + BUFFER < MAX_SECRET) ? BUFFER : secret + MAX_SECRET - p, in))
        p += n;

    fclose(in);
    int size = p - secret;
    secret[size] = '\0';

    char *d_secret = NULL;
    cudaMalloc((void **) &d_secret, size+10);
    cudaMemcpy(d_secret, secret, size+10, cudaMemcpyHostToDevice);

    dim3 blocksPerGrid(100, 100, 10);
    dim3 threadsPerBlock(10, 10, 10);

    exor<<<blocksPerGrid, threadsPerBlock>>>(d_secret);

    cudaMemcpy(secret, d_secret, size+10, cudaMemcpyDeviceToHost);
    cudaFree(d_secret);

    cudaDeviceReset();
    
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
	fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
	return -1;
    }
    
    FILE *out = fopen("out", "w");
    fprintf(out, "%s\n\n", secret);
    fclose(out);
    
    fprintf(stderr, "Done\n");
    return 0;
}
