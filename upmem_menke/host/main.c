#include <assert.h>
#include <dpu.h>
#include <dpu_log.h>
#include <stdint.h>
#include <stdio.h>
#include "timer.h"
#include <stdlib.h>

// This is replaced in compile time
#ifndef DPU_BINARY
#define DPU_BINARY "./dpu"
#endif

#ifndef FEATURE_SIZE
    #define FEATURE_SIZE 90
#endif

#ifndef NUM_FEATURES
    #define NUM_FEATURES 1
#endif

#ifndef KERNEL_SIZE
    #define KERNEL_SIZE 3
#endif

#ifndef DPU_COUNT
    #define DPU_COUNT 1
#endif

#ifndef SEED
    #define SEED 45
#endif

#ifndef PADDING
    #define PADDING 1
#endif

#define INPUT_SIZE (FEATURE_SIZE * FEATURE_SIZE)
#define OUTPUT_SIZE ((FEATURE_SIZE - KERNEL_SIZE + 1 + (2 * PADDING)) * (FEATURE_SIZE - KERNEL_SIZE + 1 + (2 * PADDING)))
#define MAX(A,B) ((A)>(B)?(A):(B))

// uint32_t buffer[DPU_COUNT][MAX(INPUT_SIZE, OUTPUT_SIZE)];
// /* in/output is just another 2D "view" of the same memory */
// uint32_t (*input)[INPUT_SIZE] = (uint32_t (*)[INPUT_SIZE]) &buffer[0][0];
// uint32_t (*output)[OUTPUT_SIZE] = (uint32_t (*)[OUTPUT_SIZE]) &buffer[0][0];

// uint8_t config[8] = {
//   // Number of features
//   NUM_FEATURES,
//   // Kernel mapping
//   0x00, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0
// };

// Helper function to round to the nearest multiple of 8
#define ALIGN8(x) (((x) + 7) & ~7)

struct kernel_t {
    uint8_t size;           // e.g. 3 for a 3x3 kernel
    int8_t  shifts[KERNEL_SIZE][KERNEL_SIZE];      // the kernel weights
    int8_t  signs[KERNEL_SIZE][KERNEL_SIZE];      // the kernel weights
};

struct kernel_t kernel;

static void alloc_dpus(struct dpu_set_t *set, uint32_t *nr_dpus, int dpu_to_allocate)
{
	DPU_ASSERT(dpu_alloc(dpu_to_allocate, NULL, set));
	//DPU_ASSERT(dpu_load(*set, KDOT_BINARY, NULL));
	DPU_ASSERT(dpu_get_nr_dpus(*set, nr_dpus));
  printf("Allocated %d/%d DPUs\n", *nr_dpus, dpu_to_allocate);
}

// static float rand_float_range(float min, float max)
// {
//     return min + (float)rand() / (float)RAND_MAX * (max - min);
// }

int main(void) {
  // struct dpu_set_t set, dpu;

  printf("DPU_BINARY: %s\n", DPU_BINARY);
  printf("Running %d number of features\n", NUM_FEATURES);

  // Load the input buffer
  // for(int k = 0; k < NUM_FEATURES; k++) {
  //   for (int i = 0; i < FEATURE_SIZE; i++) {
  //     for (int j = 0; j < FEATURE_SIZE; j++) {
  //       input[(k * FEATURE_SIZE * FEATURE_SIZE) + (i * FEATURE_SIZE) + j] = i;
  //     }
  //   }
  // }

  // // Pseudo-random filling of the input
  // // srand(SEED);
  // for (int i = 0; i < DPU_COUNT; i++) {
  //   for (int k = 0; k < NUM_FEATURES * INPUT_SIZE; k++) {
  //     input[i][k] = rand();
  //   }
  // }

  // // Filling the kernel
  // kernel.size = KERNEL_SIZE;
  // for (int i = 0; i < KERNEL_SIZE; i++) {
  //   for (int j = 0; j < KERNEL_SIZE; j++) {
  //     // random integer between ±16
  //     kernel.shifts[i][j] = (rand() % 33) - 16;
  //   }
  // }
  // for (int i = 0; i < KERNEL_SIZE; i++) {
  //   for (int j = 0; j < KERNEL_SIZE; j++) {
  //     kernel.signs[i][j] = (rand() % 2) ? 1 : -1;
  //   }
  // }

  // printf("Input size: %d elements\n", sizeof(input)/sizeof(uint32_t));
  // printf("Which is %d elements per DPU\n", (sizeof(input)/sizeof(uint32_t))/DPU_COUNT);

  // printf("random input check: %u\n", input[12345]);

  printf("Input loaded\n");

  Timer timer2;
	start(&timer2, 0, 0);
  // uint32_t nr_dpus;
  // uint32_t dpus_to_alloc = 1;
  // DPU_ASSERT(dpu_alloc(dpus_to_alloc, NULL, &set));
  // DPU_ASSERT(dpu_get_nr_dpus(set, &nr_dpus));
  // DPU_ASSERT(dpu_load(set, DPU_BINARY, NULL));

  //Allocate set of DPUs
	struct dpu_set_t set, dpu;
  uint32_t nr_dpus, each_dpu;
  alloc_dpus(&set, &nr_dpus, DPU_COUNT); 

  // DPU_FOREACH(set, dpu) {
  //   DPU_ASSERT(dpu_load(dpu, DPU_BINARY, NULL));
  //   // Load the input data into MRAM
  //   DPU_ASSERT(dpu_copy_to(dpu, "input_buffer", 0, &input, sizeof(input)));
  //   // Load the configuration
  //   DPU_ASSERT(dpu_copy_to(dpu, "mconfig", 0, &config, sizeof(config)));
  //   // Load the first kernel
  //   DPU_ASSERT(dpu_copy_to(dpu, "mkernels", 0, &kernel, sizeof(kernel)));
  // }

  // DPU_FOREACH(set, dpu) {
  DPU_ASSERT(dpu_load(set, DPU_BINARY, NULL));
  // Load the input data into MRAM
  // DPU_ASSERT(dpu_copy_to(set, "input_buffer", 0, &input, sizeof(input)));
  // Load the configuration
  // DPU_ASSERT(dpu_copy_to(set, "mconfig", 0, &config, sizeof(config)));
  // Load the first kernel
  // DPU_ASSERT(dpu_copy_to(set, "mkernels", 0, &kernel, ALIGN8(sizeof(kernel))));
  // }
  // DPU_FOREACH(set, dpu, each_dpu) {
  //   // DPU_ASSERT(dpu_copy_to(dpu, "input_buffer", 0, &input, sizeof(input)));
  // }

  // //send input with parallel transfers
  // DPU_FOREACH(set,dpu,each_dpu)
  // {
  //   DPU_ASSERT(dpu_prepare_xfer(dpu, &input[each_dpu][0]));
  // }
  // //DPU_ASSERT(dpu_prepare_xfer(dpu, &test1[NR_DPUS* (last_block_rows*test1_cols)]));
  // DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, "input_buffer", 0, sizeof(uint32_t) * INPUT_SIZE * NUM_FEATURES, DPU_XFER_DEFAULT));


  struct dpu_program_t *program = NULL;

  Timer timer;
	start(&timer, 0, 0);
  // dpu_load_from_memory(set, dpu_binary, dpu_binary_len, &program);
  DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));
  stop(&timer, 0);
  stop(&timer2, 0);
	printf("GEMM Time (ms): ");
	print(&timer, 0, 1);

  // Only run this if the dpus have a printf functionality
  // DPU_FOREACH(set, dpu) {
  //   DPU_ASSERT(dpu_log_read(dpu, stdout));
  // }
  // DPU_ASSERT(dpu_log_read(set, stdout));
  

  // Copy back the result:
  // uint32_t output_size = (FEATURE_SIZE - KERNEL_SIZE + 1 + (2 * PADDING)) * (FEATURE_SIZE - KERNEL_SIZE + 1 + (2 * PADDING));
  // DPU_FOREACH(set,dpu,each_dpu)
  // {
  //   DPU_ASSERT(dpu_prepare_xfer(dpu, &output[each_dpu][0]));
  // }
  // DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_FROM_DPU, "output_buffer", 0, sizeof(uint32_t) * OUTPUT_SIZE * NUM_FEATURES, DPU_XFER_DEFAULT));

  printf("Total Time (ms): ");
  print(&timer2, 0, 1);

  // Print the output
  // for (int i = 0; i < NUM_FEATURES; i++) {
  //   for (int j = 0; j < FEATURE_SIZE; j++) {
  //     for (int k = 0; k < FEATURE_SIZE; k++) {
  //       printf("%d ", input[(i * FEATURE_SIZE * FEATURE_SIZE) + (j * FEATURE_SIZE) + k]);
  //     }
  //     // printf("%d ", input[i * FEATURE_SIZE + j]);
  //   }
  //   printf("\n");
  // }

  DPU_ASSERT(dpu_free(set));

  return 0;
}