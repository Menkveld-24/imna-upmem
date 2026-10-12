// Optimized deepshift implementation without alignment compensation
#include <string.h>
#include <stdio.h>
#include <defs.h>
#include "stdint.h"
#include <profiling.h>
#include <perfcounter.h>
#include <barrier.h>

#define NR_WORKERS NR_TASKLETS

#define KERNEL_SIZE 3

#define INPUT_WIDTH 80
#define INPUT_HEIGHT 80

#define OUTPUT_WIDTH (INPUT_WIDTH - 2 * (KERNEL_SIZE - 1))
#define OUTPUT_HEIGHT (INPUT_HEIGHT - 2 * (KERNEL_SIZE - 1))

#define PAYLOAD_SIZE (INPUT_WIDTH * INPUT_HEIGHT)

#define MULTIPLEOF8BYTES(x) (((x) + 7) & ~7)

uint8_t output_alignment_padding[INPUT_HEIGHT] = {0};
uint8_t alignment_padding[INPUT_HEIGHT] = {0};

__host int32_t payload[PAYLOAD_SIZE];
__host int32_t output[PAYLOAD_SIZE];
__host uint8_t input_shifts[MULTIPLEOF8BYTES(KERNEL_SIZE * KERNEL_SIZE)];
__host uint8_t input_signs[MULTIPLEOF8BYTES(KERNEL_SIZE * KERNEL_SIZE)];
uint8_t shifts[KERNEL_SIZE][KERNEL_SIZE];
uint8_t signs[KERNEL_SIZE][KERNEL_SIZE];

__host uint32_t nb_cycles;
BARRIER_INIT(barrier, NR_TASKLETS);

// PROFILING_INIT(measure_convolution);

void test() {
    // Each tasklet processes multiple output pixels in a strided manner
    // ToDo: Do loop unrolling

    int tid = me();                  // worker ID
    int stride = NR_WORKERS * 1;       // number of workers

    uint16_t row = 0;
    uint16_t outputIndex = tid;// + output_alignment_padding[row];
    uint16_t relativeRowPosition = tid;

    for (int inputIndex = relativeRowPosition /*+ alignment_padding[row]*/; inputIndex < PAYLOAD_SIZE; inputIndex += stride) {

        // if (me() == 0) {
        //     printf("Processing output index %u (input index %u, row %u)\n", outputIndex, inputIndex, row);
        // }

        // we start at [0][0], but need to jump to the next row if the kernel is all the way to the right
        // The row width is 48, so the output is 46x46
        // So indexes range from 0 to 45 for each row
        // So if the index reaches 46, we need to jump to the next row
        if (relativeRowPosition > OUTPUT_WIDTH - 1) {
            // Reset the row position
            relativeRowPosition -= OUTPUT_WIDTH;
            // Jump the inputIndex to the next row
            // -1 because were actually already 1 too far
            inputIndex += KERNEL_SIZE - 1;
            row++;
            // Compensate for alignment issues of the memory
            // inputIndex += alignment_padding[row];
            // outputIndex += output_alignment_padding[row];
        }

        // int32_t acc = output[outputIndex];
        for (int kx = 0; kx < KERNEL_SIZE; kx++) {
            // uint8_t alignmentCompensation = 0;
            for (int ky = 0; ky < KERNEL_SIZE; ky++) {
                // Current row is already compensated for
                // if (ky > 0) {
                //     alignmentCompensation += alignment_padding[row + ky] + INPUT_WIDTH;
                // }

                int inputId = inputIndex + kx + INPUT_WIDTH;// + alignmentCompensation;
                int32_t input = payload[inputId];
                
                int8_t sign = signs[ky][kx];
                int8_t shift = shifts[ky][kx];

                // if (me() == 0) {
                //     printf("Processing kernel element [%u][%u] (input index %u, shift %d, sign %d, input %u)\n", ky, kx, inputId, shift, sign, input);
                // }

                // So sign of 0 means useless shift
                if (sign == 0) {
                    continue; // skip
                }

                // // Intermediate result
                int32_t result;

                // I have no idea if this direction is correct
                // ToDo: This is useless, the shift range is [-14, 0] iirc
                // if (shift < 0) {
                //     // right shift
                //     result = input >> -shift;
                // } else {
                //     // left shift
                //     result = input << shift;
                // }
                result = input >> shift;

                // Add or subtract based on sign
                if (sign > 0) {
                    // acc += result;
                    output[outputIndex] += result;
                    // intermediate += result;
                    // output_buffer[0][outputIndex] += result;
                } else {
                    output[outputIndex] -= result;
                    // acc -= result;
                    // intermediate -= result;
                    // output_buffer[0][outputIndex] -= result;
                }
            }
        }
        // output[outputIndex] = acc;

        // Move to the next output index
        outputIndex += stride;
        relativeRowPosition += stride;
    }
}

int main() {
    // perfcounter_config(COUNT_CYCLES, true);
    printf("Hello from implementation F, tasklet %d!\n", me());

    // memcpy(&shifts, &input_shifts, KERNEL_SIZE*KERNEL_SIZE);
    // memcpy(&signs, &input_signs, KERNEL_SIZE*KERNEL_SIZE);
    for (int i = 0; i < KERNEL_SIZE; i++) {
        for (int j = 0; j < KERNEL_SIZE; j++) {
            shifts[i][j] = input_shifts[i * KERNEL_SIZE + j];
            signs[i][j] = input_signs[i * KERNEL_SIZE + j];
        }
    }

    
    // profiling_start(&measure_convolution);
    barrier_wait(&barrier);
    perfcounter_config(COUNT_CYCLES, true);
    test();
    nb_cycles = perfcounter_get();
    // profiling_stop(&measure_convolution);
    return 0;
}
