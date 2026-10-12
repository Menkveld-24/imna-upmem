#include "../dpu.h"


#ifndef K_SIZE
    #error Template argument missing.
#endif

#if K_SIZE > MAX_KERNEL_SIZE
    #error Kernel size too large, exceeds maximum supported kernel size. Or update MAX_KERNEL_SIZE.
#endif

#ifndef K_STRIDE
    #error Template argument missing.
#endif

// #ifndef PREFIX
//     #error Prefix argument missing.
// #endif
#define CCAT2(x, y) x ## y
#define CCAT(x, y) CCAT2(x, y)
#define CCAT4_(a, b, c, d) a ## b ## c ## d
#define CCAT4(a, b, c, d) CCAT4_(a, b, c, d)
#define PREFIX CCAT4(conv2d_, K_SIZE, _, K_STRIDE)
#define FN(x) CCAT(CCAT(PREFIX, _), x)

void FN(doConvolutionForWindow) (
    const struct tasklet_t *tasklet,
    struct window_t *working_window
) {
    // Each tasklet processes multiple output pixels in a strided manner
    // ToDo: Do loop unrolling

    int tid = tasklet->id;                  // worker ID
    int stride = NR_WORKERS * K_STRIDE;       // number of workers

    // DPU_PRINTF("Thread idx: %d, stride: %d\n", tid, stride);

    uint16_t row = 0;
    uint16_t outputIndex = tid + working_window->output_alignment_padding[row];
    uint16_t relativeRowPosition = tid;

    for (int inputIndex = relativeRowPosition + working_window->alignment_padding[row]; inputIndex < working_window->header.payload_size; inputIndex += stride) {

        // we start at [0][0], but need to jump to the next row if the kernel is all the way to the right
        // The row width is 48, so the output is 46x46
        // So indexes range from 0 to 45 for each row
        // So if the index reaches 46, we need to jump to the next row
        if (relativeRowPosition > working_window->header.tile_info.output_width - 1) {
            // Reset the row position
            relativeRowPosition -= working_window->header.tile_info.output_width;
            // Jump the inputIndex to the next row
            // -1 because were actually already 1 too far
            inputIndex += K_SIZE - 1;
            row++;
            // Compensate for alignment issues of the memory
            inputIndex += working_window->alignment_padding[row];
            outputIndex += working_window->output_alignment_padding[row];
            // DPU_PRINTF("[Worker %d] Moving to row %d, new inputIndex: %d, new outputIndex: %d\n", tasklet->id, row, inputIndex, outputIndex);
        }

        // Reset the output window pixel
        // working_window->output[outputIndex] = 0;
        // output_buffer[working_window->header.input_feature_index][
        //     working_window->header.tile_info.output_horizontal_offset
        //     + working_window->header.tile_info.output_vertical_offset
        //     + outputIndex
        // ] = 0;
        // fxp32_t intermediate = 0;
        // DPU_PRINTF("Writing to outputIndex %d\n", outputIndex);

        // DPU_PRINTF("[Worker %d] Working on window id:%d\n", tasklet->id, working_window->header.id);
        // if (working_window->header.id == 4) {
        //     DPU_PRINTF("Starting to calculate pixel %d, with starting value: %d\n", outputIndex, working_window->output[outputIndex]);
        // }

        int32_t acc = working_window->output[outputIndex];
        for (int kx = 0; kx < K_SIZE; kx++) {
            uint8_t alignmentCompensation = 0;
            for (int ky = 0; ky < K_SIZE; ky++) {
                // Current row is already compensated for
                if (ky > 0) {
                    alignmentCompensation += working_window->alignment_padding[row + ky] + working_window->header.tile_info.width;
                }

                int8_t sign = working_window->kernel.signs[ky][kx];

                // So sign of 0 means useless shift
                if (sign == 0) {
                    continue; // skip
                }

                int32_t r = working_window->payload[inputIndex + kx + alignmentCompensation] >> working_window->kernel.shifts[ky][kx];
                // acc += (sign > 0) ? r : -r;

                // int inputId = inputIndex + kx + alignmentCompensation;
                // // int inputId = inputIndex + kx + ky;
                // // DPU_PRINTF("InputId %d for kernel[%d][%d]\n", inputId, ky, kx);
                // fxp32_t input = working_window->payload[inputId];
                // // fxp32_t input = input_buffer[working_window->header.inout_buffer_index][inputId];
                
                // int8_t sign = working_window->kernel.signs[ky][kx];
                // // float shift = kernel->shifts[ky][kx];
                // int8_t shift = working_window->kernel.shifts[ky][kx];
                
                // // equivalent multipication; 2^shift * sign
                // // working_window->output[outputIndex] += input * shift;
                // // output_buffer[working_window->header.input_feature_index][
                // //     working_window->header.tile_info.output_horizontal_offset
                // //     + working_window->header.tile_info.output_vertical_offset
                // //     + outputIndex
                // // ] += input * shift;

                // // So sign of 0 means useless shift
                // if (sign == 0) {
                //     continue; // skip
                // }

                // // // Intermediate result
                // int32_t result;

                // I have no idea if this direction is correct
                // ToDo: This is useless, the shift range is [-14, 0] iirc
                // if (shift < 0) {
                //     // right shift
                //     result = input >> -shift;
                // } else {
                //     // left shift
                //     result = input << shift;
                // }
                // result = input >> shift;

                // // Add or subtract based on sign
                if (sign > 0) {
                    // working_window->output[outputIndex] += result;
                    acc += r;
                    // intermediate += result;
                    // output_buffer[0][outputIndex] += result;
                } else {
                    // working_window->output[outputIndex] -= result;
                    acc -= r;
                    // intermediate -= result;
                    // output_buffer[0][outputIndex] -= result;
                }
            }
        }
        working_window->output[outputIndex] = acc;
        // Move to the next output index
        outputIndex += stride;
        relativeRowPosition += stride;
    }
}


#undef K_SIZE
#undef K_STRIDE
#undef PREFIX
#undef CCAT2
#undef CCAT
#undef CCAT4_
#undef CCAT4
#undef FN