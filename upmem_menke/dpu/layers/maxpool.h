#include "../dpu.h"


#ifndef K_SIZE
    #error Template argument missing.
#endif

#ifndef K_STRIDE
    #error Template argument missing.
#endif

#define CCAT2(x, y) x ## y
#define CCAT(x, y) CCAT2(x, y)
#define CCAT4_(a, b, c, d) a ## b ## c ## d
#define CCAT4(a, b, c, d) CCAT4_(a, b, c, d)
#define PREFIX CCAT4(maxpool_, K_SIZE, _, K_STRIDE)
#define FN(x) CCAT(CCAT(PREFIX, _), x)


void FN(doMaxPoolingForWindow)(
    const struct tasklet_t *tasklet,
    struct window_t *working_window
) {
    // Each tasklet processes multiple output pixels in a strided manner
    // ToDo: Do loop unrolling

    int tid = tasklet->id;
    int input_stride = NR_WORKERS * K_STRIDE;
    int output_stride = NR_WORKERS;

    uint16_t inputRow = 0;
    uint16_t outputRow = 0;
    uint16_t outputIndex = tid + working_window->output_alignment_padding[outputRow];
    uint16_t relativeRowPosition = tid;  // output-space column index

    for (int inputIndex = tid * K_STRIDE + working_window->alignment_padding[inputRow]; inputIndex < working_window->header.payload_size; inputIndex += input_stride) {

        if (relativeRowPosition > working_window->header.tile_info.output_width - 1) {
            relativeRowPosition -= working_window->header.tile_info.output_width;
            // Skip the remaining input columns in this row that have no valid kernel start,
            // then land at the correct strided position in the next row.
            inputIndex += working_window->header.tile_info.width - working_window->header.tile_info.output_width * K_STRIDE;
            // Advance K_STRIDE input rows: one full row per stride step.
            inputRow++;
            inputIndex += working_window->alignment_padding[inputRow];
#if K_STRIDE > 1
            inputIndex += working_window->header.tile_info.width;
            inputRow++;
            inputIndex += working_window->alignment_padding[inputRow];
#endif
            outputRow++;
            outputIndex += working_window->output_alignment_padding[outputRow];
        }

        fxp32_t max = working_window->payload[inputIndex];

        for (int kx = 0; kx < K_SIZE; kx++) {
            uint8_t alignmentCompensation = 0;

            for (int ky = 0; ky < K_SIZE; ky++) {
                if (ky > 0) {
                    alignmentCompensation += working_window->alignment_padding[inputRow + ky] + working_window->header.tile_info.width;
                }

                int inputId = inputIndex + kx + alignmentCompensation;
                fxp32_t input = working_window->payload[inputId];
                if (input > max) max = input;
            }
        }

        working_window->output[outputIndex] = max;

        outputIndex += output_stride;
        relativeRowPosition += output_stride;
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