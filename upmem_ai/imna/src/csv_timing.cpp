#include "csv_timing.h"
#include <cstdio>
#include <cstring>
#include <unistd.h>
#include <cerrno>

struct CsvFile {
    FILE *fp      = nullptr;
    char path[512] = {};

    bool open(const char *file_path) {
        if (fp != nullptr && strncmp(path, file_path, sizeof(path)) == 0)
            return true;
        if (fp != nullptr) fclose(fp);

        bool write_header = (access(file_path, F_OK) != 0);
        fp = fopen(file_path, "a");
        if (!fp) {
            fprintf(stderr, "[accel] fopen(%s): %s\n", file_path, strerror(errno));
            path[0] = '\0';
            return false;
        }
        if (write_header) {
            fprintf(fp, "batch_id,cin,cout,kernel_size,batch_size,"
                        "tensor_alloc_ms,build_dist_ms,dpu_alloc_ms,dpu_load_ms,"
                        "xfer_config_ms,sg_xfer_ms,dpu_launch_ms,xfer_results_ms,"
                        "total_ms\n");
        }
        strncpy(path, file_path, sizeof(path) - 1);
        return true;
    }

    ~CsvFile() { if (fp) fclose(fp); }
};

void write_timing_to_csv(
    const char      *csv_file,
    int64_t          batch_id,
    const Context *ctx,
    double tensor_alloc_ms,
    double build_dist_ms,
    double dpu_alloc_ms,
    double dpu_load_ms,
    double xfer_config_ms,
    double sg_xfer_ms,
    double dpu_launch_ms,
    double xfer_results_ms,
    double total_ms
) {
    static CsvFile csv;
    if (!csv.open(csv_file)) return;

    fprintf(csv.fp, "%lld,%d,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
        (long long)batch_id,
        ctx->input.channels, ctx->output.channels,
        ctx->kernel_size,    ctx->batch_size,
        tensor_alloc_ms, build_dist_ms,
        dpu_alloc_ms,    dpu_load_ms,
        xfer_config_ms,  sg_xfer_ms,
        dpu_launch_ms,   xfer_results_ms,
        total_ms);
    fflush(csv.fp);
}
