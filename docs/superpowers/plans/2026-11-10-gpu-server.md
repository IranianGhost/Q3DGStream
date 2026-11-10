# Real GPU validation

User authorized completing server setup and previous pending QUIC→GPU player test.
Use the existing feature branch and dedicated /local/users/abeigi scratch paths.
Do not change shared driver/system packages. Use GPU 0 only if idle before the run,
limit compilation to 2 jobs, and preserve setup/build logs.

1. Verify SSH and inventory GPU/driver/compiler/storage (done).
2. Install isolated Python 3.12 to retain aioquic 1.7.2; CUDA 12.1/PyTorch 2.5.1
   match the older driver. Compile tiny-cuda-nn for SM75 and native rasterizer.
3. Transfer original initial scene + 3 checkpoints/additions. Verify receiver
   hashes via existing transport, run native lazy NTC providers without CPU stubs.
4. Headless adapter bypasses only OpenGL presentation initialization. Reuse native
   upload/motion/rotation/additions/camera/rasterization functions; assert exact
   loaded weights, nonzero Gaussian motion, finite nonblack images and image change.
5. Download PNG/report/build provenance, inspect images, run scoped tests, review
   and push. Keep compressed-model and interactive-UI limitations explicit.
