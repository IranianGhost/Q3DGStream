# Native player network evaluation

1. Add an opt-in playback clock and telemetry to the real GLFW/CUDA viewer. Test startup gating, rebuffering, and final frame completion with controlled times.
2. Compare original FP32 fixed workload across clean, jitter/bandwidth, and repeatable random-loss profiles on gpu1. Use a private network namespace and a separate authenticated X display.
3. Compare fixed and adaptive optional bytes with identical required FP32 updates. Feed committed payload rate and contiguous playable frames from the real viewer to the sender through a private same-host feedback file.
4. Record the exact configuration, startup, stalls, late availability, transfer size, completion and all failures. Keep one-second smoke cases separate from sustained playback or quality evidence.
5. Run scoped tests on gpu1 and publish the review branch and concise evidence.

Compressed variants and saliency assignments depend on Publio. The adjustable experimental payload contains no model updates and cannot change rendered quality.
