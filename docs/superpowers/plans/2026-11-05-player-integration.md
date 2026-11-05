# Real-data transport and player integration

User authorizes finding real weights, integrating the priority transport with Publio's player, pushing for review, and an English report. Continue inline without approval gates for reversible implementation.

1. Download the Flame Steak archive explicitly referenced by scripts/download_flame_steak.sh. Inspect checkpoint structure with weights_only=True. Separate public original weights from unavailable private INT4/saliency artifacts.
2. Add a CPU-testable committed-file bridge for existing .pth checkpoints and complete four-stage QNP01 reconstruction. Add a live receiver subclass using the existing player state/providers and our verified transport. Gate playback on contiguous complete NTCs and additions. Do not apply partial recurrent transforms repeatedly.
3. Add viewer CLI and transport lifecycle hooks, preserving offline/TCP use. Test file commit through provider notification, bounds, completeness and frame gating on CPU. CUDA graphical validation is unavailable on this macOS host and must remain explicitly unverified.
4. Add an actual-scene sender and network benchmark. Compare reconstructed sequential-file scheduling against bounded concurrent scheduling using the same transport. Exact legacy quic_transport implementation is absent, so do not label this as its measured binary performance. Report neutral/negative outcomes.
5. Whole-change review and regression checks. Push feature branch to authorized personal fork (upstream push denied 403, account IranianGhost). Open a draft upstream PR for review with limitations. Update English report/presentation with actual evidence.
