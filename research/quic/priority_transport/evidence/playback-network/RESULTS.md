# Playback and loss experiments

|Run|Transfer s|Initial buffer s|Startup s|Stalls|Stall s|Played s|Valid|
|---|---:|---:|---:|---:|---:|---:|---|
|burst-timeout-120|57.89|2|57.49|0|0.00|1.00|True|
|burst-timeout-15|15.22|2|none|0|0.00|0.00|False|
|live-buffer-1|17.80|2|3.24|0|0.00|18.00|True|
|live-buffer-2|17.80|2|3.24|0|0.00|18.00|True|
|loss-1-128-cubic|48.09|2|47.81|0|0.00|2.00|True|
|loss-1-128-reno|53.80|2|53.45|0|0.00|2.00|True|
|loss-1-16-reno|50.87|2|50.61|0|0.00|2.00|True|
|loss-2-128-cubic|49.72|2|49.34|0|0.00|2.00|True|
|loss-2-128-reno|54.98|2|54.68|0|0.00|2.00|True|
|loss-2-16-reno|56.99|2|56.74|0|0.00|2.00|True|
|mtu1400-cubic|44.60|2|44.00|0|0.00|2.00|True|
|mtu1400-reno|42.99|2|42.69|0|0.00|2.00|True|
|vod-1-buffer|4.38|2|3.25|0|0.00|8.00|True|
|vod-1-composite|4.38|2|3.25|0|0.00|8.00|True|
|vod-1-fixed-high|58.84|2|18.67|3|32.99|8.00|True|
|vod-2-buffer|4.39|2|3.26|0|0.00|8.00|True|
|vod-2-composite|4.37|2|3.31|0|0.00|8.00|True|
|vod-2-fixed-high|59.42|2|19.86|3|33.14|8.00|True|

## Limits

Receiver playback is a media-time simulation, not GPU rendering. Playback requires contiguous complete intervals. Incomplete sequences are censored at observation end; fully received sequences can finish locally without more network data.

VOD transfer latency from producer enqueue is not a media deadline. Use startup, stalls and lateness against the initial playback schedule. Initial buffer sensitivity is replayed offline; only the 2 s setting drives the online buffer controller.

FP32 files are real; 62,296/66,741-byte alternatives are synthetic loads matching documented external formats. They do not establish quality or recurrent-model decodability.

Window/CC ablation uses 1,200-byte datagrams; VOD and follow-up trials use 1,400-byte datagrams. Loss realizations differ between runs; two repeats are exploratory. A larger timeout is a recovery budget, not a throughput repair.
