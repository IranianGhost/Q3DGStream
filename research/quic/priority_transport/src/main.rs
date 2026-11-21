//! Deadline-aware QUIC sender and receiver with live CUDA playback feedback.
//! The companion Python process applies cumulative updates to actual Gaussians.

use anyhow::{anyhow, bail, Context, Result};
use quinn::{Connection, Endpoint, RecvStream, SendStream};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    env, fs,
    io::{self, Write},
    net::{IpAddr, Ipv4Addr, SocketAddr},
    path::PathBuf,
    sync::{atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering}, Arc},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::sync::{mpsc, Barrier, Mutex};

const HEADER_LEN: usize = 14; // frame:u32, tier:u8, selected:u8, length:u64
const MAX_GROUP: usize = 2 * 1024 * 1024;
const CHUNK_SIZE: usize = 8192; // Match Amirali's 8 KiB application chunks.
const CHUNK_META_LEN: usize = 16; // offset:u64, total-group-bytes:u64
const CHUNK_FLAG: u8 = 0x80;

#[derive(Clone)]
struct Options {
    role: String, host: String, port: u16,
    cert: PathBuf, payload_dir: PathBuf, received_dir: PathBuf,
    metrics_json: Option<PathBuf>, session_clock_file: Option<PathBuf>,
    start_frame: u32, frames: u32, fps: f64,
    initial_mbps: f64, feedback_ms: u64, playout_ms: f64,
    utilization: f64, inference_ms: f64, render_ms: f64, decode_ms: f64,
    scheduling_mode: String, tier_selection: String, fixed_tiers: usize,
    quinn_priority: String,
    delivery_timeout_ms: u64, max_inflight: usize,
    tier_delays_ms: [u64; 4],
}

fn parse_tier_delays(text: &str) -> Result<[u64; 4]> {
    let values = text.split(',').collect::<Vec<_>>();
    if values.len() != 4 {
        bail!("--tier-delays-ms requires four comma-separated values (P0,P1,P2,P3)");
    }
    let mut delays = [0; 4];
    for (index, value) in values.iter().enumerate() {
        delays[index] = value.trim().parse().context("invalid tier delay in milliseconds")?;
        if delays[index] > 60_000 {
            bail!("tier delays must be at most 60000 ms");
        }
    }
    Ok(delays)
}

fn options() -> Result<Options> {
    let mut o = Options {
        role: String::new(), host: "127.0.0.1".into(), port: 55443,
        cert: "ntc_cert.der".into(), payload_dir: "send".into(),
        received_dir: "received".into(), metrics_json: None, session_clock_file: None,
        start_frame: 0, frames: 299, fps: 30.0, initial_mbps: 5.0,
        feedback_ms: 100, playout_ms: 150.0, utilization: 0.92,
        inference_ms: 8.0, render_ms: 8.0, decode_ms: 1.0,
        scheduling_mode: "adaptive".into(), tier_selection: "full".into(),
        fixed_tiers: 4, quinn_priority: "legacy".into(),
        delivery_timeout_ms: 30_000, max_inflight: 16,
        tier_delays_ms: [0; 4],
    };
    let mut args = env::args().skip(1);
    o.role = args.next().unwrap_or_default();
    if o.role != "sender" && o.role != "receiver" {
        bail!("usage: quinn-ntc-deadline sender|receiver [--frames N] [--fps 30] [--initial-mbps 5] [--playout-ms 150] ...");
    }
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--host" => o.host = args.next().context("--host requires IP")?,
            "--port" => o.port = args.next().context("--port requires number")?.parse()?,
            "--cert" => o.cert = args.next().context("--cert requires path")?.into(),
            "--payload-dir" => o.payload_dir = args.next().context("--payload-dir requires path")?.into(),
            "--received-dir" => o.received_dir = args.next().context("--received-dir requires path")?.into(),
            "--metrics-json" => o.metrics_json = Some(args.next().context("--metrics-json requires path")?.into()),
            "--session-clock-file" => o.session_clock_file = Some(args.next().context("--session-clock-file requires path")?.into()),
            "--start-frame" => o.start_frame = args.next().context("--start-frame requires number")?.parse()?,
            "--frames" => o.frames = args.next().context("--frames requires number")?.parse()?,
            "--fps" => o.fps = args.next().context("--fps requires number")?.parse()?,
            "--initial-mbps" => o.initial_mbps = args.next().context("--initial-mbps requires number")?.parse()?,
            "--feedback-ms" => o.feedback_ms = args.next().context("--feedback-ms requires number")?.parse()?,
            "--playout-ms" => o.playout_ms = args.next().context("--playout-ms requires number")?.parse()?,
            "--utilization" => o.utilization = args.next().context("--utilization requires number")?.parse()?,
            "--inference-ms" => o.inference_ms = args.next().context("--inference-ms requires number")?.parse()?,
            "--render-ms" => o.render_ms = args.next().context("--render-ms requires number")?.parse()?,
            "--decode-ms" => o.decode_ms = args.next().context("--decode-ms requires number")?.parse()?,
            "--scheduling-mode" => o.scheduling_mode = args.next().context("--scheduling-mode requires name")?,
            "--quinn-priority" => o.quinn_priority = args.next().context("--quinn-priority requires policy")?,
            "--tier-selection" => o.tier_selection = args.next().context("--tier-selection requires name")?,
            "--fixed-tiers" => o.fixed_tiers = args.next().context("--fixed-tiers requires number")?.parse()?,
            "--delivery-timeout-ms" => o.delivery_timeout_ms = args.next().context("--delivery-timeout-ms requires number")?.parse()?,
            "--max-inflight" => o.max_inflight = args.next().context("--max-inflight requires number")?.parse()?,
            "--tier-delays-ms" => o.tier_delays_ms =
                parse_tier_delays(&args.next().context("--tier-delays-ms requires P0,P1,P2,P3")?)?,
            _ => bail!("unknown argument: {arg}"),
        }
    }
    if o.frames == 0 || o.start_frame.checked_add(o.frames).is_none()
        || !o.fps.is_finite() || o.fps <= 0.0
        || !o.initial_mbps.is_finite() || o.initial_mbps <= 0.0
        || !o.playout_ms.is_finite() || o.playout_ms < 0.0
        || !o.utilization.is_finite() || !(0.0 < o.utilization && o.utilization <= 1.0)
        || o.feedback_ms < 20 || o.delivery_timeout_ms == 0
        || !matches!(o.scheduling_mode.as_str(), "adaptive" | "sequential_equal" |
            "concurrent_equal" | "concurrent_priority" | "sequential_priority" |
            "p0_first_priority" | "sequential" | "concurrent" | "p0_first" |
            "original" | "amirali_fifo" | "amirali_strict" | "amirali_weighted")
        || !matches!(o.quinn_priority.as_str(), "equal" | "tiered" | "legacy")
        || !matches!(o.tier_selection.as_str(), "full" | "fixed" | "adaptive")
        || !(1..=4).contains(&o.fixed_tiers)
        || !(1..=128).contains(&o.max_inflight)
        || [o.inference_ms, o.render_ms, o.decode_ms].iter().any(|v| !v.is_finite() || *v < 0.0) {
        bail!("invalid scheduler parameters");
    }
    Ok(o)
}

fn log(s: impl AsRef<str>) {
    println!("{}", s.as_ref());
    let _ = io::stdout().flush();
}

// An explicit policy is mandatory in the priority matrix. "legacy" keeps
// earlier standalone scripts' historical Quinn priority behavior intact.
fn native_tiered(mode: &str, policy: &str) -> bool {
    match policy {
        "equal" => false,
        "tiered" => true,
        _ => matches!(mode, "adaptive" | "sequential_priority" |
                     "concurrent_priority" | "p0_first_priority"),
    }
}

fn native_priority_value(tier: usize, tiered: bool) -> i32 {
    if tiered { ((3 - tier) * 10) as i32 } else { 0 }
}

async fn wait_for_tier(frame_start: Instant, delay_ms: u64) {
    if let Some(remaining) = Duration::from_millis(delay_ms).checked_sub(frame_start.elapsed()) {
        tokio::time::sleep(remaining).await;
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Feedback {
    sequence: u64,
    elapsed_ms: f64,
    app_goodput_mbps: f64,
    total_received_bytes: u64,
    completed_ntcs: u64,
    last_completed_ntc: i64,
    missed_deadlines: u64,
    buffer_frames: i64,
    last_applied_ntc: i64,
    playback_deadline_misses: u64,
    receiver_done: bool,
    decode_ms: f64,
    ntc_inference_ms: f64,
    render_ms: f64,
    gpu_timing_source: String,
}

impl Feedback {
    fn initial(o: &Options) -> Self {
        Self {
            sequence: 0, elapsed_ms: 0.0, app_goodput_mbps: 0.0,
            total_received_bytes: 0, completed_ntcs: 0,
            last_completed_ntc: o.start_frame as i64 - 1, missed_deadlines: 0,
            buffer_frames: 0,
            last_applied_ntc: o.start_frame as i64 - 1,
            playback_deadline_misses: 0,
            receiver_done: false,
            decode_ms: o.decode_ms,
            ntc_inference_ms: o.inference_ms, render_ms: o.render_ms,
            gpu_timing_source: "placeholder_not_measured".into(),
        }
    }
}

#[derive(Default)]
struct Counters {
    bytes: AtomicU64,
    completed: AtomicU64,
    last: AtomicI64,
    missed: AtomicU64,
    done: AtomicBool,
    frames: Mutex<HashMap<u32, (u8, u8)>>,
}

struct ChunkAccumulator {
    selected: u8,
    bytes: Vec<u8>,
    seen: Vec<bool>,
    count: usize,
}

type ChunkStates = Arc<Mutex<HashMap<(u32, u8), ChunkAccumulator>>>;

#[tokio::main]
async fn main() -> Result<()> {
    let o = options()?;
    if o.role == "receiver" { run_receiver(&o).await } else { run_sender(&o).await }
}

async fn run_receiver(o: &Options) -> Result<()> {
    let generated = rcgen::generate_simple_self_signed(vec!["localhost".into()])?;
    let cert = generated.cert.der().clone();
    let key = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(
        generated.signing_key.serialize_der(),
    ));
    fs::write(&o.cert, cert.as_ref())?;
    let server = quinn::ServerConfig::with_single_cert(vec![cert], key)?;
    let addr: SocketAddr = format!("{}:{}", o.host, o.port).parse()?;
    let endpoint = Endpoint::server(server, addr)?;
    log(format!("READY {}", endpoint.local_addr()?));
    let connection = endpoint.accept().await.context("no incoming connection")?.await?;
    let started = Instant::now();
    if let Some(path) = &o.session_clock_file {
        let unix_ms = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs_f64() * 1000.0;
        // The live viewer is already preloaded; this defines the receiver's
        // real connection-relative playout epoch in the same host clock.
        let temporary = path.with_extension("pending");
        fs::write(&temporary, serde_json::to_vec(&serde_json::json!({"unix_ms": unix_ms}))?)?;
        fs::rename(&temporary, path)?;
    }
    let counters = Arc::new(Counters::default());
    counters.last.store(o.start_frame as i64 - 1, Ordering::Relaxed);
    let task = tokio::spawn(feedback_loop(connection.clone(), counters.clone(), started, o.clone()));
    let received = receive_all(&connection, counters.clone(), started, o).await;
    counters.done.store(true, Ordering::Relaxed);
    let _ = task.await;
    tokio::time::sleep(Duration::from_millis(100)).await;
    endpoint.close(0u32.into(), b"done");
    endpoint.wait_idle().await;
    received
}

async fn feedback_loop(conn: Connection, c: Arc<Counters>, started: Instant, o: Options) {
    let mut previous_bytes = 0;
    let mut previous_time = Instant::now();
    let mut rate_ewma = 0.0;
    let mut sequence = 0;
    loop {
        tokio::time::sleep(Duration::from_millis(o.feedback_ms)).await;
        sequence += 1;
        let now = Instant::now();
        let bytes = c.bytes.load(Ordering::Relaxed);
        let seconds = (now - previous_time).as_secs_f64().max(1e-3);
        let rate = bytes.saturating_sub(previous_bytes) as f64 * 8.0 / (seconds * 1e6);
        rate_ewma = if sequence == 1 { rate } else { 0.25 * rate + 0.75 * rate_ewma };
        previous_bytes = bytes;
        previous_time = now;
        let mut fb = Feedback::initial(&o);
        fb.sequence = sequence;
        fb.elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        fb.app_goodput_mbps = rate_ewma;
        fb.total_received_bytes = bytes;
        fb.completed_ntcs = c.completed.load(Ordering::Relaxed);
        fb.last_completed_ntc = c.last.load(Ordering::Relaxed);
        fb.missed_deadlines = c.missed.load(Ordering::Relaxed);
        fb.receiver_done = c.done.load(Ordering::Relaxed);
        let expected_played = ((fb.elapsed_ms - o.playout_ms).max(0.0) * o.fps / 1000.0).floor() as i64;
        fb.buffer_frames = (fb.completed_ntcs as i64 - expected_played).max(0);
        if let Some(path) = &o.metrics_json {
            if let Ok(bytes) = fs::read(path) {
                if let Ok(v) = serde_json::from_slice::<serde_json::Value>(&bytes) {
                    let now_unix_ms = SystemTime::now().duration_since(UNIX_EPOCH)
                        .map(|d| d.as_secs_f64() * 1000.0).unwrap_or(0.0);
                    let fresh = v.get("updated_at_unix_ms").and_then(|a| a.as_f64())
                        .map(|t| now_unix_ms >= t && now_unix_ms - t < 500.0)
                        .unwrap_or(false);
                    if fresh && v.get("status").and_then(|s| s.as_str()) == Some("live") {
                    for (key, target) in [
                        ("decode_ms", &mut fb.decode_ms),
                        ("ntc_inference_ms", &mut fb.ntc_inference_ms),
                        ("render_ms", &mut fb.render_ms),
                    ] {
                        if let Some(measured) = v.get(key).and_then(|a| a.as_f64()) {
                            if measured.is_finite() && (0.0..=10000.0).contains(&measured) {
                                *target = measured;
                            }
                        }
                    }
                    if let Some(n) = v.get("buffer_frames").and_then(|a| a.as_i64()) {
                        fb.buffer_frames = n.max(0);
                    }
                    if let Some(n) = v.get("last_applied_ntc").and_then(|a| a.as_i64()) {
                        fb.last_applied_ntc = n;
                    }
                    if let Some(n) = v.get("playback_deadline_misses").and_then(|a| a.as_u64()) {
                        fb.playback_deadline_misses = n;
                    }
                    // Transport and playback deadline misses are distinct.
                    fb.gpu_timing_source = "live_cuda_viewer".into();
                    }
                }
            }
        }
        let final_playout_ms = o.playout_ms + o.frames as f64 * 1000.0 / o.fps;
        let final_report = fb.receiver_done && fb.elapsed_ms >= final_playout_ms;
        if let Ok(mut stream) = conn.open_uni().await {
            if let Ok(json) = serde_json::to_vec(&fb) {
                let _ = stream.write_all(&json).await;
                let _ = stream.finish();
                if final_report {
                    // Let the sender ACK the final receiver-done report before
                    // tearing down the connection. Completion counts alone do
                    // not prove the end-marker stream was consumed.
                    let _ = tokio::time::timeout(Duration::from_secs(5), stream.stopped()).await;
                }
            }
        } else { break; }
        if sequence % 10 == 0 {
            log(format!("CLIENT_FEEDBACK seq={} rate={:.3}Mbps buffer={} completed={} missed={} gpu={} last_applied={} playback_misses={}",
                sequence, rate_ewma, fb.buffer_frames, fb.completed_ntcs,
                fb.missed_deadlines, fb.gpu_timing_source,
                fb.last_applied_ntc, fb.playback_deadline_misses));
        }
        // Keep the feedback channel alive through the final GPU playout tick,
        // even for a 3-NTC smoke test whose transport finishes in ~130 ms.
        if final_report { break; }
    }
}

async fn receive_all(conn: &Connection, c: Arc<Counters>, started: Instant, o: &Options) -> Result<()> {
    let mut tasks = tokio::task::JoinSet::new();
    let chunks: ChunkStates = Arc::new(Mutex::new(HashMap::new()));
    let mut received_eof = false;
    // A low-priority group can arrive AFTER the end-marker stream. An end
    // marker alone is not sufficient to stop accepting streams.
    while !received_eof || c.completed.load(Ordering::Relaxed) < o.frames as u64 {
        tokio::select! {
            stream = conn.accept_uni() => {
                let stream = stream?;
                let cc = c.clone();
                let folder = o.received_dir.clone();
                let fps = o.fps;
                let playout = o.playout_ms;
                let start = o.start_frame;
                let end = start + o.frames;
                let chunk_states = chunks.clone();
                tasks.spawn(async move {
                    receive_one(stream, folder, cc, chunk_states, started, start, end, fps, playout).await
                });
            },
            stream = conn.accept_bi() => {
                let (send, recv) = stream?;
                let cc = c.clone();
                let chunk_states = chunks.clone();
                let folder = o.received_dir.clone();
                let (start, end) = (o.start_frame, o.start_frame + o.frames);
                let (fps, playout) = (o.fps, o.playout_ms);
                tasks.spawn(async move {
                    receive_bi(send, recv, folder, cc, chunk_states,
                               started, start, end, fps, playout).await
                });
            },
            item = tasks.join_next(), if !tasks.is_empty() => {
                // Completion tasks can finish out of order. Once the EOF
                // marker was observed, a later P0..P3 completion must not
                // reset the marker state to false.
                if let Some(done) = item { received_eof |= done??; }
            }
        }
    }
    while let Some(done) = tasks.join_next().await { done??; }
    let remaining = c.frames.lock().await;
    if !remaining.is_empty() { bail!("{} incomplete NTCs at EOF", remaining.len()); }
    if !chunks.lock().await.is_empty() { bail!("incomplete chunks at EOF"); }
    log(format!("ALL_COMPLETE {}/{} missed={} bytes={}",
                c.completed.load(Ordering::Relaxed), o.frames,
                c.missed.load(Ordering::Relaxed), c.bytes.load(Ordering::Relaxed)));
    Ok(())
}

async fn receive_one(mut recv: RecvStream, folder: PathBuf, c: Arc<Counters>, chunks: ChunkStates,
                     started: Instant, start: u32, end: u32, fps: f64,
                     playout: f64) -> Result<bool> {
    let data = recv.read_to_end(MAX_GROUP + HEADER_LEN + CHUNK_META_LEN).await?;
    receive_payload(data, folder, c, chunks, started, start, end, fps, playout).await
}

async fn receive_bi(mut send: SendStream, mut recv: RecvStream, folder: PathBuf,
                    c: Arc<Counters>, chunks: ChunkStates, started: Instant,
                    start: u32, end: u32, fps: f64, playout: f64) -> Result<bool> {
    let data = recv.read_to_end(MAX_GROUP + HEADER_LEN + CHUNK_META_LEN).await?;
    let eof = receive_payload(data, folder, c, chunks, started, start, end, fps, playout).await?;
    // Amirali-style admission frees a slot on a validated application ACK,
    // not merely on write_all or a QUIC transport-level acknowledgment.
    send.write_all(b"OK").await?;
    send.finish()?;
    Ok(eof)
}

async fn receive_payload(data: Vec<u8>, folder: PathBuf, c: Arc<Counters>,
                         chunks: ChunkStates, started: Instant, start: u32,
                         end: u32, fps: f64, playout: f64) -> Result<bool> {
    if data.len() < HEADER_LEN { bail!("short NTC header"); }
    let frame = u32::from_le_bytes(data[..4].try_into()?);
    let raw_tier = data[4];
    let selected = data[5];
    let length = u64::from_le_bytes(data[6..14].try_into()?);
    if frame == u32::MAX && raw_tier == 255 && selected == 0 && length == 0 {
        return Ok(true);
    }
    let is_chunk = raw_tier & CHUNK_FLAG != 0;
    let tier = if is_chunk { raw_tier & !CHUNK_FLAG } else { raw_tier };
    if !(start..end).contains(&frame) || tier >= 4 || selected == 0 ||
        selected > 4 || tier >= selected || length > MAX_GROUP as u64 {
        bail!("bad NTC frame={frame} tier={tier} selected={selected} bytes={length}");
    }
    if is_chunk {
        if data.len() != HEADER_LEN + CHUNK_META_LEN + length as usize ||
           length == 0 || length > CHUNK_SIZE as u64 {
            bail!("invalid chunk frame={frame} tier={tier}");
        }
        let offset = usize::try_from(u64::from_le_bytes(data[14..22].try_into()?))?;
        let total = usize::try_from(u64::from_le_bytes(data[22..30].try_into()?))?;
        if total == 0 || total > MAX_GROUP || offset % CHUNK_SIZE != 0 ||
            offset >= total || length as usize != CHUNK_SIZE.min(total - offset) {
            bail!("invalid chunk range frame={frame} tier={tier}");
        }
        let group = {
            let mut active = chunks.lock().await;
            let entry = active.entry((frame, tier)).or_insert_with(|| ChunkAccumulator {
                selected, bytes: vec![0; total], seen: vec![false; total.div_ceil(CHUNK_SIZE)], count: 0,
            });
            let index = offset / CHUNK_SIZE;
            if entry.selected != selected || entry.bytes.len() != total || entry.seen[index] {
                bail!("conflicting/duplicate chunk frame={frame} tier={tier} offset={offset}");
            }
            entry.bytes[offset..offset + length as usize].copy_from_slice(&data[30..]);
            entry.seen[index] = true;
            entry.count += 1;
            if entry.count == entry.seen.len() {
                active.remove(&(frame, tier)).map(|done| done.bytes)
            } else { None }
        };
        if let Some(whole_group) = group {
            // All chunks of this group have reached the application and been
            // reassembled. Timestamp BEFORE the filesystem commit, so the
            // observed QUIC completion order is separate from disk readiness.
            log(format!("GROUP_NET_READY {frame} tier={tier} t_ms={:.3}",
                        started.elapsed().as_secs_f64() * 1000.0));
            commit_group(folder, c, started, start, fps, playout,
                         frame, tier, selected, &whole_group).await?;
        }
        return Ok(false);
    }
    if data.len() != HEADER_LEN + length as usize {
        bail!("invalid whole-group length frame={frame} tier={tier}");
    }
    log(format!("GROUP_NET_READY {frame} tier={tier} t_ms={:.3}",
                started.elapsed().as_secs_f64() * 1000.0));
    commit_group(folder, c, started, start, fps, playout,
                 frame, tier, selected, &data[HEADER_LEN..]).await?;
    Ok(false)
}

async fn commit_group(folder: PathBuf, c: Arc<Counters>, started: Instant,
                      start: u32, fps: f64, playout: f64,
                      frame: u32, tier: u8, selected: u8, body: &[u8]) -> Result<()> {
    let ntc_folder = folder.join(format!("NTC_{frame:06}"));
    fs::create_dir_all(&ntc_folder)?;
    // Atomic rename: Python must never read a partially written group.
    let final_file = ntc_folder.join(format!("P{tier}.bin"));
    let temp_file = ntc_folder.join(format!("P{tier}.pending"));
    fs::write(&temp_file, body)?;
    fs::rename(&temp_file, &final_file)?;
    c.bytes.fetch_add(body.len() as u64, Ordering::Relaxed);
    log(format!("GROUP_COMPLETE {frame} tier={tier} t_ms={:.3}",
                started.elapsed().as_secs_f64() * 1000.0));
    let ready = {
        let mut states = c.frames.lock().await;
        let entry = states.entry(frame).or_insert((selected, 0));
        if entry.0 != selected || entry.1 & (1 << tier) != 0 {
            bail!("duplicate/conflicting group frame={frame} P{tier}");
        }
        entry.1 |= 1 << tier;
        let complete = entry.1 == (1u8 << selected) - 1;
        if complete { states.remove(&frame); }
        complete
    };
    if ready {
        let elapsed = started.elapsed().as_secs_f64() * 1000.0;
        let deadline = playout + (frame - start) as f64 * 1000.0 / fps;
        let late = elapsed > deadline;
        c.completed.fetch_add(1, Ordering::Relaxed);
        c.last.fetch_max(frame as i64, Ordering::Relaxed);
        if late { c.missed.fetch_add(1, Ordering::Relaxed); }
        log(format!("FRAME_COMPLETE {frame} tiers={selected} t_ms={elapsed:.3} deadline_ms={deadline:.3} late={late}"));
    }
    Ok(())
}

async fn write_ntc_group(mut stream: SendStream, frame: u32, tier: usize,
                         selected: usize, body: &[u8]) -> Result<()> {
    let mut header = [0u8; HEADER_LEN];
    header[..4].copy_from_slice(&frame.to_le_bytes());
    header[4] = tier as u8;
    header[5] = selected as u8;
    header[6..].copy_from_slice(&(body.len() as u64).to_le_bytes());
    stream.write_all(&header).await?;
    stream.write_all(body).await?;
    stream.finish()?;
    Ok(())
}

// Original Q3DGStream behavior: P0..P3 complete stage files in order, with
// an application-level commit acknowledgment before starting the next stage.
async fn submit_original(conn: &Connection, frame: u32, bodies: &[Vec<u8>],
                         selected: usize, started: Instant, tiered: bool,
                         frame_start: Instant, delays: [u64; 4]) -> Result<()> {
    for tier in 0..selected {
        wait_for_tier(frame_start, delays[tier]).await;
        let (send, mut recv) = conn.open_bi().await?;
        let value = native_priority_value(tier, tiered);
        send.set_priority(value)?;
        log(format!("QUINN_PRIORITY frame={frame} tier={tier} value={value}"));
        log(format!("GROUP_WRITE_BEGIN {frame} tier={tier} t_ms={:.3}",
                    started.elapsed().as_secs_f64() * 1000.0));
        write_ntc_group(send, frame, tier, selected, &bodies[tier]).await?;
        log(format!("GROUP_WRITE_END {frame} tier={tier} t_ms={:.3}",
                    started.elapsed().as_secs_f64() * 1000.0));
        let receipt = tokio::time::timeout(Duration::from_secs(120), recv.read_to_end(64)).await??;
        if receipt != b"OK" { bail!("original stage commit was not acknowledged"); }
        log(format!("GROUP_ACKED {frame} tier={tier}"));
    }
    Ok(())
}

#[derive(Clone)]
struct ChunkJob {
    frame: u32,
    tier: usize,
    selected: usize,
    offset: usize,
    group: Arc<Vec<u8>>,
    frame_start: Instant,
    delay_ms: u64,
}

struct AdmissionQueue {
    policy: String,
    fifo: VecDeque<ChunkJob>,
    tiers: [VecDeque<ChunkJob>; 4],
    cycle: Vec<usize>,
    position: usize,
}

impl AdmissionQueue {
    fn new(policy: &str) -> Self {
        let mut cycle = Vec::new();
        for (tier, count) in [8, 4, 2, 1].iter().enumerate() {
            cycle.extend(std::iter::repeat(tier).take(*count));
        }
        Self { policy: policy.into(), fifo: VecDeque::new(),
               tiers: std::array::from_fn(|_| VecDeque::new()), cycle, position: 0 }
    }

    fn push(&mut self, job: ChunkJob) {
        if self.policy == "amirali_fifo" { self.fifo.push_back(job); }
        else { self.tiers[job.tier].push_back(job); }
    }

    fn empty(&self) -> bool {
        self.fifo.is_empty() && self.tiers.iter().all(VecDeque::is_empty)
    }

    fn pop(&mut self) -> Option<ChunkJob> {
        match self.policy.as_str() {
            "amirali_fifo" => self.fifo.pop_front(),
            "amirali_strict" => self.tiers.iter_mut().find_map(VecDeque::pop_front),
            "amirali_weighted" => {
                for _ in 0..self.cycle.len() {
                    let tier = self.cycle[self.position];
                    self.position = (self.position + 1) % self.cycle.len();
                    if let Some(job) = self.tiers[tier].pop_front() { return Some(job); }
                }
                None
            },
            _ => None,
        }
    }
}

async fn send_chunk(conn: Connection, job: ChunkJob, started: Instant,
                    tiered: bool) -> Result<()> {
    wait_for_tier(job.frame_start, job.delay_ms).await;
    let (mut send, mut recv) = conn.open_bi().await?;
    // Every Amirali 8 KiB chunk is an independent QUIC stream. The matrix
    // varies Quinn's native priority independently of application admission.
    let value = native_priority_value(job.tier, tiered);
    send.set_priority(value)?;
    if job.offset == 0 {
        log(format!("QUINN_PRIORITY frame={} tier={} value={value}",
                    job.frame, job.tier));
    }
    let count = CHUNK_SIZE.min(job.group.len() - job.offset);
    let mut header = [0u8; HEADER_LEN + CHUNK_META_LEN];
    header[..4].copy_from_slice(&job.frame.to_le_bytes());
    header[4] = CHUNK_FLAG | job.tier as u8;
    header[5] = job.selected as u8;
    header[6..14].copy_from_slice(&(count as u64).to_le_bytes());
    header[14..22].copy_from_slice(&(job.offset as u64).to_le_bytes());
    header[22..30].copy_from_slice(&(job.group.len() as u64).to_le_bytes());
    send.write_all(&header).await?;
    send.write_all(&job.group[job.offset..job.offset + count]).await?;
    send.finish()?;
    log(format!("CHUNK_SUBMITTED frame={} tier={} offset={} bytes={} t_ms={:.3}",
                job.frame, job.tier, job.offset, count,
                started.elapsed().as_secs_f64() * 1000.0));
    let receipt = tokio::time::timeout(Duration::from_secs(120), recv.read_to_end(64)).await??;
    if receipt != b"OK" { bail!("chunk ACK was not validated"); }
    log(format!("CHUNK_ACKED frame={} tier={} offset={} t_ms={:.3}",
                job.frame, job.tier, job.offset,
                started.elapsed().as_secs_f64() * 1000.0));
    Ok(())
}

async fn admission_loop(conn: Connection, mut incoming: mpsc::UnboundedReceiver<ChunkJob>,
                        mode: String, max_inflight: usize, started: Instant,
                        tiered: bool) -> Result<()> {
    let mut scheduler = AdmissionQueue::new(&mode);
    let mut active = tokio::task::JoinSet::new();
    let mut first_admission: HashSet<(u32, usize)> = HashSet::new();
    let mut producer_closed = false;
    loop {
        while active.len() < max_inflight {
            let Some(job) = scheduler.pop() else { break; };
            if first_admission.insert((job.frame, job.tier)) {
                log(format!("GROUP_WRITE_BEGIN {} tier={} t_ms={:.3}",
                            job.frame, job.tier,
                            started.elapsed().as_secs_f64() * 1000.0));
            }
            active.spawn(send_chunk(conn.clone(), job, started, tiered));
        }
        if producer_closed && scheduler.empty() && active.is_empty() { break; }
        tokio::select! {
            next = incoming.recv(), if !producer_closed => {
                match next {
                    Some(job) => scheduler.push(job),
                    None => producer_closed = true,
                }
            },
            completed = active.join_next(), if !active.is_empty() => {
                if let Some(result) = completed { result??; }
            }
        }
    }
    Ok(())
}

async fn submit_ntc_groups(conn: &Connection, frame: u32, bodies: &[Vec<u8>],
                           selected: usize, mode: &str, started: Instant,
                           tiered: bool, frame_start: Instant,
                           delays: [u64; 4]) -> Result<()> {
    let concurrent = matches!(mode, "concurrent_equal" | "concurrent_priority" |
        "p0_first_priority" | "concurrent" | "p0_first");
    if concurrent {
        // Open every stream and set native Quinn priorities BEFORE any writes.
        // This keeps stream creation identical across both concurrent arms.
        let mut streams = Vec::with_capacity(selected);
        for tier in 0..selected {
            let stream = conn.open_uni().await?;
            let value = native_priority_value(tier, tiered);
            stream.set_priority(value)?;
            log(format!("QUINN_PRIORITY frame={frame} tier={tier} value={value}"));
            streams.push(stream);
        }

        // The third treatment isolates an application submission decision:
        // P0 is accepted into Quinn before any enhancement stream is written.
        // It does NOT wait for P0 delivery or an application ACK.
        let first_tier = if mode == "p0_first_priority" || mode == "p0_first" {
            let p0 = streams.remove(0);
            wait_for_tier(frame_start, delays[0]).await;
            log(format!("GROUP_WRITE_BEGIN {frame} tier=0 t_ms={:.3}",
                        started.elapsed().as_secs_f64() * 1000.0));
            write_ntc_group(p0, frame, 0, selected, &bodies[0]).await?;
            log(format!("GROUP_WRITE_END {frame} tier=0 t_ms={:.3}",
                        started.elapsed().as_secs_f64() * 1000.0));
            1usize
        } else { 0usize };

        // The remaining group writers are released together. The original
        // concurrent arms still release P0..P3 together, as before.
        let gate = Arc::new(Barrier::new(streams.len() + 1));
        let mut tasks = tokio::task::JoinSet::new();
        for (offset, stream) in streams.into_iter().enumerate() {
            let tier = first_tier + offset;
            let body = bodies[tier].clone();
            let delay_ms = delays[tier];
            let gate = gate.clone();
            tasks.spawn(async move {
                gate.wait().await;
                wait_for_tier(frame_start, delay_ms).await;
                log(format!("GROUP_WRITE_BEGIN {frame} tier={tier} t_ms={:.3}",
                            started.elapsed().as_secs_f64() * 1000.0));
                write_ntc_group(stream, frame, tier, selected, &body).await?;
                log(format!("GROUP_WRITE_END {frame} tier={tier} t_ms={:.3}",
                            started.elapsed().as_secs_f64() * 1000.0));
                Ok::<usize, anyhow::Error>(tier)
            });
        }
        gate.wait().await;
        while let Some(result) = tasks.join_next().await {
            let tier = result??;
            log(format!("GROUP_ENQUEUED {frame} tier={tier}"));
        }
    } else {
        for tier in 0..selected {
            wait_for_tier(frame_start, delays[tier]).await;
            let stream = conn.open_uni().await?;
            let value = native_priority_value(tier, tiered);
            stream.set_priority(value)?;
            log(format!("QUINN_PRIORITY frame={frame} tier={tier} value={value}"));
            log(format!("GROUP_WRITE_BEGIN {frame} tier={tier} t_ms={:.3}",
                        started.elapsed().as_secs_f64() * 1000.0));
            write_ntc_group(stream, frame, tier, selected, &bodies[tier]).await?;
            log(format!("GROUP_WRITE_END {frame} tier={tier} t_ms={:.3}",
                        started.elapsed().as_secs_f64() * 1000.0));
            log(format!("GROUP_ENQUEUED {frame} tier={tier}"));
        }
    }
    Ok(())
}

async fn run_sender(o: &Options) -> Result<()> {
    let cert = fs::read(&o.cert)?;
    let mut roots = rustls::RootCertStore::empty();
    roots.add(CertificateDer::from(cert))?;
    let client = quinn::ClientConfig::with_root_certificates(Arc::new(roots))?;
    let mut endpoint = Endpoint::client(SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), 0))?;
    endpoint.set_default_client_config(client);
    let peer: SocketAddr = format!("{}:{}", o.host, o.port).parse()?;
    let conn = endpoint.connect(peer, "localhost")?.await?;
    let started = Instant::now();
    let tiered = native_tiered(&o.scheduling_mode, &o.quinn_priority);
    let is_amirali = o.scheduling_mode.starts_with("amirali_");
    let (chunk_sender, chunk_worker) = if is_amirali {
        let (tx, rx) = mpsc::unbounded_channel::<ChunkJob>();
        let worker_conn = conn.clone();
        let worker_mode = o.scheduling_mode.clone();
        let window = o.max_inflight;
        (Some(tx), Some(tokio::spawn(async move {
            admission_loop(worker_conn, rx, worker_mode, window, started, tiered).await
        })))
    } else { (None, None) };
    // Both endpoints run on the same Linux host in this netns experiment.
    // Wall-clock epochs align sender forecasts with receiver/GL timestamps;
    // elapsed timings themselves remain monotonic.
    let started_unix_ms = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs_f64() * 1000.0;
    log(format!("SENDER_EPOCH unix_ms={started_unix_ms:.3}"));
    let current_fb = Arc::new(Mutex::new(Feedback::initial(o)));
    let reader = tokio::spawn(read_feedback(conn.clone(), current_fb.clone()));
    let mut queued_bytes = 0u64;
    let mut last_loss = 0;
    let mut last_congestion = 0;
    let mut last_sent_packets = 0;
    let mut last_rtt_ms = 0.0;
    let mut observed_rtt_jitter_ms = 0.0;
    let mut tier_counts = [0u64; 5];
    let mut infeasible = 0u64;
    log(format!("SCHEDULING_MODE {} TIER_SELECTION {} FIXED_TIERS {} QUINN_PRIORITY {}",
                o.scheduling_mode, o.tier_selection, o.fixed_tiers,
                if tiered { "tiered" } else { "equal" }));
    for offset in 0..o.frames {
        let target = Duration::from_secs_f64(offset as f64 / o.fps);
        if let Some(sleep) = target.checked_sub(started.elapsed()) {
            tokio::time::sleep(sleep).await;
        }
        let frame = o.start_frame + offset;
        let mut bodies = Vec::with_capacity(4);
        for tier in 0..4 {
            let path = o.payload_dir.join(format!("NTC_{frame:06}/P{tier}.bin"));
            let body = fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
            if body.len() > MAX_GROUP { bail!("NTC {frame} P{tier} payload too large"); }
            bodies.push(body);
        }
        let frame_start = Instant::now();
        let stats = conn.stats();
        let rtt_ms = (stats.path.rtt.as_secs_f64() * 1000.0).max(1.0);
        let cwnd_mbps = stats.path.cwnd as f64 * 8.0 / (rtt_ms * 1000.0);
        let extra_loss = stats.path.lost_packets.saturating_sub(last_loss);
        let extra_congestion = stats.path.congestion_events.saturating_sub(last_congestion);
        let sent_packets = stats.path.sent_packets.saturating_sub(last_sent_packets);
        let loss_fraction = if sent_packets > 0 {
            extra_loss as f64 / sent_packets as f64
        } else { 0.0 };
        if last_rtt_ms > 0.0 {
            observed_rtt_jitter_ms = 0.75 * observed_rtt_jitter_ms +
                0.25 * (rtt_ms - last_rtt_ms).abs();
        }
        last_rtt_ms = rtt_ms;
        last_loss = stats.path.lost_packets;
        last_congestion = stats.path.congestion_events;
        last_sent_packets = stats.path.sent_packets;
        let fb = current_fb.lock().await.clone();
        // Application goodput is only a lower bound during application-limited
        // sending. Use it as an upper bound only under sustained backlog.
        let raw_path_mbps = o.initial_mbps.min(cwnd_mbps.max(0.1));
        let waiting_bytes = queued_bytes.saturating_sub(fb.total_received_bytes);
        let backlog_threshold = raw_path_mbps * 1e6 / (8.0 * o.fps) * 2.0;
        let capacity_mbps = if fb.sequence >= 3 && fb.app_goodput_mbps > 0.0 &&
            waiting_bytes as f64 > backlog_threshold {
            raw_path_mbps.min(fb.app_goodput_mbps * 1.15)
        } else { raw_path_mbps };
        let rate_mbps = (capacity_mbps * o.utilization *
            if extra_loss + extra_congestion > 0 { 0.85 } else { 1.0 }).max(0.1);
        let queue_ms = waiting_bytes as f64 * 8.0 / (rate_mbps * 1000.0);
        let client_ms = fb.decode_ms + fb.ntc_inference_ms + fb.render_ms;
        let gpu_feasible = client_ms <= 1000.0 / o.fps;
        let expected_played = (fb.elapsed_ms - o.playout_ms).max(0.0) * o.fps / 1000.0;
        let viewer_is_behind = fb.gpu_timing_source == "live_cuda_viewer"
            && expected_played.floor() as i64 + o.start_frame as i64 > fb.last_applied_ntc + 1;
        if !gpu_feasible {
            log(format!("GPU_BUDGET_INFEASIBLE frame={frame} client_ms={client_ms:.2} tick_ms={:.2}",
                        1000.0 / o.fps));
        }
        let deadline_ms = o.playout_ms + offset as f64 * 1000.0 / o.fps;
        let now_ms = started.elapsed().as_secs_f64() * 1000.0;
        let bytes_per_tick = rate_mbps * 1e6 / (8.0 * o.fps);
        let mut selected = 0usize;
        let mut size = 0usize;
        if o.scheduling_mode == "adaptive" || o.tier_selection == "adaptive" {
            for (tier, body) in bodies.iter().enumerate() {
                let candidate = size + body.len();
                let arrival_ms = now_ms + queue_ms + candidate as f64 * 8.0 /
                    (rate_mbps * 1000.0) + rtt_ms / 2.0 + client_ms + 2.0 * observed_rtt_jitter_ms;
                if candidate as f64 <= bytes_per_tick && arrival_ms <= deadline_ms &&
                    (!viewer_is_behind || tier == 0) &&
                    (gpu_feasible || tier == 0) {
                    selected = tier + 1;
                    size = candidate;
                } else { break; }
            }
            if selected == 0 {
                // The NTCs are cumulative: never silently omit an update.
                selected = 1;
                size = bodies[0].len();
                infeasible += 1;
                log(format!("DEADLINE_INFEASIBLE frame={frame} P0_bytes={size} tick_budget={bytes_per_tick:.0}"));
            }
        } else {
            // Controlled comparison: force identical tier counts across all
            // scheduling arms for this network condition. The fixed-tier
            // scenario is provisioned independently of measured feedback.
            selected = if o.tier_selection == "fixed" { o.fixed_tiers } else { 4 };
            size = bodies.iter().take(selected).map(Vec::len).sum();
        }
        tier_counts[selected] += 1;
        log(format!("DECISION frame={frame} tiers={selected} bytes={size} capacity_mbps={rate_mbps:.3} measured_goodput_mbps={:.3} cwnd={} rtt_ms={rtt_ms:.2} min_rtt_ms={:.2} jitter_sample_ms={observed_rtt_jitter_ms:.2} backlog_ms={queue_ms:.2} client_ms={client_ms:.2} gpu_source={} deadline_ms={deadline_ms:.2} loss_delta={extra_loss} sent_delta={sent_packets} loss_fraction={loss_fraction:.5} congestion_delta={extra_congestion}",
                    fb.app_goodput_mbps, stats.path.cwnd,
                    stats.path.min_rtt.as_secs_f64() * 1000.0, fb.gpu_timing_source));
        // The predictor evaluator needs the exact bytes awaiting delivery and
        // each *candidate* cumulative tier size, including unsent candidates.
        let stage_sizes = bodies.iter().map(|body| body.len().to_string())
            .collect::<Vec<_>>().join(",");
        log(format!("PREDICT_CONTEXT frame={frame} send_ms={now_ms:.3} outstanding_bytes={waiting_bytes} stage_bytes={stage_sizes} initial_mbps={:.3} utilization={:.3} decode_ms={:.3} inference_ms={:.3} render_ms={:.3} buffer_frames={} last_applied_ntc={}",
                    o.initial_mbps, o.utilization, fb.decode_ms,
                    fb.ntc_inference_ms, fb.render_ms, fb.buffer_frames, fb.last_applied_ntc));
        if let Some(tx) = &chunk_sender {
            for (tier, body) in bodies.into_iter().take(selected).enumerate() {
                let group = Arc::new(body);
                for offset in (0..group.len()).step_by(CHUNK_SIZE) {
                    tx.send(ChunkJob { frame, tier, selected, offset, group: group.clone(),
                                       frame_start, delay_ms: o.tier_delays_ms[tier] })
                        .map_err(|_| anyhow!("chunk worker exited before all NTCs were queued"))?;
                }
            }
        } else if o.scheduling_mode == "original" {
            submit_original(&conn, frame, &bodies, selected, started, tiered,
                            frame_start, o.tier_delays_ms).await?;
        } else {
            submit_ntc_groups(&conn, frame, &bodies, selected, &o.scheduling_mode,
                              started, tiered, frame_start, o.tier_delays_ms).await?;
        }
        queued_bytes += size as u64;
    }
    // Do not send EOF until every chunk has been application-ACKed. This also
    // ensures a final delayed P3 cannot be silently discarded.
    drop(chunk_sender);
    if let Some(worker) = chunk_worker { worker.await??; }
    let mut finish = conn.open_uni().await?;
    let mut header = [0u8; HEADER_LEN];
    header[..4].copy_from_slice(&u32::MAX.to_le_bytes());
    header[4] = 255;
    // Never let the terminator preempt pending P3 payloads.
    finish.set_priority(-1000)?;
    finish.write_all(&header).await?;
    finish.finish()?;
    log(format!("SENDER_FINISH frames={} P0={} P0P1={} P0P1P2={} full={} infeasible={} elapsed_ms={:.3}",
        o.frames, tier_counts[1], tier_counts[2], tier_counts[3], tier_counts[4],
        infeasible, started.elapsed().as_secs_f64() * 1000.0));
    // Do not close the connection while lower-priority or retransmitted data
    // is still being delivered. Final feedback acknowledges all selected
    // application payloads, without ACK-gating every intermediate NTC.
    let wait_started = Instant::now();
    loop {
        let fb = current_fb.lock().await.clone();
        if fb.completed_ntcs >= o.frames as u64 && fb.receiver_done { break; }
        if wait_started.elapsed() > Duration::from_millis(o.delivery_timeout_ms) {
            bail!("receiver did not acknowledge all {} NTCs AND end-marker within final delivery timeout; completed={}, receiver_done={}",
                  o.frames, fb.completed_ntcs, fb.receiver_done);
        }
        tokio::time::sleep(Duration::from_millis(o.feedback_ms)).await;
    }
    // Do not tear down the connection before the receiver has sent live GPU
    // feedback through the final playout deadline.
    let final_playout = Duration::from_secs_f64(
        (o.playout_ms + o.frames as f64 * 1000.0 / o.fps + o.feedback_ms as f64) / 1000.0,
    );
    if let Some(remaining) = final_playout.checked_sub(started.elapsed()) {
        tokio::time::sleep(remaining).await;
    }
    reader.abort();
    endpoint.close(0u32.into(), b"done");
    endpoint.wait_idle().await;
    Ok(())
}

async fn read_feedback(conn: Connection, current: Arc<Mutex<Feedback>>) -> Result<()> {
    loop {
        let mut stream = conn.accept_uni().await?;
        let raw = stream.read_to_end(8192).await?;
        let received: Feedback = serde_json::from_slice(&raw)
            .map_err(|e| anyhow!("bad feedback JSON: {e}"))?;
        let mut fb = current.lock().await;
        if received.sequence > fb.sequence { *fb = received; }
    }
}

#[cfg(test)]
mod admission_tests {
    use super::*;

    #[test]
    fn quinn_priority_is_independent_of_application_admission() {
        assert!(native_tiered("amirali_fifo", "tiered"));
        assert!(native_tiered("original", "tiered"));
        assert!(!native_tiered("concurrent_priority", "equal"));
        assert_eq!(native_priority_value(0, true), 30);
        assert_eq!(native_priority_value(3, true), 0);
        assert_eq!(native_priority_value(0, false), 0);
    }

    fn job(tier: usize) -> ChunkJob {
        ChunkJob { frame: 0, tier, selected: 4, offset: 0,
                   group: Arc::new(vec![17; CHUNK_SIZE]),
                   frame_start: Instant::now(), delay_ms: 0 }
    }

    #[test]
    fn parse_four_tier_release_delays() {
        assert_eq!(parse_tier_delays("0,250,500,750").unwrap(), [0, 250, 500, 750]);
        assert!(parse_tier_delays("0,250,500").is_err());
        assert!(parse_tier_delays("0,-1,0,0").is_err());
    }

    #[test]
    fn fifo_preserves_arrival_order() {
        let mut q = AdmissionQueue::new("amirali_fifo");
        for tier in [3, 2, 0, 1] { q.push(job(tier)); }
        let got: Vec<_> = (0..4).map(|_| q.pop().unwrap().tier).collect();
        assert_eq!(got, vec![3, 2, 0, 1]);
        assert!(q.empty());
    }

    #[test]
    fn strict_selects_highest_priority_pending() {
        let mut q = AdmissionQueue::new("amirali_strict");
        for tier in [3, 1, 0, 2] { q.push(job(tier)); }
        let got: Vec<_> = (0..4).map(|_| q.pop().unwrap().tier).collect();
        assert_eq!(got, vec![0, 1, 2, 3]);
    }

    #[test]
    fn weighted_has_eight_four_two_one_admission_cycle() {
        let mut q = AdmissionQueue::new("amirali_weighted");
        for tier in 0..4 { for _ in 0..16 { q.push(job(tier)); } }
        let got: Vec<_> = (0..15).map(|_| q.pop().unwrap().tier).collect();
        assert_eq!(got, [vec![0; 8], vec![1; 4], vec![2; 2], vec![3]].concat());
    }
}
