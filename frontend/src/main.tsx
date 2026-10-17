import {
  useEffect,
  useRef,
  useState,
  createContext,
  useContext,
  type ReactNode,
} from "react";
import { createRoot } from "react-dom/client";
import "./style.css";

// The tags Fish Audio's own editor offers for Drama 3.
// Sampling defaults that went silent least often in our tests: top-k off as in
// Fish Audio's own demo, and a frame limit that follows the text length.
const DEFAULTS = {
  temperature: 1.0,
  topP: 0.9,
  topK: 0,
  autoLength: true,
  maxTokens: 512,
  partSize: 300,
  context: true,
};
const WORD_TAGS = [
  "whisper",
  "emphasis",
  "soft",
  "fast",
  "slow",
  "stress",
  "forceful",
];
const TONE_TAGS = [
  "angry",
  "sad",
  "embarrassed",
  "emphasis",
  "whispering",
  "soft",
  "breathy",
  "excited",
];
const SOUND_TAGS = [
  "laughing",
  "chuckling",
  "moaning",
  "clear throat",
  "sobbing",
  "crying loudly",
  "sighing",
  "panting",
  "groaning",
  "crowd laughing",
  "background laughter",
  "audience laughing",
  "pause",
  "long pause",
];

type Metrics = Record<string, number | string>;
type Status = {
  state: string;
  stage: string;
  message: string;
  error?: string;
  model_path?: string;
  compatibility: string;
  metrics: Metrics;
  hardware?: {
    available: boolean;
    name?: string | null;
    architecture?: string;
    total_vram_bytes?: number;
    suggested_vram_limit_gib?: number;
  };
  memory_budget?: {
    limit_bytes: number;
    cache_bytes: number;
    mode: string;
  } | null;
  resources?: {
    gpu?: Record<string, Record<string, string>>;
    process_rss_bytes?: number;
    error?: string;
  };
};
type Model = {
  name: string;
  path: string;
  compatibility: string;
  quantization: string;
};
type Voice = { id: string; name: string; text: string; duration_s: number };
type Take = {
  id: string;
  text: string;
  created: number;
  status: string;
  metrics?: Metrics;
  parameters?: { seed?: number };
  error?: string;
};
type Settings = {
  model_directory: string;
  output_directory: string;
  vram_limit_gib: number | null;
};
const paths: Record<string, string> = {
  settings:
    "M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8ZM12 2v3m0 14v3M2 12h3m14 0h3M5 5l2 2m10 10 2 2M5 19l2-2M17 7l2-2",
  close: "m6 6 12 12M6 18 18 6",
  folder: "M3 6h7l2 3h9v11H3V6Z",
};
function Icon({ name, size = 16 }: { name: string; size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d={paths[name]} />
    </svg>
  );
}
async function api<T>(
  path: string,
  method = "GET",
  body?: unknown,
): Promise<T> {
  const form = body instanceof FormData;
  const response = await fetch("/api" + path, {
    method,
    headers: body && !form ? { "Content-Type": "application/json" } : undefined,
    body: body ? (form ? body : JSON.stringify(body)) : undefined,
  });
  const data = await response.json();
  if (!response.ok)
    throw new Error(
      typeof data.detail === "string"
        ? data.detail
        : JSON.stringify(data.detail || data),
    );
  return data as T;
}
const gib = (n: unknown) => (n ? (Number(n) / 1024 ** 3).toFixed(1) : "–");
const seconds = (n: unknown) => (n ? `${Number(n).toFixed(1)} s` : "–");
const ErrorContext = createContext("");
function Modal({
  title,
  close,
  children,
}: {
  title: string;
  close: () => void;
  children: ReactNode;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  const error = useContext(ErrorContext);
  useEffect(() => {
    ref.current?.showModal();
  }, []);
  return (
    <dialog
      ref={ref}
      className="modal"
      onCancel={(e) => {
        e.preventDefault();
        close();
      }}
    >
      <div className="modal-heading">
        <h2>{title}</h2>
        <button className="icon-button" aria-label="Close" onClick={close}>
          <Icon name="close" />
        </button>
      </div>
      {error && (
        <p role="alert" className="modal-error">
          {error}
        </p>
      )}
      {children}
    </dialog>
  );
}
function Waveform({ id }: { id: string }) {
  const [peaks, setPeaks] = useState<number[]>([]);
  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();
    async function decode() {
      const context = new AudioContext();
      try {
        const response = await fetch("/api/audio/" + id, {
          signal: controller.signal,
        });
        const buffer = await context.decodeAudioData(
          await response.arrayBuffer(),
        );
        const samples = buffer.getChannelData(0),
          count = 160,
          stride = Math.max(1, Math.floor(samples.length / count));
        const values = Array.from({ length: count }, (_, i) => {
          let max = 0;
          for (
            let j = i * stride;
            j < Math.min(samples.length, (i + 1) * stride);
            j++
          )
            max = Math.max(max, Math.abs(samples[j]));
          return max;
        });
        const maximum = Math.max(...values, 0.001);
        if (!cancelled) setPeaks(values.map((x) => x / maximum));
      } catch {
        if (!cancelled) setPeaks([]);
      } finally {
        await context.close();
      }
    }
    void decode();
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [id]);
  return (
    <svg
      className="waveform"
      viewBox="0 0 640 48"
      preserveAspectRatio="none"
      aria-label="Waveform of the generated audio"
    >
      {peaks.map((p, i) => (
        <rect
          key={i}
          x={i * 4}
          width="2"
          y={24 - Math.max(0.5, p * 22)}
          height={Math.max(1, p * 44)}
          fill="currentColor"
        />
      ))}
    </svg>
  );
}

function App() {
  const [status, setStatus] = useState<Status>({
    state: "connecting",
    stage: "idle",
    message: "Connecting to backend…",
    compatibility: "unloaded",
    metrics: {},
  });
  const [models, setModels] = useState<Model[]>([]),
    [voices, setVoices] = useState<Voice[]>([]),
    [takes, setTakes] = useState<Take[]>([]);
  const [settings, setSettings] = useState<Settings>({
    model_directory: "models",
    output_directory: "outputs",
    vram_limit_gib: null,
  });
  const [settingsDraft, setSettingsDraft] = useState<Settings>(settings);
  const [model, setModel] = useState(""),
    [text, setText] = useState(
      () =>
        localStorage.getItem("drama-script") ||
        "Hello, this is my first locally generated voice.",
    );
  const [page, setPage] = useState("studio"),
    [modal, setModal] = useState(""),
    [pending, setPending] = useState(false),
    [error, setError] = useState("");
  const [selected, setSelected] = useState(""),
    [autoLength, setAutoLength] = useState(DEFAULTS.autoLength),
    [maxTokens, setMaxTokens] = useState(DEFAULTS.maxTokens),
    [temperature, setTemperature] = useState(DEFAULTS.temperature),
    [topP, setTopP] = useState(DEFAULTS.topP),
    [topK, setTopK] = useState(DEFAULTS.topK),
    [partSize, setPartSize] = useState(DEFAULTS.partSize),
    [context, setContext] = useState(DEFAULTS.context),
    [seed, setSeed] = useState(42);
  const [fixedSeed, setFixedSeed] = useState(false),
    [tagsOpen, setTagsOpen] = useState(false);
  const [assignments, setAssignments] = useState<Record<number, string>>({
    0: "",
    1: "",
  });
  const [voiceName, setVoiceName] = useState(""),
    [voiceText, setVoiceText] = useState(""),
    [voiceFile, setVoiceFile] = useState<File>();
  const [externalPath, setExternalPath] = useState(""),
    [logs, setLogs] = useState("");
  const textarea = useRef<HTMLTextAreaElement>(null),
    latestTake = useRef("");
  const busy = ["loading", "generating", "unloading", "downloading"].includes(
    status.state,
  );
  async function refresh() {
    const [s, m, v, t] = await Promise.all([
      api<Status>("/status"),
      api<Model[]>("/models"),
      api<Voice[]>("/voices"),
      api<Take[]>("/generations"),
    ]);
    setStatus(s);
    setModels(m);
    setVoices(v);
    setTakes(t);
    setModel(
      (old) =>
        old || m.find((x) => x.compatibility !== "unsupported")?.path || "",
    );
    const last = t.find((x) => x.status === "audio_generated_unreviewed");
    if (last && latestTake.current !== last.id) {
      latestTake.current = last.id;
      setSelected(last.id);
    }
  }
  useEffect(() => {
    let active = true;
    const tick = async () => {
      try {
        if (active) await refresh();
      } catch (e) {
        if (active) setError("Can't reach the backend. " + String(e));
      }
    };
    void tick();
    void api<Settings>("/settings")
      .then(setSettings)
      .catch((e) => setError(String(e)));
    const interval = setInterval(() => void tick(), 1500);
    return () => {
      active = false;
      clearInterval(interval);
    };
  }, []);
  useEffect(() => {
    localStorage.setItem("drama-script", text);
  }, [text]);
  async function act(action: () => Promise<unknown>) {
    setPending(true);
    setError("");
    try {
      await action();
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setPending(false);
    }
  }
  function edit(
    make: (before: string, selection: string, after: string) => string[],
  ) {
    const element = textarea.current,
      start = element?.selectionStart ?? text.length,
      end = element?.selectionEnd ?? start,
      [head, tail] = make(
        text.slice(0, start),
        text.slice(start, end),
        text.slice(end),
      );
    setText(head + tail);
    requestAnimationFrame(() => {
      element?.focus();
      element?.setSelectionRange(head.length, head.length);
    });
  }
  // Speaker tags start a new line, point tags get a space on each side;
  // both go in front of a selection instead of replacing it.
  const speaker = (n: number) =>
    edit((before, selection, after) => [
      before +
        (before && !before.endsWith("\n") ? "\n" : "") +
        `<|speaker:${n}|> `,
      selection + after,
    ]);
  const point = (tag: string) =>
    edit((before, selection, after) => [
      before +
        (before && !/\s$/.test(before) ? " " : "") +
        tag +
        (/^\s/.test(selection + after) ? "" : " "),
      selection + after,
    ]);
  const wrap = (tag: string) =>
    edit((before, selection, after) =>
      selection
        ? [before + `<${tag}>${selection}</${tag}>`, after]
        : [before + `<${tag}>`, `</${tag}>` + after],
    );
  const gpu = Object.values(status.resources?.gpu || {})[0] || {};
  const gpuName =
    status.hardware?.name || gpu["Card Series"] || "No ROCm GPU found";
  const monitoringNotice =
    gpu["Monitoring notice"] || status.resources?.error || "";
  const temperatures = Object.entries(gpu).filter(
    ([key, value]) =>
      key.startsWith("Temperature (Sensor ") &&
      value.trim() !== "" &&
      Number.isFinite(Number(value)),
  );
  const temperatureReading =
    temperatures.find(([key]) => key.includes("junction")) ||
    temperatures.find(([key]) => key.includes("edge")) ||
    temperatures[0];
  const suggestedLimit = status.hardware?.suggested_vram_limit_gib;
  const totalVramGiB = (status.hardware?.total_vram_bytes || 0) / 1024 ** 3;
  const memoryUsed = Number(gpu["VRAM Total Used Memory (B)"] || 0),
    memoryTotal = Number(gpu["VRAM Total Memory (B)"] || 0);
  const current = takes.find(
    (x) => x.id === selected && x.status === "audio_generated_unreviewed",
  );
  const currentModel = models.find(
    (m) => m.path === (status.model_path || model),
  );
  const takeMetrics = current?.metrics;
  const labels: Record<string, string> = {
    connecting: "Connecting",
    unloaded: "No model loaded",
    ready: "Ready",
    loading: "Loading model",
    generating: "Generating",
    unloading: "Unloading",
    downloading: "Downloading",
    error: "Error",
  };
  function resetSampling() {
    setTemperature(DEFAULTS.temperature);
    setTopP(DEFAULTS.topP);
    setTopK(DEFAULTS.topK);
    setAutoLength(DEFAULTS.autoLength);
    setMaxTokens(DEFAULTS.maxTokens);
    setPartSize(DEFAULTS.partSize);
    setContext(DEFAULTS.context);
  }
  async function generate() {
    await api("/generate", "POST", {
      text,
      max_new_tokens: autoLength ? null : maxTokens,
      temperature,
      top_p: topP,
      top_k: topK,
      part_size: partSize,
      context,
      seed: fixedSeed ? seed : crypto.getRandomValues(new Uint32Array(1))[0],
      voices: Object.entries(assignments)
        .filter(([, id]) => id)
        .map(([speaker, id]) => ({ id, speaker: Number(speaker) })),
    });
  }
  return (
    <ErrorContext.Provider value={error || status.error || ""}>
      <header className="topbar">
        <span className="name">Red Curtain</span>
        <nav className="tabs" aria-label="Pages">
          <button
            className={page === "studio" ? "active" : ""}
            onClick={() => setPage("studio")}
          >
            Studio
          </button>
          <button
            className={page === "voices" ? "active" : ""}
            onClick={() => setPage("voices")}
          >
            Voices <span className="num">{voices.length}</span>
          </button>
        </nav>
        <span
          className={"gpu " + (status.hardware?.available ? "" : "missing")}
          title={status.hardware?.architecture}
        >
          {gpuName}
        </span>
        <button
          className="icon-button"
          aria-label="Settings"
          title="Settings"
          onClick={() => {
            setSettingsDraft({ ...settings });
            setModal("settings");
          }}
        >
          <Icon name="settings" />
        </button>
      </header>
      <main>
        {monitoringNotice && (
          <p className="notice" role="status">
            {monitoringNotice}
          </p>
        )}
        {(error || status.error) && (
          <div className="error-banner" role="alert">
            <span>{error || status.error}</span>
            <button
              className="icon-button"
              aria-label="Dismiss"
              onClick={() => setError("")}
            >
              <Icon name="close" size={14} />
            </button>
          </div>
        )}
        {page === "studio" ? (
          <>
            <section className="model-row" aria-label="Model">
              <label htmlFor="model">Model</label>
              <select
                id="model"
                value={model}
                disabled={busy || status.state === "ready"}
                onChange={(e) => setModel(e.target.value)}
              >
                {!models.length && <option value="">No model found</option>}
                {models.map((m) => (
                  <option
                    key={m.path}
                    value={m.path}
                    disabled={m.compatibility === "unsupported"}
                  >
                    {m.name}
                    {m.compatibility === "unsupported" ? " (unsupported)" : ""}
                  </option>
                ))}
                {model && !models.some((m) => m.path === model) && (
                  <option value={model}>{model.split("/").pop()}</option>
                )}
              </select>
              <button
                className="icon-button"
                title="Use a model folder somewhere else"
                aria-label="Choose model folder"
                disabled={busy || status.state === "ready"}
                onClick={() => setModal("path")}
              >
                <Icon name="folder" />
              </button>
              <span className={"state state-" + status.state}>
                {labels[status.state] || status.state}
                {currentModel?.compatibility === "experimental_unverified" &&
                  status.state !== "connecting" && (
                    <span className="hint"> · untested on this GPU</span>
                  )}
              </span>
              {["ready", "error"].includes(status.state) ? (
                <button
                  className="button"
                  disabled={pending}
                  onClick={() => void act(() => api("/model/unload", "POST"))}
                >
                  {status.state === "error" ? "Reset" : "Unload"}
                </button>
              ) : (
                <button
                  className="button"
                  disabled={
                    busy ||
                    pending ||
                    !model ||
                    currentModel?.compatibility === "unsupported"
                  }
                  onClick={() =>
                    void act(() =>
                      api("/model/load", "POST", {
                        path: model,
                        vram_limit_gib: settings.vram_limit_gib,
                      }),
                    )
                  }
                >
                  Load
                </button>
              )}
            </section>
            {busy && (
              <section className="progress" role="status">
                <div className="bar" />
                <span>{status.message}</span>
              </section>
            )}
            <div className="studio-grid">
              <section className="script">
                <div className="section-head">
                  <label htmlFor="script">Script</label>
                  <span className="num">{text.length} / 10000</span>
                </div>
                <textarea
                  id="script"
                  ref={textarea}
                  value={text}
                  maxLength={10000}
                  spellCheck={false}
                  onChange={(e) => setText(e.target.value)}
                />
                <div className="script-foot">
                  <div className="tags" aria-label="Insert tag">
                    <button onClick={() => speaker(0)}>speaker 0</button>
                    <button onClick={() => speaker(1)}>speaker 1</button>
                    <button
                      className={tagsOpen ? "open" : ""}
                      aria-expanded={tagsOpen}
                      onClick={() => setTagsOpen(!tagsOpen)}
                    >
                      tags {tagsOpen ? "▴" : "▾"}
                    </button>
                  </div>
                  {busy ? (
                    <button
                      className="button"
                      disabled={pending || status.state === "unloading"}
                      onClick={() => void act(() => api("/cancel", "POST"))}
                    >
                      Cancel
                    </button>
                  ) : (
                    <button
                      className="button primary"
                      disabled={
                        pending || status.state !== "ready" || !text.trim()
                      }
                      onClick={() => void act(generate)}
                    >
                      Generate
                    </button>
                  )}
                </div>
                {tagsOpen && (
                  <div className="tag-panel">
                    <div className="tag-group">
                      <span>Words</span>
                      <div className="tags">
                        {WORD_TAGS.map((tag) => (
                          <button key={tag} onClick={() => wrap(tag)}>
                            {tag}
                          </button>
                        ))}
                      </div>
                    </div>
                    <div className="tag-group">
                      <span>Tone</span>
                      <div className="tags">
                        {TONE_TAGS.map((tag) => (
                          <button key={tag} onClick={() => point(`[${tag}]`)}>
                            [{tag}]
                          </button>
                        ))}
                      </div>
                    </div>
                    <div className="tag-group">
                      <span>Sounds</span>
                      <div className="tags">
                        {SOUND_TAGS.map((tag) => (
                          <button key={tag} onClick={() => point(`[${tag}]`)}>
                            [{tag}]
                          </button>
                        ))}
                      </div>
                    </div>
                    <p className="hint">
                      Word tags wrap the selected words. Put sounds after the
                      first words of a line, not right at the start, or the
                      take can go silent. Free directions work too, like [tired,
                      end of a long shift].
                    </p>
                  </div>
                )}
              </section>
              <aside className="side">
                <section>
                  <div className="section-head">
                    <span className="label">Voices</span>
                    <button className="link" onClick={() => setModal("voice")}>
                      Add
                    </button>
                  </div>
                  {[0, 1].map((s) => (
                    <label className="row" key={s}>
                      <span>Speaker {s}</span>
                      <select
                        value={assignments[s] || ""}
                        onChange={(e) =>
                          setAssignments({
                            ...assignments,
                            [s]: e.target.value,
                          })
                        }
                      >
                        <option value="">Model picks</option>
                        {voices.map((v) => (
                          <option key={v.id} value={v.id}>
                            {v.name}
                          </option>
                        ))}
                      </select>
                    </label>
                  ))}
                </section>
                <section>
                  <div className="section-head">
                    <span className="label">Sampling</span>
                    <button className="link" onClick={resetSampling}>
                      Defaults
                    </button>
                  </div>
                  <label className="slider">
                    <span>
                      Temperature
                      <span className="num">{temperature.toFixed(2)}</span>
                    </span>
                    <input
                      type="range"
                      min="0.1"
                      max="1.5"
                      step="0.05"
                      value={temperature}
                      onChange={(e) => setTemperature(+e.target.value)}
                    />
                  </label>
                  <label className="slider">
                    <span>
                      Top-p<span className="num">{topP.toFixed(2)}</span>
                    </span>
                    <input
                      type="range"
                      min="0.1"
                      max="1"
                      step="0.05"
                      value={topP}
                      onChange={(e) => setTopP(+e.target.value)}
                    />
                  </label>
                  <label className="row">
                    <span>Top-k</span>
                    <input
                      type="number"
                      min="0"
                      max="4096"
                      value={topK}
                      onChange={(e) => setTopK(+e.target.value)}
                    />
                  </label>
                  <p className="hint">
                    {topK ? `Only the ${topK} likeliest tokens.` : "Off."} With
                    top-k 30 takes go silent a lot more often.
                  </p>
                  <label className="row">
                    <span className="check">
                      <input
                        type="checkbox"
                        checked={!autoLength}
                        onChange={(e) => setAutoLength(!e.target.checked)}
                      />
                      Max frames per part
                    </span>
                    <input
                      type="number"
                      min="1"
                      max="1024"
                      value={maxTokens}
                      disabled={autoLength}
                      onChange={(e) => setMaxTokens(+e.target.value)}
                    />
                  </label>
                  <p className="hint">
                    {autoLength
                      ? "Auto: the limit follows the length of the text, " +
                        "so a take that goes silent stops early."
                      : `≈ ${((maxTokens * 2048) / 44100).toFixed(1)} s ` +
                        "max per part."}
                  </p>
                  <label className="slider">
                    <span>
                      Part size<span className="num">{partSize}</span>
                    </span>
                    <input
                      type="range"
                      min="100"
                      max="300"
                      step="10"
                      value={partSize}
                      onChange={(e) => setPartSize(+e.target.value)}
                    />
                  </label>
                  <label className="row">
                    <span className="check">
                      <input
                        type="checkbox"
                        checked={context}
                        onChange={(e) => setContext(e.target.checked)}
                      />
                      Earlier parts as context
                    </span>
                  </label>
                  <p className="hint">
                    Long text is split into parts of about this many characters.
                    With context on, each part hears the first and the last one,
                    so voices stay the same.
                  </p>
                  <label className="row">
                    <span className="check">
                      <input
                        type="checkbox"
                        checked={fixedSeed}
                        onChange={(e) => setFixedSeed(e.target.checked)}
                      />
                      Fixed seed
                    </span>
                    <input
                      type="number"
                      min="0"
                      max="4294967295"
                      value={seed}
                      disabled={!fixedSeed}
                      onChange={(e) => setSeed(+e.target.value)}
                    />
                  </label>
                </section>
              </aside>
            </div>
            <section className="output">
              <div className="section-head">
                <span className="label">Output</span>
              </div>
              {current ? (
                <div className="take">
                  <div className="take-head">
                    <span className="take-text">
                      {current.text.replace(/<\|speaker:\d+\|>/g, "").trim()}
                    </span>
                    <a
                      className="button"
                      href={"/api/audio/" + current.id}
                      download
                    >
                      Download WAV
                    </a>
                  </div>
                  <Waveform id={current.id} />
                  <audio
                    key={current.id}
                    controls
                    preload="metadata"
                    src={"/api/audio/" + current.id}
                  />
                  {takeMetrics?.audio_warning && (
                    <p className="warning-box" role="status">
                      {takeMetrics.audio_warning}
                    </p>
                  )}
                  <dl className="stats">
                    <div>
                      <dt>Length</dt>
                      <dd>{seconds(takeMetrics?.duration_s)}</dd>
                    </div>
                    <div>
                      <dt>Took</dt>
                      <dd>{seconds(takeMetrics?.generation_seconds)}</dd>
                    </div>
                    <div>
                      <dt>Peak VRAM</dt>
                      <dd>{gib(takeMetrics?.peak_gpu_allocated_bytes)} GiB</dd>
                    </div>
                    <div>
                      <dt>Frames</dt>
                      <dd>{takeMetrics?.frames ?? "–"}</dd>
                    </div>
                    {current.parameters?.seed !== undefined && (
                      <div>
                        <dt>Seed</dt>
                        <dd>{current.parameters.seed}</dd>
                      </div>
                    )}
                    <div>
                      <dt>Format</dt>
                      <dd>
                        {Number(takeMetrics?.sample_rate) / 1000 || 44.1} kHz,
                        16-bit
                      </dd>
                    </div>
                  </dl>
                  {takeMetrics?.finish_reason === "length" && (
                    <p className="warning-box">
                      Hit the frame limit, a part is probably cut off.
                    </p>
                  )}
                </div>
              ) : (
                <p className="empty">Nothing generated yet.</p>
              )}
            </section>
            {!!takes.length && (
              <section className="history">
                <div className="section-head">
                  <span className="label">History</span>
                  <span className="num">{takes.length}</span>
                </div>
                <ol>
                  {takes.slice(0, 10).map((t) => (
                    <li key={t.id}>
                      <button
                        className={selected === t.id ? "selected" : ""}
                        disabled={t.status === "failed"}
                        onClick={() => setSelected(t.id)}
                      >
                        <span className="time num">
                          {new Date(t.created * 1000).toLocaleTimeString([], {
                            hour: "2-digit",
                            minute: "2-digit",
                          })}
                        </span>
                        <span className="history-text">
                          {t.text.replace(/<\|speaker:\d+\|>/g, "").trim()}
                        </span>
                        <span
                          className={
                            "num " + (t.status === "failed" ? "failed" : "")
                          }
                        >
                          {t.status === "failed"
                            ? t.error || "failed"
                            : seconds(t.metrics?.duration_s)}
                        </span>
                      </button>
                    </li>
                  ))}
                </ol>
              </section>
            )}
          </>
        ) : (
          <section className="voice-list">
            <div className="section-head">
              <span className="label">Reference voices</span>
              <button className="button" onClick={() => setModal("voice")}>
                Add voice
              </button>
            </div>
            {voices.length ? (
              <ul>
                {voices.map((v) => (
                  <li key={v.id}>
                    <div>
                      <strong>{v.name}</strong>
                      <span className="num">{seconds(v.duration_s)}</span>
                    </div>
                    <p>{v.text}</p>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="empty">
                No voices yet. Add a short WAV or FLAC with its transcript.
              </p>
            )}
          </section>
        )}
      </main>
      <footer className="statusbar">
        <span>
          VRAM{" "}
          <span className="num">
            {gib(memoryUsed)} / {gib(memoryTotal)} GiB
          </span>
        </span>
        <span>
          RAM{" "}
          <span className="num">
            {gib(status.resources?.process_rss_bytes)} GiB
          </span>
        </span>
        <span>
          GPU{" "}
          <span className="num">
            {gpu["GPU use (%)"] ? gpu["GPU use (%)"] + " %" : "–"}
          </span>
        </span>
        <span title={temperatureReading?.[0]}>
          <span className="num">
            {temperatureReading ? temperatureReading[1] + " °C" : "– °C"}
          </span>
        </span>
        <span className="credit">
          Built with{" "}
          <a href="https://fish.audio" target="_blank" rel="noreferrer">
            Fish Audio
          </a>{" "}
          · not affiliated with Fish Audio
        </span>
      </footer>
      {modal === "path" && (
        <Modal title="Model folder" close={() => setModal("")}>
          <p>
            Folder with config.json, the safetensors files and codec.pth.
          </p>
          <label className="field">
            Path
            <input
              autoFocus
              value={externalPath}
              onChange={(e) => setExternalPath(e.target.value)}
              placeholder="/path/to/drama-3"
            />
          </label>
          <button
            className="button primary full"
            disabled={!externalPath.trim()}
            onClick={() => {
              setModel(externalPath);
              setModal("");
            }}
          >
            Use folder
          </button>
        </Modal>
      )}
      {modal === "voice" && (
        <Modal title="Add voice" close={() => setModal("")}>
          <p>
            WAV or FLAC, 0.2 to 15 seconds, max 20 MiB. Only use recordings of
            people who agreed to it.
          </p>
          <form
            onSubmit={(e) => {
              e.preventDefault();
              void act(async () => {
                if (!voiceFile) throw new Error("Pick a recording first.");
                const data = new FormData();
                data.set("name", voiceName);
                data.set("text", voiceText);
                data.set("audio", voiceFile);
                await api("/voices", "POST", data);
                setModal("");
                setVoiceName("");
                setVoiceText("");
                setVoiceFile(undefined);
              });
            }}
          >
            <label className="field">
              Name
              <input
                required
                maxLength={80}
                value={voiceName}
                onChange={(e) => setVoiceName(e.target.value)}
              />
            </label>
            <label className="field">
              Recording
              <input
                required
                type="file"
                accept=".wav,.flac,audio/wav,audio/flac"
                onChange={(e) => setVoiceFile(e.target.files?.[0])}
              />
            </label>
            <label className="field">
              Transcript (exactly what is said)
              <textarea
                required
                maxLength={2000}
                value={voiceText}
                onChange={(e) => setVoiceText(e.target.value)}
              />
            </label>
            <button className="button primary full" disabled={pending}>
              Save
            </button>
          </form>
        </Modal>
      )}
      {modal === "settings" && (
        <Modal title="Settings" close={() => setModal("")}>
          {status.state !== "unloaded" && (
            <p>Unload the model to change these.</p>
          )}
          <form
            onSubmit={(e) => {
              e.preventDefault();
              void act(async () => {
                const value = await api<Settings>(
                  "/settings",
                  "PUT",
                  settingsDraft,
                );
                setSettings(value);
                setModel("");
                setModal("");
              });
            }}
          >
            <fieldset disabled={status.state !== "unloaded" || pending}>
              <label className="field">
                Model directory
                <input
                  value={settingsDraft.model_directory}
                  onChange={(e) =>
                    setSettingsDraft({
                      ...settingsDraft,
                      model_directory: e.target.value,
                    })
                  }
                />
              </label>
              <label className="field">
                Output directory
                <input
                  value={settingsDraft.output_directory}
                  onChange={(e) =>
                    setSettingsDraft({
                      ...settingsDraft,
                      output_directory: e.target.value,
                    })
                  }
                />
              </label>
              <label className="check">
                <input
                  type="checkbox"
                  checked={settingsDraft.vram_limit_gib === null}
                  onChange={(e) =>
                    setSettingsDraft({
                      ...settingsDraft,
                      vram_limit_gib: e.target.checked
                        ? null
                        : Math.max(
                            1,
                            Math.floor(
                              (suggestedLimit || totalVramGiB * 0.9 || 1) * 2,
                            ) / 2,
                          ),
                    })
                  }
                />
                Pick VRAM limit automatically
              </label>
              <label className="field">
                VRAM limit (GiB)
                <input
                  type="number"
                  min="1"
                  max={totalVramGiB || undefined}
                  step="0.5"
                  disabled={settingsDraft.vram_limit_gib === null}
                  value={
                    settingsDraft.vram_limit_gib ??
                    (suggestedLimit === undefined
                      ? ""
                      : Math.floor(suggestedLimit * 2) / 2)
                  }
                  onChange={(e) =>
                    setSettingsDraft({
                      ...settingsDraft,
                      vram_limit_gib: +e.target.value,
                    })
                  }
                />
              </label>
              <p className="hint">
                Auto leaves some VRAM for the desktop and other apps. Whatever
                is left after the fixed weights becomes expert cache. If it
                doesn't fit, loading stops before anything big is allocated.
              </p>
              {status.memory_budget && (
                <p className="hint num">
                  Current: {gib(status.memory_budget.limit_bytes)} GiB limit,{" "}
                  {gib(status.memory_budget.cache_bytes)} GiB expert cache
                </p>
              )}
              <button className="button primary full">Save</button>
            </fieldset>
          </form>
          <div className="settings-links">
            <button
              className="link"
              onClick={() =>
                void act(async () => {
                  const data = await api<{ text: string }>("/logs");
                  setLogs(data.text);
                  setModal("logs");
                })
              }
            >
              Show logs
            </button>
            <button
              className="link"
              disabled={busy || status.state === "ready" || pending}
              onClick={() => setModal("download")}
            >
              Download Drama 3
            </button>
          </div>
        </Modal>
      )}
      {modal === "logs" && (
        <Modal title="Logs" close={() => setModal("")}>
          <pre className="log-view">{logs || "Empty."}</pre>
        </Modal>
      )}
      {modal === "download" && (
        <Modal title="Download Drama 3" close={() => setModal("")}>
          <p>
            About 41.5 GB from huggingface.co/fishaudio/drama-3 into your model
            directory. By downloading you accept the Fish Audio Community
            License. The INT4 version doesn't work here.
          </p>
          <button
            className="button primary full"
            disabled={pending}
            onClick={() =>
              void act(async () => {
                await api("/models/download", "POST", {
                  confirm_download: true,
                });
                setModal("");
              })
            }
          >
            Download 41.5 GB
          </button>
        </Modal>
      )}
    </ErrorContext.Provider>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
