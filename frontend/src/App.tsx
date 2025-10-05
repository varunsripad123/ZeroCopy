import { FormEvent, useMemo, useState } from "react";
import axios from "axios";
import { clsx } from "clsx";

declare const __APP_VERSION__: string;

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://localhost:8080").replace(/\/$/, "");

interface ChunkEntry {
  chunk_id: string;
  start_ts: number;
  end_ts: number;
  chunk_path: string;
  embedding_path?: string;
  metadata?: Record<string, unknown>;
  chunk_url?: string;
  source_video?: string;
}

interface QueryResult extends ChunkEntry {
  score: number;
  source_video: string;
}

interface CompressionResponsePayload {
  chunk_count: number;
  entries: ChunkEntry[];
}

interface QueryResponsePayload {
  results: QueryResult[];
}

interface MetadataField {
  id: string;
  key: string;
  value: string;
}

const createMetadataField = (): MetadataField => ({
  id: typeof crypto !== "undefined" && typeof crypto.randomUUID === "function"
    ? crypto.randomUUID()
    : Math.random().toString(36).slice(2),
  key: "",
  value: ""
});

const toAbsoluteUrl = (path?: string): string | null => {
  if (!path) {
    return null;
  }
  try {
    return new URL(path, `${API_BASE_URL}/`).toString();
  } catch (error) {
    console.warn("Unable to build absolute URL", error);
    return path;
  }
};

const formatSeconds = (value: number): string => {
  const minutes = Math.floor(value / 60);
  const seconds = Math.round(value % 60);
  return minutes > 0 ? `${minutes}m ${seconds}s` : `${seconds}s`;
};

export default function App(): JSX.Element {
  const [segmentLength, setSegmentLength] = useState(5);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [metadataFields, setMetadataFields] = useState<MetadataField[]>([createMetadataField()]);
  const [compressing, setCompressing] = useState(false);
  const [compressionEntries, setCompressionEntries] = useState<ChunkEntry[]>([]);
  const [compressionMessage, setCompressionMessage] = useState<string | null>(null);
  const [compressionError, setCompressionError] = useState<string | null>(null);

  const [queryText, setQueryText] = useState("");
  const [topK, setTopK] = useState(5);
  const [querying, setQuerying] = useState(false);
  const [queryError, setQueryError] = useState<string | null>(null);
  const [queryResults, setQueryResults] = useState<QueryResult[]>([]);

  const [activeChunkId, setActiveChunkId] = useState<string | null>(null);
  const activePreviewUrl = useMemo(() => {
    const chunk = [...compressionEntries, ...queryResults].find((entry) => entry.chunk_id === activeChunkId);
    return toAbsoluteUrl(chunk?.chunk_url);
  }, [activeChunkId, compressionEntries, queryResults]);

  const metadataPayload = useMemo(() => {
    const payload: Record<string, string> = {};
    metadataFields.forEach((field) => {
      const key = field.key.trim();
      if (key) {
        payload[key] = field.value;
      }
    });
    return payload;
  }, [metadataFields]);

  const handleCompress = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!videoFile) {
      setCompressionError("Please select a video file to upload.");
      return;
    }
    setCompressing(true);
    setCompressionMessage("Uploading and compressing your video...");
    setCompressionError(null);

    try {
      const formData = new FormData();
      formData.append("file", videoFile);
      formData.append("segment_length", String(segmentLength));
      if (Object.keys(metadataPayload).length > 0) {
        formData.append("metadata", JSON.stringify(metadataPayload));
      }

      const response = await axios.post<CompressionResponsePayload>(`${API_BASE_URL}/compress/upload`, formData, {
        headers: { "Content-Type": "multipart/form-data" }
      });

      setCompressionEntries(response.data.entries);
      setCompressionMessage(`Stored ${response.data.chunk_count} chunk(s). Ready to search!`);
      if (response.data.entries.length > 0) {
        setActiveChunkId(response.data.entries[0].chunk_id);
      }
    } catch (error) {
      console.error("Compression failed", error);
      const message = axios.isAxiosError(error) ? error.response?.data?.detail || error.message : "Compression failed";
      setCompressionError(message);
      setCompressionEntries([]);
    } finally {
      setCompressing(false);
    }
  };

  const handleQuery = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!queryText.trim()) {
      setQueryError("Enter a description to search your latent space.");
      return;
    }
    setQuerying(true);
    setQueryError(null);

    try {
      const response = await axios.post<QueryResponsePayload>(`${API_BASE_URL}/query`, {
        query: queryText,
        top_k: topK
      });
      setQueryResults(response.data.results);
      if (response.data.results.length > 0) {
        setActiveChunkId(response.data.results[0].chunk_id);
      }
    } catch (error) {
      console.error("Query failed", error);
      const message = axios.isAxiosError(error) ? error.response?.data?.detail || error.message : "Query failed";
      setQueryError(message);
      setQueryResults([]);
    } finally {
      setQuerying(false);
    }
  };

  const updateMetadataField = (id: string, changes: Partial<MetadataField>) => {
    setMetadataFields((fields) => fields.map((field) => (field.id === id ? { ...field, ...changes } : field)));
  };

  const addMetadataField = () => {
    setMetadataFields((fields) => [...fields, createMetadataField()]);
  };

  const removeMetadataField = (id: string) => {
    setMetadataFields((fields) => (fields.length > 1 ? fields.filter((field) => field.id !== id) : fields));
  };

  return (
    <div className="min-h-screen pb-16 text-slate-100">
      <header className="mx-auto flex w-full max-w-7xl flex-col gap-4 px-6 pt-12 pb-8">
        <div className="flex flex-col gap-3">
          <span className="inline-flex w-fit items-center gap-2 rounded-full border border-slate-700 bg-slate-900/60 px-3 py-1 text-xs uppercase tracking-[0.2em] text-slate-300">
            Zero-Copy AI · Console v{__APP_VERSION__}
          </span>
          <h1 className="text-4xl font-semibold text-slate-50 sm:text-5xl">
            Manage, search, and decode compressed video intelligence.
          </h1>
          <p className="max-w-3xl text-lg text-slate-300">
            Upload surveillance or operations footage, compress it into semantic latents, and query with natural language in real time. Every chunk is streamable without rebuilding the original video.
          </p>
        </div>
      </header>

      <main className="mx-auto grid w-full max-w-7xl gap-8 px-6 md:grid-cols-[3fr,2fr]">
        <section className="rounded-3xl border border-slate-800/60 bg-slate-900/60 p-6 shadow-[0_20px_60px_-30px_rgba(15,23,42,0.9)] backdrop-blur">
          <div className="flex items-center justify-between gap-4">
            <h2 className="text-xl font-semibold text-white">Ingestion & Compression</h2>
            <span className="rounded-full bg-emerald-500/20 px-3 py-1 text-xs font-medium text-emerald-300">
              {compressionEntries.length} chunks stored
            </span>
          </div>

          <form className="mt-6 flex flex-col gap-6" onSubmit={handleCompress}>
            <label className="flex flex-col gap-2">
              <span className="text-sm font-medium text-slate-300">Video asset</span>
              <div className="group relative flex flex-col items-center justify-center gap-3 rounded-2xl border border-dashed border-slate-700 bg-slate-950/40 px-6 py-10 transition hover:border-brand">
                <input
                  required
                  type="file"
                  accept="video/*"
                  className="absolute inset-0 cursor-pointer opacity-0"
                  onChange={(event) => setVideoFile(event.target.files?.[0] || null)}
                />
                <div className="flex h-16 w-16 items-center justify-center rounded-full bg-brand/10 text-brand">
                  <svg viewBox="0 0 24 24" fill="currentColor" className="h-8 w-8">
                    <path d="M12 16a1 1 0 0 1-.894-.553l-4-8A1 1 0 0 1 8 6h8a1 1 0 0 1 .894 1.447l-4 8A1 1 0 0 1 12 16Zm0-2.618L14.764 8H9.236Z" />
                    <path d="M5 19a1 1 0 0 1-1-1v-2.382a1 1 0 0 1 .276-.685l2-2.118a1 1 0 0 1 1.448 1.382L6 15.382V18h12v-2.618l-1.724-1.821a1 1 0 0 1 1.448-1.382l2 2.118A1 1 0 0 1 20 15.618V18a1 1 0 0 1-1 1Z" />
                  </svg>
                </div>
                <div className="text-center">
                  <p className="text-sm font-semibold text-slate-200">
                    {videoFile ? videoFile.name : "Drop a file or browse your library"}
                  </p>
                  <p className="text-xs text-slate-400">MP4, MOV or MKV up to your server limits.</p>
                </div>
              </div>
            </label>

            <div className="grid gap-4 sm:grid-cols-2">
              <label className="flex flex-col gap-2">
                <span className="text-sm font-medium text-slate-300">Segment length</span>
                <div className="rounded-2xl border border-slate-800 bg-slate-950/70 px-4 py-3">
                  <input
                    type="range"
                    min={0.5}
                    max={30}
                    step={0.5}
                    value={segmentLength}
                    onChange={(event) => setSegmentLength(Number(event.target.value))}
                    className="h-2 w-full cursor-pointer appearance-none rounded-full bg-slate-800 accent-brand"
                  />
                  <div className="mt-2 flex items-center justify-between text-xs text-slate-400">
                    <span>0.5s</span>
                    <span className="text-sm font-semibold text-brand-light">{segmentLength.toFixed(1)}s</span>
                    <span>30s</span>
                  </div>
                </div>
              </label>

              <div className="flex flex-col gap-2">
                <span className="text-sm font-medium text-slate-300">Metadata tags</span>
                <div className="space-y-3 rounded-2xl border border-slate-800 bg-slate-950/70 p-4">
                  {metadataFields.map((field) => (
                    <div key={field.id} className="flex items-center gap-3">
                      <input
                        value={field.key}
                        placeholder="Label"
                        onChange={(event) => updateMetadataField(field.id, { key: event.target.value })}
                        className="flex-1 rounded-xl border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-slate-200 placeholder:text-slate-500"
                      />
                      <input
                        value={field.value}
                        placeholder="Value"
                        onChange={(event) => updateMetadataField(field.id, { value: event.target.value })}
                        className="flex-1 rounded-xl border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-slate-200 placeholder:text-slate-500"
                      />
                      <button
                        type="button"
                        onClick={() => removeMetadataField(field.id)}
                        className="rounded-xl border border-transparent bg-slate-800 px-2 py-2 text-xs text-slate-400 transition hover:border-slate-700 hover:text-slate-200"
                      >
                        Remove
                      </button>
                    </div>
                  ))}
                  <button
                    type="button"
                    onClick={addMetadataField}
                    className="w-full rounded-xl border border-dashed border-slate-700 bg-slate-900/70 px-3 py-2 text-sm font-medium text-slate-300 transition hover:border-brand hover:text-brand"
                  >
                    + Add metadata field
                  </button>
                </div>
              </div>
            </div>

            <button
              type="submit"
              disabled={compressing}
              className={clsx(
                "inline-flex items-center justify-center gap-2 rounded-xl bg-brand px-5 py-3 text-sm font-semibold text-white shadow-lg transition", 
                compressing ? "opacity-60" : "hover:bg-brand-light hover:shadow-brand/30"
              )}
            >
              {compressing ? "Processing..." : "Compress and index"}
            </button>

            {compressionMessage && <p className="text-sm font-medium text-emerald-300">{compressionMessage}</p>}
            {compressionError && <p className="text-sm font-medium text-rose-400">{compressionError}</p>}
          </form>

          {compressionEntries.length > 0 && (
            <div className="mt-8 space-y-4">
              <h3 className="text-sm font-semibold uppercase tracking-[0.2em] text-slate-400">Latest chunks</h3>
              <div className="space-y-3">
                {compressionEntries.map((entry) => (
                  <article
                    key={entry.chunk_id}
                    className={clsx(
                      "flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-slate-800/60 bg-slate-950/60 px-4 py-3 transition",
                      activeChunkId === entry.chunk_id ? "border-brand/70 shadow-[0_0_0_1px_rgba(108,99,255,0.4)]" : "hover:border-slate-700"
                    )}
                  >
                    <div>
                      <p className="text-sm font-semibold text-slate-100">Chunk #{entry.chunk_id.slice(0, 8)}</p>
                      <p className="text-xs text-slate-400">
                        {formatSeconds(entry.start_ts)} – {formatSeconds(entry.end_ts)} · {entry.metadata && Object.keys(entry.metadata).length > 0 ? "Tagged" : "Untitled"}
                      </p>
                    </div>
                    <div className="flex items-center gap-2">
                      {entry.metadata && Object.keys(entry.metadata).length > 0 && (
                        <div className="hidden sm:flex flex-wrap gap-2">
                          {Object.entries(entry.metadata).map(([key, value]) => (
                            <span key={key} className="rounded-full bg-slate-800 px-3 py-1 text-xs text-slate-300">
                              {key}: {String(value)}
                            </span>
                          ))}
                        </div>
                      )}
                      <button
                        type="button"
                        onClick={() => setActiveChunkId(entry.chunk_id)}
                        className="rounded-lg border border-slate-700 px-3 py-1.5 text-xs font-medium text-slate-200 transition hover:border-brand hover:text-brand"
                      >
                        Preview
                      </button>
                    </div>
                  </article>
                ))}
              </div>
            </div>
          )}
        </section>

        <section className="flex h-fit flex-col gap-6 rounded-3xl border border-slate-800/60 bg-slate-900/60 p-6 shadow-[0_20px_60px_-30px_rgba(15,23,42,0.9)] backdrop-blur">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-semibold text-white">Semantic search</h2>
            <span className="text-xs font-medium text-slate-400">Top {topK} results</span>
          </div>
          <form className="flex flex-col gap-4" onSubmit={handleQuery}>
            <label className="flex flex-col gap-2">
              <span className="text-sm font-medium text-slate-300">Describe the moment</span>
              <textarea
                rows={4}
                value={queryText}
                onChange={(event) => setQueryText(event.target.value)}
                placeholder="e.g. red forklift reversing or employee entering loading dock"
                className="w-full resize-none rounded-2xl border border-slate-800 bg-slate-950/70 px-4 py-3 text-sm text-slate-200 placeholder:text-slate-500"
              />
            </label>
            <label className="flex flex-col gap-2">
              <span className="text-sm font-medium text-slate-300">Results per query</span>
              <input
                type="range"
                min={1}
                max={20}
                value={topK}
                onChange={(event) => setTopK(Number(event.target.value))}
                className="h-2 w-full cursor-pointer appearance-none rounded-full bg-slate-800 accent-brand"
              />
              <span className="text-xs text-slate-400">{topK} neighbours</span>
            </label>
            <button
              type="submit"
              disabled={querying}
              className={clsx(
                "inline-flex items-center justify-center gap-2 rounded-xl border border-brand/40 bg-transparent px-5 py-3 text-sm font-semibold text-brand transition",
                querying ? "opacity-60" : "hover:border-brand hover:bg-brand/10 hover:text-brand-light"
              )}
            >
              {querying ? "Searching..." : "Search latent space"}
            </button>
            {queryError && <p className="text-sm font-medium text-rose-400">{queryError}</p>}
          </form>

          <div className="space-y-4">
            {queryResults.map((result) => (
              <article
                key={result.chunk_id}
                className={clsx(
                  "rounded-2xl border border-slate-800/60 bg-slate-950/60 p-4 transition",
                  activeChunkId === result.chunk_id ? "border-brand/70 shadow-[0_0_0_1px_rgba(108,99,255,0.4)]" : "hover:border-slate-700"
                )}
              >
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <h3 className="text-sm font-semibold text-slate-100">Chunk #{result.chunk_id.slice(0, 8)}</h3>
                    <p className="text-xs text-slate-400">
                      Score {result.score.toFixed(3)} · {formatSeconds(result.start_ts)} – {formatSeconds(result.end_ts)}
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={() => setActiveChunkId(result.chunk_id)}
                    className="rounded-lg border border-slate-700 px-3 py-1.5 text-xs font-medium text-slate-200 transition hover:border-brand hover:text-brand"
                  >
                    Preview
                  </button>
                </div>
                {result.metadata && Object.keys(result.metadata).length > 0 && (
                  <div className="mt-3 flex flex-wrap gap-2">
                    {Object.entries(result.metadata).map(([key, value]) => (
                      <span key={key} className="rounded-full bg-slate-800 px-3 py-1 text-xs text-slate-300">
                        {key}: {String(value)}
                      </span>
                    ))}
                  </div>
                )}
              </article>
            ))}
            {queryResults.length === 0 && (
              <p className="text-sm text-slate-400">
                Run a search to populate semantic matches. Results stream instantly from the latent index once available.
              </p>
            )}
          </div>

          <div className="rounded-2xl border border-slate-800 bg-slate-950/70 p-4">
            <h3 className="text-sm font-semibold text-slate-200">Instant preview</h3>
            {activePreviewUrl ? (
              <video key={activePreviewUrl} controls className="mt-3 w-full rounded-xl border border-slate-800" src={activePreviewUrl} />
            ) : (
              <p className="mt-3 text-sm text-slate-400">
                Select any chunk from compression history or query results to stream the decoded clip without full video restore.
              </p>
            )}
          </div>
        </section>
      </main>

      <footer className="mx-auto mt-12 w-full max-w-7xl px-6 text-xs text-slate-500">
        Built for operations teams that need instant answers from petabyte-scale video archives.
      </footer>
    </div>
  );
}
