import { McpServer } from "skybridge/server";
import { z } from "zod";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

/** ---------- Stop words + theme taxonomy ---------- */

const STOP_WORDS = new Set([
  "a","an","the","and","or","but","if","then","else","when","while","of","to","in","on","for","from","with","by",
  "is","am","are","was","were","be","been","being","it","its","this","that","these","those",
  "as","at","not","no","yes","do","does","did","doing","can","could","will","would","should","may","might","must",
  "i","you","he","she","we","they","me","him","her","us","them","my","your","his","their","our",
  "what","which","who","whom","why","how",
  // extra “news-y” glue
  "into","over","after","before","amid","new","more","most","says","say","said","report","reports",
]);

const THEME_RULES: Array<{ label: string; keywords: string[] }> = [
  { label: "AI / ML", keywords: ["ai","agent","agents","model","models","llm","voice","speech","robot","robotics","compute","gpu","chip","chips","inference","training"] },
  { label: "Fintech", keywords: ["fintech","payment","payments","bank","banking","neobank","card","lending","credit","invoice","cfo","revenue","crypto","blockchain","wallet"] },
  { label: "Climate / Energy", keywords: ["climate","energy","solar","wind","battery","hydrogen","carbon","emissions","renewable","power","grid","sustainability"] },
  { label: "Health / Biotech", keywords: ["health","biotech","medical","clinical","life","science","drug","therapy","patient","dental","diagnostic","genomics"] },
  { label: "Cybersecurity", keywords: ["cyber","security","secure","breach","threat","vulnerability","defence","defense","privacy","identity","auth"] },
  { label: "Defence / Dual-use", keywords: ["defence","defense","military","drone","aerospace","nato","dual","army","air","force"] },
  { label: "Quantum", keywords: ["quantum","qubit","cryogenic","ion","photon","photonic","superconducting"] },
  { label: "Enterprise SaaS", keywords: ["enterprise","b2b","platform","workflow","compliance","legal","legaltech","hr","payroll","customer","support","service","crm"] },
  { label: "Hardware / Deeptech", keywords: ["semiconductor","chip","factory","manufacturing","robot","satellite","space","infrastructure","datacenter","hardware"] },
];

/** ---------- Data dir (robust) ---------- */
// server/src/server.ts -> repo root -> data
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const DATA_DIR = path.resolve(__dirname, "..", "..", "data");

function sanitizeFileName(fileName: string): string {
  const base = path.basename(fileName);
  if (!base.toLowerCase().endsWith(".json")) throw new Error("fileName must end with .json");
  return base;
}

async function loadJsonFromDataDir(fileName: string): Promise<{ json: unknown; loadedFrom: string }> {
  const safe = sanitizeFileName(fileName);
  const filePath = path.join(DATA_DIR, safe);
  const raw = await fs.readFile(filePath, "utf-8");
  return { json: JSON.parse(raw), loadedFrom: filePath };
}

/** ---------- Tokenization + docs ---------- */

function tokenize(text: string): string[] {
  return (text.toLowerCase().match(/[\p{L}\p{N}]+/gu) ?? [])
    .filter((w) => w.length > 2)
    .filter((w) => !STOP_WORDS.has(w));
}

function normForDedupe(s: string): string {
  return s.toLowerCase().replace(/\s+/g, " ").trim();
}

type ArticleDoc = {
  id: number;
  title: string;
  excerpt: string;
  text: string;
  tokens: string[];
};

function extractDocs(json: unknown): { docs: ArticleDoc[]; articlesRead: number; articlesDeduped: number } {
  const records = Array.isArray(json) ? json : [json];

  const seen = new Set<string>();
  const docs: ArticleDoc[] = [];

  let read = 0;

  for (const rec of records) {
    if (!rec || typeof rec !== "object") continue;
    read++;

    const obj = rec as Record<string, unknown>;
    const title = typeof obj.title === "string" ? obj.title : "";
    const excerpt = typeof obj.excerpt === "string" ? obj.excerpt : "";

    const key = `${normForDedupe(title)}|${normForDedupe(excerpt)}`;
    if (seen.has(key)) continue;
    seen.add(key);

    const text = `${title} ${excerpt}`.trim();
    const tokens = tokenize(text);

    if (tokens.length === 0) continue;
    docs.push({ id: docs.length, title, excerpt, text, tokens });
  }

  return { docs, articlesRead: read, articlesDeduped: docs.length };
}

/** ---------- TF-IDF sparse vectors ---------- */

type SparseVec = Map<string, number>;

function normalize(vec: SparseVec): SparseVec {
  let norm = 0;
  for (const v of vec.values()) norm += v * v;
  norm = Math.sqrt(norm) || 1;
  const out = new Map<string, number>();
  for (const [k, v] of vec) out.set(k, v / norm);
  return out;
}

function cosine(a: SparseVec, b: SparseVec): number {
  const [small, big] = a.size <= b.size ? [a, b] : [b, a];
  let sum = 0;
  for (const [k, v] of small) sum += v * (big.get(k) ?? 0);
  return sum;
}

function addInto(target: Map<string, number>, src: SparseVec): void {
  for (const [k, v] of src) target.set(k, (target.get(k) ?? 0) + v);
}

function topEntries(map: Map<string, number>, k: number): Array<{ term: string; weight: number }> {
  return [...map.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, k)
    .map(([term, weight]) => ({ term, weight }));
}

function buildTfidfVectors(
  docs: ArticleDoc[],
  opts: { minDf: number; maxDfRatio: number; maxVocab: number },
): { vectors: SparseVec[]; keptVocab: Set<string> } {
  const N = docs.length;

  const df = new Map<string, number>();
  for (const d of docs) {
    const unique = new Set(d.tokens);
    for (const t of unique) df.set(t, (df.get(t) ?? 0) + 1);
  }

  const pruned = new Set<string>();
  for (const [t, c] of df) {
    if (c < opts.minDf) pruned.add(t);
    else if (c / N > opts.maxDfRatio) pruned.add(t);
  }

  const vocabSorted = [...df.entries()]
    .filter(([t]) => !pruned.has(t))
    .sort((a, b) => b[1] - a[1])
    .slice(0, opts.maxVocab)
    .map(([t]) => t);

  const keptVocab = new Set(vocabSorted);

  const idf = new Map<string, number>();
  for (const t of keptVocab) {
    const c = df.get(t) ?? 0;
    idf.set(t, Math.log((N + 1) / (c + 1)) + 1);
  }

  const vectors: SparseVec[] = [];
  for (const d of docs) {
    const tf = new Map<string, number>();
    let len = 0;

    for (const t of d.tokens) {
      if (!keptVocab.has(t)) continue;
      tf.set(t, (tf.get(t) ?? 0) + 1);
      len++;
    }

    const vec = new Map<string, number>();
    if (len === 0) {
      vectors.push(vec);
      continue;
    }

    for (const [t, c] of tf) {
      const tfNorm = c / len;
      vec.set(t, tfNorm * (idf.get(t) ?? 0));
    }

    vectors.push(normalize(vec));
  }

  return { vectors, keptVocab };
}

/** ---------- k-means (cosine) ---------- */

function shuffleIndices(n: number): number[] {
  const arr = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

function kMeansCosine(vectors: SparseVec[], k: number, maxIter: number): number[] {
  const N = vectors.length;
  if (N === 0) return [];

  const order = shuffleIndices(N);

  const centroids: SparseVec[] = [];
  for (const idx of order) {
    if (vectors[idx].size > 0) centroids.push(vectors[idx]);
    if (centroids.length === k) break;
  }
  while (centroids.length < k) centroids.push(new Map());

  let assign = new Array<number>(N).fill(0);

  for (let iter = 0; iter < maxIter; iter++) {
    let changed = 0;

    for (let i = 0; i < N; i++) {
      let best = 0;
      let bestSim = -Infinity;

      for (let c = 0; c < k; c++) {
        const sim = cosine(vectors[i], centroids[c]);
        if (sim > bestSim) {
          bestSim = sim;
          best = c;
        }
      }

      if (assign[i] !== best) {
        assign[i] = best;
        changed++;
      }
    }

    const sums: Array<Map<string, number>> = Array.from({ length: k }, () => new Map());
    const counts = new Array<number>(k).fill(0);

    for (let i = 0; i < N; i++) {
      counts[assign[i]]++;
      addInto(sums[assign[i]], vectors[i]);
    }

    for (let c = 0; c < k; c++) {
      if (counts[c] === 0) continue;
      const mean = new Map<string, number>();
      for (const [t, v] of sums[c]) mean.set(t, v / counts[c]);
      centroids[c] = normalize(mean);
    }

    if (changed === 0) break;
  }

  return assign;
}

/** ---------- Labeling + themes ---------- */

function labelFromKeywords(keywords: string[]): string {
  const set = new Set(keywords);
  let best = { label: "Other", score: 0 };

  for (const rule of THEME_RULES) {
    let score = 0;
    for (const kw of rule.keywords) if (set.has(kw)) score++;
    if (score > best.score) best = { label: rule.label, score };
  }

  if (best.score === 0) return keywords.slice(0, 3).join(" / ") || "Other";
  return best.label;
}

function ensureUniqueLabels(themes: Array<{ label: string; keywords: Array<{ term: string; weight: number }> }>) {
  const seen = new Map<string, number>();
  for (const t of themes) {
    const base = t.label;
    const count = (seen.get(base) ?? 0) + 1;
    seen.set(base, count);

    if (count > 1) {
      const hint = t.keywords?.[0]?.term ?? `theme-${count}`;
      t.label = `${base} (${hint})`;
    }
  }
}

function buildThemes(
  docs: ArticleDoc[],
  vectors: SparseVec[],
  assign: number[],
  opts: { k: number; topTerms: number; sampleTitles: number },
) {
  const clusters: number[][] = Array.from({ length: opts.k }, () => []);
  for (let i = 0; i < assign.length; i++) clusters[assign[i]].push(i);

  const themes = clusters
    .map((idxs, clusterId) => {
      const agg = new Map<string, number>();
      for (const i of idxs) {
        for (const [t, w] of vectors[i]) agg.set(t, (agg.get(t) ?? 0) + w);
      }

      const top = topEntries(agg, opts.topTerms);
      const keywords = top.map((x) => x.term);
      const label = labelFromKeywords(keywords);

      const sampleTitles = idxs
        .slice(0, opts.sampleTitles)
        .map((i) => docs[i]?.title)
        .filter((t): t is string => Boolean(t));

      return {
        id: clusterId,
        label,
        size: idxs.length,
        keywords: top,
        sampleTitles,
      };
    })
    .filter((t) => t.size > 0)
    .sort((a, b) => b.size - a.size);

  ensureUniqueLabels(themes);
  return themes;
}

/** ---------- Widgets ---------- */

const server = new McpServer(
  { name: "alpic-openai-app", version: "0.0.1" },
  { capabilities: {} },
)
  .registerWidget(
    "startup-themes",
    { description: "Clusters articles into themes using TF-IDF + cosine k-means; returns labeled themes + keywords." },
    {
      description: "Loads a JSON file from ./data and extracts the main themes across articles.",
      inputSchema: {
        fileName: z.string().optional().describe("JSON file in ./data (default: sifted_last7d.json)"),
        kThemes: z.number().int().min(2).max(20).optional().describe("Number of themes (default: 8)"),
        topTerms: z.number().int().min(3).max(30).optional().describe("Keywords per theme (default: 10)"),
        sampleTitles: z.number().int().min(0).max(10).optional().describe("Sample titles per theme (default: 3)"),
        minDf: z.number().int().min(1).max(10).optional().describe("Min document frequency (default: 2)"),
        maxDfRatio: z.number().min(0.05).max(0.6).optional().describe("Drop terms in >X of docs (default: 0.25)"),
        maxVocab: z.number().int().min(200).max(20000).optional().describe("Cap vocab size (default: 5000)"),
      },
    },
    async ({ fileName, kThemes, topTerms, sampleTitles, minDf, maxDfRatio, maxVocab }) => {
      try {
        const name = fileName ?? "sifted_last7d.json";
        const { json, loadedFrom } = await loadJsonFromDataDir(name);

        const { docs, articlesRead, articlesDeduped } = extractDocs(json);

        const k = kThemes ?? 8;
        const tt = topTerms ?? 10;
        const st = sampleTitles ?? 3;

        const { vectors } = buildTfidfVectors(docs, {
          minDf: minDf ?? 2,
          maxDfRatio: maxDfRatio ?? 0.25,
          maxVocab: maxVocab ?? 5000,
        });

        const assign = kMeansCosine(vectors, k, 25);
        const themes = buildThemes(docs, vectors, assign, { k, topTerms: tt, sampleTitles: st });

        return {
          structuredContent: {
            fileName: name,
            loadedFrom,
            articlesRead,
            articlesDeduped,
            kThemes: k,
            themes,
          },
          content: [],
          isError: false,
        };
      } catch (error) {
        return {
          content: [{ type: "text", text: `Error: ${String(error)}` }],
          isError: true,
        };
      }
    },
  );

export default server;
export type AppType = typeof server;
