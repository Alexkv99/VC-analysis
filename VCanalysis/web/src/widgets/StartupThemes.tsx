import "@/index.css";

import { mountWidget } from "skybridge/web";
import { useToolInfo } from "../helpers";

function StartupThemes() {
  const { input, output } = useToolInfo<"startup-themes">();

  if (!output) return <div>Analyzing file…</div>;

  const themes = output.themes ?? [];

  return (
    <div style={{ padding: 12, fontFamily: "system-ui, sans-serif" }}>
      <h3 style={{ margin: "0 0 8px 0" }}>Startup themes</h3>

      <div style={{ fontSize: 12, opacity: 0.75, marginBottom: 10 }}>
        file: {output.fileName ?? input.fileName ?? "sifted_last7d.json"} • articles read:{" "}
        {output.articlesRead ?? "?"} • kThemes: {output.kThemes ?? input.kThemes ?? "?"}
      </div>

      {themes.length === 0 ? (
        <div>No themes found.</div>
      ) : (
        <div style={{ display: "grid", gap: 10 }}>
          {themes.map((t: any) => (
            <div
              key={t.id}
              style={{
                border: "1px solid rgba(0,0,0,0.12)",
                borderRadius: 10,
                padding: 10,
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", gap: 10 }}>
                <div style={{ fontWeight: 700 }}>{t.label}</div>
                <div style={{ fontSize: 12, opacity: 0.7 }}>{t.size} articles</div>
              </div>

              <div style={{ marginTop: 8, fontSize: 12, opacity: 0.9 }}>
                <strong>Keywords:</strong>{" "}
                {(t.keywords ?? [])
                  .slice(0, 10)
                  .map((k: any) => k.term)
                  .join(", ")}
              </div>

              {(t.sampleTitles?.length ?? 0) > 0 && (
                <div style={{ marginTop: 8 }}>
                  <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 4 }}>Examples:</div>
                  <ul style={{ margin: 0, paddingLeft: 18 }}>
                    {t.sampleTitles.slice(0, 3).map((s: string) => (
                      <li key={s} style={{ fontSize: 12 }}>
                        {s}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default StartupThemes;

mountWidget(<StartupThemes />);
