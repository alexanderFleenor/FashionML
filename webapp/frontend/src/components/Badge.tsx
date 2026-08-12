/** Small badge for labels detected by the ML pipeline. */

export function PatternBadge({ pattern }: { pattern: string | null }) {
  if (!pattern) return null;
  const style: Record<string, string> = {
    solid: "bg-emerald-900/40 text-emerald-300 border-emerald-700/50",
    "two-tone": "bg-amber-900/40 text-amber-300 border-amber-700/50",
    "multi-color": "bg-rose-900/40 text-rose-300 border-rose-700/50",
  };
  return (
    <span
      className={`inline-block rounded-full border px-2 py-0.5 text-[10px] uppercase tracking-wide ${
        style[pattern] || "bg-zinc-900 text-zinc-400 border-zinc-700"
      }`}
    >
      {pattern}
    </span>
  );
}

const HARMONY_DESCRIPTIONS: Record<string, { label: string; blurb: string; tint: string }> = {
  analogous: {
    label: "Analogous",
    blurb: "Colors next to each other on the color wheel. Cohesive and easy.",
    tint: "bg-sky-900/40 text-sky-300 border-sky-700/50",
  },
  complementary: {
    label: "Complementary",
    blurb: "Colors opposite each other on the wheel. Bold contrast.",
    tint: "bg-fuchsia-900/40 text-fuchsia-300 border-fuchsia-700/50",
  },
  triadic: {
    label: "Triadic",
    blurb: "Three evenly-spaced colors. Balanced and lively.",
    tint: "bg-violet-900/40 text-violet-300 border-violet-700/50",
  },
  "split-complementary": {
    label: "Split-complementary",
    blurb: "A color with the two neighbors of its opposite. Softer contrast.",
    tint: "bg-indigo-900/40 text-indigo-300 border-indigo-700/50",
  },
  "neutral-pairing": {
    label: "Neutrals",
    blurb: "Low-saturation colors that go with almost anything.",
    tint: "bg-zinc-800 text-zinc-300 border-zinc-600",
  },
  unknown: {
    label: "Mixed",
    blurb: "No single harmony rule dominates this outfit.",
    tint: "bg-zinc-800 text-zinc-400 border-zinc-700",
  },
};

export function HarmonyBadge({ type }: { type: string | null }) {
  if (!type) return null;
  const info = HARMONY_DESCRIPTIONS[type] || HARMONY_DESCRIPTIONS.unknown;
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs ${info.tint}`}
      title={info.blurb}
    >
      <span className="font-medium">{info.label}</span>
    </span>
  );
}

export function HarmonyExplain({ type }: { type: string | null }) {
  if (!type) return null;
  const info = HARMONY_DESCRIPTIONS[type] || HARMONY_DESCRIPTIONS.unknown;
  return <p className="text-xs italic text-zinc-500">{info.blurb}</p>;
}
