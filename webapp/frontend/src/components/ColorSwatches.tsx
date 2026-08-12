import { DominantColor } from "../api";

/** Horizontal bar of color swatches, widths proportional to the percentage of
 * each color in the garment. Used on item tiles and outfit cards. */
export default function ColorSwatches({
  colors,
  height = 8,
  className = "",
}: {
  colors: DominantColor[];
  height?: number;
  className?: string;
}) {
  if (!colors.length) return null;
  const total = colors.reduce((s, c) => s + c.percentage, 0) || 1;
  return (
    <div
      className={`flex w-full overflow-hidden rounded-full ${className}`}
      style={{ height }}
      title={colors.map((c) => `${c.name} ${Math.round((c.percentage / total) * 100)}%`).join(", ")}
    >
      {colors.map((c, i) => (
        <div
          key={i}
          style={{
            backgroundColor: c.hex,
            width: `${(c.percentage / total) * 100}%`,
          }}
        />
      ))}
    </div>
  );
}

/** Same data as a row of labeled chips, with hex + percentage. Used inside
 * the closet item-detail view. */
export function ColorLegend({ colors }: { colors: DominantColor[] }) {
  if (!colors.length) return null;
  return (
    <div className="flex flex-wrap gap-1.5">
      {colors.map((c, i) => (
        <span
          key={i}
          className="inline-flex items-center gap-1.5 rounded-full bg-zinc-900/60 px-2 py-0.5 text-[11px] text-zinc-300"
        >
          <span
            className="h-3 w-3 rounded-full border border-zinc-700"
            style={{ backgroundColor: c.hex }}
          />
          {c.name} {Math.round(c.percentage * 100)}%
        </span>
      ))}
    </div>
  );
}
