import { CATEGORIES, Category } from "../api";

export default function CategoryChips({
  value,
  onChange,
}: {
  value: Category | null;
  onChange: (cat: Category) => void;
}) {
  return (
    <div className="flex flex-wrap gap-2">
      {CATEGORIES.map((c) => (
        <button
          key={c}
          type="button"
          onClick={() => onChange(c)}
          className={`rounded-full border px-3 py-1.5 text-sm capitalize ${
            value === c
              ? "border-accent bg-accent/20 text-accent"
              : "border-zinc-700 bg-zinc-900 text-zinc-300"
          }`}
        >
          {c}
        </button>
      ))}
    </div>
  );
}
