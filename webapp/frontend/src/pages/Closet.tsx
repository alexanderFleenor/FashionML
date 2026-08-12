import { useEffect, useState } from "react";
import { api, CATEGORIES, Category, Item } from "../api";
import AddItemSheet from "../components/AddItemSheet";
import ColorSwatches, { ColorLegend } from "../components/ColorSwatches";
import { PatternBadge } from "../components/Badge";

export default function Closet({ onLogout }: { onLogout: () => void }) {
  const [items, setItems] = useState<Item[]>([]);
  const [loading, setLoading] = useState(true);
  const [showAdd, setShowAdd] = useState(false);
  const [filter, setFilter] = useState<Category | "all">("all");
  const [inspected, setInspected] = useState<Item | null>(null);

  const refresh = async () => {
    setLoading(true);
    try {
      const data = await api.listItems();
      setItems(data.items);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refresh();
  }, []);

  const handleDelete = async (id: string) => {
    if (!confirm("Remove this item?")) return;
    await api.deleteItem(id);
    setItems((xs) => xs.filter((i) => i.item_id !== id));
    setInspected(null);
  };

  const shown = filter === "all" ? items : items.filter((i) => i.category === filter);
  const counts: Record<string, number> = { all: items.length };
  for (const c of CATEGORIES) counts[c] = items.filter((i) => i.category === c).length;

  return (
    <div className="px-4 pt-4">
      <header className="mb-4 flex items-baseline justify-between">
        <div>
          <h1 className="font-serif text-2xl">My closet</h1>
          <p className="text-xs text-zinc-500">
            {items.length} items catalogued by the ML pipeline
          </p>
        </div>
        <button onClick={onLogout} className="text-xs text-zinc-500 underline-offset-2 hover:underline">
          Sign out
        </button>
      </header>

      <div className="mb-4 -mx-4 flex gap-2 overflow-x-auto px-4 pb-1">
        {(["all", ...CATEGORIES] as const).map((c) => (
          <button
            key={c}
            onClick={() => setFilter(c)}
            className={`shrink-0 rounded-full border px-3 py-1.5 text-sm capitalize ${
              filter === c
                ? "border-accent bg-accent/20 text-accent"
                : "border-zinc-800 bg-zinc-900 text-zinc-400"
            }`}
          >
            {c} <span className="ml-1 text-[11px] text-zinc-500">{counts[c] ?? 0}</span>
          </button>
        ))}
      </div>

      {loading ? (
        <div className="py-20 text-center text-zinc-500">Loading...</div>
      ) : shown.length === 0 ? (
        <div className="py-20 text-center text-zinc-500">
          {items.length === 0
            ? "No items yet. Tap + to add your first piece."
            : "Nothing in this category."}
        </div>
      ) : (
        <div className="grid grid-cols-3 gap-2">
          {shown.map((item) => (
            <ItemTile
              key={item.item_id}
              item={item}
              onTap={() => setInspected(item)}
            />
          ))}
        </div>
      )}

      <button
        onClick={() => setShowAdd(true)}
        aria-label="Add item"
        className="fixed bottom-24 right-5 z-20 h-14 w-14 rounded-full bg-accent text-3xl leading-none text-white shadow-lg"
        style={{ marginBottom: "env(safe-area-inset-bottom)" }}
      >
        +
      </button>

      {showAdd && (
        <AddItemSheet
          onClose={() => setShowAdd(false)}
          onAdded={(item) => setItems((xs) => [item, ...xs])}
        />
      )}

      {inspected && (
        <ItemInspector
          item={inspected}
          onDelete={handleDelete}
          onClose={() => setInspected(null)}
        />
      )}
    </div>
  );
}

function ItemTile({ item, onTap }: { item: Item; onTap: () => void }) {
  return (
    <button onClick={onTap} className="group relative aspect-square overflow-hidden rounded-xl bg-zinc-900 text-left">
      <img src={item.image_url} alt={item.category} className="h-full w-full object-cover" />
      {/* Color strip pinned to the bottom of the thumbnail. */}
      {item.dominant_colors.length > 0 && (
        <div className="absolute bottom-0 left-0 right-0">
          <ColorSwatches colors={item.dominant_colors} height={6} className="rounded-none" />
        </div>
      )}
      <div className="absolute left-1 top-1 flex flex-wrap gap-1">
        <span className="rounded bg-black/65 px-1.5 py-0.5 text-[10px] capitalize">
          {item.category}
        </span>
        <PatternBadge pattern={item.color_pattern} />
      </div>
    </button>
  );
}

function ItemInspector({
  item,
  onClose,
  onDelete,
}: {
  item: Item;
  onClose: () => void;
  onDelete: (id: string) => void;
}) {
  return (
    <div className="fixed inset-0 z-30 flex items-end bg-black/70" onClick={onClose}>
      <div
        className="w-full rounded-t-2xl bg-zinc-900 p-5"
        style={{ paddingBottom: "calc(env(safe-area-inset-bottom) + 1rem)" }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex gap-3">
          <img src={item.image_url} alt="" className="h-28 w-28 rounded-xl object-cover" />
          <div className="flex flex-1 flex-col gap-1.5">
            <div className="flex flex-wrap items-baseline gap-2">
              <span className="text-base capitalize">{item.category}</span>
              <PatternBadge pattern={item.color_pattern} />
            </div>
            <p className="text-xs text-zinc-400">
              Classifier said{" "}
              <span className="capitalize text-zinc-300">{item.predicted_category}</span>{" "}
              <span className="text-zinc-500">
                ({Math.round(item.predicted_confidence * 100)}% confident)
              </span>
            </p>
            <p className="text-xs text-zinc-500">{item.color_summary}</p>
          </div>
        </div>

        <div className="mt-4">
          <div className="mb-2 text-[10px] uppercase tracking-wider text-zinc-500">
            Dominant colors (K-means in LAB space)
          </div>
          <ColorLegend colors={item.dominant_colors} />
        </div>

        <button
          onClick={() => onDelete(item.item_id)}
          className="mt-5 w-full rounded-xl border border-red-700/40 bg-red-900/20 py-2.5 text-sm text-red-300"
        >
          Remove from closet
        </button>
        <button onClick={onClose} className="mt-2 w-full py-2 text-sm text-zinc-500">
          Close
        </button>
      </div>
    </div>
  );
}
