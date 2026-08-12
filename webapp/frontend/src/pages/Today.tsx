import { useEffect, useState } from "react";
import { api, ApiError, Item, OutfitSuggestion } from "../api";
import { HarmonyBadge, HarmonyExplain, PatternBadge } from "../components/Badge";

export default function Today() {
  const [outfits, setOutfits] = useState<OutfitSuggestion[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [anchor, setAnchor] = useState<Item | null>(null);
  const [pickingAnchor, setPickingAnchor] = useState(false);
  const [wornJustNow, setWornJustNow] = useState<number | null>(null);

  const generate = async (anchorId?: string) => {
    setLoading(true);
    setError(null);
    setWornJustNow(null);
    try {
      const data = await api.todaysOutfits({ anchor_item_id: anchorId });
      setOutfits(data.outfits);
    } catch (e) {
      setError((e as ApiError).message || "Could not generate outfits");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    generate(anchor?.item_id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [anchor?.item_id]);

  const wearOutfit = async (idx: number) => {
    const outfit = outfits[idx];
    await api.logWear(outfit.items.map((i) => i.item_id));
    setWornJustNow(idx);
  };

  return (
    <div className="px-4 pt-4">
      <header className="mb-4">
        <h1 className="font-serif text-2xl">Today's outfits</h1>
        <p className="text-xs text-zinc-500">
          {anchor ? (
            <>
              Built around your{" "}
              <span className="capitalize text-zinc-300">{anchor.category}</span>{" "}
              <button
                onClick={() => setAnchor(null)}
                className="ml-1 text-accent underline underline-offset-2"
              >
                clear
              </button>
            </>
          ) : (
            <>
              Picked by the Siamese compatibility model, ranked by color harmony.
            </>
          )}
        </p>
      </header>

      <div className="mb-4 flex gap-2">
        <button
          onClick={() => generate(anchor?.item_id)}
          className="flex-1 rounded-xl border border-zinc-800 bg-zinc-900 py-2.5 text-sm"
        >
          Shuffle
        </button>
        <button
          onClick={() => setPickingAnchor(true)}
          className="flex-1 rounded-xl border border-zinc-800 bg-zinc-900 py-2.5 text-sm"
        >
          Build around...
        </button>
      </div>

      {loading && (
        <div className="py-20 text-center text-zinc-500">
          <div className="animate-pulse">Scoring outfit combinations...</div>
          <p className="mt-2 text-[11px] text-zinc-600">Running the compatibility model on candidate pairs</p>
        </div>
      )}
      {error && (
        <div className="py-20 text-center text-zinc-500">
          <p>{error}</p>
          {error.includes("empty") && (
            <p className="mt-2 text-xs">Add some items in your closet first.</p>
          )}
        </div>
      )}

      <div className="space-y-4">
        {outfits.map((outfit, idx) => (
          <OutfitCard
            key={idx}
            outfit={outfit}
            worn={wornJustNow === idx}
            onWear={() => wearOutfit(idx)}
          />
        ))}
      </div>

      {pickingAnchor && (
        <AnchorPicker
          onPick={(item) => {
            setAnchor(item);
            setPickingAnchor(false);
          }}
          onClose={() => setPickingAnchor(false)}
        />
      )}
    </div>
  );
}

function OutfitCard({
  outfit,
  worn,
  onWear,
}: {
  outfit: OutfitSuggestion;
  worn: boolean;
  onWear: () => void;
}) {
  return (
    <div className="overflow-hidden rounded-2xl border border-zinc-800 bg-zinc-900">
      {/* Palette strip from the dominant colors across the outfit. */}
      {outfit.palette.length > 0 && (
        <div className="flex h-3 w-full">
          {outfit.palette.map((hex, i) => (
            <div key={i} className="flex-1" style={{ backgroundColor: hex }} />
          ))}
        </div>
      )}

      <div className="p-3">
        <div className="mb-3 flex items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <span className="text-[10px] uppercase tracking-wider text-zinc-500">compatibility</span>
            <span className="font-mono text-sm text-zinc-200">{Math.round(outfit.score * 100)}</span>
            <HarmonyBadge type={outfit.harmony_type} />
          </div>
          <button
            onClick={onWear}
            disabled={worn}
            className={`rounded-full px-3 py-1 text-xs ${
              worn ? "bg-zinc-800 text-zinc-500" : "bg-accent text-white"
            }`}
          >
            {worn ? "Logged" : "Wore this"}
          </button>
        </div>

        <div className={`mb-3 grid gap-2`} style={{ gridTemplateColumns: `repeat(${outfit.items.length}, minmax(0, 1fr))` }}>
          {outfit.items.map((item) => (
            <div key={item.item_id} className="flex flex-col gap-1">
              <div className="aspect-square overflow-hidden rounded-lg bg-zinc-950">
                <img src={item.image_url} alt={item.category} className="h-full w-full object-cover" />
              </div>
              <div className="flex items-center justify-between gap-1">
                <span className="text-[10px] capitalize text-zinc-400">{item.category}</span>
                <PatternBadge pattern={item.color_pattern} />
              </div>
            </div>
          ))}
        </div>

        <p className="text-sm text-zinc-300">{outfit.summary}</p>
        <HarmonyExplain type={outfit.harmony_type} />
      </div>
    </div>
  );
}

function AnchorPicker({
  onPick,
  onClose,
}: {
  onPick: (item: Item) => void;
  onClose: () => void;
}) {
  const [items, setItems] = useState<Item[]>([]);
  useEffect(() => {
    api.listItems().then((d) => setItems(d.items));
  }, []);
  return (
    <div className="fixed inset-0 z-30 flex items-end bg-black/70" onClick={onClose}>
      <div
        className="max-h-[80vh] w-full overflow-y-auto rounded-t-2xl bg-zinc-900 p-4"
        style={{ paddingBottom: "calc(env(safe-area-inset-bottom) + 1rem)" }}
        onClick={(e) => e.stopPropagation()}
      >
        <h2 className="mb-1 font-serif text-lg">Pick an item to build around</h2>
        <p className="mb-3 text-xs text-zinc-500">
          The generator will force this item into every outfit and pick the rest to harmonize.
        </p>
        <div className="grid grid-cols-3 gap-2">
          {items.map((item) => (
            <button
              key={item.item_id}
              onClick={() => onPick(item)}
              className="aspect-square overflow-hidden rounded-xl bg-zinc-950"
            >
              <img src={item.image_url} alt={item.category} className="h-full w-full object-cover" />
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
