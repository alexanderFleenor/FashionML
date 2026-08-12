import { useState } from "react";
import { api, Category, Item } from "../api";
import CategoryChips from "./CategoryChips";
import ColorSwatches from "./ColorSwatches";
import { PatternBadge } from "./Badge";

type Stage =
  | { kind: "pick" }
  | { kind: "uploading" }
  | { kind: "confirm"; item: Item; previewUrl: string }
  | { kind: "saving" }
  | { kind: "error"; message: string };

export default function AddItemSheet({
  onClose,
  onAdded,
}: {
  onClose: () => void;
  onAdded: (item: Item) => void;
}) {
  const [stage, setStage] = useState<Stage>({ kind: "pick" });
  const [chosenCategory, setChosenCategory] = useState<Category | null>(null);

  const onFile = async (file: File) => {
    setStage({ kind: "uploading" });
    try {
      const item = await api.addItem(file);
      setChosenCategory(item.category as Category);
      setStage({ kind: "confirm", item, previewUrl: URL.createObjectURL(file) });
    } catch (e) {
      setStage({ kind: "error", message: (e as Error).message });
    }
  };

  const confirm = async () => {
    if (stage.kind !== "confirm") return;
    setStage({ kind: "saving" });
    try {
      let final = stage.item;
      if (chosenCategory && chosenCategory !== stage.item.category) {
        final = await api.updateItem(stage.item.item_id, chosenCategory);
      }
      onAdded(final);
      onClose();
    } catch (e) {
      setStage({ kind: "error", message: (e as Error).message });
    }
  };

  const cancel = async () => {
    if (stage.kind === "confirm") {
      // The upload already saved the item, so closing just leaves it as-is.
      await api.deleteItem(stage.item.item_id).catch(() => {});
    }
    onClose();
  };

  return (
    <div className="fixed inset-0 z-30 flex items-end bg-black/60" onClick={cancel}>
      <div
        className="w-full rounded-t-2xl bg-zinc-900 p-5"
        style={{ paddingBottom: "calc(env(safe-area-inset-bottom) + 1rem)" }}
        onClick={(e) => e.stopPropagation()}
      >
        {stage.kind === "pick" && (
          <div className="space-y-3">
            <h2 className="text-lg font-semibold">Add an item</h2>
            <p className="text-sm text-zinc-400">Snap a photo or pick from your library.</p>
            <label className="block">
              <input
                type="file"
                accept="image/*"
                capture="environment"
                onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])}
                className="hidden"
              />
              <span className="block w-full cursor-pointer rounded-xl bg-accent py-3 text-center text-base font-medium text-white">
                Take photo
              </span>
            </label>
            <label className="block">
              <input
                type="file"
                accept="image/*"
                onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])}
                className="hidden"
              />
              <span className="block w-full cursor-pointer rounded-xl border border-zinc-700 bg-zinc-800 py-3 text-center text-base">
                Choose from library
              </span>
            </label>
            <button
              onClick={cancel}
              className="block w-full py-3 text-center text-sm text-zinc-400"
            >
              Cancel
            </button>
          </div>
        )}

        {stage.kind === "uploading" && (
          <div className="py-12 text-center text-zinc-400">Analyzing...</div>
        )}
        {stage.kind === "saving" && (
          <div className="py-12 text-center text-zinc-400">Saving...</div>
        )}

        {stage.kind === "confirm" && (
          <div className="space-y-4">
            <img
              src={stage.previewUrl}
              alt=""
              className="mx-auto h-48 w-48 rounded-xl object-cover"
            />
            {stage.item.dominant_colors.length > 0 && (
              <ColorSwatches colors={stage.item.dominant_colors} height={10} />
            )}
            <div className="flex flex-wrap items-center justify-center gap-2 text-sm">
              <span className="text-zinc-400">EfficientNet says:</span>
              <span className="font-medium capitalize text-zinc-100">
                {stage.item.predicted_category}
              </span>
              <span className="rounded-full bg-zinc-800 px-2 py-0.5 font-mono text-[11px] text-zinc-400">
                {Math.round(stage.item.predicted_confidence * 100)}%
              </span>
              <PatternBadge pattern={stage.item.color_pattern} />
            </div>
            {stage.item.color_summary && (
              <div className="text-center text-xs text-zinc-500">
                {stage.item.color_summary}
              </div>
            )}
            <div>
              <div className="mb-2 text-sm text-zinc-400">Confirm or correct the category:</div>
              <CategoryChips
                value={chosenCategory}
                onChange={(c) => setChosenCategory(c)}
              />
            </div>
            <button
              onClick={confirm}
              className="w-full rounded-xl bg-accent py-3 text-base font-medium text-white"
            >
              Save to closet
            </button>
            <button
              onClick={cancel}
              className="w-full py-2 text-sm text-zinc-400"
            >
              Discard
            </button>
          </div>
        )}

        {stage.kind === "error" && (
          <div className="space-y-3">
            <div className="text-red-400">Could not add item: {stage.message}</div>
            <button
              onClick={() => setStage({ kind: "pick" })}
              className="w-full rounded-xl border border-zinc-700 py-3"
            >
              Try again
            </button>
            <button onClick={cancel} className="w-full py-2 text-sm text-zinc-400">
              Cancel
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
