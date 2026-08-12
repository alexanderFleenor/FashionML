// Thin API client. All requests carry the session cookie via `credentials: include`.

export interface DominantColor {
  name: string;
  hex: string;
  percentage: number;
}

export interface Item {
  item_id: string;
  category: string;
  predicted_category: string;
  predicted_confidence: number;
  color_summary: string;
  color_pattern: string | null; // "solid" | "two-tone" | "multi-color"
  dominant_colors: DominantColor[];
  image_url: string;
}

export interface OutfitSuggestion {
  items: Item[];
  score: number;
  explanation: string;
  summary: string;
  harmony_type: string | null;
  palette: string[];
}

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function call<T>(path: string, init: RequestInit = {}): Promise<T> {
  const res = await fetch(path, {
    credentials: "include",
    ...init,
    headers: {
      ...(init.body && !(init.body instanceof FormData) ? { "Content-Type": "application/json" } : {}),
      ...(init.headers || {}),
    },
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch {}
    throw new ApiError(res.status, detail);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

export const api = {
  me: () => call<{ authed: boolean }>("/api/auth/me"),
  login: (password: string) =>
    call<{ ok: true }>("/api/auth/login", {
      method: "POST",
      body: JSON.stringify({ password }),
    }),
  logout: () => call<{ ok: true }>("/api/auth/logout", { method: "POST" }),

  listItems: () =>
    call<{ items: Item[]; summary: { total_items: number; categories: Record<string, number> } }>(
      "/api/items"
    ),
  addItem: (file: File, categoryOverride?: string) => {
    const fd = new FormData();
    fd.append("image", file);
    if (categoryOverride) fd.append("category_override", categoryOverride);
    return call<Item>("/api/items", { method: "POST", body: fd });
  },
  updateItem: (itemId: string, category: string) =>
    call<Item>(`/api/items/${itemId}`, {
      method: "PATCH",
      body: JSON.stringify({ category }),
    }),
  deleteItem: (itemId: string) =>
    call<void>(`/api/items/${itemId}`, { method: "DELETE" }),

  todaysOutfits: (opts: { anchor_item_id?: string; template?: string; max_outfits?: number } = {}) =>
    call<{ outfits: OutfitSuggestion[] }>("/api/outfits/today", {
      method: "POST",
      body: JSON.stringify({ max_outfits: 3, ...opts }),
    }),
  logWear: (item_ids: string[]) =>
    call<{ worn_at: string; item_ids: string[] }>("/api/outfits/log", {
      method: "POST",
      body: JSON.stringify({ item_ids }),
    }),
};

export const CATEGORIES = ["tops", "bottoms", "dresses", "shoes", "accessories"] as const;
export type Category = typeof CATEGORIES[number];
