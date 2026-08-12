import { FormEvent, useState } from "react";
import { api, ApiError } from "../api";

export default function Login({ onLoggedIn }: { onLoggedIn: () => void }) {
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setPending(true);
    setError(null);
    try {
      await api.login(password);
      onLoggedIn();
    } catch (err) {
      const e = err as ApiError;
      setError(e.status === 401 ? "Wrong password" : e.message);
    } finally {
      setPending(false);
    }
  };

  return (
    <div className="flex h-full flex-col items-center justify-center px-6">
      <form onSubmit={submit} className="w-full max-w-sm space-y-4">
        <div className="text-center">
          <h1 className="font-serif text-4xl">Closet</h1>
          <p className="mt-1 text-sm text-zinc-500">
            An ML outfit picker for my closet.
          </p>
        </div>
        <input
          type="password"
          inputMode="text"
          autoComplete="current-password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          className="w-full rounded-xl border border-zinc-800 bg-zinc-900 px-4 py-3 text-base focus:border-accent focus:outline-none"
        />
        {error && <p className="text-sm text-red-400">{error}</p>}
        <button
          type="submit"
          disabled={pending || !password}
          className="w-full rounded-xl bg-accent py-3 text-base font-medium text-white disabled:opacity-50"
        >
          {pending ? "Signing in..." : "Sign in"}
        </button>
      </form>
      <p className="mt-10 max-w-sm text-center text-[11px] leading-relaxed text-zinc-600">
        EfficientNet-B0 classifies garments. K-means in LAB space extracts colors. A Siamese
        compatibility network scores outfits.
      </p>
    </div>
  );
}
