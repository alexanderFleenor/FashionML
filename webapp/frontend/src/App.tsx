import { useEffect, useState } from "react";
import { Navigate, NavLink, Route, Routes, useNavigate } from "react-router-dom";
import { api, ApiError } from "./api";
import Login from "./pages/Login";
import Closet from "./pages/Closet";
import Today from "./pages/Today";

type AuthState = "checking" | "authed" | "anon";

export default function App() {
  const [auth, setAuth] = useState<AuthState>("checking");

  useEffect(() => {
    api
      .me()
      .then(() => setAuth("authed"))
      .catch((e: ApiError) => setAuth(e.status === 401 ? "anon" : "anon"));
  }, []);

  if (auth === "checking") {
    return <div className="flex h-full items-center justify-center text-zinc-500">Loading...</div>;
  }

  return (
    <Routes>
      <Route
        path="/login"
        element={
          auth === "authed" ? (
            <Navigate to="/today" replace />
          ) : (
            <Login onLoggedIn={() => setAuth("authed")} />
          )
        }
      />
      <Route
        path="/*"
        element={
          auth === "authed" ? (
            <Shell onLogout={() => setAuth("anon")} />
          ) : (
            <Navigate to="/login" replace />
          )
        }
      />
    </Routes>
  );
}

function Shell({ onLogout }: { onLogout: () => void }) {
  const nav = useNavigate();
  const handleLogout = async () => {
    await api.logout().catch(() => {});
    onLogout();
    nav("/login");
  };

  return (
    <div className="flex h-full flex-col">
      <main className="flex-1 overflow-y-auto pb-20">
        <Routes>
          <Route index element={<Navigate to="/today" replace />} />
          <Route path="/today" element={<Today />} />
          <Route path="/closet" element={<Closet onLogout={handleLogout} />} />
          <Route path="*" element={<Navigate to="/today" replace />} />
        </Routes>
      </main>
      <TabBar />
    </div>
  );
}

function TabBar() {
  const tabClass = ({ isActive }: { isActive: boolean }) =>
    `flex flex-1 flex-col items-center justify-center py-3 text-xs ${
      isActive ? "text-accent" : "text-zinc-500"
    }`;
  return (
    <nav
      className="fixed bottom-0 left-0 right-0 z-10 flex border-t border-zinc-800 bg-zinc-950/95 backdrop-blur"
      style={{ paddingBottom: "env(safe-area-inset-bottom)" }}
    >
      <NavLink to="/today" className={tabClass}>
        <span className="text-lg">★</span>
        <span>Today</span>
      </NavLink>
      <NavLink to="/closet" className={tabClass}>
        <span className="text-lg">▦</span>
        <span>Closet</span>
      </NavLink>
    </nav>
  );
}
