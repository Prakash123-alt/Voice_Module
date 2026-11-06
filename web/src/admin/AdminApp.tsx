import { useCallback, useEffect, useState } from "react";

const api = async (path: string, opts: RequestInit = {}) => {
  const token = localStorage.getItem("admin_token");
  const res = await fetch(path, {
    ...opts,
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...opts.headers,
    },
  });
  if (!res.ok) throw Object.assign(new Error(await res.text()), { status: res.status });
  return res.json();
};

const TABS = ["Analytics", "Calls", "Leads", "Persona", "Knowledge"] as const;
type Tab = (typeof TABS)[number];

export function AdminApp() {
  // auth is bypassed server-side unless ADMIN_AUTH=on; if a 401 ever comes
  // back we fall back to the login form.
  const [needLogin, setNeedLogin] = useState(false);
  useEffect(() => {
    api("/api/admin/analytics").catch(e => {
      if (e.status === 401) setNeedLogin(true);
    });
  }, []);
  if (needLogin && !localStorage.getItem("admin_token"))
    return <Login onOk={() => setNeedLogin(false)} />;
  return <Dashboard />;
}

function Login({ onOk }: { onOk: () => void }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [err, setErr] = useState("");
  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    const res = await fetch("/api/admin/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });
    if (!res.ok) return setErr("Invalid credentials");
    const { token } = await res.json();
    localStorage.setItem("admin_token", token);
    onOk();
  };
  return (
    <form className="login" onSubmit={submit}>
      <h1>Admin</h1>
      <input type="email" placeholder="Email" value={email} required
        onChange={e => setEmail(e.target.value)} />
      <input type="password" placeholder="Password" value={password} required
        onChange={e => setPassword(e.target.value)} />
      {err && <div className="err">{err}</div>}
      <button className="btn btn-primary" type="submit">Sign in</button>
    </form>
  );
}

function Dashboard() {
  const [tab, setTab] = useState<Tab>("Analytics");
  return (
    <div className="admin">
      <h1>Voice Agent — Admin</h1>
      <nav>
        {TABS.map(t => (
          <button key={t} className={t === tab ? "active" : ""}
            onClick={() => setTab(t)}>{t}</button>
        ))}
      </nav>
      {tab === "Analytics" && <Analytics />}
      {tab === "Calls" && <Calls />}
      {tab === "Leads" && <LeadsBookings />}
      {tab === "Persona" && <PersonaEditor />}
      {tab === "Knowledge" && <KbEditor />}
    </div>
  );
}

function Analytics() {
  const [a, setA] = useState<Record<string, number> | null>(null);
  const [enroll, setEnroll] = useState<any>(null);
  useEffect(() => {
    api("/api/admin/analytics").then(setA).catch(() => {});
    api("/api/admin/enrollment").then(setEnroll).catch(() => {});
  }, []);
  if (!a) return <p>Loading…</p>;
  return (
    <div>
      <div className="stat-row">
        <Stat label="Calls" v={a.calls} />
        <Stat label="Avg duration" v={`${a.avgDurationS}s`} />
        <Stat label="Resolution" v={`${a.resolutionRate}%`} />
        <Stat label="Transferred" v={a.transferred} />
        <Stat label="Leads" v={a.leads} />
        <Stat label="Bookings" v={a.bookings} />
      </div>
      <div className="card" style={{ marginTop: 16 }}>
        <b>Voice enrollment</b>
        <p style={{ color: "var(--muted)", fontSize: 14 }}>
          Speaker gate: {enroll?.models?.speaker_gate ? "on" : "off"} ·{" "}
          Enrolled voice: {enroll?.models?.enrolled ? "yes" : "no"}<br />
          Enroll via <code>python enroll.py --record 6</code> in voice/.
        </p>
      </div>
    </div>
  );
}
const Stat = ({ label, v }: { label: string; v: React.ReactNode }) => (
  <div className="stat"><b>{v}</b><span>{label}</span></div>
);

function Calls() {
  const [calls, setCalls] = useState<any[]>([]);
  const [sel, setSel] = useState<any>(null);
  const load = useCallback(() => {
    api("/api/admin/calls").then(setCalls).catch(() => {});
  }, []);
  useEffect(load, [load]);
  const del = async (id: string) => {
    if (!confirm("Delete this call and its transcript?")) return;
    await api(`/api/admin/calls/${id}`, { method: "DELETE" });
    setSel(null); load();
  };
  return (
    <div>
      {sel?.call ? (
        <div className="card">
          <button className="link" onClick={() => setSel(null)}>← back</button>
          <h3>Call {sel.call.id.slice(0, 8)} · {sel.call.status} ·{" "}
            {Math.round(sel.call.duration_s ?? 0)}s</h3>
          <p style={{ color: "var(--muted)" }}>{sel.call.summary}</p>
          <table><tbody>
            {(sel.messages as any[]).map(m => (
              <tr key={m.id}>
                <td style={{ width: 90, color: "var(--muted)" }}>{m.role}</td>
                <td>{m.text}</td>
              </tr>
            ))}
          </tbody></table>
          <h4>Tool calls</h4>
          <table><tbody>
            {(sel.toolCalls as any[]).map(t => (
              <tr key={t.id}>
                <td style={{ width: 160 }}>{t.name}</td>
                <td>{t.args_json}</td>
                <td style={{ width: 110 }}>{t.status}</td>
              </tr>
            ))}
          </tbody></table>
          <button className="btn btn-danger" onClick={() => del(sel.call.id)}>
            Delete call</button>
        </div>
      ) : (
        <table>
          <thead><tr><th>Started</th><th>Status</th><th>Lang</th>
            <th>Duration</th><th>E2E audio ms</th><th /></tr></thead>
          <tbody>
            {calls.map(c => (
              <tr key={c.id}>
                <td>{new Date(c.started_at + "Z").toLocaleString()}</td>
                <td>{c.status}</td>
                <td>{c.lang}</td>
                <td>{Math.round(c.duration_s ?? 0)}s</td>
                <td>{JSON.parse(c.metrics_json ?? "{}")
                  .userEndToFirstAudioMs ?? "—"}</td>
                <td>
                  <button className="link" onClick={async () =>
                    setSel(await api(`/api/admin/calls/${c.id}`))}>view</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

function LeadsBookings() {
  const [leads, setLeads] = useState<any[]>([]);
  const [bookings, setBookings] = useState<any[]>([]);
  useEffect(() => {
    api("/api/admin/leads").then(setLeads).catch(() => {});
    api("/api/admin/bookings").then(setBookings).catch(() => {});
  }, []);
  return (
    <div>
      <h3>Leads</h3>
      <table><thead><tr><th>When</th><th>Name</th><th>Contact</th>
        <th>Interest</th></tr></thead>
        <tbody>{leads.map(l => (
          <tr key={l.id}><td>{l.created_at}</td><td>{l.name}</td>
            <td>{l.contact}</td><td>{l.interest}</td></tr>))}
        </tbody></table>
      <h3 style={{ marginTop: 24 }}>Bookings</h3>
      <table><thead><tr><th>When</th><th>Name</th><th>Phone</th>
        <th>Service</th><th>Slot</th><th>Status</th></tr></thead>
        <tbody>{bookings.map(b => (
          <tr key={b.id}><td>{b.created_at}</td><td>{b.name}</td>
            <td>{b.phone}</td><td>{b.service}</td><td>{b.slot_ts}</td>
            <td>{b.status}</td></tr>))}
        </tbody></table>
    </div>
  );
}

function PersonaEditor() {
  const [yaml, setYaml] = useState("");
  const [msg, setMsg] = useState("");
  useEffect(() => {
    api("/api/admin/persona").then(r => setYaml(r.yaml)).catch(() => {});
  }, []);
  const save = async () => {
    try {
      await api("/api/admin/persona", {
        method: "PUT", body: JSON.stringify({ yaml }),
      });
      setMsg("Saved — applies to the next call.");
    } catch (e) {
      setMsg(String(e));
    }
  };
  return (
    <div className="card">
      <h3>Persona (persona.yaml)</h3>
      <textarea value={yaml} onChange={e => setYaml(e.target.value)}
        spellCheck={false} />
      <div style={{ marginTop: 12, display: "flex", gap: 12, alignItems: "center" }}>
        <button className="btn btn-primary" onClick={save}>Save</button>
        <span style={{ color: "var(--muted)", fontSize: 14 }}>{msg}</span>
      </div>
    </div>
  );
}

function KbEditor() {
  const [files, setFiles] = useState<Record<string, string>>({});
  const [cur, setCur] = useState("");
  const [msg, setMsg] = useState("");
  useEffect(() => {
    api("/api/admin/kb").then(r => {
      setFiles(r.files);
      const first = Object.keys(r.files)[0];
      if (first) setCur(first);
    }).catch(() => {});
  }, []);
  const save = async () => {
    await api(`/api/admin/kb/${cur}`, {
      method: "PUT", body: JSON.stringify({ content: files[cur] }),
    });
    setMsg("Saved.");
    setTimeout(() => setMsg(""), 2000);
  };
  return (
    <div className="card">
      <h3>Knowledge base</h3>
      <nav>
        {Object.keys(files).map(f => (
          <button key={f} className={f === cur ? "active" : ""}
            onClick={() => setCur(f)}>{f}</button>
        ))}
      </nav>
      {cur && (
        <>
          <textarea value={files[cur] ?? ""} spellCheck={false}
            onChange={e => setFiles({ ...files, [cur]: e.target.value })} />
          <div style={{ marginTop: 12, display: "flex", gap: 12, alignItems: "center" }}>
            <button className="btn btn-primary" onClick={save}>Save</button>
            <span style={{ color: "var(--muted)", fontSize: 14 }}>{msg}</span>
          </div>
        </>
      )}
      <p style={{ color: "var(--muted)", fontSize: 13 }}>
        Chunks are split on <code>## headings</code> and matched with TF-IDF
        retrieval — add a file under <code>server/config/kb/</code> for more.
      </p>
    </div>
  );
}
