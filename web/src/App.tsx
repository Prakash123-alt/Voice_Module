import { Route, Routes } from "react-router-dom";
import { AdminApp } from "./admin/AdminApp";
import { CallView } from "./components/CallView";

export function App() {
  return (
    <Routes>
      <Route path="/" element={<CallView />} />
      <Route path="/admin/*" element={<AdminApp />} />
    </Routes>
  );
}
