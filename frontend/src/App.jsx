import { Suspense, lazy, useEffect } from "react";
import { Navigate, Route, Routes, useLocation } from "react-router-dom";
import AOS from "aos";
import AppShell from "./components/AppShell";
import { SkeletonBlock, SkeletonLines, SkeletonMetric } from "./components/Skeleton";
import { VisionPlayProvider } from "./context/VisionPlayContext";

const WorkspacePage = lazy(() => import("./pages/WorkspacePage"));
const LibraryPage = lazy(() => import("./pages/LibraryPage"));
const ResultsPage = lazy(() => import("./pages/ResultsPage"));
const LiveCameraPage = lazy(() => import("./pages/LiveCameraPage"));

function RouteFallback() {
  return (
    <main className="route-fallback">
      <section className="panel route-fallback__panel">
        <div className="route-fallback__header">
          <SkeletonBlock className="skeleton-block--chip" />
          <SkeletonBlock className="skeleton-block--button" />
        </div>
        <SkeletonLines lines={2} className="route-fallback__lines" />
        <div className="route-fallback__metrics">
          <SkeletonMetric />
          <SkeletonMetric />
          <SkeletonMetric />
        </div>
      </section>
    </main>
  );
}

function ScrollToTop() {
  const location = useLocation();

  useEffect(() => {
    window.scrollTo(0, 0);
    AOS.refreshHard();
  }, [location.pathname]);

  return null;
}

function AOSBootstrap() {
  useEffect(() => {
    AOS.init({
      duration: 700,
      easing: "ease-out-cubic",
      offset: 18,
      once: true,
      mirror: false
    });
  }, []);

  return null;
}

export default function App() {
  return (
    <VisionPlayProvider>
      <div className="app-shell">
        <Suspense fallback={<RouteFallback />}>
          <AOSBootstrap />
          <ScrollToTop />
          <Routes>
            <Route path="/" element={<Navigate to="/workspace" replace />} />

            <Route path="/workspace" element={<AppShell />}>
              <Route index element={<WorkspacePage />} />
              <Route path="upload" element={<WorkspacePage />} />
              <Route path="live" element={<LiveCameraPage />} />
              <Route path="library" element={<LibraryPage />} />
              <Route path="library/upload" element={<LibraryPage />} />
              <Route path="results/:videoId" element={<ResultsPage />} />
              <Route path="results/:videoId/upload" element={<ResultsPage />} />
            </Route>

            <Route path="*" element={<Navigate to="/workspace" replace />} />
          </Routes>
        </Suspense>
      </div>
    </VisionPlayProvider>
  );
}
